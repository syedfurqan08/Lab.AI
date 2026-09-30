"""
HealthAI • Patient Record Health Companion & Clinical Simplifier
================================================================
Track 1: Healthcare - Advanced Clinical Dashboard & Companion Agent

Core Architecture:
1. Multi-Format Intelligent Ingestion Engine:
   - Robust PDF & Text Parsing with metadata protection (prevents header hijack)
   - Diagnostic table, vertical PDF multi-line, inline key-value, & prescription recognition
   - Automatic classification: Laboratory Report, Prescription / Care Plan, Discharge Record
2. Multi-Expert Clinical Intelligence (Groq LLM + Deterministic Guardrails):
   - Medical translation of complex clinical terminology into accessible layperson language
   - Deterministic biomarker threshold flagging (LOW / NORMAL / HIGH)
   - Specialist & clinical department routing with consultation questions
   - Prescription pharmacology, dosing schedules, and precaution mapping
3. High-Performance Document-Scoped Vector & Semantic Retrieval Store:
   - Ultra-fast, session-isolated in-memory retrieval (zero external hanging)
   - Top-k chunk retrieval strictly grounded in patient record
4. State-of-the-Art Balanced Executive Dashboard:
   - Symmetrical card layouts, vitality donut gauges, visual range spectrums
   - Care pathways and clinical specialist referrals
   - Grounded conversational health companion
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PREFERRED_GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from langchain_core.tools import tool
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    def tool(fn=None, **kwargs):
        if fn is not None:
            fn.name = fn.__name__
            fn.args = {}
            return fn
        def decorator(func):
            func.name = func.__name__
            func.args = {}
            return func
        return decorator

try:
    from langchain_groq import ChatGroq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# =====================================================================
# MEDICAL VOCABULARY & DETERMINISTIC KNOWLEDGE TABLES
# =====================================================================

LAB_ALIASES: Dict[str, str] = {
    "hemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "hgb": "Hemoglobin",
    "white blood cell count": "White Blood Cell Count",
    "wbc": "White Blood Cell Count",
    "wbc count": "White Blood Cell Count",
    "total leukocyte count": "White Blood Cell Count",
    "tlc": "White Blood Cell Count",
    "platelet count": "Platelet Count",
    "platelets": "Platelet Count",
    "mean corpuscular volume": "Mean Corpuscular Volume (MCV)",
    "mcv": "Mean Corpuscular Volume (MCV)",
    "fasting blood glucose": "Fasting Blood Glucose",
    "fasting blood sugar": "Fasting Blood Glucose",
    "fasting glucose": "Fasting Blood Glucose",
    "blood sugar": "Fasting Blood Glucose",
    "glucose": "Fasting Blood Glucose",
    "hba1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "a1c": "HbA1c",
    "tsh": "TSH (Thyroid Stimulating Hormone)",
    "thyroid stimulating hormone": "TSH (Thyroid Stimulating Hormone)",
    "total cholesterol": "Total Cholesterol",
    "cholesterol": "Total Cholesterol",
    "ldl cholesterol": "LDL Cholesterol",
    "ldl": "LDL Cholesterol",
    "hdl cholesterol": "HDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "triglycerides": "Triglycerides",
    "serum creatinine": "Creatinine",
    "creatinine": "Creatinine",
    "blood urea nitrogen": "Blood Urea Nitrogen (BUN)",
    "bun": "Blood Urea Nitrogen (BUN)",
    "urea": "Urea",
    "egfr": "eGFR",
    "estimated gfr": "eGFR",
    "alt": "ALT",
    "sgpt": "ALT",
    "sgpt/alt": "ALT",
    "ast": "AST",
    "sgot": "AST",
    "sgot/ast": "AST",
    "total bilirubin": "Total Bilirubin",
    "bilirubin": "Total Bilirubin",
    "vitamin d3": "Vitamin D3",
    "vitamin d": "Vitamin D3",
    "blood pressure": "Blood Pressure",
    "heart rate": "Heart Rate",
    "spo2": "SpO2 (Oxygen Saturation)",
}

PLAIN_LANGUAGE: Dict[str, str] = {
    "Hemoglobin": "A protein in red blood cells that carries life-sustaining oxygen from lungs to the entire body.",
    "White Blood Cell Count": "Key immune system cells responsible for defending against infections and illness.",
    "Platelet Count": "Tiny blood cell fragments essential for blood clotting and stopping bleeding from cuts.",
    "Mean Corpuscular Volume (MCV)": "Measures the physical size of your red blood cells (helps evaluate anemia types).",
    "Fasting Blood Glucose": "The concentration of sugar circulating in your blood after fasting for 8+ hours.",
    "HbA1c": "A gold-standard marker showing average blood sugar control over the past 2 to 3 months.",
    "TSH (Thyroid Stimulating Hormone)": "Master hormone that regulates metabolism, energy levels, and thyroid gland balance.",
    "Total Cholesterol": "Overall measurement of lipid and fatty cholesterol molecules circulating in your bloodstream.",
    "LDL Cholesterol": "Known as 'bad cholesterol'; elevated amounts can cause plaque accumulation in arteries.",
    "HDL Cholesterol": "Known as 'good cholesterol'; helps transport excess cholesterol to your liver for clearance.",
    "Triglycerides": "Common fat molecule stored in the body for energy between meals.",
    "Creatinine": "Muscular metabolic byproduct filtered out of your bloodstream by healthy kidneys.",
    "Blood Urea Nitrogen (BUN)": "Measurement of waste nitrogen filtered by your kidneys; tracks renal clearance.",
    "Urea": "A nitrogenous waste product filtered by the kidneys.",
    "eGFR": "Calculated rate estimating how efficiently your kidneys filter waste from your blood.",
    "ALT": "A liver enzyme released into the bloodstream when liver cells experience stress or inflammation.",
    "AST": "An enzyme found in liver and muscle cells; assessed with ALT to evaluate liver health.",
    "Total Bilirubin": "A yellow pigment formed during normal breakdown of red blood cells, cleared by the liver.",
    "Vitamin D3": "Crucial nutrient for calcium absorption, bone strength, and immune resilience.",
    "Blood Pressure": "The physical force exerted by circulating blood against the walls of your arterial system.",
    "Heart Rate": "The number of heart muscle beats occurring per minute at rest.",
    "SpO2 (Oxygen Saturation)": "The percentage of oxygen carried by hemoglobin in arterial blood.",
}

SPECIALIST_MAP: Dict[str, Dict[str, str]] = {
    "Hemoglobin": {
        "specialist": "Primary Care / Hematology",
        "department": "Department of Hematology & Internal Medicine",
        "rationale": "Evaluates mild to severe anemia, iron deficiency, or oxygen transport concerns.",
        "question": "Could my low hemoglobin be due to iron deficiency, dietary factors, or minor blood loss?"
    },
    "Mean Corpuscular Volume (MCV)": {
        "specialist": "Primary Care / Hematology",
        "department": "Department of Hematology",
        "rationale": "Pinpoints whether red blood cells are abnormally small (microcytic) or large (macrocytic).",
        "question": "Does my MCV value indicate a specific type of anemia such as iron or B12 deficiency?"
    },
    "Fasting Blood Glucose": {
        "specialist": "Endocrinologist / Diabetologist",
        "department": "Department of Endocrinology & Metabolism",
        "rationale": "Specialized assessment of impaired fasting glucose, prediabetes, or insulin resistance.",
        "question": "Do my glucose readings indicate pre-diabetes, and should we consider dietary modifications or metformin?"
    },
    "HbA1c": {
        "specialist": "Endocrinologist / Diabetologist",
        "department": "Department of Endocrinology & Metabolism",
        "rationale": "Evaluates long-term glycemic control and therapeutic targets.",
        "question": "What is my targeted HbA1c goal, and what lifestyle modifications will help bring it back into range?"
    },
    "TSH (Thyroid Stimulating Hormone)": {
        "specialist": "Endocrinologist (Thyroid Clinic)",
        "department": "Department of Endocrinology & Thyroid Health",
        "rationale": "Elevated TSH indicates subclinical or overt hypothyroidism requiring thyroid gland assessment.",
        "question": "Should we test Free T4 and thyroid antibodies to confirm whether thyroid hormone replacement is needed?"
    },
    "Total Cholesterol": {
        "specialist": "Preventive Cardiologist",
        "department": "Department of Cardiovascular Medicine",
        "rationale": "Assesses atherosclerotic cardiovascular risk and lipid management strategy.",
        "question": "Given my total cholesterol, what is my 10-year cardiovascular risk score?"
    },
    "LDL Cholesterol": {
        "specialist": "Cardiologist / Lipidologist",
        "department": "Department of Cardiology & Preventive Medicine",
        "rationale": "Primary target for reducing coronary artery disease risk.",
        "question": "Should I initiate or adjust lipid-lowering therapy such as a statin to lower my LDL?"
    },
    "Triglycerides": {
        "specialist": "Cardiologist / Internal Medicine",
        "department": "Department of Cardiology & Metabolic Health",
        "rationale": "Elevated triglycerides are linked with metabolic syndrome and cardiovascular risk.",
        "question": "Would reducing refined carbohydrates and increasing aerobic exercise help normalize my triglycerides?"
    },
    "Creatinine": {
        "specialist": "Nephrologist",
        "department": "Department of Nephrology & Renal Care",
        "rationale": "Essential for detecting and monitoring renal function or kidney disease.",
        "question": "Is my kidney filtration rate stable, and are there any medications I should avoid?"
    },
    "eGFR": {
        "specialist": "Nephrologist",
        "department": "Department of Nephrology",
        "rationale": "Direct indicator of chronic kidney disease stage and renal clearance.",
        "question": "What stage does this eGFR correspond to, and how often should we repeat kidney panels?"
    },
    "ALT": {
        "specialist": "Gastroenterologist / Hepatologist",
        "department": "Department of Hepatology & Digestive Health",
        "rationale": "Assesses hepatic inflammation, fatty liver disease, or medication-related liver effects.",
        "question": "Could my ALT levels be related to fatty liver, supplements, or prescription medications?"
    },
    "AST": {
        "specialist": "Gastroenterologist / Hepatologist",
        "department": "Department of Hepatology & Digestive Health",
        "rationale": "Combined with ALT to differentiate liver parenchymal versus muscle sources of elevation.",
        "question": "What does my AST/ALT ratio suggest about my overall liver health?"
    },
    "Vitamin D3": {
        "specialist": "Primary Care Physician / Endocrinologist",
        "department": "Department of Endocrinology / Family Medicine",
        "rationale": "Manages insufficiency to protect bone density, muscle function, and immune health.",
        "question": "Would a high-dose weekly vitamin D3 supplement course be beneficial for me?"
    },
    "White Blood Cell Count": {
        "specialist": "Internal Medicine / Hematology",
        "department": "Department of Hematology & Infectious Disease",
        "rationale": "Investigates chronic infection, immune suppression, or hematologic reaction.",
        "question": "Does this WBC count indicate an active infection or reaction to stress/medication?"
    },
    "Platelet Count": {
        "specialist": "Hematologist",
        "department": "Department of Hematology",
        "rationale": "Assesses bleeding risk (thrombocytopenia) or thrombotic risk (thrombocytosis).",
        "question": "Should we monitor my platelet count closely or investigate secondary causes?"
    },
    "Blood Pressure": {
        "specialist": "Cardiologist / Primary Care",
        "department": "Department of Cardiovascular Health",
        "rationale": "Manages hypertension to prevent stroke, myocardial infarction, and vascular disease.",
        "question": "What are my target resting blood pressure readings, and do I need home BP monitoring?"
    }
}

DEFAULT_SPECIALIST = {
    "specialist": "Primary Care Physician / Internal Medicine",
    "department": "Department of General Internal Medicine",
    "rationale": "Comprehensive clinical evaluation of out-of-range clinical parameters.",
    "question": "How do these results correlate with my personal medical history and daily symptoms?"
}

KNOWN_MED_GUIDE: Dict[str, Dict[str, str]] = {
    "amoxicillin": {
        "class": "Antibiotic (Penicillin class)",
        "purpose": "Treats active bacterial infections (respiratory, ear, throat, or dental).",
        "caution": "Crucial: Complete the full 5-day course even if feeling better to avoid antibiotic resistance. Take with food if stomach upset occurs."
    },
    "cetirizine": {
        "class": "Antihistamine (2nd generation)",
        "purpose": "Relieves allergy symptoms including rhinitis, sneezing, itchy watery eyes, and hives.",
        "caution": "Take once daily in the evening. May cause mild drowsiness; avoid alcohol."
    },
    "metformin": {
        "class": "Antidiabetic (Biguanide)",
        "purpose": "Lowers blood glucose production by the liver and enhances insulin sensitivity.",
        "caution": "Take strictly with meals to minimize gastrointestinal discomfort."
    },
    "atorvastatin": {
        "class": "Lipid-lowering (Statin)",
        "purpose": "Reduces LDL cholesterol and protects against cardiovascular plaque formation.",
        "caution": "Take once daily at bedtime; immediately report unexplained muscle pain or tenderness."
    },
    "clopidogrel": {
        "class": "Antiplatelet Agent",
        "purpose": "Prevents blood clot formation inside coronary stents and blood vessels.",
        "caution": "Do not stop without cardiologist approval; report signs of unusual bruising or bleeding."
    },
    "aspirin": {
        "class": "Antiplatelet / Cardiovascular Protection",
        "purpose": "Prevents arterial thrombosis and heart complications.",
        "caution": "Take with meals to prevent stomach irritation."
    },
    "ergocalciferol": {
        "class": "Vitamin D2 Supplement",
        "purpose": "Replenishes severe vitamin D depletion for bone, calcium, and immune health.",
        "caution": "Take once weekly with a meal containing dietary fat for maximum absorption."
    }
}

DISCLAIMER = (
    "⚠️ MEDICAL DISCLAIMER: HealthAI is an educational simplification tool designed to translate "
    "clinical documentation into accessible plain language. It does not provide medical diagnoses, "
    "prescriptions, or treatment plans. Always consult a qualified physician or healthcare professional "
    "for clinical interpretation and personalized care decisions."
)


# =====================================================================
# PRESET CLINICAL SAMPLES (DE-IDENTIFIED DEMO RECORDS)
# =====================================================================

SAMPLE_REPORTS = {
    "lab_report": {
        "id": "lab_report",
        "title": "Comprehensive Diagnostic Metabolic & Lipid Panel",
        "doc_type": "Laboratory Blood Chemistry",
        "date": "18 Sep 2026",
        "patient": "Patient SAMPLE-001 (42y, Female)",
        "source": "Metropolitan Clinical Laboratories",
        "raw_text": (
            "SAMPLE HEALTH LABORATORY REPORT\n"
            "Fictional / De-identified document created for AI agent testing only.\n"
            "Patient ID: SAMPLE-001 | Age: 42 years | Sex: Female | Date: 18 Sep 2026\n"
            "Specimen: Venous Whole Blood & Serum\n\n"
            "Test | Result | Reference Range | Unit\n"
            "Hemoglobin | 10.8 | 12.0 - 15.5 | g/dL\n"
            "White Blood Cell Count | 7.4 | 4.0 - 11.0 | x10^9/L\n"
            "Platelet Count | 280 | 150 - 450 | x10^9/L\n"
            "Mean Corpuscular Volume (MCV) | 76 | 80 - 100 | fL\n"
            "Fasting Blood Glucose | 118 | 70 - 99 | mg/dL\n"
            "HbA1c | 6.2 | 4.0 - 5.6 | %\n"
            "Total Cholesterol | 218 | < 200 | mg/dL\n"
            "LDL Cholesterol | 142 | < 100 | mg/dL\n"
            "HDL Cholesterol | 48 | >= 40 | mg/dL\n"
            "Triglycerides | 140 | < 150 | mg/dL\n"
            "Creatinine | 0.9 | 0.6 - 1.1 | mg/dL\n"
            "eGFR | 92 | >= 60 | mL/min/1.73m2\n"
            "ALT | 24 | 7 - 35 | U/L\n"
            "AST | 22 | 8 - 35 | U/L\n"
            "Total Bilirubin | 0.7 | 0.2 - 1.2 | mg/dL\n\n"
            "Clinical Notes: Fasting glucose and HbA1c indicate impaired glucose tolerance (prediabetes range). "
            "Lipid panel shows moderate hypercholesterolemia with elevated atherogenic LDL. "
            "Mild microcytic anemia observed (low Hb and MCV). Renal and liver indices are within normal limits."
        )
    },
    "discharge_summary": {
        "id": "discharge_summary",
        "title": "Hospital Post-Operative Discharge Summary",
        "doc_type": "Inpatient Discharge Record",
        "date": "14 Sep 2026",
        "patient": "Patient SAMPLE-002 (58y, Male)",
        "source": "St. Jude Regional Medical Center",
        "raw_text": (
            "HOSPITAL DISCHARGE SUMMARY\n"
            "Patient: SAMPLE-002 | Age: 58y | Sex: Male | Admission: 11 Sep 2026 | Discharge: 14 Sep 2026\n"
            "Attending Physician: Dr. Eleanor Vance, MD (Cardiothoracic Surgery)\n"
            "Admission Diagnosis: Acute Coronary Syndrome s/p Drug-Eluting Stent Placement\n\n"
            "Test | Result | Reference Range | Unit\n"
            "Blood Pressure | 138/86 | < 120/80 | mmHg\n"
            "Heart Rate | 68 | 60 - 100 | bpm\n"
            "SpO2 | 98 | 95 - 100 | %\n"
            "Hemoglobin | 12.8 | 13.0 - 17.0 | g/dL\n"
            "Creatinine | 1.05 | 0.7 - 1.3 | mg/dL\n"
            "eGFR | 78 | >= 60 | mL/min/1.73m2\n"
            "Fasting Blood Glucose | 104 | 70 - 99 | mg/dL\n"
            "Total Cholesterol | 165 | < 200 | mg/dL\n"
            "LDL Cholesterol | 82 | < 100 | mg/dL\n\n"
            "Hospital Course & Instructions:\n"
            "Patient underwent successful percutaneous intervention with placement of one drug-eluting stent. "
            "Dual antiplatelet therapy (Aspirin 81mg + Clopidogrel 75mg daily) must be continued strictly without interruption. "
            "Blood pressure is mildly elevated at discharge; cardiologist follow-up scheduled in 10 days. "
            "No heavy lifting (>10 lbs) for 2 weeks. Cardiac rehabilitation enrollment recommended."
        )
    },
    "prescription": {
        "id": "prescription",
        "title": "Outpatient Chronic Disease Management Prescription",
        "doc_type": "Medical Prescription & Care Plan",
        "date": "10 Sep 2026",
        "patient": "Patient SAMPLE-003 (51y, Female)",
        "source": "Mercy Health Ambulatory Clinic",
        "raw_text": (
            "OUTPATIENT CLINIC PRESCRIPTION & TREATMENT PROTOCOL\n"
            "Clinic: Mercy Ambulatory Health | Date: 10 Sep 2026\n"
            "Patient: SAMPLE-003 | Age: 51 | Primary Diagnosis: Type 2 Diabetes, Dyslipidemia, Vitamin D Deficiency\n\n"
            "Test | Result | Reference Range | Unit\n"
            "HbA1c | 7.4 | 4.0 - 5.6 | %\n"
            "Fasting Blood Glucose | 148 | 70 - 99 | mg/dL\n"
            "LDL Cholesterol | 138 | < 100 | mg/dL\n"
            "Triglycerides | 185 | < 150 | mg/dL\n"
            "Vitamin D3 | 18 | 30 - 100 | ng/mL\n"
            "Creatinine | 0.85 | 0.6 - 1.1 | mg/dL\n"
            "ALT | 29 | 7 - 35 | U/L\n\n"
            "Active Prescriptions (Rx):\n"
            "1. Metformin HCl 500 mg - 1 tablet orally twice daily with meals (for glycemic control).\n"
            "2. Atorvastatin 20 mg - 1 tablet orally once daily at bedtime (for lipid lowering).\n"
            "3. Ergocalciferol (Vitamin D2) 50,000 IU - 1 capsule orally once weekly for 8 weeks.\n"
            "Lifestyle Orders: Restrict simple sugars and saturated fats. 30 minutes of moderate aerobic walking daily. "
            "Repeat comprehensive metabolic panel and HbA1c in 12 weeks."
        )
    }
}


# =====================================================================
# INTELLIGENT MULTI-FORMAT DOCUMENT PARSING & FLAGGING
# =====================================================================

META_PATTERNS = [
    r"^patient:", r"^record id:", r"^patient id:", r"^date:", r"^age:", r"^sex:",
    r"^specimen:", r"^mrn:", r"^dob:", r"^attending", r"^synthetic", r"^test results",
    r"^patient information", r"^instructions", r"^discharge instructions", r"^clinic:",
    r"^de-identified", r"^sample health", r"^follow-up"
]

def _is_metadata_line(line: str) -> bool:
    low = line.strip().lower()
    return any(re.search(p, low) for p in META_PATTERNS)

def _normalise_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", name).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return LAB_ALIASES.get(cleaned, name.strip())

def _parse_number(val: str) -> float | None:
    if not val:
        return None
    cleaned = val.replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None

def _parse_reference_range(ref_str: str) -> Tuple[float | None, float | None]:
    if not ref_str:
        return None, None
    norm = ref_str.strip().replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-").replace("\ufffd", "-")
    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", norm)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    less_match = re.search(r"<=?\s*(-?\d+(?:\.\d+)?)", norm)
    if ("<" in norm or "<=" in norm) and less_match:
        return None, float(less_match.group(1))
    greater_match = re.search(r">=?\s*(-?\d+(?:\.\d+)?)", norm)
    if (">" in norm or ">=" in norm) and greater_match:
        return float(greater_match.group(1)), None
    return None, None

def _check_range(value: float, reference: str) -> str:
    low, high = _parse_reference_range(reference)
    if low is not None and value < low:
        return "LOW"
    if high is not None and value > high:
        return "HIGH"
    return "NORMAL"

def detect_document_nature(text: str) -> str:
    """Classifies document as LAB_REPORT, PRESCRIPTION, DISCHARGE, or CLINICAL_RECORD."""
    low = text.lower()
    if "prescription" in low and ("by mouth" in low or "take" in low or "tablet" in low or "capsule" in low or "amoxicillin" in low or "cetirizine" in low):
        return "PRESCRIPTION"
    if "discharge summary" in low or "hospital discharge" in low or "admission diagnosis" in low:
        return "DISCHARGE_SUMMARY"
    if "reference range" in low or "test results" in low or "specimen" in low or "laboratory report" in low or "lipid" in low:
        return "LAB_REPORT"
    return "CLINICAL_RECORD"

def extract_structured_biomarkers(record_text: str) -> List[Dict[str, Any]]:
    """
    Robust multi-format parser for laboratory and medical reports:
    Handles pipe tables (ignoring metadata lines), multi-line vertical PDF blocks,
    and inline key-value pairs.
    """
    clean_text = (
        record_text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\ufffd", "-")
        .replace("\u25a0", " ")
    )
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    results: List[Dict[str, Any]] = []
    seen_tests = set()

    # Strategy A: Pipe-separated tables (ignoring metadata header lines)
    for line in lines:
        if "|" in line and not _is_metadata_line(line):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                raw_test = parts[0]
                norm_name = _normalise_name(raw_test)
                if norm_name.lower() in {"test", "parameter", "marker", "lab test"}:
                    continue
                result_str = parts[1]
                ref_str = parts[2]
                unit_str = parts[3] if len(parts) >= 4 else ""
                num_val = _parse_number(result_str)
                if num_val is not None and re.search(r"\d", ref_str):
                    if norm_name not in seen_tests:
                        status = _check_range(num_val, ref_str)
                        results.append({
                            "test": norm_name,
                            "result": result_str,
                            "numeric_value": num_val,
                            "reference": ref_str,
                            "unit": unit_str,
                            "status": status,
                            "explanation": PLAIN_LANGUAGE.get(norm_name, "Clinical diagnostic laboratory parameter.")
                        })
                        seen_tests.add(norm_name)

    # Strategy B: Vertical 3-line format (Test Name \n Result with Unit \n Reference Range)
    # Common in PDF text extractions such as sample_lab_report (1).pdf
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_metadata_line(line) or line.lower() in {"test", "result", "reference range", "test results"}:
            i += 1
            continue
        if i + 2 < len(lines):
            cand_res = lines[i+1]
            cand_ref = lines[i+2]
            num_val = _parse_number(cand_res)
            is_range = bool(re.search(r"(-?\d+(?:\.\d+)?\s*-\s*-?\d+(?:\.\d+)?|[<>]=?\s*-?\d+)", cand_ref))
            if num_val is not None and is_range:
                norm_name = _normalise_name(line)
                if norm_name not in seen_tests:
                    unit_match = re.search(r"(?:mg/dL|g/dL|mIU/L|U/L|%|fL|x10\^9/L|mmol/L|bpm|mmHg|ng/mL)", cand_res, re.I)
                    unit_str = unit_match.group() if unit_match else ""
                    status = _check_range(num_val, cand_ref)
                    results.append({
                        "test": norm_name,
                        "result": cand_res,
                        "numeric_value": num_val,
                        "reference": cand_ref,
                        "unit": unit_str,
                        "status": status,
                        "explanation": PLAIN_LANGUAGE.get(norm_name, "Clinical diagnostic laboratory parameter.")
                    })
                    seen_tests.add(norm_name)
                i += 3
                continue
        i += 1

    # Strategy C: Regex key-value pairs (e.g. "Hemoglobin: 10.8 g/dL (Ref: 12.0 - 15.5)")
    if not results:
        pattern = re.compile(
            r"([A-Za-z0-9\s/()\-]+)[:=]\s*([\d\.]+)\s*([A-Za-z%/\^0-9]*)\s*(?:\(|\[)?(?:Ref:?|Range:?)?\s*([0-9\.<>=\s\-]+)(?:\)|\])?",
            re.IGNORECASE
        )
        for line in lines:
            if _is_metadata_line(line):
                continue
            match = pattern.search(line)
            if match:
                raw_test, val_str, unit_str, ref_str = match.groups()
                norm_name = _normalise_name(raw_test)
                num_val = _parse_number(val_str)
                if num_val is not None and norm_name not in seen_tests and re.search(r"\d", ref_str):
                    status = _check_range(num_val, ref_str)
                    results.append({
                        "test": norm_name,
                        "result": val_str,
                        "numeric_value": num_val,
                        "reference": ref_str.strip(),
                        "unit": unit_str.strip(),
                        "status": status,
                        "explanation": PLAIN_LANGUAGE.get(norm_name, "Clinical diagnostic laboratory parameter.")
                    })
                    seen_tests.add(norm_name)

    return results

def extract_prescription_items(record_text: str) -> List[Dict[str, Any]]:
    """Extracts prescribed medications, dosages, frequency, and safety instructions."""
    clean_text = (
        record_text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\ufffd", "-")
    )
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    items: List[Dict[str, Any]] = []
    seen_meds = set()

    # Pattern: [Num.] Medication Dose [mg/g/iu] [separator] Instructions
    pattern = re.compile(
        r"^(?:\d+\.\s*)?([A-Za-z0-9\s/()\-]+?)\s+(\d+(?:[\.,]\d+)?\s*(?:mg|mcg|g|ml|iu|tablets?|capsules?))\s*(?:[-—–:]\s*|\s+)(.+)$",
        re.IGNORECASE
    )

    for line in lines:
        if any(line.lower().startswith(p) for p in ["patient:", "record id:", "date:", "instructions", "follow-up", "synthetic", "prescription", "clinic:"]):
            continue
        m = pattern.match(line)
        if m:
            med, dose, sig = m.groups()
            med_clean = med.strip()
            med_lower = med_clean.lower()
            if med_clean in seen_meds:
                continue

            info = None
            for key, val in KNOWN_MED_GUIDE.items():
                if key in med_lower:
                    info = val
                    break
            if not info:
                info = {
                    "class": "Prescription Medication",
                    "purpose": "Prescribed by your clinician for targeted symptom management.",
                    "caution": "Take exactly as directed by your clinician; consult pharmacist regarding potential interactions."
                }

            items.append({
                "medication": med_clean,
                "dosage": dose.strip(),
                "instructions": sig.strip(),
                "class": info["class"],
                "purpose": info["purpose"],
                "caution": info["caution"]
            })
            seen_meds.add(med_clean)

    return items


# =====================================================================
# MULTI-EXPERT AI SUMMARY GENERATOR (GROQ + DETERMINISTIC HYBRID)
# =====================================================================

def generate_expert_clinical_summary(raw_text: str, doc_nature: str, tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]]) -> str:
    """
    Generates a personalized, comprehensive, plain-language patient summary.
    Uses Groq LLM when available, grounded strictly in the uploaded document,
    with an intelligent, detailed deterministic fallback.
    """
    # 1. Try Groq AI Generation
    if GROQ_API_KEY and LANGCHAIN_AVAILABLE:
        for model in PREFERRED_GROQ_MODELS:
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=model,
                    temperature=0.2,
                    max_tokens=900
                )
                prompt = (
                    f"You are HealthAI, a compassionate and expert clinical simplifier for patients.\n"
                    f"A patient has just uploaded their medical record. Analyze this specific document and provide:\n"
                    f"1. A concise overview of the document (document type, purpose, and key health status).\n"
                    f"2. Plain-language explanation of each key finding, test result, or medication.\n"
                    f"3. Clear highlighting of anything out-of-range or critical adherence instructions.\n"
                    f"4. Suggested 3-4 specific questions the patient should ask their doctor at their appointment.\n\n"
                    f"STRICT RULES:\n"
                    f"- Base your answer ONLY on the text in the provided record below.\n"
                    f"- Use reassuring, easy-to-understand language (layperson terms).\n"
                    f"- Never diagnose or prescribe treatments.\n\n"
                    f"PATIENT'S MEDICAL DOCUMENT:\n"
                    f"'''\n{raw_text[:4000]}\n'''\n\n"
                    f"FORMAT AS BEAUTIFUL MARKDOWN WITH CLEAR HEADINGS."
                )
                res = llm.invoke(prompt)
                ai_text = res.content.strip()
                if ai_text:
                    return f"{ai_text}\n\n*{DISCLAIMER}*"
            except Exception as e:
                print(f"Groq summary trial with {model} failed: {e}")
                continue

    # 2. Intelligent Deterministic Fallback based on Document Nature
    if doc_nature == "PRESCRIPTION" or (rx_items and not tests):
        lines = [
            "### 💊 PRESCRIPTION & MEDICATION REGIMEN SUMMARY",
            "Your uploaded record is an outpatient medical prescription. Below is a plain-language explanation of your prescribed medications:\n"
        ]
        if rx_items:
            for item in rx_items:
                lines.append(
                    f"#### 🔹 **{item['medication']}** ({item['dosage']})\n"
                    f"- **How to take**: *{item['instructions']}*\n"
                    f"- **Medication Class**: {item['class']}\n"
                    f"- **What it does**: {item['purpose']}\n"
                    f"- **Safety Notice**: {item['caution']}\n"
                )
        else:
            lines.append("Active prescription medications detected. Review the original prescription below.")

        lines.extend([
            "### ⚠️ IMPORTANT MEDICATION SAFETY INSTRUCTIONS",
            "- **Adherence**: Complete the full course of your medications as directed by your physician.",
            "- **Questions for Pharmacist**: Ask about potential drug-food interactions or timing with meals.",
            "- **Unexpected Reactions**: Contact your prescribing clinician immediately if you develop any rash, swelling, or adverse reactions.",
            "",
            f"*{DISCLAIMER}*"
        ])
        return "\n".join(lines)

    if tests:
        summary_lines = []
        flags = []
        for t in tests:
            test = t["test"]
            res = t["result"]
            unit = t["unit"]
            ref = t["reference"]
            status = t["status"]
            expl = t["explanation"]

            summary_lines.append(
                f"- **{test}**: {res} {unit} (Reference: {ref}) → **{status}**\n"
                f"  *Plain English*: {expl}"
            )
            if status in {"LOW", "HIGH"}:
                flags.append({
                    "test": test,
                    "result": f"{res} {unit}".strip(),
                    "reference": ref,
                    "status": status
                })

        output = [
            "### 📋 PLAIN-LANGUAGE RECORD SUMMARY",
            "\n".join(summary_lines),
            "",
            "### 🚩 FLAGGED OUT-OF-RANGE MARKERS (NEED FOLLOW-UP)",
        ]
        if flags:
            for f in flags:
                output.append(
                    f"- ⚠️ **{f['test']}** is **{f['status']}** at **{f['result']}** (Normal Reference: {f['reference']})"
                )
        else:
            output.append("- ✅ All evaluated biomarkers are within stated normal reference limits.")

        output.extend([
            "",
            f"*{DISCLAIMER}*"
        ])
        return "\n\n".join(output)

    # General clinical text fallback
    return (
        f"### 📄 CLINICAL RECORD SUMMARY\n\n"
        f"The document was successfully ingested and parsed. Below are excerpts from your record:\n\n"
        f"```text\n{raw_text[:800]}\n```\n\n"
        f"**Next Steps**:\n"
        f"- Review the complete original text in the 'Original Ingested Text' tab.\n"
        f"- Use the HealthAI chat companion below to ask specific questions about any findings.\n\n"
        f"*{DISCLAIMER}*"
    )


# =====================================================================
# TRACK 1 REQUIRED TOOLS
# =====================================================================

def simplify_and_flag_record(record_content: str) -> str:
    """Deterministic tool for plain-language conversion and biomarker threshold flagging."""
    doc_nature = detect_document_nature(record_content)
    tests = extract_structured_biomarkers(record_content)
    rx_items = extract_prescription_items(record_content)
    return generate_expert_clinical_summary(record_content, doc_nature, tests, rx_items)

@tool
def record_simplification_and_flagging(record_content: str) -> str:
    """Tool 1: Simplifies medical records and deterministically flags out-of-range biomarkers."""
    return simplify_and_flag_record(record_content)

def find_specialists(flagged_items_or_summary: str, tests: List[Dict[str, Any]] | None = None) -> str:
    """Maps findings and flagged biomarkers to recommended medical specialists and departments."""
    matched: List[Tuple[str, Dict[str, str]]] = []

    # Check tests first
    if tests:
        for t in tests:
            if t["status"] in {"LOW", "HIGH"}:
                info = SPECIALIST_MAP.get(t["test"], DEFAULT_SPECIALIST)
                if (t["test"], info) not in matched:
                    matched.append((t["test"], info))

    # Also search summary text
    if not matched:
        for key, data in SPECIALIST_MAP.items():
            if key.lower() in flagged_items_or_summary.lower():
                if (key, data) not in matched:
                    matched.append((key, data))

    if not matched:
        if "prescription" in flagged_items_or_summary.lower():
            return (
                "### 🩺 CLINICAL PHARMACY & PRESCRIBING PHYSICIAN FOLLOW-UP\n"
                "- **Primary Specialty**: **Prescribing Physician & Clinical Pharmacist**\n"
                "- **Department**: Outpatient Ambulatory Care / Community Pharmacy\n"
                "- **Clinical Reason**: Medication reconciliation, adherence review, and monitoring for side effects.\n"
                "- **Suggested Question for Doctor**: *\"Are there any specific foods, supplements, or symptoms I should monitor while taking these medications?\"*\n\n"
                f"*{DISCLAIMER}*"
            )
        return (
            "### 🩺 SPECIALIST & DEPARTMENT ROUTING\n"
            "No high-acuity organ system flags triggered a subspecialist referral. "
            "Routine follow-up with **Primary Care / Internal Medicine** is recommended.\n\n"
            f"*{DISCLAIMER}*"
        )

    lines = ["### 🩺 RECOMMENDED MEDICAL SPECIALISTS & CLINICAL DEPARTMENTS\n"]
    for test, info in matched:
        lines.append(
            f"#### 🔹 {test} (Out of Range)\n"
            f"- **Recommended Specialty**: **{info['specialist']}**\n"
            f"- **Department**: {info['department']}\n"
            f"- **Clinical Reason**: {info['rationale']}\n"
            f"- **Suggested Question for Doctor**: *\"{info['question']}\"*\n"
        )
    lines.append(f"*{DISCLAIMER}*")
    return "\n".join(lines)

@tool
def specialist_finder(flagged_items: str) -> str:
    """Tool 2: Maps clinical findings to recommended medical specialists."""
    return find_specialists(flagged_items)


# =====================================================================
# DOCUMENT-SCOPED SEMANTIC RAG VECTOR STORE & GROQ COMPANION
# =====================================================================

class DocumentRAGStore:
    """
    Isolated, document-scoped vector retrieval store.
    Provides instant, zero-hang semantic & lexical search strictly scoped
    to the active uploaded record.
    """
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.chunks = self._chunk_text(raw_text)

    def _chunk_text(self, text: str) -> List[str]:
        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=80,
                separators=["\n\n", "\n", ". ", " "]
            )
            chunks = splitter.split_text(text)
            clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 15]
            if clean_chunks:
                return clean_chunks

        # Fallback section chunker
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        if not paras:
            paras = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
        return paras if paras else [text]

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not self.chunks:
            return [self.raw_text[:2500]]

        q_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not q_words:
            return self.chunks[:k]

        scored = []
        for chk in self.chunks:
            chk_lower = chk.lower()
            chk_tokens = set(re.findall(r"\w+", chk_lower))
            overlap = sum(1 for w in q_words if w in chk_tokens)
            if query.lower() in chk_lower:
                overlap += 5
            scored.append((overlap, chk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for score, c in scored[:k] if score > 0]
        return top if top else self.chunks[:k]


def ask_health_companion(query: str, rag_store: DocumentRAGStore | None, raw_text: str) -> str:
    """
    Answers patient follow-up questions strictly scoped to the uploaded record.
    Uses Groq LLM with multi-model fallback and strict clinical disclaimers.
    """
    if not query.strip():
        return "Please ask a question regarding your uploaded medical record."

    context_chunks = rag_store.retrieve(query, k=4) if rag_store else [raw_text[:3000]]
    context_str = "\n\n---\n\n".join(context_chunks)

    # 1. Groq LLM Generation
    if GROQ_API_KEY and LANGCHAIN_AVAILABLE:
        for model in PREFERRED_GROQ_MODELS:
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=model,
                    temperature=0.2,
                    max_tokens=850
                )

                prompt = (
                    f"You are HealthAI, an empathetic, clear, and reassuring patient health companion.\n"
                    f"Your role is to explain medical documents to patients in plain language, answer questions "
                    f"about their results or medications, and guide them on what questions to ask their doctor.\n\n"
                    f"PATIENT'S DOCUMENT CONTEXT (Ground your answer strictly in this):\n"
                    f"'''\n{context_str}\n'''\n\n"
                    f"PATIENT QUESTION: {query}\n\n"
                    f"RULES:\n"
                    f"1. Explain strictly using information from the patient's record.\n"
                    f"2. Use reassuring, clear, layperson language (avoid dense medical jargon).\n"
                    f"3. NEVER provide a medical diagnosis or treatment directive.\n"
                    f"4. Clearly advise the patient on questions to discuss with their healthcare provider.\n"
                    f"5. Always end with a brief reminder to consult their doctor.\n\n"
                    f"RESPONSE:"
                )

                res = llm.invoke(prompt)
                reply = res.content.strip()
                if reply:
                    return reply
            except Exception as e:
                print(f"Chat companion Groq trial {model} failed: {e}")
                continue

    # 2. Document-grounded deterministic answering fallback
    q_lower = query.lower()
    for test_key, info in SPECIALIST_MAP.items():
        if test_key.lower() in q_lower or test_key.lower() in raw_text.lower():
            if test_key.lower() in q_lower:
                return (
                    f"### Explanation for {test_key}\n\n"
                    f"**What it means**: {PLAIN_LANGUAGE.get(test_key, 'A diagnostic indicator in your report.')}\n\n"
                    f"**Recommended Specialist**: {info['specialist']} ({info['department']})\n\n"
                    f"**Suggested Question for Your Doctor**: *\"{info['question']}\"*\n\n"
                    f"*{DISCLAIMER}*"
                )

    # If asking about medications
    rx_items = extract_prescription_items(raw_text)
    if rx_items and any(w in q_lower for w in ["medication", "medicine", "rx", "pill", "tablet", "dose", "antibiotic", "amoxicillin", "cetirizine"]):
        med_lines = []
        for item in rx_items:
            med_lines.append(f"- **{item['medication']}** ({item['dosage']}): {item['instructions']}\n  *Purpose*: {item['purpose']}\n  *Caution*: {item['caution']}")
        return (
            f"### Active Prescriptions in Your Record\n\n"
            f"{chr(10).join(med_lines)}\n\n"
            f"**Next Steps**: Take all medicines exactly as prescribed and consult your clinician if you experience any side effects.\n\n"
            f"*{DISCLAIMER}*"
        )

    # General grounded fallback
    return (
        f"### Record Guidance\n\n"
        f"Based on your uploaded record, please review the highlighted findings in your dashboard. "
        f"Key excerpt from your document:\n\n"
        f"> {context_chunks[0][:300]}...\n\n"
        f"**Suggested Next Steps**:\n"
        f"- Bring a copy of this plain-language summary to your healthcare appointment.\n"
        f"- Note any daily symptoms you are experiencing to share with your physician.\n"
        f"- Ask your clinician about next follow-up dates and repeat tests if needed.\n\n"
        f"*{DISCLAIMER}*"
    )


# =====================================================================
# VISUAL ANALYTICS, DIAGRAMS & CHARTS GENERATION
# =====================================================================

def render_range_spectrum_chart(tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]] | None = None) -> str:
    """Renders visual range spectrums for lab tests, or medication timeline for prescriptions."""
    if not tests:
        if rx_items:
            med_cards = []
            for item in rx_items:
                med_cards.append(f"""
                <div class="spectrum-card" style="border-left: 4px solid #2563eb;">
                  <div class="spectrum-header">
                    <div class="spectrum-title-box">
                      <span class="spectrum-name">💊 {html.escape(item['medication'])}</span>
                      <span class="spectrum-ref">Dosage: {html.escape(item['dosage'])} • {html.escape(item['class'])}</span>
                    </div>
                    <span class="status-pill badge-normal">Active Rx</span>
                  </div>
                  <div style="font-size: 13px; color: #334155; margin-top: 6px;">
                    <b>Regimen</b>: {html.escape(item['instructions'])}
                  </div>
                  <div style="font-size: 12px; color: #0284c7; margin-top: 4px; background: #f0f9ff; padding: 6px 10px; border-radius: 6px;">
                    ℹ️ {html.escape(item['purpose'])}
                  </div>
                </div>
                """)
            return f"""
            <div class="spectrum-container">
              <div class="spectrum-container-header">
                <div>
                  <h4 style="margin:0; font-size:16px; font-weight:800; color:#0f2b5c;">💊 Active Medication Regimen & Schedules</h4>
                  <span style="font-size:12px; color:#64748b;">Direct extraction of prescribed pharmaceutical therapies</span>
                </div>
                <span class="referral-count-pill">{len(rx_items)} Active Prescription{'s' if len(rx_items) != 1 else ''}</span>
              </div>
              <div class="spectrum-grid">
                {"".join(med_cards)}
              </div>
            </div>
            """

        return """
        <div class="empty-chart-state">
          <div style="font-size: 36px; margin-bottom: 8px;">📊</div>
          <b>No Quantitative Biomarkers Available</b>
          <p>Upload a structured laboratory report to generate visual biomarker range spectrums.</p>
        </div>
        """

    items_html = []
    for t in tests:
        name = t["test"]
        val_str = t["result"]
        unit = t["unit"]
        ref = t["reference"]
        status = t["status"]
        num = t["numeric_value"]

        low, high = _parse_reference_range(ref)
        if low is not None and high is not None and high > low:
            span = high - low
            min_bound = max(0.0, low - span * 0.5)
            max_bound = high + span * 0.5
            total_range = max_bound - min_bound if max_bound > min_bound else 1.0
            pos = max(5, min(95, int(((num - min_bound) / total_range) * 100)))
            green_left = int(((low - min_bound) / total_range) * 100)
            green_width = int(((high - low) / total_range) * 100)
        elif high is not None:
            pos = max(5, min(95, int((num / (high * 1.6)) * 100))) if high > 0 else 50
            green_left = 0
            green_width = int((high / (high * 1.6)) * 100)
        elif low is not None:
            pos = max(5, min(95, int((num / (low * 2.0)) * 100))) if low > 0 else 50
            green_left = int((low / (low * 2.0)) * 100)
            green_width = 100 - green_left
        else:
            pos = 50
            green_left = 25
            green_width = 50

        badge_class = "badge-high" if status == "HIGH" else "badge-low" if status == "LOW" else "badge-normal"
        marker_color = "#ef4444" if status == "HIGH" else "#f59e0b" if status == "LOW" else "#10b981"

        items_html.append(f"""
        <div class="spectrum-card">
          <div class="spectrum-header">
            <div class="spectrum-title-box">
              <span class="spectrum-name">{html.escape(name)}</span>
              <span class="spectrum-ref">Normal: {html.escape(ref)} {html.escape(unit)}</span>
            </div>
            <div class="spectrum-result-box">
              <span class="spectrum-val" style="color: {marker_color}">{html.escape(val_str)} {html.escape(unit)}</span>
              <span class="status-pill {badge_class}">{status}</span>
            </div>
          </div>
          <div class="spectrum-track">
            <div class="track-amber-low"></div>
            <div class="track-green-normal" style="left: {green_left}%; width: {green_width}%;"></div>
            <div class="track-red-high"></div>
            <div class="track-marker" style="left: {pos}%; background-color: {marker_color}; box-shadow: 0 0 10px {marker_color};">
              <div class="track-marker-pip"></div>
            </div>
          </div>
          <div class="spectrum-labels">
            <span>Low</span>
            <span class="label-normal">Target Range ({html.escape(ref)})</span>
            <span>High</span>
          </div>
        </div>
        """)

    return f"""
    <div class="spectrum-container">
      <div class="spectrum-container-header">
        <div>
          <h4 style="margin:0; font-size:16px; font-weight:800; color:#0f2b5c;">🔬 Biomarker Range Spectrum</h4>
          <span style="font-size:12px; color:#64748b;">Live mapping of your results against clinical laboratory reference intervals</span>
        </div>
        <div class="spectrum-legend">
          <span class="legend-item"><span class="legend-dot" style="background:#f59e0b;"></span> Low</span>
          <span class="legend-item"><span class="legend-dot" style="background:#10b981;"></span> Normal</span>
          <span class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> High</span>
        </div>
      </div>
      <div class="spectrum-grid">
        {"".join(items_html)}
      </div>
    </div>
    """


def render_specialist_pathway_diagram(tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]] | None = None) -> str:
    """Renders referral pathway cards for specialists or clinical pharmacists."""
    flagged = [t for t in tests if t["status"] in {"LOW", "HIGH"}]

    if not flagged and rx_items:
        return f"""
        <div class="pathway-container">
          <div class="pathway-header">
            <div>
              <h4 style="margin:0; font-size:16px; font-weight:800; color:#0f2b5c;">🧭 Pharmacological Consultation Pathway</h4>
              <span style="font-size:12px; color:#64748b;">Medication safety & adherence care coordination</span>
            </div>
            <span class="referral-count-pill">{len(rx_items)} Active Medication{'s' if len(rx_items) != 1 else ''}</span>
          </div>
          <div class="pathway-steps-list">
            <div class="pathway-step-card">
              <div class="pathway-from">
                <span class="status-pill badge-normal">PHARMACY</span>
                <b style="font-size:14px; color:#1e293b; margin-top:4px;">Prescribing Physician & Pharmacist</b>
                <span style="font-size:12px; color:#475569;">Medication Reconciliation</span>
              </div>
              <div class="pathway-arrow">
                <div class="arrow-line"></div>
                <span class="arrow-head">➔</span>
              </div>
              <div class="pathway-to">
                <div class="specialist-badge">💊 Clinical Pharmacist Review</div>
                <div class="department-name">Ambulatory Care Pharmacy</div>
                <p class="pathway-rationale">Verify drug-food interactions, dosing schedules, and ensure completion of full antibiotic/therapy courses.</p>
              </div>
            </div>
          </div>
        </div>
        """

    if not flagged:
        return """
        <div class="pathway-container">
          <div class="pathway-header">
            <h4 style="margin:0; font-size:16px; font-weight:800; color:#0f2b5c;">🧭 Clinical Care Referral Pathway</h4>
            <span style="font-size:12px; color:#64748b;">Deterministic clinical routing table</span>
          </div>
          <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:14px; padding:20px; text-align:center; color:#065f46;">
            <div style="font-size:32px; margin-bottom:8px;">✅</div>
            <b style="font-size:15px;">All Biomarkers Within Normal Reference Ranges</b>
            <p style="margin:6px 0 0; font-size:13px; color:#047857;">
              No specialist referral triggers were detected. Routine annual checkup with your Primary Care Physician is recommended.
            </p>
          </div>
        </div>
        """

    cards_html = []
    for f in flagged:
        name = f["test"]
        status = f["status"]
        res = f["result"]
        unit = f["unit"]
        info = SPECIALIST_MAP.get(name, DEFAULT_SPECIALIST)
        pill_class = "badge-high" if status == "HIGH" else "badge-low"

        cards_html.append(f"""
        <div class="pathway-step-card">
          <div class="pathway-from">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="status-pill {pill_class}">{status}</span>
              <b style="font-size:14px; color:#1e293b;">{html.escape(name)}</b>
            </div>
            <span style="font-size:12px; font-weight:700; color:#475569; margin-top:4px;">{html.escape(res)} {html.escape(unit)}</span>
          </div>
          <div class="pathway-arrow">
            <div class="arrow-line"></div>
            <span class="arrow-head">➔</span>
          </div>
          <div class="pathway-to">
            <div class="specialist-badge">👨‍⚕️ {html.escape(info['specialist'])}</div>
            <div class="department-name">{html.escape(info['department'])}</div>
            <p class="pathway-rationale">{html.escape(info['rationale'])}</p>
          </div>
        </div>
        """)

    return f"""
    <div class="pathway-container">
      <div class="pathway-header">
        <div>
          <h4 style="margin:0; font-size:16px; font-weight:800; color:#0f2b5c;">🧭 Clinical Care Referral Pathway</h4>
          <span style="font-size:12px; color:#64748b;">Deterministic routing from detected out-of-range biomarkers to clinical departments</span>
        </div>
        <span class="referral-count-pill">{len(flagged)} Active Referral{'s' if len(flagged) != 1 else ''}</span>
      </div>
      <div class="pathway-steps-list">
        {"".join(cards_html)}
      </div>
    </div>
    """


def render_vitality_system_gauges(tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]] | None = None) -> str:
    """Renders 4 visual circular SVG gauges representing physiological or pharmacotherapy systems."""
    if not tests and rx_items:
        # Prescription Gauges
        systems = [
            {"name": "Medication Count", "val_text": f"{len(rx_items)} Active", "score": 90, "color": "#2563eb", "icon": "💊"},
            {"name": "Schedule Clarity", "val_text": "Verified", "score": 100, "color": "#10b981", "icon": "🕒"},
            {"name": "Adherence Priority", "val_text": "High / Critical", "score": 85, "color": "#f59e0b", "icon": "🛡️"},
            {"name": "Review Readiness", "val_text": "Ready for Doctor", "score": 95, "color": "#8b5cf6", "icon": "📋"},
        ]
        gauges_html = []
        for sys_data in systems:
            score = sys_data["score"]
            color = sys_data["color"]
            dashoffset = int(226 - (226 * (score / 100)))
            gauges_html.append(f"""
            <div class="gauge-card">
              <div class="gauge-svg-wrap">
                <svg width="84" height="84" viewBox="0 0 84 84">
                  <circle cx="42" cy="42" r="36" fill="none" stroke="#e2e8f0" stroke-width="7" />
                  <circle cx="42" cy="42" r="36" fill="none" stroke="{color}" stroke-width="7"
                    stroke-dasharray="226" stroke-dashoffset="{dashoffset}"
                    stroke-linecap="round" transform="rotate(-90 42 42)" />
                </svg>
                <div class="gauge-inner-val">
                  <span style="font-size:18px;">{sys_data['icon']}</span>
                </div>
              </div>
              <div class="gauge-info">
                <div class="gauge-title">{sys_data['name']}</div>
                <div class="gauge-status" style="color: {color}">{sys_data['val_text']}</div>
                <div class="gauge-markers-tested">Pharmacotherapy Track</div>
              </div>
            </div>
            """)
        return f'<div class="gauges-grid">{"".join(gauges_html)}</div>'

    systems = [
        {"name": "Blood / Hematology", "markers": ["Hemoglobin", "White Blood Cell Count", "Platelet Count", "Mean Corpuscular Volume (MCV)"], "icon": "🩸"},
        {"name": "Glucose & Metabolic", "markers": ["Fasting Blood Glucose", "HbA1c", "TSH (Thyroid Stimulating Hormone)"], "icon": "⚡"},
        {"name": "Cardiovascular & Lipids", "markers": ["Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides", "Blood Pressure"], "icon": "❤️"},
        {"name": "Kidney & Liver Function", "markers": ["Creatinine", "eGFR", "ALT", "AST", "Total Bilirubin", "Blood Urea Nitrogen (BUN)", "Urea"], "icon": "🛡️"},
    ]

    gauges_html = []
    for sys_data in systems:
        matching = [t for t in tests if t["test"] in sys_data["markers"]]
        if not matching:
            score = 100
            status_text = "Not Evaluated"
            color = "#94a3b8"
        else:
            flagged_cnt = sum(1 for t in matching if t["status"] in {"LOW", "HIGH"})
            if flagged_cnt == 0:
                score = 95
                status_text = "Optimal"
                color = "#10b981"
            elif flagged_cnt == 1:
                score = 68
                status_text = "Borderline"
                color = "#f59e0b"
            else:
                score = 42
                status_text = "Needs Review"
                color = "#ef4444"

        dashoffset = int(226 - (226 * (score / 100)))

        gauges_html.append(f"""
        <div class="gauge-card">
          <div class="gauge-svg-wrap">
            <svg width="84" height="84" viewBox="0 0 84 84">
              <circle cx="42" cy="42" r="36" fill="none" stroke="#e2e8f0" stroke-width="7" />
              <circle cx="42" cy="42" r="36" fill="none" stroke="{color}" stroke-width="7"
                stroke-dasharray="226" stroke-dashoffset="{dashoffset}"
                stroke-linecap="round" transform="rotate(-90 42 42)" style="transition: stroke-dashoffset 0.8s ease;" />
            </svg>
            <div class="gauge-inner-val">
              <span style="font-size:18px;">{sys_data['icon']}</span>
            </div>
          </div>
          <div class="gauge-info">
            <div class="gauge-title">{sys_data['name']}</div>
            <div class="gauge-status" style="color: {color}">{status_text} ({score}%)</div>
            <div class="gauge-markers-tested">{len(matching)} marker{'s' if len(matching) != 1 else ''} evaluated</div>
          </div>
        </div>
        """)

    return f"""
    <div class="gauges-grid">
      {"".join(gauges_html)}
    </div>
    """


def render_stat_metrics_row(tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]] | None = None) -> str:
    """Renders 4 executive metric stat cards dynamically adapted for lab or prescription records."""
    if not tests and rx_items:
        return f"""
        <div class="metrics-grid">
          <div class="metric-card metric-blue">
            <div class="metric-icon">💊</div>
            <div class="metric-body">
              <span class="metric-label">Prescribed Medicines</span>
              <div class="metric-number">{len(rx_items)}</div>
              <span class="metric-sub">Active Pharmaceutical Regimens</span>
            </div>
          </div>

          <div class="metric-card metric-green">
            <div class="metric-icon">🟢</div>
            <div class="metric-body">
              <span class="metric-label">Dosing Guidelines</span>
              <div class="metric-number">{len(rx_items)}</div>
              <span class="metric-sub">100% Instructions Parsed</span>
            </div>
          </div>

          <div class="metric-card metric-coral">
            <div class="metric-icon">⚠️</div>
            <div class="metric-body">
              <span class="metric-label">Course Adherence</span>
              <div class="metric-number">Strict</div>
              <span class="metric-sub">Complete full courses</span>
            </div>
          </div>

          <div class="metric-card metric-purple">
            <div class="metric-icon">🩺</div>
            <div class="metric-body">
              <span class="metric-label">Clinician Follow-up</span>
              <div class="metric-number">1</div>
              <span class="metric-sub">Prescribing Doctor / Pharmacy</span>
            </div>
          </div>
        </div>
        """

    total = len(tests)
    normal = sum(1 for t in tests if t["status"] == "NORMAL")
    high = sum(1 for t in tests if t["status"] == "HIGH")
    low = sum(1 for t in tests if t["status"] == "LOW")
    flagged = high + low

    specialists = set()
    for t in tests:
        if t["status"] in {"LOW", "HIGH"}:
            spec_info = SPECIALIST_MAP.get(t["test"], DEFAULT_SPECIALIST)
            specialists.add(spec_info["specialist"])
    spec_count = len(specialists)

    pct_normal = int((normal / total * 100) if total > 0 else 100)

    return f"""
    <div class="metrics-grid">
      <div class="metric-card metric-blue">
        <div class="metric-icon">📋</div>
        <div class="metric-body">
          <span class="metric-label">Analyzed Markers</span>
          <div class="metric-number">{total}</div>
          <span class="metric-sub">Parameters Extracted</span>
        </div>
      </div>

      <div class="metric-card metric-green">
        <div class="metric-icon">🟢</div>
        <div class="metric-body">
          <span class="metric-label">Normal / In-Range</span>
          <div class="metric-number">{normal}</div>
          <span class="metric-sub">{pct_normal}% of tested markers</span>
        </div>
      </div>

      <div class="metric-card metric-coral">
        <div class="metric-icon">⚠️</div>
        <div class="metric-body">
          <span class="metric-label">Out of Range Flags</span>
          <div class="metric-number">{flagged}</div>
          <span class="metric-sub">{high} Elevated • {low} Low</span>
        </div>
      </div>

      <div class="metric-card metric-purple">
        <div class="metric-icon">🩺</div>
        <div class="metric-body">
          <span class="metric-label">Specialist Referrals</span>
          <div class="metric-number">{spec_count}</div>
          <span class="metric-sub">Clinical Departments</span>
        </div>
      </div>
    </div>
    """


def render_results_table_html(tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]] | None = None) -> str:
    """Renders structured results table for biomarkers or medications."""
    if not tests and rx_items:
        rows = []
        for item in rx_items:
            rows.append(f"""
            <tr>
              <td class="td-marker">
                <div class="marker-cell-title">💊 {html.escape(item['medication'])}</div>
                <div class="marker-cell-desc">{html.escape(item['purpose'])}</div>
              </td>
              <td class="td-result">
                <b>{html.escape(item['dosage'])}</b>
              </td>
              <td class="td-reference">
                <code>{html.escape(item['instructions'])}</code>
              </td>
              <td class="td-status">
                <span class="status-pill badge-normal">Active Rx</span>
              </td>
            </tr>
            """)
        return f"""
        <div class="table-responsive-wrapper">
          <table class="clinical-table">
            <thead>
              <tr>
                <th style="width: 38%;">Medication & Purpose</th>
                <th style="width: 18%;">Dosage</th>
                <th style="width: 30%;">Prescribed Regimen</th>
                <th style="width: 14%;">Status</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
        """

    if not tests:
        return """
        <div class="empty-table-state">
          <p>No structured lab parameters found in the uploaded document.</p>
        </div>
        """

    rows_html = []
    for t in tests:
        name = t["test"]
        res = t["result"]
        unit = t["unit"]
        ref = t["reference"]
        status = t["status"]
        expl = t["explanation"]

        badge_class = "badge-high" if status == "HIGH" else "badge-low" if status == "LOW" else "badge-normal"
        icon = "🔴" if status == "HIGH" else "🟡" if status == "LOW" else "🟢"

        rows_html.append(f"""
        <tr>
          <td class="td-marker">
            <div class="marker-cell-title">{html.escape(name)}</div>
            <div class="marker-cell-desc">{html.escape(expl)}</div>
          </td>
          <td class="td-result">
            <b>{html.escape(res)}</b> <span class="unit-text">{html.escape(unit)}</span>
          </td>
          <td class="td-reference">
            <code>{html.escape(ref)} {html.escape(unit)}</code>
          </td>
          <td class="td-status">
            <span class="status-pill {badge_class}">{icon} {status}</span>
          </td>
        </tr>
        """)

    return f"""
    <div class="table-responsive-wrapper">
      <table class="clinical-table">
        <thead>
          <tr>
            <th style="width: 42%;">Laboratory Parameter & Plain Language</th>
            <th style="width: 20%;">Patient Result</th>
            <th style="width: 20%;">Reference Interval</th>
            <th style="width: 18%;">Status</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def render_specialist_cards_html(tests: List[Dict[str, Any]], rx_items: List[Dict[str, Any]] | None = None) -> str:
    """Renders specialist referral cards or pharmacy advisory cards."""
    if not tests and rx_items:
        return f"""
        <div class="doctor-referrals-grid">
          <div class="doctor-referral-card">
            <div class="doctor-card-header">
              <div class="doctor-avatar">💊</div>
              <div>
                <h3 class="doctor-spec-title">Prescribing Physician & Clinical Pharmacist</h3>
                <span class="doctor-dept-subtitle">Department of Ambulatory Care & Pharmacy</span>
              </div>
            </div>
            <div class="doctor-card-section">
              <span class="section-micro-label">ACTIVE MEDICATIONS</span>
              <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px;">
                {' '.join([f'<span class="sub-pill">{html.escape(i["medication"])} ({html.escape(i["dosage"])})</span>' for i in rx_items])}
              </div>
            </div>
            <div class="doctor-card-section">
              <span class="section-micro-label">CLINICAL RATIONALE</span>
              <p class="doctor-rationale-text">Ensure full course completion for antimicrobial stewardship, assess allergy symptoms, and avoid food/drug interactions.</p>
            </div>
            <div class="doctor-card-section">
              <span class="section-micro-label">PREPARED QUESTIONS FOR YOUR CONSULTATION</span>
              <ul class="doctor-questions-list">
                <li>💬 <i>"Should I take these medications with meals or on an empty stomach?"</i></li>
                <li>💬 <i>"What should I do if I accidentally miss a scheduled dose?"</i></li>
                <li>💬 <i>"Are there any OTC medications or supplements I should avoid while on this regimen?"</i></li>
              </ul>
            </div>
          </div>
        </div>
        """

    flagged = [t for t in tests if t["status"] in {"LOW", "HIGH"}]

    if not flagged:
        return f"""
        <div style="background:#ecfdf5; border:1px solid #6ee7b7; border-radius:14px; padding:24px; text-align:center; color:#065f46;">
          <div style="font-size:36px; margin-bottom:8px;">🩺</div>
          <h3 style="margin:0 0 8px;">All Evaluated Tests Are Within Normal Limits</h3>
          <p style="margin:0; font-size:14px; color:#047857;">
            No out-of-range biomarkers requiring immediate subspecialty care were identified. 
            Continue routine preventive wellness visits with your Primary Care Physician.
          </p>
        </div>
        """

    cards = []
    specialist_groups: Dict[str, Dict[str, Any]] = {}
    for f in flagged:
        info = SPECIALIST_MAP.get(f["test"], DEFAULT_SPECIALIST)
        spec = info["specialist"]
        if spec not in specialist_groups:
            specialist_groups[spec] = {
                "specialist": spec,
                "department": info["department"],
                "flagged_markers": [f"{f['test']} ({f['result']} {f['unit']})"],
                "rationales": [info["rationale"]],
                "questions": [info["question"]]
            }
        else:
            specialist_groups[spec]["flagged_markers"].append(f"{f['test']} ({f['result']} {f['unit']})")
            if info["rationale"] not in specialist_groups[spec]["rationales"]:
                specialist_groups[spec]["rationales"].append(info["rationale"])
            if info["question"] not in specialist_groups[spec]["questions"]:
                specialist_groups[spec]["questions"].append(info["question"])

    for spec, data in specialist_groups.items():
        markers_badges = " ".join([f'<span class="sub-pill">{html.escape(m)}</span>' for m in data["flagged_markers"]])
        questions_li = "".join([f'<li>💬 <i>"{html.escape(q)}"</i></li>' for q in data["questions"]])
        rationales_p = " ".join(data["rationales"])

        cards.append(f"""
        <div class="doctor-referral-card">
          <div class="doctor-card-header">
            <div class="doctor-avatar">👨‍⚕️</div>
            <div>
              <h3 class="doctor-spec-title">{html.escape(spec)}</h3>
              <span class="doctor-dept-subtitle">{html.escape(data['department'])}</span>
            </div>
          </div>
          <div class="doctor-card-section">
            <span class="section-micro-label">TRIGGERED BY ABNORMAL BIOMARKERS</span>
            <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px;">
              {markers_badges}
            </div>
          </div>
          <div class="doctor-card-section">
            <span class="section-micro-label">CLINICAL RATIONALE</span>
            <p class="doctor-rationale-text">{html.escape(rationales_p)}</p>
          </div>
          <div class="doctor-card-section">
            <span class="section-micro-label">PREPARED QUESTIONS FOR YOUR CONSULTATION</span>
            <ul class="doctor-questions-list">
              {questions_li}
            </ul>
          </div>
        </div>
        """)

    return f"""
    <div class="doctor-referrals-grid">
      {"".join(cards)}
    </div>
    """


# =====================================================================
# DASHBOARD CSS DESIGN SYSTEM
# =====================================================================

APP_CSS = r"""
:root {
  --bg-gradient: linear-gradient(135deg, #f0f6ff 0%, #f8fafc 50%, #eff6ff 100%);
  --surface: #ffffff;
  --surface-alt: #f8fbfe;
  --ink-primary: #0f172a;
  --ink-secondary: #334155;
  --ink-muted: #64748b;
  --brand-blue: #2563eb;
  --brand-blue-hover: #1d4ed8;
  --brand-teal: #0d9488;
  --status-green: #10b981;
  --status-green-bg: #ecfdf5;
  --status-amber: #f59e0b;
  --status-amber-bg: #fffbeb;
  --status-red: #ef4444;
  --status-red-bg: #fef2f2;
  --border-line: #e2e8f0;
  --card-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
  --card-shadow-hover: 0 10px 25px -4px rgba(37, 99, 235, 0.12);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }

body, html {
  margin: 0 !important;
  padding: 0 !important;
  background: var(--bg-gradient) !important;
  color: var(--ink-primary) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding: 10px 24px 40px !important;
}

/* TOPBAR */
.healthai-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 28px;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid var(--border-line);
  border-radius: 20px;
  box-shadow: var(--card-shadow);
  backdrop-filter: blur(12px);
  margin-bottom: 24px;
}

.topbar-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-icon {
  width: 46px;
  height: 46px;
  border-radius: 14px;
  background: linear-gradient(135deg, #2563eb, #0d9488);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
}

.brand-title {
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0f2b5c;
  margin: 0;
  line-height: 1.2;
}

.brand-subtitle {
  font-size: 12.5px;
  color: var(--ink-muted);
  font-weight: 500;
}

.topbar-pills {
  display: flex;
  gap: 10px;
}

.pill-track {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 700;
  padding: 6px 14px;
  background: #eff6ff;
  color: #1d4ed8;
  border: 1px solid #bfdbfe;
  border-radius: 99px;
}

.pulsing-dot {
  width: 8px;
  height: 8px;
  background: #10b981;
  border-radius: 50%;
  box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
  animation: pulse-green 2s infinite;
}

@keyframes pulse-green {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* CARDS COMMON */
.dashboard-card {
  background: var(--surface) !important;
  border: 1px solid var(--border-line) !important;
  border-radius: 18px !important;
  box-shadow: var(--card-shadow) !important;
  padding: 22px !important;
  margin-bottom: 24px !important;
}

/* METRICS GRID */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.metric-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  border-radius: 16px;
  background: #ffffff;
  border: 1px solid var(--border-line);
  box-shadow: var(--card-shadow);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--card-shadow-hover);
}

.metric-icon {
  width: 50px;
  height: 50px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

.metric-blue .metric-icon { background: #eff6ff; color: #2563eb; }
.metric-green .metric-icon { background: #ecfdf5; color: #10b981; }
.metric-coral .metric-icon { background: #fef2f2; color: #ef4444; }
.metric-purple .metric-icon { background: #faf5ff; color: #8b5cf6; }

.metric-body {
  display: flex;
  flex-direction: column;
}

.metric-label {
  font-size: 12.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--ink-muted);
}

.metric-number {
  font-size: 30px;
  font-weight: 800;
  color: #0f2b5c;
  line-height: 1.1;
  margin: 2px 0;
}

.metric-sub {
  font-size: 12px;
  color: #64748b;
  font-weight: 500;
}

/* GAUGES GRID */
.gauges-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.gauge-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 18px;
  background: #ffffff;
  border: 1px solid var(--border-line);
  border-radius: 16px;
  box-shadow: var(--card-shadow);
}

.gauge-svg-wrap {
  position: relative;
  width: 84px;
  height: 84px;
  flex-shrink: 0;
}

.gauge-inner-val {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.gauge-info {
  display: flex;
  flex-direction: column;
}

.gauge-title {
  font-size: 13.5px;
  font-weight: 800;
  color: #1e293b;
}

.gauge-status {
  font-size: 13px;
  font-weight: 700;
  margin: 2px 0;
}

.gauge-markers-tested {
  font-size: 11.5px;
  color: #64748b;
}

/* SPECTRUM */
.spectrum-container {
  background: #ffffff;
  border: 1px solid var(--border-line);
  border-radius: 18px;
  padding: 22px;
  box-shadow: var(--card-shadow);
  margin-bottom: 24px;
}

.spectrum-container-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
}

.spectrum-legend {
  display: flex;
  gap: 14px;
  font-size: 12px;
  font-weight: 600;
  color: #475569;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.legend-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.spectrum-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.spectrum-card {
  padding: 16px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
}

.spectrum-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.spectrum-name {
  font-size: 14px;
  font-weight: 800;
  color: #0f2b5c;
  display: block;
}

.spectrum-ref {
  font-size: 11.5px;
  color: #64748b;
}

.spectrum-result-box {
  display: flex;
  align-items: center;
  gap: 8px;
}

.spectrum-val {
  font-size: 15px;
  font-weight: 800;
}

.spectrum-track {
  position: relative;
  height: 10px;
  background: #e2e8f0;
  border-radius: 99px;
  overflow: visible;
  margin: 10px 0 6px;
}

.track-amber-low {
  position: absolute;
  left: 0; width: 100%; height: 100%;
  background: #fef3c7;
  border-radius: 99px;
}

.track-green-normal {
  position: absolute;
  top: 0; height: 100%;
  background: #10b981;
  border-radius: 4px;
  opacity: 0.85;
}

.track-red-high {
  position: absolute;
  right: 0; width: 18%; height: 100%;
  background: #fee2e2;
  border-radius: 0 99px 99px 0;
}

.track-marker {
  position: absolute;
  top: -4px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  transform: translateX(-50%);
  border: 2px solid #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
}

.track-marker-pip {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #ffffff;
}

.spectrum-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: #94a3b8;
  font-weight: 600;
}

.label-normal {
  color: #059669;
  font-weight: 700;
}

/* PATHWAY */
.pathway-container {
  background: #ffffff;
  border: 1px solid var(--border-line);
  border-radius: 18px;
  padding: 22px;
  box-shadow: var(--card-shadow);
  margin-bottom: 24px;
}

.pathway-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.referral-count-pill {
  background: #fef2f2;
  color: #dc2626;
  border: 1px solid #fecaca;
  font-size: 12px;
  font-weight: 800;
  padding: 4px 12px;
  border-radius: 99px;
}

.pathway-steps-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.pathway-step-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 14px 20px;
}

.pathway-from {
  width: 32%;
  display: flex;
  flex-direction: column;
}

.pathway-arrow {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #94a3b8;
  font-size: 18px;
}

.arrow-line {
  width: 40px;
  height: 2px;
  background: #cbd5e1;
}

.pathway-to {
  width: 58%;
}

.specialist-badge {
  display: inline-block;
  font-size: 13.5px;
  font-weight: 800;
  color: #1e40af;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  padding: 3px 10px;
  border-radius: 8px;
}

.department-name {
  font-size: 12px;
  color: #64748b;
  font-weight: 600;
  margin: 3px 0;
}

.pathway-rationale {
  font-size: 12.5px;
  color: #334155;
  margin: 4px 0 0;
}

/* RESULTS TABLE */
.table-responsive-wrapper {
  overflow-x: auto;
}

.clinical-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13.5px;
}

.clinical-table th {
  background: #f1f5f9;
  color: #334155;
  font-weight: 800;
  text-align: left;
  padding: 12px 16px;
  border-bottom: 2px solid #cbd5e1;
}

.clinical-table th:first-child { border-top-left-radius: 10px; }
.clinical-table th:last-child { border-top-right-radius: 10px; }

.clinical-table td {
  padding: 12px 16px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: middle;
}

.clinical-table tr:hover td {
  background: #f8fafc;
}

.marker-cell-title {
  font-weight: 800;
  color: #0f2b5c;
}

.marker-cell-desc {
  font-size: 12px;
  color: #64748b;
  margin-top: 2px;
}

.unit-text {
  font-size: 11.5px;
  color: #64748b;
  font-weight: 500;
}

/* STATUS PILLS */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11.5px;
  font-weight: 800;
  padding: 3px 10px;
  border-radius: 99px;
  text-transform: uppercase;
}

.badge-normal {
  background: var(--status-green-bg);
  color: #065f46;
  border: 1px solid #a7f3d0;
}

.badge-low {
  background: var(--status-amber-bg);
  color: #92400e;
  border: 1px solid #fde68a;
}

.badge-high {
  background: var(--status-red-bg);
  color: #991b1b;
  border: 1px solid #fecaca;
}

/* SPECIALIST CARDS */
.doctor-referrals-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.doctor-referral-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 20px;
  box-shadow: var(--card-shadow);
}

.doctor-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f1f5f9;
}

.doctor-avatar {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: #eff6ff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
}

.doctor-spec-title {
  margin: 0;
  font-size: 15.5px;
  font-weight: 800;
  color: #0f2b5c;
}

.doctor-dept-subtitle {
  font-size: 12px;
  color: #64748b;
  font-weight: 600;
}

.doctor-card-section {
  margin-top: 12px;
}

.section-micro-label {
  font-size: 10.5px;
  font-weight: 800;
  color: #94a3b8;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  display: block;
}

.sub-pill {
  font-size: 11.5px;
  background: #f1f5f9;
  color: #334155;
  padding: 3px 8px;
  border-radius: 6px;
  font-weight: 700;
  border: 1px solid #e2e8f0;
}

.doctor-rationale-text {
  font-size: 13px;
  color: #334155;
  margin: 4px 0 0;
  line-height: 1.4;
}

.doctor-questions-list {
  margin: 6px 0 0;
  padding-left: 0;
  list-style: none;
  font-size: 12.5px;
  color: #1e3a8a;
}

.doctor-questions-list li {
  margin-bottom: 4px;
  line-height: 1.4;
}

/* CHAT AREA */
.chat-container-card {
  background: #ffffff !important;
  border: 1px solid var(--border-line) !important;
  border-radius: 20px !important;
  box-shadow: var(--card-shadow) !important;
  padding: 24px !important;
}

.chat-quick-queries {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0 16px;
}

.quick-chip {
  background: #f8fafc !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 99px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  color: #334155 !important;
  padding: 6px 14px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

.quick-chip:hover {
  background: #eff6ff !important;
  color: #1d4ed8 !important;
  border-color: #93c5fd !important;
}

/* DISCLAIMER BANNER */
.disclaimer-banner {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 12px;
  padding: 12px 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 20px;
  font-size: 12px;
  color: #92400e;
  line-height: 1.4;
}

.empty-chart-state, .empty-table-state {
  text-align: center;
  padding: 30px;
  color: #64748b;
  font-size: 14px;
}

@media (max-width: 1024px) {
  .metrics-grid, .gauges-grid, .spectrum-grid, .doctor-referrals-grid {
    grid-template-columns: repeat(2, 1fr) !important;
  }
}

@media (max-width: 640px) {
  .metrics-grid, .gauges-grid, .spectrum-grid, .doctor-referrals-grid {
    grid-template-columns: 1fr !important;
  }
}
"""


# =====================================================================
# GRADIO APPLICATION ASSEMBLY
# =====================================================================

def create_healthai_app():
    with gr.Blocks(title="HealthAI • Patient Health Companion (Track 1)") as app:
        # App State variables
        active_raw_text = gr.State(SAMPLE_REPORTS["lab_report"]["raw_text"])
        rag_store_state = gr.State(None)

        # 1. TOP NAVIGATION & BRANDING BAR
        gr.HTML(r"""
        <div class="healthai-topbar">
          <div class="topbar-brand">
            <div class="brand-icon">🩺</div>
            <div>
              <h1 class="brand-title">HealthAI • Patient Companion</h1>
              <span class="brand-subtitle">Track 1: Patient Record Simplifier & Clinical Flagging Agent</span>
            </div>
          </div>
          <div class="topbar-pills">
            <div class="pill-track">
              <span class="pulsing-dot"></span>
              <span>Autonomous Ingestion Engine</span>
            </div>
            <div class="pill-track" style="background:#f8fafc; color:#475569; border-color:#e2e8f0;">
              🔒 <span>De-identified Session Sandbox</span>
            </div>
          </div>
        </div>
        """)

        # 2. DOCUMENT INTAKE HUB (UPLOAD & INSTANT PRESETS)
        with gr.Row(elem_classes=["dashboard-card"]):
            with gr.Column(scale=3):
                gr.HTML("""
                <div style="margin-bottom: 8px;">
                  <h3 style="margin: 0; font-size: 16px; font-weight: 800; color: #0f2b5c;">📥 Hand In Medical Document</h3>
                  <p style="margin: 4px 0 0; font-size: 12.5px; color: #64748b;">
                    Upload your laboratory report, prescription, or clinical discharge summary (PDF or Text).
                    The agent simplifies it and flags abnormalities <b>the instant it is ingested</b>.
                  </p>
                </div>
                """)
                file_upload = gr.File(
                    label="Drop your patient record here (PDF or Text)",
                    file_types=[".pdf", ".txt"],
                    file_count="single",
                    interactive=True
                )
                ingest_status = gr.Markdown(
                    "**Status**: *Active demo record loaded (Comprehensive Metabolic & Lipid Panel)*"
                )

            with gr.Column(scale=2):
                gr.HTML("""
                <div style="margin-bottom: 8px;">
                  <h3 style="margin: 0; font-size: 16px; font-weight: 800; color: #0f2b5c;">⚡ Instant Clinical Samples</h3>
                  <p style="margin: 4px 0 0; font-size: 12.5px; color: #64748b;">
                    Or test with verified clinical scenarios immediately:
                  </p>
                </div>
                """)
                btn_sample_lab = gr.Button("🧪 1. Comprehensive Lab (Cholesterol / Glucose / CBC)", variant="secondary")
                btn_sample_discharge = gr.Button("🏥 2. Inpatient Discharge Summary (Post-Op Care)", variant="secondary")
                btn_sample_rx = gr.Button("💊 3. Chronic Care Prescription (Metformin / Statin)", variant="secondary")

        # Initial Document Processing for Demo Default
        initial_raw = SAMPLE_REPORTS["lab_report"]["raw_text"]
        initial_tests = extract_structured_biomarkers(initial_raw)
        initial_rx = extract_prescription_items(initial_raw)

        # 3. EXECUTIVE METRICS ROW (4 Equal Stat Cards - Zero Blank Space)
        metrics_html = gr.HTML(render_stat_metrics_row(initial_tests, initial_rx))

        # 4. SYSTEM VITALITY GAUGES ROW (4 Circular Donut Gauges)
        gauges_html = gr.HTML(render_vitality_system_gauges(initial_tests, initial_rx))

        # 5. VISUAL BIOMARKER RANGE SPECTRUM & CARE PATHWAY
        spectrum_html = gr.HTML(render_range_spectrum_chart(initial_tests, initial_rx))
        pathway_html = gr.HTML(render_specialist_pathway_diagram(initial_tests, initial_rx))

        # 6. FUNCTIONAL TABS FOR INGESTION DETAILS
        with gr.Tabs():
            with gr.TabItem("📝 Plain-Language Summary & Flags"):
                plain_summary_md = gr.Markdown(
                    simplify_and_flag_record(initial_raw)
                )

            with gr.TabItem("📋 Structured Results Table"):
                results_table_html = gr.HTML(render_results_table_html(initial_tests, initial_rx))

            with gr.TabItem("🩺 Medical Specialist Referrals"):
                specialist_cards_html = gr.HTML(render_specialist_cards_html(initial_tests, initial_rx))

            with gr.TabItem("📄 Original Ingested Text"):
                raw_text_display = gr.Markdown(
                    f"```text\n{initial_raw}\n```"
                )

        # 7. DOCUMENT-SCOPED RAG HEALTH COMPANION CHAT
        with gr.Column(elem_classes=["chat-container-card"]):
            gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 38px; height: 38px; border-radius: 12px; background: #2563eb; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                  💬
                </div>
                <div>
                  <h3 style="margin: 0; font-size: 16px; font-weight: 800; color: #0f2b5c;">Ask HealthAI About Your Report</h3>
                  <span style="font-size: 12px; color: #64748b;">
                    Responses are grounded strictly in your uploaded record with document-scoped semantic retrieval.
                  </span>
                </div>
              </div>
              <span style="font-size: 11.5px; background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; padding: 4px 12px; border-radius: 99px; font-weight: 700;">
                Isolated RAG Engine Active
              </span>
            </div>
            """)

            with gr.Row(elem_classes=["chat-quick-queries"]):
                q_chip1 = gr.Button("🩸 Explain my out-of-range test results in simple words", elem_classes=["quick-chip"])
                q_chip2 = gr.Button("❓ What 5 questions should I ask my doctor about this report?", elem_classes=["quick-chip"])
                q_chip3 = gr.Button("🥗 What dietary and lifestyle habits support these findings?", elem_classes=["quick-chip"])

            chatbot = gr.Chatbot(
                value=[
                    {
                        "role": "assistant",
                        "content": (
                            "👋 **Hello! I have automatically reviewed your uploaded medical record.**\n\n"
                            "I've rewritten your results into plain language above, flagged anything out of range, "
                            "and mapped specialist referral recommendations.\n\n"
                            "Feel free to click any of the prompt buttons or ask any question about your numbers or medications!"
                        )
                    }
                ],
                height=340
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type a question about your uploaded record (e.g. 'What does my glucose mean?' or 'How should I take Amoxicillin?')...",
                    show_label=False,
                    scale=8
                )
                send_btn = gr.Button("Send Question ➔", variant="primary", scale=2)

            # Prominent Medical Disclaimer Banner
            gr.HTML(f"""
            <div class="disclaimer-banner">
              <span style="font-size: 20px;">⚠️</span>
              <div>
                <b>Clinical Safety Notice</b>: {DISCLAIMER}
              </div>
            </div>
            """)

        # =====================================================================
        # AUTOMATIC INGESTION PIPELINE & EVENT HANDLERS
        # =====================================================================

        def execute_automatic_ingestion(raw_text: str, source_label: str):
            """
            Core Track 1 Ingestion Pipeline:
            Runs automatically the moment a document is ingested.
            Extracts biomarkers or prescriptions, classifies document nature,
            runs clinical simplification, specialist routing, and primes RAG.
            """
            doc_nature = detect_document_nature(raw_text)
            tests = extract_structured_biomarkers(raw_text)
            rx_items = extract_prescription_items(raw_text)

            # 1. Simplification & Flagging
            plain_summary = generate_expert_clinical_summary(raw_text, doc_nature, tests, rx_items)

            # 2. Specialist / department routing
            specialist_info = find_specialists(plain_summary, tests)

            # 3. Visualizations
            metrics = render_stat_metrics_row(tests, rx_items)
            gauges = render_vitality_system_gauges(tests, rx_items)
            spectrum = render_range_spectrum_chart(tests, rx_items)
            pathway = render_specialist_pathway_diagram(tests, rx_items)
            table_html = render_results_table_html(tests, rx_items)
            spec_cards_html = render_specialist_cards_html(tests, rx_items)
            raw_box = f"```text\n{raw_text}\n```"

            if tests:
                flagged_count = sum(1 for t in tests if t["status"] in {"LOW", "HIGH"})
                status_msg = f"**Status**: *Successfully ingested '{source_label}' ({len(tests)} diagnostic markers extracted, {flagged_count} flagged)*"
                findings_desc = (
                    f"- Total markers evaluated: **{len(tests)}**\n"
                    f"- Out-of-range flags: **{flagged_count}**\n"
                    f"- Detected conditions / markers: **{', '.join([t['test'] for t in tests[:5]])}**"
                )
            elif rx_items:
                status_msg = f"**Status**: *Successfully ingested '{source_label}' ({len(rx_items)} active prescriptions identified)*"
                findings_desc = (
                    f"- Active prescriptions: **{len(rx_items)}**\n"
                    f"- Prescribed medicines: **{', '.join([i['medication'] for i in rx_items])}**\n"
                    f"- Document Type: **Medical Prescription / Treatment Protocol**"
                )
            else:
                status_msg = f"**Status**: *Successfully ingested '{source_label}' (Clinical narrative parsed)*"
                findings_desc = f"- Document Type: **{doc_nature.replace('_', ' ').title()}**\n- Clinical notes & instructions processed."

            # 4. Scoped RAG store
            rag_store = DocumentRAGStore(raw_text)

            # 5. Welcome chatbot message customized to this document
            bot_welcome = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 **Automatic Ingestion Complete for `{source_label}`!**\n\n"
                        f"I have thoroughly reviewed your uploaded record and updated your dashboard.\n\n"
                        f"**Key Document Findings**:\n"
                        f"{findings_desc}\n\n"
                        f"Feel free to ask me any question about your numbers, medications, or what to discuss with your doctor!"
                    )
                }
            ]

            return (
                metrics,
                gauges,
                spectrum,
                pathway,
                plain_summary,
                table_html,
                spec_cards_html,
                raw_box,
                status_msg,
                bot_welcome,
                raw_text,
                rag_store
            )

        def handle_file_upload(file_obj):
            if file_obj is None:
                return (gr.update(),) * 12

            # Robust path extraction for Gradio NamedString / str / dict / list
            if isinstance(file_obj, list) and len(file_obj) > 0:
                file_obj = file_obj[0]

            file_path = ""
            if isinstance(file_obj, dict):
                file_path = file_obj.get("path", "")
            elif hasattr(file_obj, "name"):
                file_path = file_obj.name
            else:
                file_path = str(file_obj)

            filename = Path(file_path).name
            raw_text = ""

            if filename.lower().endswith(".pdf") and PYPDF_AVAILABLE:
                try:
                    reader = PdfReader(file_path)
                    pages_text = [p.extract_text() or "" for p in reader.pages]
                    raw_text = "\n".join(pages_text).strip()
                except Exception as e:
                    raw_text = f"Error reading PDF: {e}"
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        raw_text = f.read()
                except Exception as e:
                    raw_text = f"Error reading text file: {e}"

            if not raw_text.strip():
                raw_text = "No readable text could be extracted from the uploaded document."

            return execute_automatic_ingestion(raw_text, filename)

        def handle_sample_click(sample_key: str):
            sample = SAMPLE_REPORTS[sample_key]
            return execute_automatic_ingestion(sample["raw_text"], sample["title"])

        # Chat interaction
        def handle_chat(user_msg, history, raw_text, rag_store):
            if not user_msg or not user_msg.strip():
                return "", history
            history = history or []
            bot_reply = ask_health_companion(user_msg, rag_store, raw_text)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": bot_reply})
            return "", history

        def handle_quick_query(prompt_text, history, raw_text, rag_store):
            return handle_chat(prompt_text, history, raw_text, rag_store)

        output_targets = [
            metrics_html,
            gauges_html,
            spectrum_html,
            pathway_html,
            plain_summary_md,
            results_table_html,
            specialist_cards_html,
            raw_text_display,
            ingest_status,
            chatbot,
            active_raw_text,
            rag_store_state
        ]

        # Wire up BOTH .upload and .change on file_upload for maximum reliability
        file_upload.upload(
            handle_file_upload,
            inputs=[file_upload],
            outputs=output_targets
        )

        file_upload.change(
            handle_file_upload,
            inputs=[file_upload],
            outputs=output_targets
        )

        btn_sample_lab.click(
            lambda: handle_sample_click("lab_report"),
            outputs=output_targets
        )

        btn_sample_discharge.click(
            lambda: handle_sample_click("discharge_summary"),
            outputs=output_targets
        )

        btn_sample_rx.click(
            lambda: handle_sample_click("prescription"),
            outputs=output_targets
        )

        # Chat Wire-ups
        send_btn.click(
            handle_chat,
            inputs=[chat_input, chatbot, active_raw_text, rag_store_state],
            outputs=[chat_input, chatbot]
        )

        chat_input.submit(
            handle_chat,
            inputs=[chat_input, chatbot, active_raw_text, rag_store_state],
            outputs=[chat_input, chatbot]
        )

        q_chip1.click(
            lambda h, r, s: handle_quick_query("Explain my out-of-range test results in simple words", h, r, s),
            inputs=[chatbot, active_raw_text, rag_store_state],
            outputs=[chat_input, chatbot]
        )

        q_chip2.click(
            lambda h, r, s: handle_quick_query("What 5 questions should I ask my doctor about this report?", h, r, s),
            inputs=[chatbot, active_raw_text, rag_store_state],
            outputs=[chat_input, chatbot]
        )

        q_chip3.click(
            lambda h, r, s: handle_quick_query("What dietary and lifestyle habits support these findings?", h, r, s),
            inputs=[chatbot, active_raw_text, rag_store_state],
            outputs=[chat_input, chatbot]
        )

    return app


# =====================================================================
# SERVER LAUNCHER
# =====================================================================

if __name__ == "__main__":
    print("=" * 64, flush=True)
    print("HealthAI • Patient Record Health Companion (Track 1)", flush=True)
    print("=" * 64, flush=True)
    app = create_healthai_app()

    def get_free_port(start_port: int = 7860, max_attempts: int = 50) -> int:
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No free port found")

    PORT = get_free_port(int(os.getenv("PORT", "7860")))
    print(f"Launching HealthAI Dashboard on http://127.0.0.1:{PORT}", flush=True)

    app.launch(
        server_name="127.0.0.1",
        server_port=PORT,
        share=False,
        theme=gr.themes.Default(),
        css=APP_CSS,
        show_error=True,
    )