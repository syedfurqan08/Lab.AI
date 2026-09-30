"""
healthcare_tools.py

Track 1: Healthcare
Patient Record Simplifier & Health Companion Agent

This module contains the two required deterministic tools:

1. record_simplification_and_flagging
   - Converts common lab terminology into plain language.
   - Parses result/reference-range pairs.
   - Flags values outside the supplied reference range.
   - Does NOT use an LLM.

2. specialist_finder
   - Maps flagged tests/conditions to a suggested department.
   - Uses a local deterministic lookup table.
   - Does NOT use a live API.

Important:
- This is for a student/demo project.
- It is not a diagnostic system.
- It must not be used as a substitute for a qualified healthcare professional.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from langchain_core.tools import tool


# ============================================================
# REFERENCE-RANGE RULES
# ============================================================

# These rules are intentionally simple and deterministic.
# The project should primarily use the reference range printed
# on the patient's own report.

LAB_ALIASES = {
    "hemoglobin": "Hemoglobin",
    "white blood cell count": "White Blood Cell Count",
    "wbc": "White Blood Cell Count",
    "platelet count": "Platelet Count",
    "platelets": "Platelet Count",
    "mean corpuscular volume": "Mean Corpuscular Volume (MCV)",
    "mcv": "Mean Corpuscular Volume (MCV)",
    "fasting blood glucose": "Fasting Blood Glucose",
    "fasting glucose": "Fasting Blood Glucose",
    "hba1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "total cholesterol": "Total Cholesterol",
    "ldl cholesterol": "LDL Cholesterol",
    "ldl": "LDL Cholesterol",
    "hdl cholesterol": "HDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "triglycerides": "Triglycerides",
    "creatinine": "Creatinine",
    "egfr": "eGFR",
    "alt": "ALT",
    "ast": "AST",
    "total bilirubin": "Total Bilirubin",
}

PLAIN_LANGUAGE = {
    "Hemoglobin": "A protein in blood that carries oxygen around the body.",
    "White Blood Cell Count": "The number of infection-fighting white blood cells in the blood.",
    "Platelet Count": "The number of blood cells that help the blood clot.",
    "Mean Corpuscular Volume (MCV)": "The average size of red blood cells.",
    "Fasting Blood Glucose": "The amount of sugar in the blood after fasting.",
    "HbA1c": "An estimate of average blood sugar over roughly the previous 2–3 months.",
    "Total Cholesterol": "The total amount of cholesterol in the blood.",
    "LDL Cholesterol": "A type of cholesterol often called 'bad cholesterol' because higher levels can increase cardiovascular risk.",
    "HDL Cholesterol": "A type of cholesterol that helps carry cholesterol away from the bloodstream.",
    "Triglycerides": "A type of fat found in the blood.",
    "Creatinine": "A waste product commonly used to assess kidney function.",
    "eGFR": "An estimate of how well the kidneys are filtering blood.",
    "ALT": "A liver enzyme; abnormal levels can be associated with liver or other conditions.",
    "AST": "An enzyme found in several tissues, including the liver and muscles.",
    "Total Bilirubin": "A substance produced when red blood cells are broken down and processed by the liver.",
}

# Deterministic routing table.
SPECIALIST_MAP = {
    "Hemoglobin": "Primary care / Internal Medicine; possible Hematology review if persistent or clinically significant.",
    "Mean Corpuscular Volume (MCV)": "Primary care / Internal Medicine; possible Hematology review.",
    "Fasting Blood Glucose": "Primary care / Internal Medicine; Endocrinology if clinically indicated.",
    "HbA1c": "Primary care / Internal Medicine; Endocrinology if clinically indicated.",
    "Total Cholesterol": "Primary care / Internal Medicine; Cardiology if clinically indicated.",
    "LDL Cholesterol": "Primary care / Internal Medicine; Cardiology if clinically indicated.",
    "HDL Cholesterol": "Primary care / Internal Medicine.",
    "Triglycerides": "Primary care / Internal Medicine; Cardiology if clinically indicated.",
    "Creatinine": "Primary care / Internal Medicine; Nephrology if kidney impairment is suspected.",
    "eGFR": "Primary care / Internal Medicine; Nephrology if kidney impairment is suspected.",
    "ALT": "Primary care / Internal Medicine; Gastroenterology/Hepatology if persistent or clinically significant.",
    "AST": "Primary care / Internal Medicine; Gastroenterology/Hepatology if persistent or clinically significant.",
    "Total Bilirubin": "Primary care / Internal Medicine; Gastroenterology/Hepatology if persistent or clinically significant.",
    "White Blood Cell Count": "Primary care / Internal Medicine; Hematology if persistent or clinically significant.",
    "Platelet Count": "Primary care / Internal Medicine; Hematology if persistent or clinically significant.",
}


# ============================================================
# HELPERS
# ============================================================

def _normalise_name(name: str) -> str:
    """Convert a test name into the canonical name used by this module."""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    return LAB_ALIASES.get(cleaned, name.strip())


def _parse_number(value: str) -> float | None:
    """Extract the first numeric value from a result string."""
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None


def _parse_reference_range(reference: str) -> Tuple[float | None, float | None]:
    """
    Parse common reference formats:
      12.0 – 15.5
      70 - 99
      < 200
      <= 100
      >= 40
      > 60
    """
    ref = reference.strip().replace("–", "-").replace("—", "-")

    range_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
        ref
    )

    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    less_match = re.search(r"<=?\s*(-?\d+(?:\.\d+)?)", ref)
    if ref.startswith("<") and less_match:
        return None, float(less_match.group(1))

    greater_match = re.search(r">=?\s*(-?\d+(?:\.\d+)?)", ref)
    if ref.startswith(">") and greater_match:
        return float(greater_match.group(1)), None

    return None, None


def _check_range(value: float, reference: str) -> str:
    """Return LOW, HIGH, or NORMAL based only on the stated reference range."""
    low, high = _parse_reference_range(reference)

    if low is not None and value < low:
        return "LOW"

    if high is not None and value > high:
        return "HIGH"

    return "NORMAL"


def _extract_table_rows(record_text: str) -> List[Dict[str, str]]:
    """
    Extract simple pipe-separated lab rows.

    Expected form:
    Test | Result | Reference Range | Unit

    Also accepts whitespace-separated rows reasonably well when the
    reference range is enclosed in a recognizable numeric pattern.
    """
    rows = []

    # First handle Markdown/table-like rows.
    for line in record_text.splitlines():
        if "|" not in line:
            continue

        cells = [c.strip() for c in line.split("|")]

        if len(cells) < 3:
            continue

        test = cells[0]
        result = cells[1]
        reference = cells[2]
        unit = cells[3] if len(cells) >= 4 else ""

        if test.lower() in {"test", "parameter", "lab test"}:
            continue

        if _parse_number(result) is None:
            continue

        if not re.search(r"\d", reference):
            continue

        rows.append({
            "test": _normalise_name(test),
            "result": result,
            "reference": reference,
            "unit": unit,
        })

    return rows


# ============================================================
# TOOL 1: SIMPLIFICATION + FLAGGING
# ============================================================

@tool
def record_simplification_and_flagging(record_content: str) -> str:
    """
    Simplify a medical record into plain language and deterministically
    flag laboratory results outside their stated reference ranges.

    This tool must run automatically immediately after document ingestion.
    It does not diagnose disease and does not use an LLM.
    """
    rows = _extract_table_rows(record_content)

    if not rows:
        return (
            "No structured laboratory result rows were detected. "
            "The document should be reviewed manually or processed with "
            "a document-specific parser."
        )

    summary_lines = []
    flags = []

    for row in rows:
        test = row["test"]
        result = row["result"]
        reference = row["reference"]
        unit = row["unit"]

        explanation = PLAIN_LANGUAGE.get(
            test,
            "A laboratory measurement included in the medical report."
        )

        numeric_value = _parse_number(result)

        if numeric_value is not None:
            status = _check_range(numeric_value, reference)
        else:
            status = "UNKNOWN"

        summary_lines.append(
            f"- {test}: {result} {unit}. "
            f"Plain language: {explanation} "
            f"Reference range: {reference}. Status: {status}."
        )

        if status in {"LOW", "HIGH"}:
            flags.append({
                "test": test,
                "result": f"{result} {unit}".strip(),
                "reference": reference,
                "status": status,
            })

    output = [
        "PLAIN-LANGUAGE RECORD SUMMARY",
        "",
        *summary_lines,
        "",
        "FOLLOW-UP FLAGS",
    ]

    if flags:
        for flag in flags:
            output.append(
                f"- {flag['test']}: {flag['result']} is {flag['status']} "
                f"relative to the stated reference range "
                f"({flag['reference']})."
            )
    else:
        output.append("- No out-of-range values were detected.")

    output.extend([
        "",
        "IMPORTANT: This is an automated document explanation and "
        "reference-range check. It is not a diagnosis or treatment plan. "
        "A qualified healthcare professional should interpret results "
        "using the patient's symptoms and medical history."
    ])

    return "\n".join(output)


# ============================================================
# TOOL 2: SPECIALIST / DEPARTMENT FINDER
# ============================================================

@tool
def specialist_finder(flagged_items: str) -> str:
    """
    Map flagged laboratory findings to a relevant type of doctor or
    department using a deterministic local lookup table.

    This tool does not diagnose the patient and does not call a live API.
    """
    requested = []

    for canonical_name in SPECIALIST_MAP:
        if canonical_name.lower() in flagged_items.lower():
            requested.append(canonical_name)

    # Also check common aliases.
    for alias, canonical_name in LAB_ALIASES.items():
        if alias in flagged_items.lower() and canonical_name not in requested:
            requested.append(canonical_name)

    if not requested:
        return (
            "No matching specialist-routing rule was found. "
            "Consider Primary Care / Internal Medicine for initial review."
        )

    results = ["SPECIALIST / DEPARTMENT ROUTING"]

    for test in requested:
        results.append(
            f"- {test} → {SPECIALIST_MAP[test]}"
        )

    results.append(
        "\nRouting is informational only and does not establish a diagnosis."
    )

    return "\n".join(results)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    sample_record = """
    Test | Result | Reference Range | Unit
    Hemoglobin | 10.8 | 12.0 – 15.5 | g/dL
    White Blood Cell Count | 7.4 | 4.0 – 11.0 | ×10⁹/L
    Platelet Count | 280 | 150 – 450 | ×10⁹/L
    Mean Corpuscular Volume (MCV) | 76 | 80 – 100 | fL
    Fasting Blood Glucose | 118 | 70 – 99 | mg/dL
    HbA1c | 6.2 | 4.0 – 5.6 | %
    Total Cholesterol | 218 | < 200 | mg/dL
    LDL Cholesterol | 142 | < 100 | mg/dL
    HDL Cholesterol | 48 | ≥ 40 | mg/dL
    Triglycerides | 140 | < 150 | mg/dL
    Creatinine | 0.9 | 0.6 – 1.1 | mg/dL
    eGFR | 92 | ≥ 60 | mL/min/1.73m²
    ALT | 24 | 7 – 35 | U/L
    AST | 22 | 8 – 35 | U/L
    Total Bilirubin | 0.7 | 0.2 – 1.2 | mg/dL
    """

    print("=" * 70)
    print("TOOL 1 TEST")
    print("=" * 70)

    flag_result = record_simplification_and_flagging.invoke({
        "record_content": sample_record
    })

    print(flag_result)

    print("\n" + "=" * 70)
    print("TOOL 2 TEST")
    print("=" * 70)

    specialist_result = specialist_finder.invoke({
        "flagged_items": flag_result
    })

    print(specialist_result)

    print("\n" + "=" * 70)
    print("TOOL METADATA")
    print("=" * 70)

    print("Tool 1:", record_simplification_and_flagging.name)
    print("Arguments:", record_simplification_and_flagging.args)

    print("\nTool 2:", specialist_finder.name)
    print("Arguments:", specialist_finder.args)
