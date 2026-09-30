from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from llm_config import create_llm


# Step 1: Load and split PDF
def process_pdf(file_path="./sample_health_lab_report.pdf"):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Loaded pages: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    return chunks


# Step 2: Create embeddings
embeddings = OllamaEmbeddings(
    model="all-minilm"
)


# Step 3: Create FAISS vector database
def ingest_data():

    chunks = process_pdf()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("Data ingested successfully!")

    return vector_store


# Step 4: Create RAG chain
def rag_chain(query, vector_store):

    llm = create_llm()

    # Retrieve top 3 relevant chunks
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Create Retrieval QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    response = chain.invoke({
        "query": query
    })

    return response


# Step 5: Run application
if __name__ == "__main__":

    vector_store = ingest_data()

    while True:

        query = input("\nEnter your query : ")

        if query.lower() == "exit":
            break

        response = rag_chain(
            query,
            vector_store
        )

        print("\nAnswer:")
        print(response["result"])