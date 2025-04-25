import os
import ollama
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def create_vector_store(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

def ask_ollama(query, context):
    prompt = f"""
    Answer the question based on the following context:

    Context:
    {context}

    Question: {query}
    
    Answer:
    """
    
    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['message']['content']

def answer_question(query, vector_store):
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get answer from Ollama
    answer = ask_ollama(query, context)
    return answer

def main():
    pdf_path = "../annual-report-2024.pdf"
    try:
        # Load and process PDF
        pdf_text = load_pdf(pdf_path)
        print("PDF loaded successfully.")
        
        # Create vector store
        vector_store = create_vector_store(pdf_text)
        print("Vector store created successfully.")
        
        # Interactive question answering
        print("\nYou can now ask questions about the annual report (type 'exit' to quit):")
        while True:
            query = input("\nQuestion: ")
            if query.lower() == 'exit':
                break
            
            answer = answer_question(query, vector_store)
            print(f"\nAnswer: {answer}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()