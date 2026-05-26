from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import gradio as gr

# -----------------------------------
# Load Embedding Model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.Client()

collection_name = "pdf_collection"

# Delete old collection if exists
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(name=collection_name)


def process_pdf(pdf_file):

    global collection

    # Reset collection
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(name=collection_name)

    reader = PdfReader(pdf_file.name)

    text = ""

    for page in reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    # Create chunks
    chunks = text.split(".")

    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Create embeddings
    embeddings = model.encode(chunks).tolist()

    # Generate IDs
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return f"PDF Processed Successfully\nTotal Chunks: {len(chunks)}"


def ask_question(question):

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    final_result = ""

    for i, doc in enumerate(retrieved_docs):

        final_result += f"Result {i+1}:\n{doc}\n\n"

    return final_result



with gr.Blocks() as demo:

    gr.Markdown("# PDF Question Answering System")

    with gr.Tab("Upload PDF"):

        pdf_input = gr.File(label="Upload PDF")

        upload_output = gr.Textbox(label="Status")

        upload_button = gr.Button("Process PDF")

        upload_button.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=upload_output
        )

    with gr.Tab("Ask Questions"):

        question_input = gr.Textbox(
            label="Ask Question"
        )

        answer_output = gr.Textbox(
            label="Retrieved Chunks",
            lines=10
        )

        ask_button = gr.Button("Search")

        ask_button.click(
            fn=ask_question,
            inputs=question_input,
            outputs=answer_output
        )

demo.launch()