from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import uuid
import gradio as gr


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)



client = chromadb.Client()

collection_name = "pdf_collection"

# Delete old collection if exists
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(
    name=collection_name
)


def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_pdf(pdf_file):

    global collection

    # Reset collection
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    # Read PDF
    reader = PdfReader(pdf_file.name)

    text = ""

    for page in reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    # Create chunks
    chunks = chunk_text(text)

    # Remove empty chunks
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    # Create embeddings
    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    # Generate IDs
    ids = [
        str(uuid.uuid4())
        for _ in range(len(chunks))
    ]

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return (
        f"PDF Processed Successfully\n"
        f"Total Chunks: {len(chunks)}"
    )


def ask_question(question):

    # Check if PDF is uploaded
    if collection.count() == 0:
        return "Please upload and process a PDF first."

    # Create query embedding
    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    # Combine retrieved chunks
    context = " ".join(retrieved_docs)

    # Create prompt for LLM
    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate answer
    response = generator(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )

    answer = response[0]["generated_text"]

    return answer


with gr.Blocks() as demo:

    gr.Markdown(
        "# PDF Question Answering System using RAG"
    )

    with gr.Tab("Upload PDF"):

        pdf_input = gr.File(
            label="Upload PDF"
        )

        upload_output = gr.Textbox(
            label="Status"
        )

        upload_button = gr.Button(
            "Process PDF"
        )

        upload_button.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=upload_output
        )

    with gr.Tab("Ask Questions"):

        question_input = gr.Textbox(
            label="Ask Question",
            placeholder="Ask anything from the PDF..."
        )

        answer_output = gr.Textbox(
            label="Generated Answer",
            lines=10
        )

        ask_button = gr.Button(
            "Get Answer"
        )

        ask_button.click(
            fn=ask_question,
            inputs=question_input,
            outputs=answer_output
        )

demo.launch()