from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import os

from app.pdf_loader import load_pdf_text
from app.vector_store import create_vector_store
from app.chatbot import answer_question

app = FastAPI()

# Global vector store (single-PDF demo)
vectorstore = None


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>PDFMind</title>
    </head>
    <body>
        <h2>📄 PDFMind – PDF Chatbot</h2>

        <h3>Upload PDF</h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".pdf" required />
            <br><br>
            <button type="submit">Upload PDF</button>
        </form>

        <hr>

        <h3>Ask Question</h3>
        <form action="/ask" method="post">
            <input type="text" name="question" placeholder="Ask from the PDF..." required />
            <br><br>
            <button type="submit">Ask</button>
        </form>
    </body>
    </html>
    """


# ✅ Prevent GET /upload errors (important for Render)
@app.get("/upload")
def upload_redirect():
    return RedirectResponse(url="/")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    print("📥 Upload started")

    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", file.filename)

    # Save uploaded PDF safely
    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    print(f"✅ PDF saved, size: {len(contents)} bytes")

    # Extract text and create vector store
    text = load_pdf_text(pdf_path)
    vectorstore = create_vector_store(text)

    print("🧠 Vector store created")

    return {"message": "PDF uploaded successfully. You can now ask questions."}


@app.post("/ask")
def ask_question_endpoint(question: str = Form(...)):
    global vectorstore

    if vectorstore is None:
        return {"error": "Please upload a PDF first."}

    answer = answer_question(vectorstore, question)
    return {"answer": answer}
