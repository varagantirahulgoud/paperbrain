from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import os
import shutil

from app.pdf_loader import load_pdf_text
from app.vector_store import create_vector_store
from app.chatbot import answer_question

app = FastAPI()

vectorstore = None


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h2>PDF Chatbot</h2>

        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".pdf" required />
            <br><br>
            <button type="submit">Upload PDF</button>
        </form>

        <hr>

        <form action="/ask" method="post">
            <input type="text" name="question" required />
            <br><br>
            <button type="submit">Ask</button>
        </form>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    print("📥 Upload started")

    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", file.filename)

    # ✅ Correct way to save UploadFile in async FastAPI
    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    print("✅ PDF saved, size:", len(contents), "bytes")

    # Now read PDF safely
    text = load_pdf_text(pdf_path)
    vectorstore = create_vector_store(text)

    print("🧠 Vector store created")

    return {"message": "PDF uploaded successfully"}


@app.post("/ask")
def ask_question(question: str = Form(...)):
    global vectorstore

    if vectorstore is None:
        return {"error": "Upload PDF first"}

    answer = answer_question(vectorstore, question)
    return {"answer": answer}
