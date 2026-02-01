import os
from groq import Groq

# ✅ Correct: read the ENV VARIABLE NAME
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(vectorstore, question: str) -> str:
    # Retrieve relevant chunks from the PDF
    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        return "The answer is not present in the provided PDF."

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a strict PDF-based assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"The answer is not present in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()
