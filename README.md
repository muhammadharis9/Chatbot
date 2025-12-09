# Chatbot – RAG Chatbot over the “Agentic AI” Book (LangChain + Pinecone + OpenRouter)

This repo contains a simple end-to-end **Retrieval-Augmented Generation (RAG)** chatbot.

By default, it is designed to chat over the **“Agentic AI” book** (or any similar PDF you put in the `dataset/` folder). The bot **does not invent knowledge on its own** – instead, it retrieves relevant passages from the book and uses them as context for the answer.

> ⚠️ This project is for learning and experimentation only.  
> It must **not** be used for real medical or clinical decision-making.

---

## ✨ What This Project Actually Is

You can think of this project as:

> 🧠 **“Ask-me-anything about the Agentic AI book”** – powered by RAG.

Concretely:

1. You drop the **Agentic AI book PDF** into `dataset/`.
2. The project:
   - Loads the PDF
   - Splits it into small overlapping chunks
   - Embeds those chunks using a sentence-transformer model
   - Stores the vectors in a **Pinecone** index
3. When a user asks a question in the **Flask web UI**:
   - The system retrieves the most relevant chunks from the book
   - Injects them into a carefully designed system prompt
   - Sends everything to an LLM via **OpenRouter**
   - Returns an answer that is grounded in the *actual text* of the book

So you end up with a small, focused **Agentic AI RAG chatbot**.

You can easily swap the book for any other domain PDF (support docs, internal guides, research papers, etc.) and reuse the same pipeline.

---

## 🧰 Tech Stack

- **Language:** Python 3.11+
- **Frameworks:** Flask, LangChain
- **Vector Store:** Pinecone
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`)
- **LLM Access:** `langchain-openai` with OpenRouter as the backend
- **Env / Deps:** [`uv`](https://github.com/astral-sh/uv), `pyproject.toml`
- **Frontend:** HTML template (`templates/chat.html`) + CSS (`static/style.css`)
- **Notebook Exploration:** Jupyter notebook in `research/`

---

## 📂 Project Structure

```text
.
├── app.py                # Main Flask app and RAG pipeline
├── main.py               # Simple CLI entry point (prints “Hello from chatbot!”)
├── src/
│   ├── helper.py         # PDF loading, metadata cleanup, chunking, embeddings
│   └── prompt.py         # System prompt for the chatbot
├── templates/
│   └── chat.html         # Chat UI rendered by Flask
├── static/
│   └── style.css         # Styles for the chat interface
├── dataset/              # Folder where you place your PDFs (e.g. Agentic AI book)
├── research/
│   └── notebook_test.ipynb  # Notebook showing the RAG + Pinecone pipeline
├── pyproject.toml        # Project metadata & dependencies (managed by uv)
├── uv.lock               # Locked dependency versions
├── .python-version       # Python version used by the project
├── .env                  # Environment variables (API keys, etc.) – do NOT commit real keys
├── template.sh           # Shell script placeholder (can be used for automation)
├── LICENSE               # Apache-2.0 license
└── README.md             # This file
