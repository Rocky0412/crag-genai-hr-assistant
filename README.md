# 🤖 CRAG GenAI HR Assistant

An intelligent HR Assistant powered by **Corrective Retrieval-Augmented Generation (CRAG)**, designed to answer HR-related queries with high accuracy by combining vector-based document retrieval, relevance evaluation, and fallback web search — all through a conversational interface.

---

## 📌 Overview

Traditional RAG systems can return irrelevant or hallucinated answers when retrieved documents don't match the query well. This project addresses that limitation by implementing **CRAG (Corrective RAG)** — a self-correcting retrieval pipeline that evaluates document relevance and automatically falls back to web search when the knowledge base is insufficient.

Applied to the HR domain, this assistant can handle questions around company policies, employee records, leave management, onboarding, compliance, and more.

---

## ✨ Features

- 🔍 **Corrective RAG Pipeline** — Retrieves documents, grades their relevance, and corrects retrieval failures using web search
- 🧠 **LLM-Powered Responses** — Uses a large language model to generate grounded, context-aware answers
- 📄 **HR Knowledge Base** — Ingests HR documents (policies, SOPs, handbooks) into a vector store for fast semantic search
- 🌐 **Web Search Fallback** — Automatically queries the web when internal documents are insufficient
- 🖥️ **Streamlit UI** — Clean, interactive chat interface for HR queries
- 🔄 **LangGraph Workflow** — Modular, graph-based control flow for retrieval, grading, query rewriting, and generation

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT / Groq / Gemini |
| Orchestration | LangChain + LangGraph |
| Vector Store | ChromaDB / FAISS |
| Embeddings | HuggingFace |


---

## 📁 Project Structure



## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- An Groq API key (or compatible LLM provider)
- A Tavily API key for web search

### 1. Clone the Repository

```bash
git clone https://github.com/Rocky0412/crag-genai-hr-assistant.git
cd crag-genai-hr-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
# Optional: for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### 5. Ingest HR Documents

Place your HR policy documents (PDF, TXT, DOCX) into the `data/hr_documents/` folder, then run the ingestion script:

```bash
python retriever/vector_store.py
```

### 6. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 💬 Example Queries

- *"What is the leave encashment policy for unused annual leave?"*
- *"How many days of maternity leave are employees entitled to?"*
- *"What is the process to raise a grievance with HR?"*
- *"What are the criteria for a mid-year performance review?"*
- *"How do I apply for a remote work arrangement?"*

---

## 🔄 CRAG Workflow Explained

1. **Retrieve** — The user's query is embedded and matched against the HR document vector store to fetch the top-k relevant chunks.
2. **Grade** — Each retrieved chunk is evaluated by the LLM for relevance. Documents are scored as `relevant` or `not relevant`.
3. **Decide** — If at least one chunk is relevant, the pipeline proceeds to generation. If none are relevant, the query is rewritten and passed to web search.
4. **Rewrite** *(if needed)* — The LLM reformulates the original query to make it more suitable for web search.
5. **Web Search** *(if needed)* — Tavily fetches external results to supplement or replace the internal knowledge base.
6. **Generate** — The LLM synthesizes a final, grounded answer from the available context.

---

## ⚙️ Configuration

Key settings can be found and modified in `config.py`:

| Parameter | Description | Default |
|---|---|---|
| `CHUNK_SIZE` | Document chunk size for splitting | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `TOP_K_DOCS` | Number of documents to retrieve | `4` |
| `LLM_MODEL` | LLM model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

---

## 🧪 Running Tests

```bash
python -m pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [CRAG Paper](https://arxiv.org/abs/2401.15884) — *Corrective Retrieval Augmented Generation* by Yan et al.
- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph) for the orchestration framework
- [Tavily](https://tavily.com/) for web search integration
- [Streamlit](https://streamlit.io/) for the UI framework

---

> Built with ❤️ by [Rocky0412](https://github.com/Rocky0412)
