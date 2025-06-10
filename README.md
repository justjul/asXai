![asXai Logo](./src/frontend/public/asXai_logo_black.svg)
# Scientific RAG Pipeline

**asXai** is an open-source, containerized Retrieval-Augmented Generation (RAG) pipeline designed for the scientific research domain. It enables transparent, citation-aware chat interfaces grounded in real literature from arXiv, Semantic Scholar, and other sources.

---

## 📚 Demo
Try the live version (if deployed):
👉 asXai Live Demo
🔐 Auth required via Firebase (email/password)

## 💡Features

- **Retrieval-Augmented Generation** using LLMs grounded in live academic sources
- **Chat with citations**: Ask questions and receive answers with linked article references
- **Real-time streaming** via Server-Sent Events (SSE)
- **Citation-aware reranker** trained with triplet loss
- **Fully containerized** (Docker Compose, Ollama, Qdrant, Kafka, Selenium, ...)
- **Monitoring & MLOps**: Integrated with MLflow, Prometheus, and Grafana
- **Offline update pipeline**: Download, parse, embed and index scientific PDFs


## 👷🏻‍♀️System Architecture

### 🌐 Online Services

- `frontend/`: Vite + React UI with Firebase Auth
- `chat-api/`: FastAPI endpoint that manages conversation flow and SSE
- `chat-worker/`: Handles LLM responses and calls to search-worker to integrate citations
- `search-api/`: API to dispatch search queries to workers
- `search-worker/`: Embeds queries, retrieve documents and rerank them

### 🦖 Offline Services

- `update-service/`: Downloads articles, extracts PDFs, embeds and pushes to DB, train the reranker


## 🛠️ Technologies Used

| Layer         | Stack                                                   |
|---------------|---------------------------------------------------------|
| LLM backend   | [Ollama](https://ollama.com/) (e.g., `gemma:12b`)       |
| Vector DB     | [Qdrant](https://qdrant.tech/)                          |
| Embeddings    | `e5-large-instruct`                                     |
| Rerank model  | Transformer trained with PyTorch                        |
| Frameworks    | Docker, FastAPI, Kafka, React                           |
| ML Ops        | MLflow, Prometheus, Grafana                             |
| Auth          | Firebase                                                |


## 📦 Installation

- when using with Docker:
```bash
git clone https://github.com/yourusername/asXai.git
cd asXai/docker
cp .env.compose.example .env.compose
```
- If you prefer running everything in your environment
```bash
pip install -e .[torch,nlp,api,pdf,dev]
```


## 🚀 Start service

```bash
cd asXai/docker
docker-compose --env-file .env.compose up -d
```

## 🤝 Contributing
Pull requests and issues are welcome!
If you use this pipeline in research, please consider citing or starring the repository.