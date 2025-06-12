<div align="center">
  <img src="./front/public/asXai_logo_black.svg" alt="asXai Logo" width="200" />
</div>

**asXai** is an open-source, containerized RAG pipeline (Retrieval-Augmented Generation) designed for scientific questions. It enables transparent, citation-aware chat interfaces grounded in real literature from arXiv, Semantic Scholar, and other sources.

👉 Try the live [demo](https://goose-beloved-kit.ngrok-free.app) (if deployed)

---

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
| LLM backend   | [Ollama](https://ollama.com/) (e.g., `gemma3:12b`)       |
| Vector DB     | [Qdrant](https://qdrant.tech/)                          |
| Embeddings    | `e5-large-instruct`                                     |
| Rerank model  | Transformer trained with PyTorch                        |
| Frameworks    | Docker, FastAPI, Kafka, React                           |
| ML Ops        | MLflow, Prometheus, Grafana                             |
| Auth          | Firebase                                                |


## 📦 Installation

- Using Docker compose:
```bash
git clone https://github.com/yourusername/asXai.git
cd asXai/docker
cp .env.compose.example .env.compose
```
- If you also want a local environment
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
If you use this pipeline or want to deploy it for real, please star the repo and drop me a message.
