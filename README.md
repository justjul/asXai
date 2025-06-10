![asXai Logo](./src/frontend/public/asXai_logo_black.svg)
# Scientific RAG Pipeline

**asXai** is an open-source, containerized Retrieval-Augmented Generation (RAG) pipeline designed for the scientific research domain. It enables transparent, citation-aware chat interfaces grounded in real literature from arXiv, Semantic Scholar, and other sources.

---

## 🚀 Features

- **Retrieval-Augmented Generation** using LLMs grounded in live academic sources
- **Chat with citations**: Ask technical questions and receive answers with linked article references
- **Real-time streaming** via Server-Sent Events (SSE)
- **Citation-aware reranker** trained with triplet loss
- **Fully containerized** (Docker Compose, Kafka, Ollama, FastAPI, Qdrant, React)
- **Monitoring & MLOps**: Integrated with MLflow, Prometheus, and Grafana
- **Offline update pipeline**: Download, parse, embed and index scientific PDFs

---

## 📐 System Architecture

### 🔧 Online Services

- `frontend/`: Vite + React UI with Firebase Auth and article sidebar
- `chat-api/`: FastAPI endpoint that manages conversation flow and SSE
- `chat-worker/`: Handles LLM responses and integrates citation scores
- `search-api/`: API to dispatch search queries to workers
- `search-worker/`: Embeds queries and reranks retrieved documents
- `retrieval/`: Qdrant vector DB + reranker inference + logistic blender

### 🗃️ Offline Services

- `update-service/`: Downloads articles, extracts PDFs, embeds and pushes to DB
- `train-reranker/`: Trains reranker model using triplet loss and logs to MLflow

---

## 🧪 Technologies Used

| Layer         | Stack                                                   |
|---------------|----------------------------------------------------------|
| LLM backend   | [Ollama](https://ollama.com/) (e.g., `gemma:12b`)       |
| Vector DB     | [Qdrant](https://qdrant.tech/)                          |
| Embeddings    | `e5-large-instruct`                                     |
| Frameworks    | FastAPI, React, Kafka, Docker                           |
| ML Ops        | MLflow, Prometheus, Grafana                             |
| Auth          | Firebase                                                |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/asXai.git
cd asXai
cp .env.example .env
docker-compose up --build