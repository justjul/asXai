# 🤖 asXai — Scientific RAG Pipeline

**asXai** is an open-source, containerized Retrieval-Augmented Generation (RAG) pipeline designed for the scientific research domain. It enables transparent, citation-aware chat interfaces grounded in real literature from arXiv, Semantic Scholar, and other sources.

![asXai Logo](./assets/asxai_logo.png)

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



# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.


## Here's now my memo on how I made it work:
0- intsall Node.js using instruction in https://learn.microsoft.com/en-us/windows/dev-environment/javascript/nodejs-on-wsl#install-nvm-nodejs-and-npm
1- cd src
2- npm create vite@latest frontend -- --template react
3- npm install
4- npm install -D tailwindcss postcss autoprefixer
5- npm install tailwindcss @tailwindcss/vite
6- npm install class-variance-authority clsx tailwind-variants
7- npm install lucide-react
8- npm install react-markdown remark-gfm
9- npm install react-router-dom
10- npm run firebase
11- npm install eventsource-parser
12- npm run dev

then start ngrok tunnel: ngrok start --all