[ User (React Frontend) ]
           |
           v
    [ Chat API (FastAPI) ]
           |
           |--> [ Kafka Producer ] --> (Kafka Topic: chat-topic)
           |
           v
    [ Chat Stream (via /stream) ]  <----+
           |                            |
           v                            |
    [ Chat Worker (async) ] ------------+
           |
           |--> expand/parse --> [ Ollama (LLM) ]
                     |
                 [ Search API ]
           |         |
           |         v
           |     [Search Worker]
           |         |
           |         v
           |     [Qdrant (Vector DB) ] <--- embeddings
           |         |
           |         v
           |     [Blend scores using user feedback (logreg) ]
           |
           v
[ Final Prompt + Top-K Context Chunks ] --> [ Ollama / vLLM (LLM Call) ]
                                                    |
                                                    |
                                      <-------------+

    [ Streamed Response to Frontend ]
           |
           v
  [ Sidebar Papers | Cited Highlights ]
