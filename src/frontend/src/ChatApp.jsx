import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'
import { useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL;
const USER = 'jul'
const NOTEBOOK = 'my-notebook'

export default function ChatApp() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [papers, setPapers] = useState([]);
  const [selectedQueryId, setSelectedQueryId] = useState(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    const init = async () => {
      let taskId = localStorage.getItem('task_id');
      if (!taskId) {
        try {
          taskId = await getOrCreateTaskId();
          localStorage.setItem('task_id', taskId);
        } catch (err) {
          console.error('Failed to get or create task ID:', err);
          return;
        }
      }

      try {
        const res = await fetch(`${API_URL}/notebook/${taskId}/chat/history`);
        if (!res.ok) throw new Error('Failed to load history');
        const data = await res.json();
        if (Array.isArray(data)) {
          setMessages(data);
        }
      } catch (err) {
        console.error('Error restoring chat history:', err);
      }
    };

    init();
  }, []);

  const handleSubmit = async () => {
    if (!question.trim()) return;

    let taskId;
    try {
      taskId = localStorage.getItem('task_id') || (await getOrCreateTaskId());
      localStorage.setItem('task_id', taskId);
    } catch (err) {
      console.error('Failed to get or create task ID:', err);
      return;
    }

    let queryId;
    try {
      const res = await fetch(`${API_URL}/notebook/${taskId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, user_id: `${USER}`, notebook_id: `${NOTEBOOK}` }),
      });
      if (!res.ok) throw new Error('Failed to submit chat message');
      const data = await res.json();
      queryId = data.query_id;
    } catch (err) {
      console.error('Failed to submit chat message:', err);
      return;
    }

    setMessages((prev) => [...prev, 
      { role: 'user', content: question, query_id: queryId },
      { role: 'assistant', content: '', query_id: queryId }
    ]);
    setQuestion('');

    streamAnswer(taskId, queryId);
  };

  const streamAnswer = (taskId, query_id) => {
    const url = `${API_URL}/notebook/${taskId}/chat/stream`;
    if (eventSourceRef.current) eventSourceRef.current.close();
    const eventSource = new EventSource(url);

    let assistantResponse = '';
    let retryCount = 0;

    eventSource.onmessage = (event) => {
      if (event.data === '<END_OF_MESSAGE>') {
        eventSource.close();
        loadPapers(taskId, query_id);
        setSelectedQueryId(query_id);
      } else {
        assistantResponse += '\n' + event.data;
        setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === 'assistant') {
          return [...prev.slice(0, -1), { role: 'assistant', content: assistantResponse, query_id: query_id }];
        } else {
          return [...prev, { role: 'assistant', content: assistantResponse, query_id: query_id }];
        }
      });
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      if (retryCount < 3) {
        setTimeout(() => streamAnswer(taskId, query_id), 1000 * Math.pow(2, retryCount));
        retryCount++;
      } else {
        console.error('Failed to connect to stream after multiple attempts.');
      }
    };

    eventSourceRef.current = eventSource;
  };

  const loadPapers = async (taskId, query_id) => {
    try {
      const res = await fetch(`${API_URL}/notebook/${taskId}/content/${query_id}`);
      if (!res.ok) {
        throw new Error('Failed to fetch papers');
      }
      const data = await res.json();
      const results = data.papers.filter((p) => p.score > 0);
      setPapers(results);
    } catch (err) {
      console.error('Failed to load papers:', err)
      setPapers([]);
    }
  };

  const getOrCreateTaskId = async () => {
    console.log(`API URL is: ${API_URL}`)
    const res = await fetch(`${API_URL}/notebook/task_id?user_id=${USER}&notebook_id=${NOTEBOOK}`);
    if (!res.ok) {
      throw new Error('Failed to fetch task_id');
    }
    const data = await res.json();
    return data.task_id;
  };

  const handleMessageClick = (query_id) => {
    const taskId = localStorage.getItem('task_id');
    if (!taskId || !query_id) return;
    setSelectedQueryId(query_id);
    loadPapers(taskId, query_id);
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: 2, display: 'flex', flexDirection: 'column', padding: '1rem', borderRight: '1px solid #ccc' }}>
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1rem' }}>
          {messages.map((msg, idx) => (
            <div 
            key={idx}
            onClick={() => msg.query_id && handleMessageClick(msg.query_id)}
            style={{
              background: msg.query_id === selectedQueryId ? '#f2fcf1' : '#f3f3f3', // blue-100 vs default
              padding: '1rem',
              borderRadius: '6px',
              marginBottom: '0.5rem',
              cursor: msg.query_id ? 'pointer' : 'default',
              border: msg.query_id === selectedQueryId ? '1px solid #2563eb' : 'none'
            }}
            // style={{ background: '#f3f3f3', padding: '1rem', borderRadius: '6px', marginBottom: '0.5rem' }}
            >
              <strong>{msg.role === 'user' ? '🧑 You' : '🤖 asXai'}:</strong>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question..."
            style={{ flex: 1, padding: '0.5rem', border: '1px solid #ccc', borderRadius: '4px' }}
          />
          <button
            onClick={handleSubmit}
            style={{ padding: '0.5rem 1rem', backgroundColor: '#2563eb', color: 'white', border: 'none', borderRadius: '4px' }}
          >
            Send
          </button>
        </div>
      </div>
      <div style={{ flex: 1, padding: '1rem', overflowY: 'auto' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem' }}>Top Articles</h2>
        {papers.map((p, i) => (
          <div key={i} style={{ marginBottom: '1rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '6px' }}>
            <div style={{ fontSize: '0.9rem', fontWeight: '600' }}>{p.title}</div>
            <div style={{ fontSize: '0.75rem', color: '#666' }}>{p.authorName}</div>
            <div style={{ fontSize: '0.75rem', color: '#666' }}>{p.publicationDate}</div>
            <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', fontStyle: 'italic' }}>{p.best_chunks?.[0]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
