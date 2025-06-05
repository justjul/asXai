import { useState, useRef, useMemo  } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { createParser } from 'eventsource-parser';
import { useAuth } from './firebase-auth';
import { getAuth } from "firebase/auth";
import './cssChatApp.css';


const API_URL = import.meta.env.VITE_API_URL;

export async function authFetch(user, url, options = {}) {
  const token = await user.getIdToken();
  const headers = {
    ...options.headers,
    Authorization: `Bearer ${token}`,
    "ngrok-skip-browser-warning": "1",
  };
  return fetch(url, { ...options, headers });
}

export default function ChatApp() {
  const { notebookId } = useParams();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [papers, setPapers] = useState([]);
  const [selectedQueryIds, setSelectedQueryIds] = useState(new Set());
  const [expandedIndexes, setExpandedIndexes] = useState(new Set());
  const [highlightedPaperIds, setHighlightedPaperIds] = useState(new Set());
  const [notebooks, setNotebooks] = useState([]);
  const [lastNotebookDeleted, setLastNotebookDeleted] = useState([]);
  const lastNotebookCreated = localStorage.getItem("lastNotebookCreated")
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(true);
  const messagesEndRef = useRef(null);
  const auth = getAuth();
  const [isAdmin, setIsAdmin] = useState(false);

  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (firebaseUser) => {
      if (firebaseUser) {
        // force a token refresh so we get the latest custom claims
        const idTokenResult = await firebaseUser.getIdTokenResult(true);
        setIsAdmin(!!idTokenResult.claims.admin);
      } else {
        setIsAdmin(false);
      }
    });
    return unsubscribe;
  }, [auth]);

  const fetchNotebooks = async () => {
    try {
      const res = await authFetch(user, `${API_URL}/notebook`);
      const data = await res.json();
      setNotebooks(data);
    } catch (err) {
      console.error('Failed to load notebooks:', err);
    }
  };

  useEffect(() => {
    if (!isStreaming) {
      fetchNotebooks();
    }
  }, [isStreaming, user, notebookId]);


  useEffect(() => {
    if (!notebookId) return;
    if (!lastNotebookCreated) return;
    if (notebookId === lastNotebookCreated) return;

    if (notebooks.length === 0) {
      fetchNotebooks();
    }
  }, [notebookId, lastNotebookCreated, notebooks.length]);


  useEffect(() => {
    if (!notebookId) {
      // when “/n” is hit after login, create a new notebookId and navigate to it.
      createNewNotebook();
    }
  }, [notebookId]);


  useEffect(() => {
    console.log(notebookId, lastNotebookCreated)
    if (!notebookId) return;
    if (!lastNotebookCreated) return;
    if (notebookId === lastNotebookCreated) return;

    if (notebooks.length === 0)  return;
    if (notebookId === lastNotebookDeleted) {
      setLastNotebookDeleted([])
      return;
    }
    
    console.log(notebooks.length)
    const found = notebooks.some(nb => nb.id === notebookId);
    if (!found) {
      alert(`Unable to load notebook ${notebookId}`);
      navigate(-1)
    }
  }, [notebooks, notebookId]);


  useEffect(() => {
    const init = async () => {
      if (!notebookId) return;
      if (notebooks.length === 0) return;
      const exists = notebooks.some((nb) => nb.id === notebookId);
      if (!exists) return; // we’ll have already navigated away from invalid IDs in the previous effect

      try {
        const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/chat/history`);
        if (!res.ok) throw new Error('Failed to load history');
        const data = await res.json();
        setPapers([]);
        if (Array.isArray(data)) {
          const processed = data.map((msg) => {
            if (msg.role === 'assistant' && msg.papers?.length) {
              return {
                ...msg,
                content: linkifyPaperIds(msg.content, msg.papers),
              };
            }
            return msg;
          });
          setMessages(processed);
        }
      } catch (err) {
        console.error('Error restoring chat history:', err);
      }
    };
    init();
  }, [notebookId, notebooks, user]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);


  const createNewNotebook = async () => {
    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/new_task_id`,
        { method: 'GET', headers: { 'Content-Type': 'application/json' } }
      );
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      const new_notebookId = data.notebook_id
      localStorage.setItem("lastNotebookCreated", new_notebookId)
      navigate(`/n/${new_notebookId}`);
      setMessages([])
    } catch (err) {
      console.error("Failed to create new notebook:", err);
    }
  };


  const deleteNotebook = async (id) => {
    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/${id}/delete`,
        { method: 'DELETE' }
      );
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);

      setLastNotebookDeleted(id)

      // if we just deleted the one we were viewing, spin up a fresh one
      if (id === notebookId) {
        createNewNotebook();
      }

      // refresh the notebooks array
      await fetchNotebooks();

    } catch (err) {
      console.error(`Failed to delete notebook ${id}:`, err);
    }
  };


  const handleSubmit = async () => {
    if (!question.trim()) return;
    let queryId;
    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({ message: question, notebook_id: `${notebookId}`, topK }),
      });
      if (!res.ok) throw new Error('Failed to submit chat message');
      const data = await res.json();
      queryId = data.query_id;
    } catch (err) {
      console.error('Failed to submit chat message:', err);
      return;
    }
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question, query_id: queryId },
      { role: 'assistant', content: '', query_id: queryId },
    ]);
    setQuestion('');
    streamAnswer(notebookId, queryId);
  };

  const streamAnswer = async (notebookId, query_id) => {
    const maxRetries = 3;
    let retryCount = 0;

    let assistantResponse = '';
    let success = false;

    const startStreaming = async () => {
      setIsStreaming(true);

      try {
        const url = `${API_URL}/notebook/${notebookId}/chat/stream`;
        const res = await authFetch(user, url);

        if (!res.ok || !res.body) {
          throw new Error("Failed to open stream.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        const parser = createParser({
          onEvent: (event) => {
            
            if (event.data) {
              if (event.data === '<END_OF_MESSAGE>') {
                reader.cancel();
                success = true;

                loadPapers(notebookId, query_id).then((loadedPapers) => {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.query_id === query_id && msg.role === 'assistant'
                        ? {
                            ...msg,
                            content: linkifyPaperIds(msg.content, loadedPapers),
                          }
                        : msg
                    )
                  );
                });

                setSelectedQueryIds(prev => {
                  const next = new Set(prev);
                  next.add(query_id);
                  return next;
                });
                return;
              }

              assistantResponse += '\n' + event.data;
              setMessages((prev) => {
                const last = prev[prev.length - 1];
                if (last?.role === 'assistant') {
                  return [
                    ...prev.slice(0, -1),
                    { role: 'assistant', content: assistantResponse, query_id },
                  ];
                } else {
                  return [...prev, { role: 'assistant', content: assistantResponse, query_id }];
                }
              });
            }
          }
        });

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          parser.feed(decoder.decode(value));
        }
      } catch (err) {
        console.error(`Stream failed (attempt ${retryCount + 1}):`, err);
        retryCount += 1;
        if (retryCount <= maxRetries) {
          const delay = 1000 * Math.pow(2, retryCount); // exponential backoff
          setTimeout(() => startStreaming(), delay);
        } else {
          console.error("Stream failed after maximum retries.");
        }
      } finally {
        if (retryCount === 0 || success) setIsStreaming(false);
      }
    };

    startStreaming();
  };

  const loadPapers = async (notebookId, query_id) => {
    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/content/${query_id}`);
      if (!res.ok) throw new Error('Failed to fetch papers');
      const data = await res.json();
      const results = data.papers.filter((p) => p.score > 0);
      setPapers(results);
      return results;
    } catch (err) {
      console.error('Failed to load papers:', err);
      setPapers([]);
      return [];
    }
  };

  const handleMessageClick = (e, query_id) => {
    const newSelected = new Set(selectedQueryIds);

    if (e.ctrlKey || e.metaKey) {
      // Ctrl-click add msg to the set
      if (newSelected.has(query_id)) {
        newSelected.delete(query_id);
      } else {
        newSelected.add(query_id);
      }
    } else {
      // Plain click make a new set starting with this msg
      if (newSelected.has(query_id) && newSelected.size === 1) {
        // already solo-selected, so this click should deselect it
        newSelected.clear();
      } else {
        newSelected.clear();
        newSelected.add(query_id);
      }
    }

    setSelectedQueryIds(newSelected);
    setRightCollapsed(false);

    // rebuilding `papers` from all selected messages:
    if (newSelected.size === 0) {
      setPapers([]); 
      return;
    }

    // Collect all papers from every selected assistant message
    const allPapers = [];
    for (const id of newSelected) {
      const msg = messages.find(
        m => m.query_id === id && m.role === 'assistant' && Array.isArray(m.papers)
      );
      if (msg?.papers) {
        allPapers.push(...msg.papers);
      }
    }

    // Remove duplicates by paperId
    const seen = new Set();
    const deduped = [];
    for (const p of allPapers) {
      if (!seen.has(p.paperId)) {
        seen.add(p.paperId);
        deduped.push(p);
      }
    }

    setPapers(deduped);
    setExpandedIndexes(new Set()); // reset any expanded indexes
  };

  const toggleExpand = (index) => {
    setExpandedIndexes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  function buildFuzzyRegex(paperId) {
    const parts = paperId.split('');
    const patterns = [];

    // for every contiguous block of 5 characters, make a fuzzy subpattern
    for (let i = 0; i <= parts.length - 5; i++) {
      const slice5 = parts.slice(i, i + 5);
      const fuzzy5 = slice5.map(ch => `${ch}\\s*?`).join('');
      patterns.push(fuzzy5);
    }

    // combine with | so that any of those 5‐letter runs will match
    return new RegExp(patterns.join('|'), 'g');
  }

  function buildCollapseRegex(label) {
    return new RegExp(`(?:${label})(?:\\s*${label})+`, 'g');
  }

  const linkifyPaperIds = (text, papers) => {
    const sorted = [...papers].sort((a, b) => b.paperId.length - a.paperId.length);
    for (const { paperId, authorName, publicationDate } of sorted) {
      const year = new Date(publicationDate).getFullYear();
      const firstAuthor = (() => {
        const fullname = authorName?.split(',')[0] || authorName || '';
        const nameparts = fullname.trim().split(/\s+/);
        return nameparts[nameparts.length - 1] || 'Unknown';
      })();
      const label = `${firstAuthor} et al., ${year}`;
      const idregex = buildFuzzyRegex(paperId);
      text = text.replace(idregex, `${paperId}`);
      const labelregex = buildCollapseRegex(`${paperId}`);
      text = text.replace(labelregex, `[${label}](#${paperId})`);
    }
    return text;
  };

  const allReferencedPapers = useMemo(() => {
    const seen = new Set();
    const all = [];
    for (const msg of messages) {
      if (msg.role === 'assistant' && Array.isArray(msg.papers)) {
        for (const p of msg.papers) {
          if (!seen.has(p.paperId)) {
            seen.add(p.paperId);
            all.push(p);
          }
        }
      }
    }
    return all;
  }, [messages]);

  const displayedPapers = selectedQueryIds.size > 0 ? papers : allReferencedPapers;

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw' }}>
      {/* Sidebar */}
      <div
        className={`notebook-sidebar`}
        style={{
          width: leftCollapsed ? '2.5rem' : '20%',
          transition: 'width 0.3s ease',
          borderRight: '1px solid #ccc',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* collapse/expand toggle */}
        <button
          onClick={() => setLeftCollapsed(!leftCollapsed)}
          style={{
            padding: '0.5rem',
            border: 'none',
            background: '#eee',
            cursor: 'pointer',
            alignSelf: 'flex-start',
          }}
        >
          {leftCollapsed ? '📒' : '📝'}
        </button>

        <div style={{ flex: 1, overflowY: "auto", padding: leftCollapsed ? 0 : "0.5rem" }}>
          {!leftCollapsed && (
            <>
              {/* Notebook list container—make this grow to fill the middle */}
              <div style={{ padding: '0.5rem', flex: 1, overflowY: 'auto' }}>
                <h3 style={{ marginBottom: '0.5rem' }}>Notebooks</h3>
                {notebooks.map((nb) => (
                  <div
                    key={nb.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '0.25rem 0.5rem',
                      borderRadius: '4px',
                      backgroundColor: nb.id === notebookId ? '#d1fae5' : 'transparent',
                      marginBottom: '0.25rem',
                    }}
                  >
                    {/* clicking the title navigates into that notebook */}
                    <span
                      onClick={() => {
                        navigate(`/n/${nb.id}`)
                        if (isMobile) setLeftCollapsed(true);
                      }}
                      style={{ cursor: 'pointer', flex: 1 }}
                    >
                      {nb.title}
                    </span>

                    {/* trash icon to delete this notebook */}
                    <button
                      onClick={() => deleteNotebook(nb.id)}
                      style={{
                        background: 'none',
                        border: '1px solid #ccc',
                        cursor: 'pointer',
                        marginLeft: '0.1rem',
                        fontSize: '1.0rem',
                      }}
                      title="Delete this notebook"
                    >
                      🗑️
                    </button>
                  </div>
                ))}

                {/* "+" button at end of list */}
                <button
                  onClick={createNewNotebook}
                  style={{
                    width: "2.5rem",
                    height: "2.5rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "1px solid #ccc",
                    borderRadius: "4px",
                  }}
                  title="New notebook"
                >
                  🗒️
                </button>
              </div>
            </>
          )}
        </div>

        {/* “admin” grafana button */}
        {isAdmin && (
          <button
            onClick={() => {
              // This causes the browser to request /prometheus/ with the existing Firebase ID token
              window.open("/grafana/login", "_blank")
            }}
            style={{
              padding: "0.5rem",
              border: "none",
              background: "#eee",
              cursor: "pointer",
              alignSelf: "flex-start",
              fontSize: "1.1rem",
              marginTop: "0.5rem",
            }}
            title="Grafana dashboard"
          >
            📈
          </button>
        )}

        {/* “admin” prometheus button */}
        {isAdmin && (
          <button
            onClick={() => {
              // This causes the browser to request /prometheus/ with the existing Firebase ID token
              window.open("/prometheus/query", "_blank")
            }}
            style={{
              padding: "0.5rem",
              border: "none",
              background: "#eee",
              cursor: "pointer",
              alignSelf: "flex-start",
              fontSize: "1.1rem",
              marginTop: "0.5rem",
            }}
            title="Prometheus dashboard"
          >
            🔥
          </button>
        )}

        {/* “admin” mlflow button */}
        {isAdmin && (
          <button
            onClick={() => {
              // This causes the browser to request /mlflow/ with the existing Firebase ID token
              window.open("/mlflow/", "_blank")
            }}
            style={{
              padding: "0.5rem",
              border: "none",
              background: "#eee",
              cursor: "pointer",
              alignSelf: "flex-start",
              fontSize: "1.1rem",
              marginTop: "0.5rem",
            }}
            title="MLflow dashboard"
          >
            📦
          </button>
        )}

        {/* Logout button at the very bottom */}
        <button
          onClick={logout}
          style={{
            padding: '0.5rem',
            border: 'none',
            background: '#eee',
            cursor: 'pointer',
            alignSelf: 'flex-start',
            marginTop: "0.5rem",
            color: "red",
            fontSize: "1.5rem",
          }}
          title="Sign out"
        >
          ⏻
        </button>
      </div>

      {/* Chat and Paper Layout (your current layout) */}
      <div 
        className="main-chat-area" 
        style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '1rem', borderRight: '1px solid #ccc' }}
      >
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1rem' }}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              onClick={e => msg.query_id && handleMessageClick(e, msg.query_id)}
              style={{
                background: msg.query_id && selectedQueryIds.has(msg.query_id) ? '#f2fcf1' : '#f3f3f3',
                padding: '1rem',
                borderRadius: '6px',
                marginBottom: '0.5rem',
                cursor: msg.query_id ? 'pointer' : 'default',
                border: msg.query_id && selectedQueryIds.has(msg.query_id) ? '1px solid #2563eb' : 'none',
              }}
            >
              <strong>{msg.role === 'user' ? '🧑 You' : '🤖 asXai'}:</strong>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a({ href, children }) {
                    const paperId = href?.replace('#', '');
                    const isHighlighted = highlightedPaperIds.has(paperId);
                    return (
                      <a
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          const paperId = href?.replace('#', '');
                          setHighlightedPaperIds((prev) => {
                            const next = new Set(prev);

                            if (e.metaKey || e.ctrlKey) {
                              next.has(paperId) ? next.delete(paperId) : next.add(paperId);
                            } else {
                              if (prev.has(paperId) && prev.size === 1) {
                                next.clear();
                              } else {
                                next.clear();
                                next.add(paperId);
                              }
                            }

                            return next;
                          });
                        }}
                        style={{
                          color: isHighlighted ? '#3a6d2d' : '#2563eb',
                          textDecoration: 'underline',
                          fontWeight: isHighlighted ? 'bold' : 'normal',
                          cursor: 'pointer',
                        }}
                      >
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {isStreaming && (
            <div style={{
              textAlign: 'center',
              padding: '0.5rem',
              fontStyle: 'italic',
              color: '#2563eb',
              animation: 'pulse 1.5s infinite',
            }}>
              <span role="img" aria-label="thinking">🤖</span> Generating answer...
            </div>
          )}
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

      {/* Sidebar with top articles */}
      <div
        className="paper-sidebar"
        style={{
          flexShrink: 0,
          flexBasis: rightCollapsed ? '2.5rem' : '30%',
          transition: 'flex-basis 0.3s ease',
          borderLeft: '1px solid #ccc',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* collapse/expand toggle at top of right sidebar */}
        <button
          onClick={() => setRightCollapsed(!rightCollapsed)}
          style={{
            padding: '0.5rem',
            border: 'none',
            background: '#eee',
            cursor: 'pointer',
            alignSelf: 'flex-end',
          }}
        >
          {rightCollapsed ? '🗞️' : '🏛️'}
        </button>

        {!rightCollapsed && (
          <div style={{ padding: '1rem', overflowY: 'auto', flex: 1 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '1rem',
              }}
            >
              <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>Top Articles</h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <label
                      style={{
                        position: 'relative',
                        display: 'inline-block',
                        width: '40px',
                        height: '20px',
                      }}
                    >
                    </label>
                  </label>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label htmlFor="topK">TopK:</label>
                  <input
                    type="range"
                    id="topK"
                    min="5"
                    max="15"
                    step="5"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                  />
                  <span>{topK}</span>
                </div>
              </div>
            </div>
            {displayedPapers.map((p, i) => {
              const isExpanded = expandedIndexes.has(i);
              return (
                <div
                  key={p.paperId}
                  onClick={(e) => {
                    setRightCollapsed(false); // Auto‐open on paper click
                    const paperId = p.paperId;
                    setHighlightedPaperIds((prev) => {
                      const next = new Set(prev);
                      if (e.metaKey || e.ctrlKey) {
                        next.has(paperId) ? next.delete(paperId) : next.add(paperId);
                      } else {
                        if (prev.has(paperId) && prev.size === 1) {
                          next.clear();
                        } else {
                          next.clear();
                          next.add(paperId);
                        }
                      }
                      return next;
                    });
                  }}
                  style={{
                    marginBottom: '1rem',
                    padding: '1rem',
                    border: '1px solid #ccc',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    backgroundColor: highlightedPaperIds.has(p.paperId)
                      ? '#d1fae5'
                      : isExpanded
                      ? '#f9f9f9'
                      : '#fff',
                    transition: 'all 0.3s ease',
                  }}
                >
                  <div style={{ fontSize: '0.9rem', fontWeight: '600' }}>
                    <a
                      href={p.openAccessPdf}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: '#2563eb', textDecoration: 'underline' }}
                    >
                      {p.title}
                    </a>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>{p.authorName}</div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>{p.publicationDate}</div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>{p.paperId}</div>
                  {isExpanded && (
                    <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', fontStyle: 'italic' }}>
                      {p.best_chunks?.map((chunk, idx) => (
                        <div key={idx} style={{ marginBottom: '0.25rem' }}>
                          {chunk}
                        </div>
                      ))}
                    </div>
                  )}
                  <div
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleExpand(i);
                    }}
                    style={{
                      marginTop: '0.5rem',
                      fontSize: '0.75rem',
                      color: '#2563eb',
                      cursor: 'pointer',
                      userSelect: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.25rem',
                    }}
                  >
                    {isExpanded ? '▲ Show less' : '▼ Show more'}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
