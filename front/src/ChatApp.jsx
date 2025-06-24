import { useState, useRef, useMemo  } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { createParser } from 'eventsource-parser';
import { useAuth } from './firebase-auth';
import { getAuth } from "firebase/auth";
import './ChatApp.css';


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
  const [modelList, setModelList] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingStage, setStreamingStage] = useState('done'); //'generating', 'editing' or 'done'
  const [streamingNotebookIds, setStreamingNotebookIds] = useState(new Set());
  const streamingNotebookIdRef = useRef();
  const streamingMsgIndexRef = useRef();
  const [updatedNotebookIds, setUpdatedNotebookIds] = useState(new Set())
  const [openFilterPanels, setOpenFilterPanels] = useState(new Set());
  const [openReasoningPanels, setOpenReasoningPanels] = useState(new Set());
  const [papers, setPapers] = useState([]);
  const papersLoadingRef = useRef(false);
  const papersRef = useRef([]);
  const [selectedQueryIds, setSelectedQueryIds] = useState(new Set());
  const [paperMatchingQuery, setPaperMatchingQuery] = useState(new Set());
  const selPaperRefs = useRef({});
  const firstPaperRef = useRef(null);
  const [expandExcerptIndexes, setExpandExcerptIndexes] = useState(new Set());
  const [expandAbstractIndexes, setExpandAbstractIndexes] = useState(new Set());
  const [highlightedPaperIds, setHighlightedPaperIds] = useState(new Set());
  const [checkingTrash, setCheckingTrash] = useState(false);
  const [notebooks, setNotebooks] = useState([]);
  const [lastNotebookDeleted, setLastNotebookDeleted] = useState([]);
  const lastNotebookCreated = localStorage.getItem("lastNotebookCreated")
  const activeNotebook = notebooks.find(nb => nb.id === notebookId);
  const notebookTitle = activeNotebook ? activeNotebook.title : '';
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(true);
  const [dropdownOpen, setDropdownOpen] = useState(null);
  const [lockArticleList, setLockArticleList] = useState(false);
  const [editingMsg, setEditingMsg] = useState(null);
  const [editContent, setEditContent] = useState("");
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
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

  useEffect(() => {
    loadModelList();
  }, []);

  useEffect(() => {
    if (modelList.length > 0) {
      setSelectedModel((prev) => {
        if (!prev || !modelList.includes(prev)) {
          return modelList[0];
        }
        return prev;
      });
    }
  }, [modelList]);

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
  }, [notebookId, lastNotebookCreated, notebooks.length, updatedNotebookIds]);


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
    refreshMessages();
  }, [notebookId, notebooks, user, updatedNotebookIds]);

  useEffect(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      // Are we close to the bottom?
      const threshold = 100; // px from bottom—tune as you like
      const atBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight < threshold;

      setAutoScroll(atBottom);
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const selPaperIds = [...highlightedPaperIds];
    if (selPaperIds.length > 0) {
      const firstMatchEl = selPaperRefs.current[selPaperIds[0]];
      if (firstMatchEl) {
        firstMatchEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } else if (firstPaperRef.current) {
      firstPaperRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [highlightedPaperIds]);


  const refreshMessages = async () => {
    if (!notebookId) return;
    if (notebooks.length === 0) return;
    const exists = notebooks.some((nb) => nb.id === notebookId);
    if (!exists) return;

    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/chat/history`);
      if (!res.ok) throw new Error('Failed to load history');
      const data = await res.json();
      
      const seen = new Set();
      const allPapers = [];
      for (const msg of data) {
        if (Array.isArray(msg.papers)) {
          for (const p of msg.papers) {
            if (!seen.has(p.paperId)) {
              seen.add(p.paperId);
              allPapers.push(p);
            }
          }
        }
      }
      setPapers(allPapers);
      if (Array.isArray(data)) {
        const processed = data.map((msg) => {
          if (allPapers.length > 0) {
            return {
              ...msg,
              content: linkifyPaperIds(msg.content, allPapers),
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


  const createNewNotebook = async () => {
    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/new_task_id`,
          {method: 'GET', 
          headers: { 'Content-Type': 'application/json' }
        }
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

  
  const updateAllNotebook = async () => {
    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/update`,
        { method: 'POST', 
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedModel
          }),
        }
      );
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data['updated_notebooks'])) {
        setUpdatedNotebookIds(new Set(data['updated_notebooks']));
      } else {
        console.error("Invalid format for updated_notebooks:", data['updated_notebooks']);
      }
    } catch (err) {
      console.error("Failed to update notebooks:", err);
    }
  };


  const deleteNotebook = async (id, title) => {
    if (!window.confirm(`Are you sure you want to delete this notebook (${title})?`)) {
      return;
    }

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
      alert("Failed to delete notebook");
    }
  };

  const toggleDropdown = (id) => {
    setDropdownOpen(dropdownOpen === id ? null : id);
  };

  const toggleFilterPanel = (queryId) => {
    setOpenFilterPanels(prev => {
      const updated = new Set(prev);
      if (updated.has(queryId)) updated.delete(queryId);
      else updated.add(queryId);
      return updated;
    });
  };

  const toggleReasoningPanel = (queryId) => {
    setOpenReasoningPanels(prev => {
      const updated = new Set(prev);
      if (updated.has(queryId)) updated.delete(queryId);
      else updated.add(queryId);
      return updated;
    });
  };

  const renameNotebook = async (id, old_title) => {
    const newTitle = prompt("Enter new notebook title:", old_title);
      if (!newTitle || newTitle.trim() === "") {
        return;
      }

    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/${id}/rename/${newTitle}`,
        { method: 'PATCH' }
      );
      if (!res.ok) throw new Error(`Failed to rename notebook ${res.status}`);

      // refresh the notebooks array
      await fetchNotebooks();

    } catch (err) {
      console.error(`Failed to rename notebook ${id}:`, err);
      alert("Failed to rename notebook");
    }
  };

  const handleEdit = (msg) => {
    setEditingMsg(msg.query_id);
    setEditContent(msg.content);
  };

  const cancelEdit = () => {
    setEditingMsg(null);
    setEditContent("");
  };

  const sendEdit = async (id) => {
    handleSubmit({editedQuestion: editContent, query_id: id});
    cancelEdit();
    // 🔧 Optional: persist edit to backend here
  };

  const abortSubmit = async (id) => {
    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/${id}/abort`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);

      refreshMessages()
      setIsStreaming(false)

    } catch (err) {
      console.error(`Failed to abort generation ${id}:`, err);
      alert("Failed to stop generation");
    }
  };

  const handleSubmit = async ({
    editedQuestion = null, 
    query_id = null, 
    mode = 'reply', 
    search_query = null,
    nbId = null
  } = {}) => {
    const finalQuestion = editedQuestion ?? question;
    const notebookChatId = nbId ?? notebookId

    if (!finalQuestion.trim() && !query_id) return;

    let queryId;
    console.log(query_id)
    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookChatId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({
          message: finalQuestion,
          query_id, 
          search_query,
          topK: topK,
          paperLock: lockArticleList,
          mode: mode,
          model: selectedModel}),
      });
      if (!res.ok) throw new Error('Failed to submit chat message');
      const data = await res.json();
      queryId = data.query_id;
    } catch (err) {
      console.error('Failed to submit chat message:', err);
      return;
    }

    if (notebookChatId === notebookId) {
      if (finalQuestion.trim()) {
        setMessages((prev) => {
          const existingIdx = prev.findIndex(
            (msg) => msg.query_id === queryId && msg.role === 'user'
          );

          const newMessage = { role: 'user', content: finalQuestion, query_id: queryId };

          if (existingIdx !== -1) {
            const updated = [...prev];
            updated[existingIdx] = newMessage;
            return updated;
          } else {
            return [...prev, newMessage];
          }
        });

        setQuestion('');
      }

      
      setMessages((prev) => {
        let insertAfterIdx;
        if (mode !== 'expand') {
          insertAfterIdx = prev.findIndex(msg => msg.query_id === queryId);
        } else {
          insertAfterIdx = prev.findLastIndex(msg => msg.query_id === queryId);
        }
        streamingMsgIndexRef.current = insertAfterIdx + 1;
        if (insertAfterIdx !== -1) {
          const newMessages = [...prev];
          newMessages.splice(insertAfterIdx + 1, 0, {
            role: 'assistant',
            content: '',
            query_id: queryId,
          });
          return newMessages;
        } else {
          return [...prev, { role: 'assistant', content: '', query_id: queryId }];
        }
      });
      
    }

    streamAnswer(notebookChatId, queryId);
  };

  const streamAnswer = async (notebookStreamId, query_id) => {

    const maxRetries = 3;
    let retryCount = 0;

    let assistantResponse = '';
    let success = false;

    const startStreaming = async () => {
      setIsStreaming(true);
      setStreamingStage('generating');
      setStreamingNotebookIds(prev => {
        const next = new Set(prev);
        next.add(notebookStreamId);
        return next;
      });
      
      streamingNotebookIdRef.current = notebookStreamId;

      try {
        const url = `${API_URL}/notebook/${notebookStreamId}/chat/stream`;
        const res = await authFetch(user, url);

        if (!res.ok || !res.body) {
          throw new Error("Failed to open stream.");
        }

        setPapers([]);
        papersRef.current = [];

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        const parser = createParser({
          onEvent: (event) => {
            if (streamingNotebookIdRef.current !== notebookId) return;
            
            if (event.data === '<EDITING>') {
              setStreamingStage('editing');
              return; 
            }

            if (event.data === '<CLEAR>') {
              assistantResponse = '';
              return; 
            }
            
            if (event.data) {
              if (!papersLoadingRef.current && papersRef.current.length === 0 && assistantResponse.length > 100) {
                papersLoadingRef.current = true;
                loadPapers(notebookId, query_id).then((loadedPapers) => {
                  setPapers(loadedPapers);
                  papersRef.current = loadedPapers;
                  papersLoadingRef.current = false;
                });
              }

              if (event.data === '<END_OF_MESSAGE>') {
                reader.cancel();
                success = true;

                setSelectedQueryIds(prev => {
                  const next = new Set(prev);
                  next.add(query_id);
                  return next;
                });
                return;
              }

              assistantResponse += '\n' + event.data;
              setMessages((prev) => {
                const newMessages = [...prev];

                let content = assistantResponse;
                if (papersRef.current.length > 0) {
                  content = linkifyPaperIds(content, papersRef.current);
                }

                const idx = streamingMsgIndexRef.current;

                if (idx !== undefined && idx >= 0 && idx < newMessages.length) {
                  newMessages[idx] = { ...newMessages[idx], content };
                } else {
                  newMessages.push({ role: 'assistant', content, query_id });
                }

                return newMessages;
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
        if (retryCount === 0 || success) {
          setIsStreaming(false)
          setStreamingStage('done');
          setStreamingNotebookIds(prev => {
            const next = new Set(prev);
            next.delete(notebookStreamId);
            return next;
          });
        };
      }
    };

    startStreaming();
  };

  const loadPapers = async (notebookId, query_id) => {
    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/papers/${query_id}`);
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

  const loadModelList = async () => {
    try {
      const res = await authFetch(
        user,
        `${API_URL}/notebook/models`,
        { method: 'GET' }
      );
      if (!res.ok) throw new Error('Failed to load list of available models');
      const data = await res.json();
      setModelList(data.model_list);
    } catch (err) {
      console.error('Failed to load list of available models:', err);
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
    // setRightCollapsed(false);

    // rebuilding `papers` from all selected messages:
    if (newSelected.size === 0) {
      setPapers([]); 
      return;
    }

    // Collect all papers from every selected assistant message
    const allPapers = [];
    for (const id of newSelected) {
      const msg = messages.find(
        m => m.query_id === id && Array.isArray(m.papers)
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
    setExpandExcerptIndexes(new Set()); // reset any expanded indexes
    setExpandAbstractIndexes(new Set());
  };

  const deleteQuery = async (query_id) => {
    if (!window.confirm("Are you sure you want to delete this message?")) {
      return;
    }

    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/content/${query_id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json'},
      });

      refreshMessages();

    } catch (err) {
      console.error("Failed to delete message", err);
      alert("Failed to delete message");
    }
  }

  const deleteChatfrom = async (query_id, keepUsermsg = false) => {
    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/back/${query_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({ keepUsermsg: keepUsermsg })
      });

      if (!res.ok) throw new Error('Failed to delete chat messages');

      refreshMessages();

    } catch (err) {
      console.error("Failed to delete messages", err);
      alert("Failed to delete messages");
    }
  }

  const toggleExcerptExpand = (paperId) => {
    setExpandExcerptIndexes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(paperId)) {
        newSet.delete(paperId);
      } else {
        newSet.add(paperId);
      }
      return newSet;
    });
  };

  const toggleAbstractExpand = (paperId) => {
    setExpandAbstractIndexes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(paperId)) {
        newSet.delete(paperId);
      } else {
        newSet.add(paperId);
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


  const ScorePaper = async (paperId, score) => {
    try {
      const res = await authFetch(user, `${API_URL}/notebook/${notebookId}/scores/${paperId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({ user_score: score })
      });

      if (!res.ok) throw new Error('Failed to delete paper');

      refreshMessages();

    } catch (err) {
      console.error("Failed to delete paper", err);
      alert("Failed to delete paper");
    }
  }

  const allReferencedPapers = useMemo(() => {
    const seen = new Set();
    const all = [];
    for (const msg of messages) {
      if (Array.isArray(msg.papers)) {
        for (const p of msg.papers) {
          if (!seen.has(p.paperId) && ((checkingTrash && p.user_score < 0) || (!checkingTrash && p.user_score >= 0))) {
            seen.add(p.paperId);
            all.push(p);
          }
        }
      }
    }
    return all;
  }, [messages, checkingTrash]);

  const displayedPapers = useMemo(() => {
    if (selectedQueryIds.size === 0) {
      setPaperMatchingQuery(new Set())
      return allReferencedPapers;
    }

    const selectedPaperIds = new Set(
      messages
      .filter((msg) => selectedQueryIds.has(msg.query_id))
      .flatMap((msg) =>
        (msg.papers || []).map((p) => p.paperId)
      )
    );

    setPaperMatchingQuery(selectedPaperIds)

    const selected = [];
    const others = [];

    for (const p of allReferencedPapers) {
      if (selectedPaperIds.has(p.paperId)) {
        selected.push(p);
      } else {
        others.push(p);
      }
    }

    return [...selected, ...others];
  }, [selectedQueryIds, allReferencedPapers, messages]);

  // const displayedPapers = selectedQueryIds.size > 0 ? papers : allReferencedPapers;

  return (
    <div style={{flex: 1, display: 'flex', height: '100dvh', width: '100vw', position: 'relative' }}>
      {/* Left Sidebar */}
      <div
        className={`notebook-sidebar`}
        style={{
          width: leftCollapsed ? '0rem' : isMobile ? '90vw': '20%',
          position: isMobile ? 'absolute' : 'relative',
          left: 0,
          top: 0,
          height: isMobile ? '100vh' : '100%',
          background: 'var(--main-bg)',
          zIndex: isMobile ? 20 : 1,
          boxShadow: isMobile && !leftCollapsed ? '2px 0 12px rgba(0,0,0,0.09)' : 'none',
          transition: 'width 0.3s ease',
          borderRight: isMobile ? 'none' : '1px solid var(--main-border)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {!leftCollapsed && (
           <div 
           style={{
            display: 'flex', 
            overflowY: 'auto',
            justifyContent: 'space-between',  // <-- This is key
            alignItems: 'center',
            width: '100%',
            marginBottom: '0.5rem',
            }}>
            <h2 style={{
              marginTop: '1%',
              marginBottom: '1%', 
              marginLeft: '4%', 
              fontWeight: 'bold',
              color: 'var(--main-font-color)',
              fontFamily: 'var(--main-font)',
              }}>
                Notebooks
            </h2>
            <button
              onClick={() => setLeftCollapsed((c) => !c)}
              style={{
                border: 'none',
                background: 'transparent',
                cursor: 'pointer',
                fontSize: '1.3rem',
                padding: '0.5rem',
                marginRight: '5%',
              }}
              title={leftCollapsed ? "Open notebooks sidebar" : "Collapse notebooks sidebar"}
            >
              <img
                  src={leftCollapsed ? "/book_closed_icon.svg" : "/book_open_icon.svg"}
                  alt={leftCollapsed ? "Open notebook icon" : "Closed notebook icon"}
                  style={{
                    height: '1.7em', // or adjust as needed for your top bar
                    width: '1.7em',
                    display: 'block',
                    marginRight: '5%',
                    pointerEvents: 'none', // so clicks reach the button
                    background: 'transparent',
                  }}
                />
            </button>
          </div>
        )}
        <div style={{ flex: 1, overflowY: "auto", padding: leftCollapsed ? 0 : "0.0rem" }}>
          {!leftCollapsed && (
            <>
              {/* Notebook list container—make this grow to fill the middle */}
              <div style={{
                padding: '0.5rem', 
                flex: 1, 
                overflowY: 'auto',
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  width: '100%', // make sure it spans the container
                  padding: '0.0rem',
                  marginTop: '0.5rem',
                  }}
                >
                  <button
                    onClick={createNewNotebook}
                    style={{
                      width: "2.5rem",
                      height: "2.5rem",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: 'none',
                      borderRadius: "4px",
                      background: 'var(--main-bg)',
                    }}
                    title="New notebook"
                  >
                    <img
                      src= "/book_add_icon.svg"
                      alt= "New notebook"
                      style={{
                        height: '1.7em', // or adjust as needed for your top bar
                        width: '1.7em',
                        display: 'block',
                        marginRight: '1%',
                        pointerEvents: 'none', // so clicks reach the button
                        background: 'transparent',
                      }}
                    />
                  </button>
                  <button
                    onClick={updateAllNotebook}
                    style={{
                      width: "2.5rem",
                      height: "2.5rem",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      marginRight: '5%',
                      border: 'none',
                      borderRadius: "4px",
                      background: 'var(--main-bg)',
                      color: 'var(--main-font-color)',
                    }}
                    title="Update all Notebooks"
                  >
                    ↻
                  </button>
                </div>
                {notebooks.map((nb) => (
                  <div
                    key={nb.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '0.25rem 0.5rem',
                      borderRadius: '4px',
                      backgroundColor: 'transparent',
                      border: nb.id === notebookId ? '1px solid rgb(124, 223, 172)' : 'none',
                      marginBottom: '0.25rem',
                      fontFamily: 'var(--main-font)',
                      color: 'var(--main-font-color)',
                    }}
                  >
                    {/* clicking the title navigates into that notebook */}
                    <span
                      onClick={() => {
                        navigate(`/n/${nb.id}`)
                        setUpdatedNotebookIds(prev => {
                          const next = new Set(prev);
                          next.delete(nb.id);
                          return next;
                        });
                        if (isMobile) setLeftCollapsed(true);
                      }}
                      style={{ cursor: 'pointer', flex: 1 }}
                    >
                      {nb.title}
                      {updatedNotebookIds.has(nb.id) && <sup style={{ marginLeft: '0.3em', fontSize: '0.5em' }}>🔴</sup>}
                    </span>

                    <div style={{ position: 'relative' }}>
                      <button
                        onClick={() => toggleDropdown(nb.id)}
                        style={{
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          display: 'block',
                          marginRight: '1%',
                          fontSize: '1.2rem',
                        }}
                        title="Notebook actions"
                      >
                        ⋮
                      </button>

                      {dropdownOpen === nb.id && (
                        <div
                          style={{
                            position: 'absolute',
                            top: '100%',
                            right: 0,
                            background: 'white',
                            border: '1px solid #ccc',
                            borderRadius: '4px',
                            zIndex: 10,
                            boxShadow: '0px 2px 8px rgba(0,0,0,0.2)',
                            fontSize: '0.9rem',
                            padding: '0.2rem',
                          }}
                        >
                          <div
                            onClick={() => {
                              handleSubmit({editedQuestion: "*notebook update*", nbId: nb.id});
                              setDropdownOpen(null);
                            }}
                            style={{ padding: '0.3rem 0.6rem', cursor: 'pointer' }}
                          >
                            Update
                          </div>
                          <div
                            onClick={() => {
                              renameNotebook(nb.id, nb.title);
                              setDropdownOpen(null);
                            }}
                            style={{ padding: '0.3rem 0.6rem', cursor: 'pointer' }}
                          >
                            Rename
                          </div>
                          <div
                            onClick={() => {
                              deleteNotebook(nb.id, nb.title);
                              setDropdownOpen(null);
                            }}
                            style={{ padding: '0.3rem 0.6rem', cursor: 'pointer', color: 'red' }}
                          >
                            Delete
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
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
              background: "transparent",
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
              background: "transparent",
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
              background: "transparent",
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

        {/* Center: logout */}
          <button
            onClick={logout}
            style={{
              padding: "0.5rem",
              border: '1px solid var(--main-border)',
              background: "transparent",
              cursor: "pointer",
              justifyContent: "center",
              display: "flex",
              alignItems: "center", 
              fontSize: "1.1rem",
              margin: "2%",
              color: 'red'
            }}
            title="Sign out"
          >
            <img
              src= "/asXai_signout_icon.svg"
              alt= "Sign out"
              style={{
                height: '1.7em', // or adjust as needed for your top bar
                width: '1.7em',
                display: 'block',
                pointerEvents: 'none', // so clicks reach the button
                background: 'transparent',
              }}
            />
          </button>
      </div>
      {/* Backdrop for left sidebar on mobile */}
      {isMobile && !leftCollapsed && (
        <div
          onClick={() => setLeftCollapsed(true)}
          style={{
            position: 'fixed',
            top: 0, left: 0, width: '100vw', height: '100vh',
            background: 'rgba(0,0,0,0.2)',
            zIndex: 15,
          }}
        />
      )}

      <div 
        className="main-chat-area" 
        style={{ flex: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          background: 'var(--main-bg)',
          padding: 0}}
      >
        {/* --- TOP BAR (just above chat, not above sidebars) --- */}
        <div
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 10,
            height: '3.5rem',
            background: 'var(--main-bg)',
            borderBottom: '1px solid var(--main-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: 0,
          }}
        >
          {/* Left: collapse left sidebar */}
          
          <div style={{ flex: 1, display: "flex", justifyContent: "flex-start", paddingLeft: "1rem" }}>
            {leftCollapsed && (
              <button
              onClick={() => setLeftCollapsed((c) => !c)}
              style={{
                border: 'none',
                background: 'transparent',
                cursor: 'pointer',
                fontSize: '1.3rem',
                padding: '0.5rem',
                borderRadius: '50%',
              }}
              title={leftCollapsed ? "Open notebooks sidebar" : "Collapse notebooks sidebar"}
            >
              <img
                src={leftCollapsed ? "/book_closed_icon.svg" : "/book_open_icon.svg"}
                alt={leftCollapsed ? "Open notebook icon" : "Closed notebook icon"}
                style={{
                  height: '1.7em', // or adjust as needed for your top bar
                  width: '1.7em',
                  display: 'block',
                  margin: '0.5 rem',
                  pointerEvents: 'none', // so clicks reach the button
                }}
              />
            </button>
            )}
            {/* Notebook Title */}
            <span
              style={{
                fontWeight: "600",
                fontSize: "1.2rem",
                marginLeft: "0.75rem",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                maxWidth: "18vw",  // prevents long titles from breaking layout
                color: 'var(--main-font-color)',
                fontFamily: 'var(--main-font)',
              }}
            >
              {notebookTitle}
            </span>
            <select
              value={selectedModel}
              onFocus={loadModelList}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                flex: 1,
                maxWidth: '30%', // limit width so it fits between buttons
                margin: '0 0.5rem',
                borderRadius: '4px',
                border: '1px solid var(--main-border)',
                background: 'var(--main-bg)',
                color: 'var(--main-font-color)',
                fontFamily: 'inherit',
                fontSize: '0.9rem',
              }}
              title="Select model"
            >
              <option value="">Select model</option>
              {modelList.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
          {/* Center: Logo */}
          <div style={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "center"
          }}>
            <img
              className= "logo-img"
              alt="asXai logo"
              style={{
                height: "3.5rem", // adjust size as needed
                objectFit: "contain",
                margin: 0
              }}
            />
          </div>
          {/* Right: collapse right sidebar */}
          <div style={{ flex: 1, display: "flex", justifyContent: "flex-end", paddingRight: "1rem" }}>
            {rightCollapsed && (
              <button
              onClick={() => setLockArticleList((c) => !c)}
              style={{
                border: lockArticleList ? '1px solid rgb(236, 116, 230)' : 'none',
                background: 'transparent',
                cursor: 'pointer',
                fontSize: '1.3rem',
                padding: '0.5rem',
                borderRadius: '50%',
              }}
              title={lockArticleList ? "Unlock Article List" : "Lock Article List"}
            >
              <img
                src={lockArticleList ? "/lock_closed_icon.svg" : "/lock_open_icon.svg"}
                alt={lockArticleList ? "Locked articles icon" : "Unlocked articles icon"}
                style={{
                  height: '1.7em', // or adjust as needed for your top bar
                  width: '1.7em',
                  display: 'block',
                  margin: '0.5 rem',
                  pointerEvents: 'none', // so clicks reach the button
                }}
              />
            </button>
            )}
            {rightCollapsed && (
              <button
              onClick={() => setRightCollapsed((c) => !c)}
              style={{
                border: 'none',
                background: 'transparent',
                cursor: 'pointer',
                fontSize: '1.3rem',
                padding: '0.5rem',
                borderRadius: '50%',
              }}
              title={rightCollapsed ? "Open papers sidebar" : "Collapse papers sidebar"}
            >
              <img
                src={rightCollapsed ? "/academia_closed_icon.svg" : "/academia_open_icon.svg"}
                alt={rightCollapsed ? "Open articles icon" : "Closed articles icon"}
                style={{
                  height: '1.7em', // or adjust as needed for your top bar
                  width: '1.7em',
                  display: 'block',
                  margin: '0.5 rem',
                  pointerEvents: 'none', // so clicks reach the button
                }}
              />
            </button>
            )}
          </div>
        </div>
        {/* --- END TOP BAR --- */}

        <div 
          style={{ flex: 1, 
            overflowY: 'auto',
            marginBottom: '1rem',
            background: 'var(--main-bg)',
            border: 'none'
            }} ref={chatContainerRef}>
          {messages.map((msg, idx) => {
            const isSelected = msg.query_id && selectedQueryIds.has(msg.query_id);
            const isUpdate = msg?.mode === "update";

            const backgroundColor = isSelected
              ? 'var(--main-fg-sel)'
              : isUpdate
              ? 'var(--main-fg-update)'
              : 'var(--main-fg)';

            const font_size = '1rem'

            const marginLeft = isUpdate
              ? '15%' 
              : msg.role === 'user' 
              ? '15%'
              : '5%';
                  
            const marginRight = isUpdate
              ? '15%' 
              : msg.role === 'user' 
              ? '5%'
              : '5%';

            return (
              <div
                key={idx}
                style={{
                  background: backgroundColor,
                  padding: '1rem',
                  borderRadius: '1%',
                  cursor: 'default',
                  border: "none",
                  fontSize: font_size,
                  marginLeft: marginLeft,
                  marginRight: marginRight,
                  marginBottom: '0%',
                  marginTop: msg.role === 'user' ? '2%' : '1%',
                }}
              >
              <div 
                style={{ 
                  display: 'flex',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  alignItems: 'center',
                  width: '100%', // make sure it spans the container
                  padding: '0.0rem',
                  marginTop: '0.5rem',
                }}
                onClick={e => msg.query_id && handleMessageClick(e, msg.query_id)}
              >
                <img
                  src={msg.role === 'user' ? "/asXai_user_black_icon.svg" : "/asXai_robot_black_icon.svg"}
                  alt={msg.role === 'user' ? "User" : "asXai"}
                  style={{
                    height: '2em',
                    verticalAlign: 'middle',
                  }}
                />
                {msg.role === 'assistant' && (
                  <div 
                    style={{ 
                      fontSize: '0.7em',
                      fontStyle: 'italic',
                    }}
                  >
                  {msg.model?.split('/').at(-1) || 'unknown'}
                  </div>
                )}
              </div>
                {msg.role === 'assistant' && openReasoningPanels.has(msg.query_id) && (
                  <>
                  <div
                    style={{
                      fontStyle: 'italic',
                      fontFamily: 'var(--main-font)',
                      fontSize: '0.75rem',
                      display: 'block',
                      marginTop: '0.2rem',
                      marginBottom: '0.2rem',
                    }}
                  >
                    Reasoning:
                    {msg.think}
                  </div>
                </>
              )}

              {msg.role === 'user' && editingMsg === msg.query_id ? (
                <div>
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    style={{ width: '100%', minHeight: '5rem' }}
                  />
                  <div style={{ marginTop: '0.5rem', textAlign: 'right' }}>
                    <button onClick={() => sendEdit(msg.query_id)}>Send</button>
                    <button onClick={() => cancelEdit()}>Cancel</button>
                  </div>
                </div>
              ) : 
                (
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
                            setRightCollapsed(false)
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
                            color: isHighlighted ? 'var(--main-font-link-sel)' : 'var(--main-font-link)' ,
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
              )}

              {msg.query_id && (
                <div style={{
                  display: 'flex',
                  flexDirection: 'row-reverse',
                  flex: 2,
                  gap: '0.5rem',
                }}>
                  <span
                    style={{
                      cursor: 'pointer',
                      fontSize: '0.8em',
                      color: 'red',
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteQuery(msg.query_id);
                    }}
                    title="Delete message"
                  >
                    🗑️
                  </span>


                  {msg.role === 'user' &&
                    msg.mode !== 'update' &&
                    <span
                      style={{
                        cursor: 'pointer',
                        fontSize: '0.8em',
                        color: 'orange',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        cancelEdit()
                        handleEdit(msg);
                      }}
                      title="Edit message"
                    >
                      ✏️
                    </span>
                  }

                  {msg.search_query && (
                      <span
                        style={{
                          cursor: 'pointer',
                          fontSize: '0.8em',
                        }}
                        onClick={() => toggleFilterPanel(msg.query_id)}
                        title={openFilterPanels.has(msg.query_id) ? 'Hide filters' : 'Show filters'}
                      >
                        {openFilterPanels.has(msg.query_id) ? '▲' : '▼ '}
                      </span>
                  )}
                  {msg.role === 'assistant' &&
                    msg.mode !== 'update' &&
                      msg.mode !== 'expand' &&
                    <span
                      style={{
                        cursor: 'pointer',
                        fontSize: '0.8em',
                        color: 'blue',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSubmit({query_id: msg.query_id, mode: 'expand'});
                      }}
                      title="More detail"
                    >
                      🔎
                    </span>
                  }
                  {msg.role === 'assistant' && msg.think.length > 20 && 
                    <span
                      style={{
                        cursor: 'pointer',
                        fontSize: '0.8em',
                      }}
                      onClick={() => toggleReasoningPanel(msg.query_id)}
                      title={openReasoningPanels.has(msg.query_id) ? 'Hide Reasoning' : 'Show Reasoning'}
                    >
                      {openReasoningPanels.has(msg.query_id) ? 'ᝰ.ᐟ' : '𖡎'}
                    </span>
                  }
                </div>
              )}


              {msg.search_query && (
                <div style={{ marginTop: '0.5rem' }}>
                  {openFilterPanels.has(msg.query_id) && (
                    <div style={{
                      backgroundColor: selectedQueryIds.has(msg.query_id) ? 'var(--main-fg-sel)' : 'var(--main-fg)',
                      border: '1px solid var(--main-border)',
                      padding: '0.75rem',
                      borderRadius: '15px',
                      marginTop: '0.2rem',
                      marginBottom: '1rem',
                      fontSize: '0.75rem',
                    }}>
                      <label
                        style={{
                          fontWeight: 'bold',
                          fontStyle: 'italic',
                          fontFamily: 'var(--main-font)',
                          fontSize: '0.75rem',
                          display: 'block',
                          margintop: '0.2rem',
                          marginBottom: '0.2rem'
                        }}
                      >
                          Queries <br />
                        <textarea
                          style={{
                            width: '100%',
                            backgroundColor: 'var(--main-fg)',
                            borderRadius: '15px',
                          }}
                          rows={2*msg.search_query.queries.length + 2}
                          defaultValue={'👉' + msg.search_query.queries.join('\n👉')}
                          onChange={(e) => msg.search_query.queries = e.target.value.split('\n').strip('👉')}
                        />
                      </label>

                      <label
                        style={{
                          fontWeight: 'bold',
                          fontStyle: 'italic',
                          fontFamily: 'var(--main-font)',
                          fontSize: '0.75rem',
                          display: 'block',
                          marginBottom: '0.2rem'
                        }}
                      >
                        Authors names<br />
                        <input
                          type="text"
                          style={{
                            width: '100%',
                            backgroundColor: 'var(--main-fg)',
                            border: '1px solid var(--main-border)',
                            borderRadius: '50px',
                          }}
                          defaultValue={msg.search_query.authorName || ''}
                          placeholder="name1, name2, ..."
                          onChange={(e) => msg.search_query.authorName = e.target.value}
                        />
                      </label>

                      <div
                        style={{
                          display: 'flex',
                          gap: '1rem',
                          marginBottom: '0.5rem',
                          fontFamily: 'var(--main-font)',
                        }}
                      >
                        <div style={{ flex: 1 }}>
                          <label style={{ fontWeight: 'bold', fontStyle: 'italic', fontSize: '0.75rem' }}>
                            Start Date<br />
                            <input
                              type="text"
                              style={{
                                width: '100%',
                                backgroundColor: 'var(--main-fg)',
                                border: '1px solid var(--main-border)',
                                borderRadius: '50px',
                              }}
                              defaultValue={msg.search_query.publicationDate_start || ''}
                              placeholder="yyyy-mm-dd"
                              onChange={(e) => msg.search_query.publicationDate_start = e.target.value}
                            />
                          </label>
                        </div>

                        <div style={{ flex: 1 }}>
                          <label style={{ fontWeight: 'bold', fontStyle: 'italic', fontSize: '0.75rem' }}>
                            End Date<br />
                            <input
                              type="text"
                              style={{
                                width: '100%',
                                backgroundColor: 'var(--main-fg)',
                                border: '1px solid var(--main-border)',
                                borderRadius: '50px',
                              }}
                              defaultValue={msg.search_query.publicationDate_end || ''}
                              placeholder="yyyy-mm-dd"
                              onChange={(e) => msg.search_query.publicationDate_end = e.target.value}
                            />
                          </label>
                        </div>
                      </div>

                      <div style={{
                        width: '100%',
                        alignSelf: 'center',
                        textAlign: 'center',
                        marginTop: '0.5rem',
                        marginBottom: '0.0rem',
                        fontSize: '1.2rem',
                        }}>
                        <button
                          title='Save and submit'
                          style={{
                            backgroundColor: selectedQueryIds.has(msg.query_id) ? 'var(--main-fg-sel)': 'var(--main-fg)',
                            width: '10%',
                            alignSelf: 'center',
                            border: '1px solid var(--main-border)',
                            borderRadius: '25px',
                            padding: '0.5rem 1rem',
                            cursor: 'pointer',
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSubmit({editedQuestion: msg.content, query_id: msg.query_id, mode: 'reply', search_query: msg.search_query});
                            setOpenFilterPanels(prev => {
                              const next = new Set(prev);
                              next.delete(msg.query_id);
                              return next;
                            });
                          }}
                        >
                          🔁
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
        <div style={{
          background: 'var(--main-bg)',
          margin: "1%",
          display: "flex",
          // alignItems: "flex-center"
        }}>
          {streamingNotebookIds.has(notebookId) && (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              padding: '0.5rem',
            }}>
              <div style={{
                width: '24px',
                height: '24px',
                border: '3px solid #c3dafe',
                borderTop: '3px solid #2563eb',
                borderRadius: '25px',
                animation: 'spin 1s linear infinite',
              }} />
            </div>
          )}
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder= {(streamingNotebookIds.has(notebookId) && isStreaming) ? streamingStage : "Ask a question..."}
            rows={2}
            style={{
              background: 'var(--main-bg)',
              flex: 1,
              resize: "vertical",          // allow user to drag if you want
              minHeight: "2.5rem",
              maxHeight: "6rem",
              padding: "0.5rem",
              border: "1px solid var(--main-border)",
              borderRadius: '25px',
              fontSize: "1rem", 
              lineHeight: 1.3,
              boxSizing: "border-box",
            }}
            onKeyDown={e => {
              // Optional: send on Ctrl+Enter
              if (e.shiftKey && e.key === "Enter") handleSubmit();
            }}
          />
          <button
            onClick={() => (!streamingNotebookIds.has(notebookId) ? handleSubmit() : abortSubmit(notebookId))}
            style={{
              width:'5%',
              padding: '0.75rem 0.5rem',
              backgroundColor: '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              fontSize: '0.75rem',
              justifyContent: 'center',
              height: 'fit-content',
              alignSelf: 'flex-end',
              fontFamily: 'var(--main-font)',
              margin: '1%'
            }}
          >
            {!streamingNotebookIds.has(notebookId) ? 'Send' : '⏹'}
          </button>
        </div>
      </div>



      {/* Sidebar with top articles */}
      <div
        className="paper-sidebar"
        style={{
          width: rightCollapsed ? '0rem' : isMobile ? '90vw': '30%',
          position: isMobile ? 'absolute' : 'relative',
          right: 0,
          top: 0,
          height: isMobile ? '100vh' : '100%',
          background: 'var(--main-bg)',
          zIndex: isMobile ? 20 : 1,
          boxShadow: isMobile && !rightCollapsed ? '2px 0 12px rgba(0,0,0,0.09)' : 'none',
          transition: 'width 0.3s ease',
          borderLeft: isMobile ? 'none' : '1px solid var(--main-border)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {!rightCollapsed && (
          <div 
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',}}>
            <button
              onClick={() => setRightCollapsed((c) => !c)}
              style={{
                border: 'none',
                background: 'transparent',
                alignSelf: "flex-start",
                cursor: 'pointer',
                fontSize: '1.3rem',
                padding: '0.5rem',
                marginTop: "0.5rem",
                marginLeft: "1%",
              }}
              title={rightCollapsed ? "Open papers sidebar" : "Collapse papers sidebar"}
            >
              <img
                  src={rightCollapsed ? "/academia_closed_icon.svg" : "/academia_open_icon.svg"}
                  alt={rightCollapsed ? "Open articles icon" : "Closed articles icon"}
                  style={{
                    height: '1.7em', // or adjust as needed for your top bar
                    width: '1.7em',
                    display: 'block',
                    pointerEvents: 'none', // so clicks reach the button
                  }}
                />
            </button>
            <h2 
              style={{
                fontSize: '1.25rem',
                fontWeight: 'bold',
                color: 'var(--main-font-color)',
                fontFamily: 'var(--main-font)',
                }}>
              Top Articles
            </h2>
            <button
              onClick={() => setLockArticleList(l => !l)}
              title={lockArticleList ? "Unlock articles" : "Lock articles"}
              style={{
                border: lockArticleList ? '1px solid rgb(236, 116, 230)' : 'none',
                background: 'transparent',
                cursor: 'pointer',
                padding: 0,
                margin: '2%',
                fontSize: '1.5rem',
                lineHeight: 0, 
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <img
                  src={lockArticleList ? "/lock_closed_icon.svg" : "/lock_open_icon.svg"}
                  alt={lockArticleList ? "Article list locked" : "Article list unlocked"}
                  style={{
                    height: '2em', // or adjust as needed for your top bar
                    width: '2em',
                    display: 'block',
                    pointerEvents: 'none', // so clicks reach the button
                  }}
                />
            </button>
          </div>
        )}

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
            </div>
            {displayedPapers.map((p, idx) => {
              const isExcerptExpanded = expandExcerptIndexes.has(p.paperId);
              const isAbstractExpanded = expandAbstractIndexes.has(p.paperId);
              return (
                <div
                  key={p.paperId}
                  ref={(el) => {
                    if (highlightedPaperIds.has(p.paperId)) {
                      selPaperRefs.current[p.paperId] = el;
                    }
                    if (idx === 0) {
                      firstPaperRef.current = el;
                    }
                  }}
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
                    border: paperMatchingQuery.has(p.paperId) ? '4px solid var(--main-fg-sel)' : 'var(--main-fg)',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    backgroundColor: (highlightedPaperIds.has(p.paperId) && !checkingTrash)
                      ? 'var(--main-fg-sel)'
                      : checkingTrash
                      ? 'rgb(240, 208, 223)'
                      : 'var(--main-fg)',
                    transition: 'all 0.3s ease',
                  }}
                >
                  <div style={{ fontSize: '0.9rem', fontWeight: '600' }}>
                    <a
                      href={p.openAccessPdf}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: 'var(--main-font-link)', textDecoration: 'underline' }}
                    >
                      {p.title}
                    </a>
                  </div>
                  <div style={{ fontSize: '0.75rem'}}>{p.authorName}</div>
                  <div style={{ fontSize: '0.75rem'}}>{p.venue +' (' + p.publicationDate + ')'}</div>
                  <div style={{ fontSize: '0.75rem'}}>{p.paperId} / score {p.score?.toFixed(2)}</div>
                  <div
                    style={{
                      marginTop: '0.5rem',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      userSelect: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      flex: 1,
                      gap: '0.25rem',
                    }}
                  >
                    <span
                      onClick={(e) => {
                      e.stopPropagation();
                      toggleAbstractExpand(p.paperId);
                    }}
                    >
                      {isAbstractExpanded ? '▲ Abstract' : '▼ Abstract'}
                    </span>
                    <span
                      onClick={(e) => {
                      e.stopPropagation();
                      toggleExcerptExpand(p.paperId);
                    }}
                    >
                      {isExcerptExpanded ? '▲ Excerpt' : '▼ Excerpt'}
                    </span>
                    <span
                      title = {!checkingTrash ? 'delete' : 'restore'}
                      onClick={(e) => {
                      e.stopPropagation();
                      const score = !checkingTrash ? -1 : 0;
                      ScorePaper(p.paperId, score);
                    }}
                    >
                       {!checkingTrash ? '🗑️' : '↩'}
                    </span>
                  </div>

                  {isAbstractExpanded && (
                    <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', fontStyle: 'normal' }}>
                      {p.abstract}
                    </div>
                  )}
                  {isExcerptExpanded && (
                    <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', fontStyle: 'italic' }}>
                      {p.best_chunks?.map((chunk, idx) => (
                        <div key={idx} style={{ marginBottom: '0.25rem' }}>
                          {chunk}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
        {!rightCollapsed && (
          <div 
            style={{ 
              display: 'flex', 
              alignItems: 'right', 
              justifyContent: 'space-between',}}>
            <button
              onClick={() => setCheckingTrash(l => !l)}
              title={checkingTrash ? "Back to Articles" : "Trashed Articles"}
              style={{
                border: checkingTrash ? '1px solid rgb(236, 64, 150)' : 'none',
                background: checkingTrash ? 'rgb(240, 208, 223)' : 'transparent',
                cursor: 'pointer',
                padding: 0,
                margin: '2%',
                fontSize: '1.5rem',
                lineHeight: 0, 
                alignItems: 'center',
                justifyContent: 'center',
                height: '2em', // or adjust as needed for your top bar
                width: '2em',
                display: 'block',
              }}
            >
              ♻️
            </button>
            <div style={{ display: 'flex', alignItems: 'right', gap: '0.1rem' }}>
              <label style={{ display: 'flex', alignItems: 'right', gap: '0.1rem' }}>
                <label
                  style={{
                    position: 'relative',
                    display: 'inline-block',
                    width: '5%',
                    height: '2%',
                  }}
                >
                </label>
              </label>
            </div>
            <div style={{ 
              display: 'flex', 
              marginRight: '2%', 
              gap: '0rem',
              color: 'var(--main-font-color)',
              fontFamily: 'var(--main-font)',
              }}>
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
        )}
      </div>
      {/* Backdrop for right sidebar on mobile */}
      {isMobile && !rightCollapsed && (
        <div
          onClick={() => setRightCollapsed(true)}
          style={{
            position: 'fixed',
            top: 0, right: 0, width: '100vw', height: '100vh',
            background: 'rgba(0,0,0,0.2)',
            zIndex: 15,
          }}
        />
      )}
    </div>
  );
}
