import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from './firebase-auth';
import { getAuth } from 'firebase/auth';
import './ChatApp.css';

// Helper: fetch with Firebase token
export async function authFetch(currentUser, url, options = {}) {
  if (!currentUser) {
    throw new Error("No authenticated user available for authFetch");
  }
  const token = await currentUser.getIdToken();
  const headers = {
    ...options.headers,
    Authorization: `Bearer ${token}`,
    "ngrok-skip-browser-warning": "1",
  };
  return fetch(url, { ...options, headers });
}

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { loading, signIn } = useAuth();
  const navigate = useNavigate();

  const writeIdTokenCookie = async () => {
    const currentUser = getAuth().currentUser;
    if (!currentUser) return;

    // Force a fresh token so we pick up any new custom claims (e.g. admin).
    const idTokenResult = await currentUser.getIdTokenResult(true);
    const token = idTokenResult.token;

    // Store it in a Secure, SameSite=Strict cookie:
    document.cookie = `id_token=${token}; Secure; SameSite=Strict; path=/`;
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      //Sign in with Firebase
      await signIn(email, password);

      //get the currentUser from Firebase Auth
      const currentUser = getAuth().currentUser;
      if (!currentUser) {
        throw new Error("Firebase user not available immediately after signIn");
      }
      await writeIdTokenCookie();
      ////Navigate to the /n to get a new notebookId from ChatApp
      navigate(`/n`);
    } catch (err) {
      console.error("Login failed:", err);
      alert("Login failed. Check credentials or sign up.");
    }
  };


  // While AuthProvider is initializing, show nothing (or a loader)
  if (loading) return null;

  return (
    <div style={containerStyle}>
      <form onSubmit={handleLogin} style={formStyle}>
        <img
          className= "logo-img"
          alt="asXai logo"
          style={{
            height: "3.5rem", // adjust size as needed
            objectFit: "contain",
            margin: 0
          }}
        />
        <h2 style={{ textAlign: "center" }}>Welcome Back</h2>
        <input
          type="email"
          autoComplete="username"
          value={email}
          onChange={e => setEmail(e.target.value)}
          placeholder="Email"
          required
          style={inputStyle}
        />
        <input
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder="Password"
          required
          style={inputStyle}
        />
        <button type="submit" style={loginBtnStyle}>Login</button>
        <div style={{ textAlign: "center", marginTop: "0.5rem" }}>
          <span>
            New here?{' '}
            <a
              href="#"
              onClick={e => { e.preventDefault(); navigate('/signup'); }}
              style={{ color: "#2563eb", textDecoration: "underline", cursor: "pointer" }}
            >
              Sign Up
            </a>
          </span>
        </div>
      </form>
    </div>
  );
}


const containerStyle = {
  height: "100vh",
  width: "100vw",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  backgroundColor: `var(--main-bg)`,
};
const formStyle = {
  backgroundColor: `var(--main-fg)`,
  padding: "2rem",
  borderRadius: "8px",
  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
  width: "320px",
  display: "flex",
  flexDirection: "column",
  gap: "1rem",
};
const inputStyle = {
  padding: "0.75rem",
  fontSize: "1rem",
  borderRadius: "4px",
  border: "1px solid #ccc",
  outline: "none",
};
const loginBtnStyle = {
  padding: "0.75rem",
  fontSize: "1rem",
  backgroundColor: "#2563eb",
  color: "white",
  border: "none",
  borderRadius: "4px",
  cursor: "pointer",
};

