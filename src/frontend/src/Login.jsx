import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from './firebase-auth';
import { getAuth } from 'firebase/auth';

const API_URL = import.meta.env.VITE_API_URL;

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
  const { user, loading, signUp, signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

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

  const handleSignup = async () => {
    try {
      //Create user in Firebase
      await signUp(email, password);

      //get the currentUser from Firebase Auth
      const currentUser = getAuth().currentUser;
      if (!currentUser) {
        throw new Error("Firebase user not available immediately after signUp");
      }
      
      await writeIdTokenCookie();

      //Navigate to the /n to get a new notebookId from ChatApp
      navigate(`/n`);
    } catch (err) {
      console.error("Signup failed:", err);
      alert("Signup failed.");
    }
  };

  // While AuthProvider is initializing, show nothing (or a loader)
  if (loading) return null;

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#f5f5f5",
      }}
    >
      <form
        onSubmit={handleLogin}
        style={{
          background: "white",
          padding: "2rem",
          borderRadius: "8px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
          width: "320px",
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
        }}
      >
        <h2 style={{ textAlign: "center", marginBottom: "1rem" }}>
          Welcome Back to asXai
        </h2>

        <input
          type="email"
          autoComplete="username"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
          style={{
            padding: "0.75rem",
            fontSize: "1rem",
            borderRadius: "4px",
            border: "1px solid #ccc",
            outline: "none",
          }}
        />

        <input
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
          style={{
            padding: "0.75rem",
            fontSize: "1rem",
            borderRadius: "4px",
            border: "1px solid #ccc",
            outline: "none",
          }}
        />

        <button
          type="submit"
          style={{
            padding: "0.75rem",
            fontSize: "1rem",
            backgroundColor: "#2563eb",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Login
        </button>

        <button
          type="button"
          onClick={handleSignup}
          style={{
            padding: "0.75rem",
            fontSize: "1rem",
            backgroundColor: "#10b981",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Sign Up
        </button>
      </form>
    </div>
  );
}
