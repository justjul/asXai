import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from './firebase-auth';
import { getAuth } from 'firebase/auth';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const { loading, signUp } = useAuth();
  const navigate = useNavigate();

  const writeIdTokenCookie = async () => {
    const currentUser = getAuth().currentUser;
    if (!currentUser) return;
    const idTokenResult = await currentUser.getIdTokenResult(true);
    document.cookie = `id_token=${idTokenResult.token}; Secure; SameSite=Strict; path=/`;
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    if (password !== confirm) {
      alert("Passwords do not match");
      return;
    }
    try {
      await signUp(email, password);
      await writeIdTokenCookie();
      navigate('/n');
    } catch (err) {
      alert("Signup failed. Try again or use a different email.");
    }
  };

  if (loading) return null;

  return (
    <div style={containerStyle}>
      <form onSubmit={handleSignup} style={formStyle}>
        <h2 style={{ textAlign: "center" }}>Create an account</h2>
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
          autoComplete="new-password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder="Password"
          required
          style={inputStyle}
        />
        <input
          type="password"
          autoComplete="new-password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          placeholder="Confirm Password"
          required
          style={inputStyle}
        />
        <button type="submit" style={signupBtnStyle}>Sign Up</button>
        <div style={{ textAlign: "center", marginTop: "0.5rem" }}>
          <span>
            Already have an account?{' '}
            <a
              href="#"
              onClick={e => { e.preventDefault(); navigate('/'); }}
              style={{ color: "#2563eb", textDecoration: "underline", cursor: "pointer" }}
            >
              Login
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
  backgroundColor: "#f5f5f5",
};
const formStyle = {
  background: "white",
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
const signupBtnStyle = {
  ...loginBtnStyle,
  backgroundColor: "#10b981"
};