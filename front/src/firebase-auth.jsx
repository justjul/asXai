import { initializeApp } from 'firebase/app';
import {
  getAuth,
  signInWithEmailAndPassword as firebaseSignIn,
  createUserWithEmailAndPassword as firebaseSignUp,
  onAuthStateChanged,
  signOut as firebaseSignOut,
  sendEmailVerification,
  sendPasswordResetEmail
} from 'firebase/auth';
import { createContext, useContext, useEffect, useState } from 'react';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // sign up
  const signUp = async (email, password) => {
    const userCredential = await firebaseSignUp(auth, email, password);
    await sendEmailVerification(userCredential.user);
    await firebaseSignOut(auth);
    return userCredential;
  };

  // sign in
  const signIn = async (email, password) => {
    const userCredential = await firebaseSignIn(auth, email, password);
    if (!userCredential.user.emailVerified) {
      await sendEmailVerification(userCredential.user);
    }
    return userCredential;
  };

  // sign out
  const logout = () => firebaseSignOut(auth);

  // reset password
  const resetPassword = async (email) => {
    return await sendPasswordResetEmail(auth, email, {
      url: 'https://www.asxai.org',
      handleCodeInApp: false
    });
  };

  return (
    <AuthContext.Provider value={{ user, loading, signUp, signIn, logout, resetPassword }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
