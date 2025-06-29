import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 3) Allow HMR/WebSocket stanzas from both localhost and ngrok domain:
    allowedHosts: [
      'localhost',
      'www.asxai.org',
      'api.asxai.org'
    ],
  },
});
