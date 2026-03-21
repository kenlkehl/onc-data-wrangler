import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
      '/chat': 'http://localhost:8080',
      '/answer': 'http://localhost:8080',
      '/reset': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/summary': 'http://localhost:8080',
      '/summary-stats': 'http://localhost:8080',
      '/config': 'http://localhost:8080',
    },
  },
})
