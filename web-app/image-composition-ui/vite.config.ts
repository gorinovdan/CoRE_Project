import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload': 'http://localhost:8000',
      '/results': 'http://localhost:8000',
      '/gradcam': 'http://localhost:8000',
      '/shap': 'http://localhost:8000',
      '/logs': 'http://localhost:8000'
    }
  }
})
