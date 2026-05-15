import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import '@styles/global.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 2,
      retry: 2,
      refetchOnWindowFocus: false
    },
    mutations: {
      retry: 1
    }
  }
})

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Root element not found — ensure index.html has a div with id="root"')
}

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>
)