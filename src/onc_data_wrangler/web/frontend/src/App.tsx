import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import { SetupPage } from '@/pages/SetupPage'
import { PipelinePage } from '@/pages/PipelinePage'
import { ConfigPage } from '@/pages/ConfigPage'
import { DataExplorerPage } from '@/pages/DataExplorerPage'
import { ChatPage } from '@/pages/ChatPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/ui/setup" element={<SetupPage />} />
            <Route path="/ui/pipeline" element={<PipelinePage />} />
            <Route path="/ui/config" element={<ConfigPage />} />
            <Route path="/ui/data" element={<DataExplorerPage />} />
            <Route path="/ui/chat" element={<ChatPage />} />
            <Route path="/ui" element={<Navigate to="/ui/setup" replace />} />
            <Route path="*" element={<Navigate to="/ui/setup" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
