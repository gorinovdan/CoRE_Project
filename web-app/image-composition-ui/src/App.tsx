import { useState } from 'react'
import { Container, Box, Typography, Tabs, Tab, Paper } from '@mui/material'
import UploadPage from './components/UploadPage'
import ResultsPage from './components/ResultsPage'
import InterpretationPage from './components/InterpretationPage'
import ExportPage from './components/ExportPage'
import { CloudUpload, ListAlt, Visibility, Download } from '@mui/icons-material'

function App() {
  const [tab, setTab] = useState(0)

  return (
    <Container maxWidth="md" sx={{ pt: 5, pb: 8 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
          Image Composition Analysis
        </Typography>
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          centered
          variant="fullWidth"
          sx={{ mb: 2 }}
        >
          <Tab icon={<CloudUpload />} label="Загрузка" />
          <Tab icon={<ListAlt />} label="Результаты" />
          <Tab icon={<Visibility />} label="Интерпретация" />
          <Tab icon={<Download />} label="Экспорт" />
        </Tabs>
        <Box>
          {tab === 0 && <UploadPage />}
          {tab === 1 && <ResultsPage />}
          {tab === 2 && <InterpretationPage />}
          {tab === 3 && <ExportPage />}
        </Box>
      </Paper>
      <Box textAlign="center" sx={{ mt: 4 }}>
        <Typography variant="caption" color="text.secondary">
          © {new Date().getFullYear()} ITMO · Diplom · ViT Composition Demo
        </Typography>
      </Box>
    </Container>
  )
}

export default App
