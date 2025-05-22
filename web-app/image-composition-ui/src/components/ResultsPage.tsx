import { useEffect, useState } from 'react'
import { Box, Typography, CircularProgress, Alert, TableContainer, Table, TableHead, TableRow, TableCell, TableBody, Paper } from '@mui/material'
import axios from 'axios'

type Prediction = { rule: string; value: number }
type ImageResult = { id: string; filename: string; predictions: Prediction[] }

export default function ResultsPage() {
  const [data, setData] = useState<ImageResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    axios.get<ImageResult[]>('/results', { baseURL: 'http://localhost:8000' })
      .then(res => setData(res.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <Box>
      <Typography variant="h6" gutterBottom>Все оценки</Typography>
      {loading && <CircularProgress />}
      {error && <Alert severity="error">{error}</Alert>}
      {data.length > 0 && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Изображение</TableCell>
                {data[0].predictions.map(pred =>
                  <TableCell key={pred.rule}>{pred.rule}</TableCell>
                )}
              </TableRow>
            </TableHead>
            <TableBody>
              {data.map(row => (
                <TableRow key={row.id}>
                  <TableCell>
                    <span style={{ fontSize: 13 }}>{row.filename}</span>
                  </TableCell>
                  {row.predictions.map(pred =>
                    <TableCell key={pred.rule}>{pred.value.toFixed(2)}</TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      {!loading && data.length === 0 && <Alert severity="info">Нет загруженных изображений.</Alert>}
    </Box>
  )
}
