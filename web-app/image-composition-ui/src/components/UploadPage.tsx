import { useRef, useState } from 'react'
import { Box, Button, Typography, LinearProgress, Alert, Grid, Card, CardContent } from '@mui/material'
import axios from 'axios'

type Prediction = { rule: string; value: number }
type ImageResult = { id: string; filename: string; predictions: Prediction[] }

export default function UploadPage() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<ImageResult[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleFiles = async (files: FileList | null) => {
    if (!files) return
    setLoading(true)
    setError(null)
    const form = new FormData()
    Array.from(files).forEach((file) => form.append('files', file))
    try {
      const res = await axios.post<ImageResult[]>('/upload', form, {
        baseURL: 'http://localhost:8000',
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResults(res.data)
    } catch (e: any) {
      setError(e.message)
    }
    setLoading(false)
  }

  return (
    <Box>
      <input
        type="file"
        ref={inputRef}
        multiple
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => handleFiles(e.target.files)}
      />
      <Button
        variant="contained"
        fullWidth
        size="large"
        onClick={() => inputRef.current?.click()}
        disabled={loading}
        sx={{ mb: 3, mt: 2 }}
      >
        Загрузить изображения
      </Button>
      {loading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error">{error}</Alert>}
      {results.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Оценка изображений
          </Typography>
          <Grid container spacing={2}>
            {results.map(res => (
              <Grid item xs={12} md={6} key={res.id}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" fontWeight={600}>{res.filename}</Typography>
                    <ul style={{ paddingLeft: 20 }}>
                      {res.predictions.map(pred =>
                        <li key={pred.rule}>
                          {pred.rule}: <b>{pred.value.toFixed(2)}</b>
                        </li>
                      )}
                    </ul>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  )
}
