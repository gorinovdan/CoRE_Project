import { useEffect, useState } from 'react'
import { Box, Typography, MenuItem, Select, FormControl, InputLabel, Button, Alert, CircularProgress } from '@mui/material'
import axios from 'axios'

type ImageMeta = { id: string; filename: string }

const RULES = [
  "center", "curved", "diagonal", "fill_the_frame", "pattern",
  "rule_of_thirds", "symmetric", "triangle", "vanishing_point",
  "golden_ratio", "horizontal", "radial", "vertical"
]

export default function InterpretationPage() {
  const [images, setImages] = useState<ImageMeta[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [selectedRule, setSelectedRule] = useState<number>(0)
  const [gradcamUrl, setGradcamUrl] = useState<string>('')
  const [shapUrl, setShapUrl] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Получить только метаинфу по id/filename
    axios.get<any[]>('/results', { baseURL: 'http://localhost:8000' })
      .then(res => setImages(res.data.map((d: any) => ({ id: d.id, filename: d.filename }))))
      .catch(() => { })
  }, [])

  const fetchInterpret = async (type: 'gradcam' | 'shap') => {
    if (!selectedId) return
    setLoading(true)
    setError(null)
    try {
      const url = `/` + type + `/${selectedId}?rule=${selectedRule}`
      const apiUrl = 'http://localhost:8000' + url
      // чтобы обновлялась картинка даже если id тот же:
      const urlWithCacheBusting = apiUrl + '&t=' + Date.now()
      if (type === 'gradcam') setGradcamUrl(urlWithCacheBusting)
      if (type === 'shap') setShapUrl(urlWithCacheBusting)
    } catch (e: any) {
      setError(e.message)
    }
    setLoading(false)
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Интерпретация (Grad-CAM / SHAP)
      </Typography>
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Изображение</InputLabel>
        <Select
          value={selectedId}
          label="Изображение"
          onChange={e => setSelectedId(e.target.value)}
        >
          {images.map(img =>
            <MenuItem value={img.id} key={img.id}>{img.filename}</MenuItem>
          )}
        </Select>
      </FormControl>
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Правило</InputLabel>
        <Select
          value={selectedRule}
          label="Правило"
          onChange={e => setSelectedRule(Number(e.target.value))}
        >
          {RULES.map((r, i) =>
            <MenuItem value={i} key={i}>{r}</MenuItem>
          )}
        </Select>
      </FormControl>
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <Button variant="contained" disabled={!selectedId} onClick={() => fetchInterpret('gradcam')}>Grad-CAM</Button>
        <Button variant="outlined" disabled={!selectedId} onClick={() => fetchInterpret('shap')}>SHAP</Button>
      </Box>
      {error && <Alert severity="error">{error}</Alert>}
      {loading && <CircularProgress />}
      {gradcamUrl && (
        <Box sx={{ mb: 2 }}>
          <Typography>Grad-CAM карта:</Typography>
          <img src={gradcamUrl} alt="GradCAM" style={{ maxWidth: 400, borderRadius: 8 }} />
        </Box>
      )}
      {shapUrl && (
        <Box>
          <Typography>SHAP карта:</Typography>
          <img src={shapUrl} alt="SHAP" style={{ maxWidth: 400, borderRadius: 8 }} />
        </Box>
      )}
    </Box>
  )
}
