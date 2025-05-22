import { Box, Button, Typography, Alert } from '@mui/material'

export default function ExportPage() {
  const download = () => {
    window.open('http://localhost:8000/logs/export', '_blank')
  }
  return (
    <Box>
      <Typography variant="h6" gutterBottom>Экспорт данных</Typography>
      <Button variant="contained" color="success" onClick={download}>
        Скачать таблицу (CSV)
      </Button>
      <Alert severity="info" sx={{ mt: 2 }}>
        Будет выгружена полная таблица по всем изображениям и признакам.
      </Alert>
    </Box>
  )
}
