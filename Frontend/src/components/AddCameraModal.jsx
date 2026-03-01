import { useState } from 'react'
import { cameraApi } from '../api/services'
import styles from './AddCameraModal.module.css'

const AddCameraModal = ({ onClose, onSuccess }) => {
  const [form,    setForm]    = useState({ name: '', streamPath: '' })
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')

  const handleChange = (e) => {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }))
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name.trim() || !form.streamPath.trim()) {
      setError('Both fields are required.')
      return
    }
    setLoading(true)
    try {
      const result = await cameraApi.addCamera(form)
      onSuccess(result)
      onClose()
    } catch (err) {
      setError(err.message || 'Failed to add camera.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>

        <div className={styles.header}>
          <div className={styles.iconWrap}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00e5cc" strokeWidth="1.5">
              <path d="M23 7l-7 5 7 5V7z"/>
              <rect x="1" y="5" width="15" height="14" rx="2"/>
            </svg>
          </div>
          <div>
            <h2 className={styles.title}>Register Camera</h2>
            <p className={styles.subtitle}>Add a new surveillance feed</p>
          </div>
          <button className={styles.closeBtn} onClick={onClose}>✕</button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label className={styles.label}>Camera Name</label>
            <input
              className={styles.input}
              type="text"
              name="name"
              value={form.name}
              onChange={handleChange}
              placeholder="e.g. Gate A — Entry Camera"
              autoFocus
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Video Stream Path</label>
            <input
              className={styles.input}
              type="text"
              name="streamPath"
              value={form.streamPath}
              onChange={handleChange}
              placeholder="/streams/gate_a.mp4  or  rtsp://…"
            />
            <p className={styles.hint}>Local file path or RTSP/HTTP stream URL processed by your backend.</p>
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <div className={styles.actions}>
            <button type="button" className={styles.cancelBtn} onClick={onClose}>Cancel</button>
            <button type="submit" className={styles.submitBtn} disabled={loading}>
              {loading ? <><span className={styles.btnSpinner} />Registering…</> : 'Add Camera'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default AddCameraModal
