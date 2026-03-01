import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth()

  if (loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: '#0f1923', flexDirection: 'column', gap: '1rem'
      }}>
        <div style={{
          width: 36, height: 36,
          border: '3px solid rgba(0,229,204,0.15)',
          borderTopColor: '#00e5cc',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
        <span style={{ fontFamily: 'Rajdhani, sans-serif', color: '#475569', letterSpacing: '0.1em' }}>
          LOADING
        </span>
      </div>
    )
  }

  return isAuthenticated ? children : <Navigate to="/auth" replace />
}

export default ProtectedRoute
