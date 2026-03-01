import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import styles from './Navbar.module.css'

const Logo = () => (
  <svg width="44" height="40" viewBox="0 0 44 40" fill="none">
    <path d="M22 3L41 37H3L22 3Z" stroke="#ef4444" strokeWidth="2.2" fill="none"/>
    <text x="22" y="31" textAnchor="middle" fill="#ef4444" fontSize="16" fontWeight="bold" fontFamily="sans-serif">!</text>
    <ellipse cx="22" cy="32" rx="8" ry="3" fill="#ef4444" opacity="0.12"/>
    <circle cx="16" cy="32" r="2" fill="#ef4444" opacity="0.65"/>
    <circle cx="22" cy="31" r="2" fill="#ef4444" opacity="0.65"/>
    <circle cx="28" cy="32" r="2" fill="#ef4444" opacity="0.65"/>
  </svg>
)

const Navbar = ({ alertCount = 0 }) => {
  const { isAuthenticated, logout, user } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <nav className={styles.nav}>
      <div className={styles.logo} onClick={() => navigate('/')}>
        <Logo />
      </div>

      <div className={styles.links}>
        <a href="/" className={styles.link}>Home</a>
        <a href="/features" className={styles.link}>Features</a>
        <a href="/about" className={styles.link}>About Us</a>

        {isAuthenticated ? (
          <div className={styles.authSection}>
            {alertCount > 0 && (
              <div className={styles.alertBadge}>
                <span className={styles.alertDot} />
                {alertCount} Alert{alertCount > 1 ? 's' : ''}
              </div>
            )}
            <span className={styles.userName}>{user?.firstName || 'Admin'}</span>
            <button className={styles.outlineBtn} onClick={handleLogout}>Sign Out</button>
          </div>
        ) : (
          <button className={styles.outlineBtn} onClick={() => navigate('/auth')}>
            Sign Up / Login
          </button>
        )}
      </div>
    </nav>
  )
}

export default Navbar
