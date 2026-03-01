import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import Navbar from '../components/Navbar'
import styles from './AuthPage.module.css'

const AuthPage = () => {
  const navigate = useNavigate()
  const { login, register } = useAuth()
  const [isLogin,  setIsLogin]  = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [success,  setSuccess]  = useState('')

  const [form, setForm] = useState({
    firstName: '', lastName: '', email: '', password: '', confirmPassword: '',
  })

  const handleChange = (e) => {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }))
    setError(''); setSuccess('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')

    if (!isLogin && form.password !== form.confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    setLoading(true)
    try {
      if (isLogin) {
        await login(form)
        navigate('/dashboard')
      } else {
        await register(form)
        setSuccess('Account created! You can now sign in.')
        setIsLogin(true)
        setForm(prev => ({ ...prev, password: '', confirmPassword: '' }))
      }
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const switchMode = () => { setIsLogin(v => !v); setError(''); setSuccess('') }

  return (
    <div className={styles.page}>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.card}>

          <div className={styles.cardHeader}>
            <div className={styles.iconRing}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00e5cc" strokeWidth="1.5">
                <circle cx="12" cy="8" r="4"/>
                <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
              </svg>
            </div>
            <h1 className={styles.title}>
              {isLogin ? 'Log in to your account' : 'Create Account'}
            </h1>
            <p className={styles.subtitle}>
              {isLogin ? 'Access your Admin Control Panel' : 'Register as a new event admin'}
            </p>
          </div>

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.row}>
              <div className={styles.field}>
                <label className={styles.label}>First Name</label>
                <input className={styles.input} type="text" name="firstName"
                  value={form.firstName} onChange={handleChange} placeholder="John" required />
              </div>
              <div className={styles.field}>
                <label className={styles.label}>Last Name</label>
                <input className={styles.input} type="text" name="lastName"
                  value={form.lastName} onChange={handleChange} placeholder="Doe" required />
              </div>
            </div>

            <div className={styles.field}>
              <label className={styles.label}>E-mail</label>
              <input className={styles.input} type="email" name="email"
                value={form.email} onChange={handleChange} placeholder="admin@event.com" required />
            </div>

            <div className={isLogin ? styles.field : styles.row}>
              <div className={styles.field}>
                <label className={styles.label}>Password</label>
                <input className={styles.input} type="password" name="password"
                  value={form.password} onChange={handleChange} placeholder="••••••••" required />
              </div>
              {!isLogin && (
                <div className={styles.field}>
                  <label className={styles.label}>Confirm Password</label>
                  <input className={styles.input} type="password" name="confirmPassword"
                    value={form.confirmPassword} onChange={handleChange} placeholder="••••••••" required />
                </div>
              )}
            </div>

            {error   && <p className={styles.error}>{error}</p>}
            {success && <p className={styles.success}>{success}</p>}

            <button type="submit" className={styles.submitBtn} disabled={loading}>
              {loading
                ? <><span className={styles.spinner} />{isLogin ? 'Signing in…' : 'Creating account…'}</>
                : isLogin ? 'Sign In' : 'Sign Up'}
            </button>

            <p className={styles.switchRow}>
              {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
              <button type="button" className={styles.switchBtn} onClick={switchMode}>
                {isLogin ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}

export default AuthPage
