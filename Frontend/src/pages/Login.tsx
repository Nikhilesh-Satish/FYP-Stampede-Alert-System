import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Layout, Button, FormInput, Alert } from '../components';
import { useAuthStore } from '../store/authStore';

export const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuthStore();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email || !password) {
      setError('Please fill in all fields');
      return;
    }

    if (!email.includes('@')) {
      setError('Please enter a valid email');
      return;
    }

    setLoading(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500));

      // For demo purposes, accept any email/password combination
      login({
        email,
        userType: email.includes('admin') ? 'admin' : 'operator',
        device: navigator.userAgent,
      });

      navigate(email.includes('admin') ? '/admin-dashboard' : '/camera-dashboard');
    } catch {
      setError('Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="max-w-md mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2 text-center">Welcome Back</h1>
          <p className="text-gray-600 text-center mb-8">Sign in to your account</p>

          {error && <Alert type="error">{error}</Alert>}

          <form onSubmit={handleSubmit} className="space-y-6">
            <FormInput
              label="Email Address"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={setEmail}
              required
            />

            <FormInput
              label="Password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={setPassword}
              required
            />

            <Button
              variant="primary"
              className="w-full"
              disabled={loading}
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </Button>
          </form>

          <div className="mt-6 pt-6 border-t border-gray-200">
            <p className="text-gray-600 text-center">
              Don't have an account?{' '}
              <Link to="/signup" className="text-primary-600 font-semibold hover:text-primary-700">
                Sign up
              </Link>
            </p>
          </div>

          <div className="mt-4 p-4 bg-blue-50 rounded-lg text-sm text-blue-800">
            <p className="font-semibold mb-2">Demo Credentials:</p>
            <p>Admin: admin@example.com / password</p>
            <p>Operator: operator@example.com / password</p>
          </div>
        </div>
      </div>
    </Layout>
  );
};
