import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Layout, Button, FormInput, FormSelect, FormCheckbox, Alert } from '../components';
import { useAuthStore } from '../store/authStore';

export const Signup = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [userType, setUserType] = useState('operator');
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { signup } = useAuthStore();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email || !password || !confirmPassword) {
      setError('Please fill in all fields');
      return;
    }

    if (!email.includes('@')) {
      setError('Please enter a valid email');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!agreedToTerms) {
      setError('Please agree to terms and conditions');
      return;
    }

    setLoading(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500));

      signup({
        email,
        userType,
        device: navigator.userAgent,
      });

      navigate(userType === 'admin' ? '/admin-dashboard' : '/camera-dashboard');
    } catch {
      setError('Signup failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="max-w-md mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2 text-center">Create Account</h1>
          <p className="text-gray-600 text-center mb-8">Join Stampede Alert System</p>

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

            <FormSelect
              label="Account Type"
              options={[
                { value: 'operator', label: 'Camera Operator' },
                { value: 'admin', label: 'Administrator' },
              ]}
              value={userType}
              onChange={setUserType}
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

            <FormInput
              label="Confirm Password"
              type="password"
              placeholder="••••••••"
              value={confirmPassword}
              onChange={setConfirmPassword}
              required
            />

            <FormCheckbox
              label={
                <>
                  I agree to the{' '}
                  <a href="#" className="text-primary-600 hover:underline">
                    Terms and Conditions
                  </a>
                </>
              }
              checked={agreedToTerms}
              onChange={setAgreedToTerms}
            />

            <Button
              variant="primary"
              className="w-full"
              disabled={loading}
            >
              {loading ? 'Creating account...' : 'Create Account'}
            </Button>
          </form>

          <div className="mt-6 pt-6 border-t border-gray-200">
            <p className="text-gray-600 text-center">
              Already have an account?{' '}
              <Link to="/login" className="text-primary-600 font-semibold hover:text-primary-700">
                Sign in
              </Link>
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
};
