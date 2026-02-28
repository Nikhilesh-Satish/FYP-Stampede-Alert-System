import { Menu, X, LogOut } from 'lucide-react';
import { useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { Link } from 'react-router-dom';

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { user, logout } = useAuthStore();

  const navLinks = [
    { label: 'Home', path: '/' },
    { label: 'About', path: '/about' },
  ];

  const dashboardLinks = user ? [
    { label: user.userType === 'admin' ? 'Admin Dashboard' : 'Camera Dashboard', path: user.userType === 'admin' ? '/admin-dashboard' : '/camera-dashboard' },
  ] : [];

  const allLinks = [...navLinks, ...dashboardLinks];

  return (
    <nav className="bg-white shadow-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center text-white font-bold group-hover:bg-primary-600 transition">
              SA
            </div>
            <span className="font-bold text-lg text-gray-800 hidden sm:inline">Stampede Alert</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            {allLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className="text-gray-600 hover:text-primary-500 transition font-medium text-sm"
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* User Menu or Auth Links */}
          <div className="hidden md:flex items-center gap-4">
            {user ? (
              <>
                <div className="flex items-center gap-2 px-3 py-1 bg-gray-100 rounded-lg">
                  <div className="w-8 h-8 bg-primary-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
                    {user.email[0].toUpperCase()}
                  </div>
                  <span className="text-sm text-gray-700">{user.email}</span>
                </div>
                <button
                  onClick={logout}
                  className="flex items-center gap-2 text-gray-600 hover:text-primary-500 transition"
                >
                  <LogOut size={18} />
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className="text-gray-600 hover:text-primary-500 transition font-medium">
                  Login
                </Link>
                <Link to="/signup" className="btn btn-primary text-sm">
                  Sign Up
                </Link>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden pb-4 border-t border-gray-200">
            <div className="flex flex-col gap-2 pt-4">
              {allLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
                  onClick={() => setIsOpen(false)}
                >
                  {link.label}
                </Link>
              ))}
              {user ? (
                <button
                  onClick={() => {
                    logout();
                    setIsOpen(false);
                  }}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition text-left flex items-center gap-2"
                >
                  <LogOut size={18} />
                  Logout
                </button>
              ) : (
                <>
                  <Link
                    to="/login"
                    className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
                    onClick={() => setIsOpen(false)}
                  >
                    Login
                  </Link>
                  <Link
                    to="/signup"
                    className="px-4 py-2 btn btn-primary text-center"
                    onClick={() => setIsOpen(false)}
                  >
                    Sign Up
                  </Link>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};
