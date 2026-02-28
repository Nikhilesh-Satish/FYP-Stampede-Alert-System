import { create } from 'zustand';

export interface User {
  id: string;
  email: string;
  userType: 'admin' | 'camera';
  fullName?: string;
  organizationName?: string;
  deviceName?: string;
  deviceId?: string;
  phone?: string;
}

interface AuthStore {
  user: User | null;
  isLoggedIn: boolean;
  login: (user: User) => void;
  logout: () => void;
  signup: (user: User) => void;
}

export const useAuthStore = create<AuthStore>((set) => {
  // Load from localStorage on init
  const savedUser = localStorage.getItem('stampede_user');
  const isLoggedIn = localStorage.getItem('stampede_is_logged_in') === 'true';

  return {
    user: savedUser ? JSON.parse(savedUser) : null,
    isLoggedIn,
    login: (user: User) => {
      localStorage.setItem('stampede_user', JSON.stringify(user));
      localStorage.setItem('stampede_is_logged_in', 'true');
      set({ user, isLoggedIn: true });
    },
    logout: () => {
      localStorage.removeItem('stampede_user');
      localStorage.removeItem('stampede_is_logged_in');
      set({ user: null, isLoggedIn: false });
    },
    signup: (user: User) => {
      localStorage.setItem('stampede_user', JSON.stringify(user));
      localStorage.setItem('stampede_is_logged_in', 'true');
      set({ user, isLoggedIn: true });
    },
  };
});
