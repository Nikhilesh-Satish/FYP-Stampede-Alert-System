# 🚀 Quick Start Guide - React Frontend

## Installation & Setup (2 minutes)

### 1. Install Dependencies
```bash
cd Frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

The app opens automatically at **http://localhost:5173**

## Testing the App (5 minutes)

### Home Page
- Visit http://localhost:5173
- See hero section, features, and stats
- Click "Get Started" or "Learn More"

### Signup
- Go to /signup (or click "Sign Up")
- Fill in:
  - Email: `user@example.com`
  - Account Type: Choose "Camera Operator" or "Administrator"
  - Password: `password123`
  - Confirm Password: `password123`
  - Check "I agree to Terms"
- Click "Create Account"
- Redirects to appropriate dashboard

### Login
- Go to /login (or click "Sign In")
- Email: `admin@example.com` (auto-detected as admin)
- Password: `password123`
- Click "Sign In"
- Redirects to Admin Dashboard

### Admin Dashboard
Four sections accessible from sidebar:

**Overview Tab**
- System statistics
- Recent alerts with severity levels

**Cameras Tab**
- Card view of 3 sample cameras
- Edit/Delete buttons
- "Add Camera" button opens modal
- Modal form for new camera details

**Alerts Tab**
- Table view of alert history
- Severity badges
- Sortable/filterable (ready for API)

**Users Tab**
- Table of system users
- Edit/Delete actions per user

### Camera Dashboard
- Sidebar with camera list
- Click cameras to switch views
- Each camera shows:
  - Live feed placeholder
  - Current crowd count
  - Maximum capacity
  - Occupancy percentage
  - Recent alerts
  - Action buttons

### About Page
- Company mission statement
- Team section
- Technology overview
- Contact information

## File Structure

```
src/
├── components/           # Reusable UI components
│   ├── Button.tsx       # 4 variants: primary, secondary, danger, ghost
│   ├── Form.tsx         # Input, Select, Checkbox fields
│   ├── Modal.tsx        # Dialog boxes
│   ├── Toast.tsx        # Notifications
│   ├── Content.tsx      # Card, Badge, Alert, StatCard
│   ├── Navbar.tsx       # Top navigation
│   ├── Layout.tsx       # Page wrapper + sidebar
│   └── index.ts         # Exports all components
├── pages/               # Full page components
│   ├── Home.tsx         # Landing page
│   ├── About.tsx        # About page
│   ├── Login.tsx        # Login form
│   ├── Signup.tsx       # Registration form
│   ├── AdminDashboard.tsx
│   ├── CameraDashboard.tsx
│   └── index.ts         # Exports all pages
├── store/               # State management
│   └── authStore.ts     # User authentication (Zustand)
├── App.tsx              # Router setup
├── main.tsx             # React entry point
└── index.css            # Tailwind + custom styles
```

## Common Tasks

### Add a New Page
1. Create `src/pages/MyPage.tsx`:
```tsx
import { Layout } from '../components';

export const MyPage = () => (
  <Layout>
    <h1>My Page</h1>
  </Layout>
);
```

2. Add route in `src/App.tsx`:
```tsx
import { MyPage } from './pages/MyPage';

<Route path="/my-page" element={<MyPage />} />
```

### Add a New Component
1. Create `src/components/MyComponent.tsx`:
```tsx
interface MyComponentProps {
  title: string;
  onAction?: () => void;
}

export const MyComponent = ({ title, onAction }: MyComponentProps) => (
  <button onClick={onAction}>{title}</button>
);
```

2. Export in `src/components/index.ts`:
```tsx
export { MyComponent } from './MyComponent';
```

### Access Authentication
```tsx
import { useAuthStore } from '../store/authStore';

export const MyComponent = () => {
  const { user, login, logout } = useAuthStore();
  
  return <div>{user?.email}</div>;
};
```

### Build for Production
```bash
npm run build
```
Output in `dist/` folder - ready to deploy

## Customization

### Change Colors
Edit `tailwind.config.js`:
```js
theme: {
  colors: {
    'primary': {
      500: '#your-color',
      // ...
    }
  }
}
```

### Change Typography
Edit `src/index.css`:
```css
@layer base {
  body {
    @apply text-base;
  }
}
```

## Deployment Options

### Vercel (Recommended)
```bash
npm i -g vercel
vercel
```

### Netlify
```bash
npm run build
# Drag & drop 'dist' folder to Netlify
```

### Docker
```bash
docker build -t stampede-alert .
docker run -p 80:80 stampede-alert
```

## Troubleshooting

### Port 5173 already in use
```bash
npm run dev -- --port 3000
```

### Components not importing
Make sure to:
1. Export from component file
2. Re-export in `index.ts`
3. Import from `../components`

### TypeScript errors
Run: `npm run lint`

### Need to reset auth
- Open DevTools Console
- Type: `localStorage.removeItem('auth-store')`
- Refresh page

## Available Scripts

```bash
npm run dev      # Start dev server
npm run build    # Production build
npm run lint     # Check TypeScript/ESLint
npm run preview  # Preview production build
```

## Next: Backend Integration

To connect to your Python backend:

1. **Set API Base URL** in `.env`:
```env
VITE_API_BASE_URL=http://localhost:3000
```

2. **Create API helper** in `src/api/client.ts`:
```tsx
const API_BASE = import.meta.env.VITE_API_BASE_URL;

export const api = {
  login: (email: string, password: string) =>
    fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      body: JSON.stringify({ email, password })
    })
};
```

3. **Use in components**:
```tsx
import { api } from '../api/client';
const response = await api.login(email, password);
```

## Questions?

Check these files for more info:
- `README-REACT.md` - Full documentation
- `MIGRATION-SUMMARY.md` - What was built
- `tailwind.config.js` - Theme customization
- `vite.config.ts` - Build configuration

Happy coding! 🎉
