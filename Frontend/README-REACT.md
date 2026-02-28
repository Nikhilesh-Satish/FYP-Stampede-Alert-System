# Stampede Alert System - React Frontend

A modern React + Tailwind CSS web interface for the Stampede Alert System crowd monitoring platform.

## Tech Stack

- **React 18.2** - UI framework
- **Vite 5.0** - Build tool and dev server
- **Tailwind CSS 3.3** - Utility-first CSS framework
- **TypeScript 5.2** - Type safety
- **React Router v6** - Client-side routing
- **Zustand 4.4** - State management (auth)
- **Lucide React** - Icon library

## Getting Started

### Prerequisites
- Node.js 16+ and npm/yarn

### Installation

```bash
cd Frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

Output will be in the `dist/` directory.

### Lint

```bash
npm run lint
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Button.tsx      # Button variants (primary, secondary, danger, ghost)
│   ├── Modal.tsx       # Modal dialog component
│   ├── Toast.tsx       # Notification component
│   ├── Form.tsx        # Form inputs (Input, Select, Checkbox)
│   ├── Content.tsx     # Card, Badge, Alert, StatCard components
│   ├── Navbar.tsx      # Navigation bar with mobile menu
│   ├── Layout.tsx      # Main layout wrapper with sidebar
│   └── index.ts        # Component exports
├── pages/              # Page components
│   ├── Home.tsx        # Landing page
│   ├── About.tsx       # About page
│   ├── Login.tsx       # Login page
│   ├── Signup.tsx      # Registration page
│   ├── AdminDashboard.tsx  # Admin dashboard with camera/alert/user management
│   ├── CameraDashboard.tsx # Camera operator dashboard
│   └── index.ts        # Page exports
├── store/              # State management
│   └── authStore.ts    # Zustand auth store with localStorage persistence
├── App.tsx             # Main app component with routing
├── main.tsx            # React entry point
└── index.css           # Global styles and Tailwind directives
```

## Features

### Authentication
- Login/Signup pages with form validation
- Role-based access control (Admin/Operator)
- Persistent login with localStorage
- Protected routes

### Admin Dashboard
- **Overview**: System stats and recent alerts
- **Cameras**: Add, edit, delete camera feeds
- **Alerts**: View alert history with severity levels
- **Users**: Manage admin and operator users

### Camera Dashboard
- Live feed display area
- Real-time crowd count monitoring
- Capacity utilization percentage
- Alert history with severity indicators
- Export reports functionality

### UI Components
- Reusable button with 4 variants and 3 sizes
- Modal dialogs with customizable content
- Toast notifications (success, error, warning, info)
- Form inputs with validation
- Responsive layout with mobile menu
- Badge and alert components
- Sidebar navigation for dashboards

## Authentication

### Demo Credentials

For testing, use any email and password combination:
- **Admin**: Any email containing "admin" (e.g., admin@example.com)
- **Operator**: Any other email (e.g., operator@example.com)

Password must be at least 6 characters.

## Configuration

### Tailwind Theme

Custom colors defined in `tailwind.config.js`:
- Primary: #0066cc (blue)
- Secondary: #4ecdc4 (teal)

Easily customize by editing the config file.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Deployment

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Deploy to Netlify

```bash
npm run build
# Deploy the dist/ folder to Netlify
```

### Docker

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Contributing

1. Create feature branches
2. Follow the existing code style
3. Use TypeScript for type safety
4. Test your changes

## License

MIT

## Support

For support, contact: support@stampedealertsystem.com
