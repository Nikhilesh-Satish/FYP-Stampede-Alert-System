# Stampede Alert System — React Frontend

## How to Run

```bash
# 1. Install dependencies
npm install

# 2. Create your .env file and point it at your backend
cp .env.example .env
# Then open .env and set:  VITE_API_URL=http://localhost:8000

# 3. Start the dev server
npm run dev
```

The app will open at **http://localhost:5173**

---

## File Structure

```
stampede-alert/
├── index.html                        ← Vite entry (must be at root)
├── vite.config.js                    ← Vite config
├── package.json
├── .env.example                      ← Copy to .env
│
└── src/
    ├── main.jsx                      ← React app mount point
    ├── App.jsx                       ← Router + AuthProvider wrapper
    ├── App.css                       ← Global reset + fonts
    │
    ├── api/
    │   ├── config.js                 ← API_BASE_URL, ENDPOINTS, THRESHOLDS
    │   └── services.js               ← authApi + cameraApi (all fetch calls)
    │
    ├── context/
    │   └── AuthContext.jsx           ← Global auth state (login/logout/token)
    │
    ├── utils/
    │   └── useCameraMonitor.js       ← Polls /cameras/counts every 10 min
    │
    ├── components/
    │   ├── Navbar.jsx / .module.css
    │   ├── ProtectedRoute.jsx        ← Redirects to /auth if not logged in
    │   ├── AlertBanner.jsx / .module.css
    │   ├── CameraCard.jsx / .module.css
    │   └── AddCameraModal.jsx / .module.css
    │
    └── pages/
        ├── HomePage.jsx / .module.css
        ├── AuthPage.jsx / .module.css  ← Sign Up + Login (toggled)
        └── DashboardPage.jsx / .module.css
```

---

## API Your Backend Must Implement

All protected routes require `Authorization: Bearer <token>` header.

### Auth

| Method | Path             | Body                                              | Response                              |
|--------|------------------|---------------------------------------------------|---------------------------------------|
| POST   | /auth/register   | `{ first_name, last_name, email, password }`      | `{ message }`                         |
| POST   | /auth/login      | `{ first_name, last_name, email, password }`      | `{ token, user: { id, firstName, lastName, email } }` |

### Cameras

| Method | Path                        | Body / Params              | Response                          |
|--------|-----------------------------|----------------------------|-----------------------------------|
| GET    | /cameras                    | —                          | `[{ id, name, stream_path }]`     |
| POST   | /cameras/add                | `{ name, stream_path }`    | `{ id, name, stream_path }`       |
| DELETE | /cameras/:id                | —                          | `{ message }`                     |
| GET    | /cameras/counts             | —                          | `[{ camera_id, count, timestamp }]` |
| GET    | /cameras/counts/:id         | —                          | `{ camera_id, count, timestamp }` |

> The frontend polls `/cameras/counts` every **10 minutes** automatically.

---

## Alert Thresholds

Edit `src/api/config.js` to change thresholds:

| Level    | Count    | Color      |
|----------|----------|------------|
| Safe     | 0–100    | Green      |
| Warning  | 101–200  | Amber      |
| Danger   | 201–300  | Red        |
| Critical | 300+     | Deep Red   |

---

## Environment Variables

| Variable       | Description                    | Default                  |
|----------------|--------------------------------|--------------------------|
| VITE_API_URL   | Your backend base URL          | http://localhost:8000    |
