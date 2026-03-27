import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useLocation,
} from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import HomePage from "./pages/HomePage";
import AuthPage from "./pages/AuthPage";
import DashboardPage from "./pages/DashboardPage";
import "./App.css";
import About from "./pages/About";
import Feature from "./pages/Feature";

// Component to handle home page routing
const HomeRedirect = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          background: "#0f1923",
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            border: "3px solid rgba(0,229,204,0.15)",
            borderTopColor: "#00e5cc",
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
          }}
        />
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <HomePage />;
};

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AnimatedRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}

const AnimatedRoutes = () => {
  const location = useLocation();

  return (
    <div className="routeStage" key={location.pathname}>
      <Routes location={location}>
        <Route path="/" element={<HomeRedirect />} />
        <Route path="/auth" element={<AuthPage />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />
        <Route path="/about" element={<About />} />
        <Route path="/features" element={<Feature />} />
      </Routes>
    </div>
  );
};

export default App;
