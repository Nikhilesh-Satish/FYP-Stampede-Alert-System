import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import styles from "./Navbar.module.css";

const Logo = () => (
  <svg width="44" height="40" viewBox="0 0 44 40" fill="none">
    <path
      d="M22 3L41 37H3L22 3Z"
      stroke="#ff6b5f"
      strokeWidth="2.2"
      fill="rgba(255,107,95,0.08)"
    />
    <text
      x="22"
      y="31"
      textAnchor="middle"
      fill="#ff6b5f"
      fontSize="16"
      fontWeight="bold"
      fontFamily="Rajdhani, sans-serif"
    >
      !
    </text>
    <ellipse cx="22" cy="32" rx="8" ry="3" fill="#ff6b5f" opacity="0.18" />
    <circle cx="16" cy="32" r="2" fill="#ff6b5f" opacity="0.65" />
    <circle cx="22" cy="31" r="2" fill="#ff6b5f" opacity="0.65" />
    <circle cx="28" cy="32" r="2" fill="#ff6b5f" opacity="0.65" />
  </svg>
);

const Navbar = ({ alertCount = 0 }) => {
  const { isAuthenticated, logout, user } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <nav className={styles.nav}>
      <div className={styles.navGlow} />
      <div className={styles.logo} onClick={() => navigate("/")}>
        <div className={styles.logoMark}>
          <Logo />
        </div>
        <div className={styles.brandText}>
          <span className={styles.brandTitle}>Crowd Sentinel</span>
          <span className={styles.brandTag}>Predict. Prevent. Protect.</span>
        </div>
      </div>

      <div className={styles.links}>
        <NavLink
          to="/"
          className={({ isActive }) =>
            `${styles.link} ${isActive ? styles.activeLink : ""}`
          }
        >
          Home
        </NavLink>
        <NavLink
          to="/features"
          className={({ isActive }) =>
            `${styles.link} ${isActive ? styles.activeLink : ""}`
          }
        >
          Features
        </NavLink>
        <NavLink
          to="/about"
          className={({ isActive }) =>
            `${styles.link} ${isActive ? styles.activeLink : ""}`
          }
        >
          About
        </NavLink>

        {isAuthenticated ? (
          <div className={styles.authSection}>
            {alertCount > 0 && (
              <div className={styles.alertBadge}>
                <span className={styles.alertDot} />
                {alertCount} Alert{alertCount > 1 ? "s" : ""}
              </div>
            )}
            <div className={styles.userChip}>
              <span className={styles.userLabel}>Operator</span>
              <span className={styles.userName}>{user?.firstName || "Admin"}</span>
            </div>
            <button className={styles.outlineBtn} onClick={handleLogout}>
              Sign Out
            </button>
          </div>
        ) : (
          <button className={styles.outlineBtn} onClick={() => navigate("/auth")}>
            Access Console
          </button>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
