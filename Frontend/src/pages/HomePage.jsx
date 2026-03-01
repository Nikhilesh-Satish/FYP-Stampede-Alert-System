import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import styles from "./HomePage.module.css";

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <div className={styles.page}>
      <Navbar />

      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>AI-Powered Safety Solution</div>
          <h1 className={styles.heroTitle}>
            Real-Time
            <br />
            <span className={styles.accent}>Stampede Alert</span>
            <br />
            System
          </h1>
          <p className={styles.heroDesc}>
            Using computer vision, people-counting algorithms, and automated
            alert mechanisms, the system continuously monitors crowd size and
            movement at entry and exit points. When the crowd exceeds safe
            limits, it instantly alerts authorities — enabling quick
            intervention and reducing the risk of stampedes.
          </p>
          <div className={styles.heroActions}>
            <button
              className={styles.primaryBtn}
              onClick={() => navigate("/auth")}
            >
              Get Started
            </button>
            <button
              className={styles.secondaryBtn}
              onClick={() => navigate("/auth")}
            >
              Admin Login
            </button>
          </div>
        </div>

        <div className={styles.heroViz}>
          <div className={styles.ring3}>
            <div className={styles.ring2}>
              <div className={styles.ring1}>
                <svg width="72" height="66" viewBox="0 0 44 40" fill="none">
                  <path
                    d="M22 3L41 37H3L22 3Z"
                    stroke="#ef4444"
                    strokeWidth="2.2"
                    fill="rgba(239,68,68,0.04)"
                  />
                  <text
                    x="22"
                    y="31"
                    textAnchor="middle"
                    fill="#ef4444"
                    fontSize="16"
                    fontWeight="bold"
                    fontFamily="sans-serif"
                  >
                    !
                  </text>
                  <circle cx="16" cy="33" r="2" fill="#ef4444" opacity="0.7" />
                  <circle cx="22" cy="32" r="2" fill="#ef4444" opacity="0.7" />
                  <circle cx="28" cy="33" r="2" fill="#ef4444" opacity="0.7" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature cards */}
      

      <footer className={styles.footer}>
        <p className={styles.copyright}>Copyright © JSS STU 2025</p>
      </footer>
    </div>
  );
};

export default HomePage;
