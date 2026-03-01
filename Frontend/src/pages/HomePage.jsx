import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import styles from "./HomePage.module.css";
import StatCard from "./StatCard";
import statStyles from "./StatCard.module.css";
import Demo from "./Demo";

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

      <h2 className={statStyles.statHeading}>
        <center>
          Why are stampedes a big threat? The statistics hold the answers
        </center>
      </h2>

      <StatCard
        img="./src/assets/chart1(2).png"
        desc="Stampede-related deaths are not limited to isolated regions — both India and the global community have experienced thousands of fatalities, showing that crowd disasters are a widespread and persistent public safety threat."
      />
      <StatCard
        img="./src/assets/chart2(2).png"
        desc="Stampedes occur regularly rather than rarely, resulting in continuous monthly loss of life and reinforcing the need for proactive crowd monitoring systems."
      />

      <StatCard
        img="./src/assets/chart4(1).png"
        desc="Physical constraints and behavioral triggers — such as narrow exits, panic, and unmanaged crowd flow — repeatedly emerge as key contributors to crowd disasters worldwide."
      />
      <StatCard
        img="./src/assets/chart6.png"
        desc="As crowd density increases, individual movement becomes restricted and pressure builds rapidly, dramatically raising the likelihood of a crush situation."
      />
      <div className={styles.homeForm}>
        <Demo />
      </div>
      <center>
        <footer className={styles.footer}>
          <p className={styles.copyright}>Copyright © JSS STU 2025</p>
        </footer>
      </center>
    </div>
  );
};

export default HomePage;
