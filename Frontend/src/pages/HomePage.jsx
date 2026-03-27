import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import styles from "./HomePage.module.css";
import StatCard from "./StatCard";
import statStyles from "./StatCard.module.css";
import Demo from "./Demo";
import Reveal from "../components/Reveal";
import { usePointerGlow } from "../hooks/usePointerGlow";
import { useMagneticHover } from "../hooks/useMagneticHover";

const HomePage = () => {
  const navigate = useNavigate();
  const heroGlow = usePointerGlow();
  const primaryMagnetic = useMagneticHover();
  const secondaryMagnetic = useMagneticHover();

  return (
    <div className={styles.page}>
      <Navbar />

      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>AI Crowd Intelligence Platform</div>
          <h1 className={styles.heroTitle}>
            Real-time vision for
            <span className={styles.accent}> high-stakes crowd safety</span>
          </h1>
          <p className={styles.heroDesc}>
            Crowd Sentinel tracks live inflow, outflow, density, and predicted
            pressure build-up across entry and exit zones so authorities can act
            before a crush event escalates.
          </p>

          <div className={styles.heroActions}>
            <button
              ref={primaryMagnetic.ref}
              {...primaryMagnetic.magneticProps}
              className={`${styles.primaryBtn} magneticButton`}
              onClick={() => navigate("/auth")}
            >
              Launch Control Room
            </button>
            <button
              ref={secondaryMagnetic.ref}
              {...secondaryMagnetic.magneticProps}
              className={`${styles.secondaryBtn} magneticButton`}
              onClick={() => navigate("/features")}
            >
              Explore Features
            </button>
          </div>

          <div className={styles.metricStrip}>
            <div className={styles.metric}>
              <span className={styles.metricValue}>Live Feeds</span>
              <span className={styles.metricLabel}>Visual monitoring across registered cameras</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>Alert States</span>
              <span className={styles.metricLabel}>Escalation from safe conditions to critical risk</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>Trend View</span>
              <span className={styles.metricLabel}>Operational history and forward-looking analysis</span>
            </div>
          </div>
        </div>

        <div className={styles.heroViz}>
          <div
            ref={heroGlow.ref}
            {...heroGlow.glowProps}
            className={`${styles.vizShell} interactiveGlow`}
          >
            <div className={styles.vizShellScanline} />
            <div className={styles.vizShellBeam} />
            <div className={styles.vizShellParticles}>
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
            <div className={styles.radarRingOuter}>
              <div className={styles.radarRingMid}>
                <div className={styles.radarCore}>
                  <div className={styles.signalPulse} />
                  <svg width="76" height="70" viewBox="0 0 44 40" fill="none">
                    <path
                      d="M22 3L41 37H3L22 3Z"
                      stroke="#ff6b5f"
                      strokeWidth="2.2"
                      fill="rgba(255,107,95,0.06)"
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
                    <circle cx="16" cy="33" r="2" fill="#ff6b5f" opacity="0.7" />
                    <circle cx="22" cy="32" r="2" fill="#ff6b5f" opacity="0.7" />
                    <circle cx="28" cy="33" r="2" fill="#ff6b5f" opacity="0.7" />
                  </svg>
                </div>
              </div>
            </div>

            <div className={styles.dataCardPrimary}>
              <span className={styles.dataLabel}>Signal Inputs</span>
              <strong>Density + Flow + Direction</strong>
              <span className={styles.dataHint}>Built from live camera observations</span>
            </div>

            <div className={styles.dataCardStack}>
              <div className={styles.dataCardSecondary}>
                <span className={styles.dataLabel}>Response Loop</span>
                <strong>Observe → Assess → Intervene</strong>
              </div>

              <div className={styles.dataCardTertiary}>
                <span className={styles.dataLabel}>Operator View</span>
                <strong>One dashboard for feeds, counts, alerts, and trends</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      <Reveal as="section" className={styles.storySection}>
        <div className={styles.storyHeader}>
          <p className={styles.storyEyebrow}>Why this matters</p>
          <h2 className={statStyles.statHeading}>
            Stampedes are fast, chaotic, and preventable when crowd stress is
            measured early.
          </h2>
          <p className={styles.storyText}>
            These visual summaries frame the operational problem this platform
            is built to solve.
          </p>
        </div>

        <div className={styles.storyGrid}>
          <Reveal delay={0}>
            <StatCard
              img="./src/assets/chart1(2).png"
              desc="Crowd disasters are not isolated incidents. Fatalities recur across geographies, proving the need for repeatable safety infrastructure rather than reactive intervention."
            />
          </Reveal>
          <Reveal delay={80}>
            <StatCard
              img="./src/assets/chart2(2).png"
              desc="The pattern is recurrent rather than isolated. The case for active monitoring is strongest when venues experience repeated surges and fluctuating movement over time."
            />
          </Reveal>
          <Reveal delay={160}>
            <StatCard
              img="./src/assets/chart4(1).png"
              desc="Narrow exits, unmanaged movement, and panic cascades show up repeatedly. These are operational signals that technology can surface before conditions collapse."
            />
          </Reveal>
          <Reveal delay={240}>
            <StatCard
              img="./src/assets/chart6.png"
              desc="As density rises, individual freedom of movement disappears quickly. Pressure compounds nonlinearly, which is why predictive visibility matters more than static counting."
            />
          </Reveal>
        </div>
      </Reveal>

      <Reveal as="section" className={styles.contactSection}>
        <div className={styles.contactIntro}>
          <p className={styles.storyEyebrow}>Contact</p>
          <h2 className={styles.contactTitle}>Bring the system into a real venue</h2>
          <p className={styles.contactText}>
            Share your event context and the team can help assess deployment,
            camera strategy, and operational fit.
          </p>
        </div>
        <div className={styles.homeForm}>
          <Demo />
        </div>
      </Reveal>

      <footer className={styles.footer}>
        <p className={styles.copyright}>Copyright © JSS STU 2025</p>
      </footer>
    </div>
  );
};

export default HomePage;
