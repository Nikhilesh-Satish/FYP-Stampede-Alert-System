import Navbar from "../components/Navbar";
import styles from "./Feature.module.css";
import Demo from "./Demo";
import Reveal from "../components/Reveal";
import { usePointerGlow } from "../hooks/usePointerGlow";

const features = [
  {
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#65f2d7" strokeWidth="1.5">
        <path d="M23 7l-7 5 7 5V7z" />
        <rect x="1" y="5" width="15" height="14" rx="2" />
      </svg>
    ),
    title: "Live Camera Monitoring",
    desc: "Register surveillance cameras and track crowd levels across entry and exit points in real time.",
  },
  {
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#ffb84d" strokeWidth="1.5">
        <path d="M10.3 21h3.4M12 3a7 7 0 0 1 7 7c0 2.5 1 4 1 6H4c0-2 1-3.5 1-6a7 7 0 0 1 7-7z" />
      </svg>
    ),
    title: "Threshold Alerts",
    desc: "Escalate from Safe to Critical with visual warnings that are easy to interpret under pressure.",
  },
  {
    icon: (
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7dd3fc" strokeWidth="1.5">
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
      </svg>
    ),
    title: "Unified Admin Dashboard",
    desc: "Manage zones, operators, streams, and analytics from one command surface.",
  },
  {
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="#9db0ff">
        <path d="M12 2a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h1a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1h-1ZM6.5 6a1 1 0 0 1 1-1h1a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-1a1 1 0 0 1-1-1V6ZM2 9a1 1 0 0 1 1-1h1a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V9Z" />
      </svg>
    ),
    title: "Historical Analytics",
    desc: "Review density evolution, identify patterns, and understand how risk builds over time.",
  },
  {
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="#65f2d7">
        <path d="M12.416 3.376a.75.75 0 0 1 .208 1.04l-5 7.5a.75.75 0 0 1-1.154.114l-3-3a.75.75 0 0 1 1.06-1.06l2.353 2.353 4.493-6.74a.75.75 0 0 1 1.04-.207Z" />
      </svg>
    ),
    title: "Configurable Safety Limits",
    desc: "Tune operational capacity thresholds to the specific constraints of each monitored area.",
  },
  {
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="#ff6b5f">
        <path d="M6.5 2.25a.75.75 0 0 0-1.5 0v3a.75.75 0 0 0 1.5 0V4.5h6.75a.75.75 0 0 0 0-1.5H6.5v-.75ZM11 6.5a.75.75 0 0 0-1.5 0v3a.75.75 0 0 0 1.5 0v-.75h2.25a.75.75 0 0 0 0-1.5H11V6.5ZM5.75 10a.75.75 0 0 1 .75.75v.75h6.75a.75.75 0 0 1 0 1.5H6.5v.75a.75.75 0 0 1-1.5 0v-3A.75.75 0 0 1 5.75 10ZM2.75 7.25H8.5v1.5H2.75a.75.75 0 0 1 0-1.5ZM4 3H2.75a.75.75 0 0 0 0 1.5H4V3ZM2.75 11.5H4V13H2.75a.75.75 0 0 1 0-1.5Z" />
      </svg>
    ),
    title: "Fast Setup",
    desc: "Connect a stream, set a capacity, and begin monitoring without changing the core workflow.",
  },
];

const Feature = () => {
  return (
    <div className={styles.page}>
      <Navbar />

      <Reveal as="section" className={styles.hero}>
        <p className={styles.kicker}>Platform capabilities</p>
        <h1 className={styles.title}>Built like a control system, not a demo.</h1>
        <p className={styles.subtitle}>
          The interface is designed to keep operators oriented, aware, and ahead
          of dangerous crowd escalation.
        </p>
      </Reveal>

      <section className={styles.features}>
        {features.map(({ icon, title, desc }, index) => (
          <FeatureCard
            key={title}
            icon={icon}
            title={title}
            desc={desc}
            delay={index * 70}
          />
        ))}
      </section>

      <Reveal as="section" className={styles.cta}>
        <Demo />
      </Reveal>
    </div>
  );
};

const FeatureCard = ({ icon, title, desc, delay }) => {
  const glow = usePointerGlow();

  return (
    <Reveal delay={delay}>
      <div
        ref={glow.ref}
        {...glow.glowProps}
        className={`${styles.featureCard} interactiveGlow`}
      >
        <div className={styles.featureIcon}>{icon}</div>
        <h3 className={styles.featureTitle}>{title}</h3>
        <p className={styles.featureDesc}>{desc}</p>
      </div>
    </Reveal>
  );
};

export default Feature;
