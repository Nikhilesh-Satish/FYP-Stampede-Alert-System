import styles from "./Reveal.module.css";
import { useRevealOnScroll } from "../hooks/useRevealOnScroll";

const Reveal = ({ as: Tag = "div", children, className = "", delay = 0 }) => {
  const { ref, isVisible } = useRevealOnScroll();

  return (
    <Tag
      ref={ref}
      className={`${styles.reveal} ${isVisible ? styles.visible : ""} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </Tag>
  );
};

export default Reveal;
