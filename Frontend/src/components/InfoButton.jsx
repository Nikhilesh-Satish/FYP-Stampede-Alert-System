import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import styles from "./InfoButton.module.css";

const InfoButton = ({ text, label = "Info" }) => {
  const buttonRef = useRef(null);
  const tooltipRef = useRef(null);
  const closeTimerRef = useRef(null);
  const tooltipId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  const clearCloseTimer = () => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const updatePosition = () => {
    const button = buttonRef.current;
    if (!button) return;

    const rect = button.getBoundingClientRect();
    const tooltipWidth = 320;
    const minMargin = 12;
    const viewportWidth = window.innerWidth;
    const centeredLeft = rect.left + rect.width / 2 - tooltipWidth / 2;
    const left = Math.min(
      Math.max(centeredLeft, minMargin),
      viewportWidth - tooltipWidth - minMargin,
    );

    setPosition({
      top: rect.bottom + 10,
      left,
    });
  };

  const openTooltip = () => {
    clearCloseTimer();
    updatePosition();
    setIsOpen(true);
  };

  const closeTooltip = () => {
    clearCloseTimer();
    closeTimerRef.current = window.setTimeout(() => {
      setIsOpen(false);
    }, 120);
  };

  const toggleTooltip = (event) => {
    event.preventDefault();
    event.stopPropagation();
    clearCloseTimer();
    if (isOpen) {
      setIsOpen(false);
      return;
    }
    openTooltip();
  };

  useEffect(() => {
    if (!isOpen) return undefined;

    const handlePointerDown = (event) => {
      const target = event.target;
      if (
        buttonRef.current?.contains(target) ||
        tooltipRef.current?.contains(target)
      ) {
        return;
      }
      setIsOpen(false);
    };

    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    const handleReposition = () => {
      updatePosition();
    };

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleEscape);
    window.addEventListener("resize", handleReposition);
    window.addEventListener("scroll", handleReposition, true);

    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleEscape);
      window.removeEventListener("resize", handleReposition);
      window.removeEventListener("scroll", handleReposition, true);
    };
  }, [isOpen]);

  useEffect(() => {
    return () => {
      clearCloseTimer();
    };
  }, []);

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={styles.infoBtn}
        aria-label={label}
        aria-expanded={isOpen}
        aria-describedby={isOpen ? tooltipId : undefined}
        onClick={toggleTooltip}
        onMouseEnter={openTooltip}
        onMouseLeave={closeTooltip}
        onFocus={openTooltip}
        onBlur={closeTooltip}
      >
        i
      </button>

      {isOpen &&
        createPortal(
          <div
            ref={tooltipRef}
            id={tooltipId}
            className={styles.tooltip}
            role="tooltip"
            style={{
              top: `${position.top}px`,
              left: `${position.left}px`,
            }}
            onMouseEnter={openTooltip}
            onMouseLeave={closeTooltip}
          >
            <strong className={styles.tooltipTitle}>How This Is Calculated</strong>
            <p className={styles.tooltipText}>{text}</p>
          </div>,
          document.body,
        )}
    </>
  );
};

export default InfoButton;
