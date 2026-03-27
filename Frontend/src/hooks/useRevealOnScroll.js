import { useEffect, useRef, useState } from "react";

export const useRevealOnScroll = (options = {}) => {
  const ref = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(entry.target);
        }
      },
      {
        threshold: 0.14,
        rootMargin: "0px 0px -8% 0px",
        ...options,
      },
    );

    observer.observe(node);

    return () => observer.disconnect();
  }, [options]);

  return { ref, isVisible };
};
