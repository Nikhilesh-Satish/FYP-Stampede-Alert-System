import { useRef } from "react";

export const useMagneticHover = (strength = 0.18) => {
  const ref = useRef(null);

  const handlePointerMove = (event) => {
    const node = ref.current;
    if (!node) return;

    const rect = node.getBoundingClientRect();
    const x = event.clientX - rect.left - rect.width / 2;
    const y = event.clientY - rect.top - rect.height / 2;

    node.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
  };

  const handlePointerLeave = () => {
    const node = ref.current;
    if (!node) return;
    node.style.transform = "translate(0, 0)";
  };

  return {
    ref,
    magneticProps: {
      onPointerMove: handlePointerMove,
      onPointerLeave: handlePointerLeave,
    },
  };
};
