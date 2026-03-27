import { useRef } from "react";

export const usePointerGlow = () => {
  const ref = useRef(null);

  const handlePointerMove = (event) => {
    const node = ref.current;
    if (!node) return;

    const rect = node.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const rotateY = ((x / rect.width) - 0.5) * 7;
    const rotateX = (((y / rect.height) - 0.5) * -7);

    node.style.setProperty("--pointer-x", `${x}px`);
    node.style.setProperty("--pointer-y", `${y}px`);
    node.style.setProperty("--rotate-x", `${rotateX.toFixed(2)}deg`);
    node.style.setProperty("--rotate-y", `${rotateY.toFixed(2)}deg`);
  };

  const handlePointerLeave = () => {
    const node = ref.current;
    if (!node) return;

    node.style.setProperty("--rotate-x", "0deg");
    node.style.setProperty("--rotate-y", "0deg");
    node.style.setProperty("--pointer-x", "50%");
    node.style.setProperty("--pointer-y", "50%");
  };

  return {
    ref,
    glowProps: {
      onPointerMove: handlePointerMove,
      onPointerLeave: handlePointerLeave,
    },
  };
};
