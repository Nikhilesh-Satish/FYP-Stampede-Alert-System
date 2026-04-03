const pickNiceStep = (maxValue) => {
  if (maxValue <= 10) return 2;
  if (maxValue <= 50) return 10;
  if (maxValue <= 100) return 20;
  if (maxValue <= 300) return 50;
  if (maxValue <= 700) return 100;
  if (maxValue <= 1500) return 200;
  if (maxValue <= 3000) return 500;
  return 1000;
};

export const getNiceAxisConfig = (values = []) => {
  const safeValues = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value >= 0);

  const maxValue = safeValues.length > 0 ? Math.max(...safeValues) : 0;
  const step = pickNiceStep(maxValue);
  const maxDomain = Math.max(step, Math.ceil((maxValue + step * 0.35) / step) * step);
  const ticks = [];

  for (let value = 0; value <= maxDomain; value += step) {
    ticks.push(value);
  }

  return {
    domain: [0, maxDomain],
    ticks,
  };
};
