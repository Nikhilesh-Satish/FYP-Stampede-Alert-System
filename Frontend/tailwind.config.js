/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f7ff',
          100: '#e0f2fe',
          500: '#0066cc',
          600: '#0052a3',
          700: '#003d7a',
          800: '#002851',
          900: '#001328',
        },
        secondary: {
          500: '#4ecdc4',
          600: '#3ab5b1',
          700: '#289d9a',
        },
      },
    },
  },
  plugins: [],
}
