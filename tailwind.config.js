/** Tailwind scans the Django templates; there is no JavaScript build step. */
module.exports = {
  content: [
    "./src/signaldesk/web/templates/**/*.html",
    "./src/signaldesk/web/**/templates/**/*.html",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
