/**
 * Copy the front-end libraries out of node_modules into the static directory.
 *
 * The copies are committed so the production image needs no Node and the pages
 * load no third-party CDN. Re-run with `npm run vendor` after changing a pinned
 * version in package.json.
 */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const target = path.join(root, "src", "signaldesk", "web", "static", "js");

const assets = [
  ["htmx.org/dist/htmx.min.js", "htmx.min.js"],
  ["alpinejs/dist/cdn.min.js", "alpine.min.js"],
];

fs.mkdirSync(target, { recursive: true });

for (const [source, name] of assets) {
  const from = path.join(root, "node_modules", source);
  if (!fs.existsSync(from)) {
    console.error(`missing ${source}; run npm install first`);
    process.exit(1);
  }
  fs.copyFileSync(from, path.join(target, name));
  console.log(`vendored ${name}`);
}
