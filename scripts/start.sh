#!/bin/sh
# Starts the colorization server and the Stremio addon (Linux/macOS).
cd "$(dirname "$0")/.."
python3 -m colorstream serve &
(cd addon && node index.js) &
echo "Dashboard: http://127.0.0.1:7001"
echo "Stremio addon URL: http://127.0.0.1:7000/manifest.json"
wait
