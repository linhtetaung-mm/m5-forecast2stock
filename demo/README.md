# Stockdesk

This static demo lets a visitor make a simulated replenishment decision for three CA_1 teaching SKUs and inspect the all-item study results. It has no API or database. Browser decisions stay in the current tab.

From the project root, open `demo/dist/index.html` or serve the static directory.

```sh
python -m http.server 8770 --bind 127.0.0.1 --directory demo/dist
```

With the project's Python environment active, regenerate the demo data and verify the JavaScript replay.

```sh
python demo/export_data.py
node demo/test.mjs
```

The exporter reads frozen forecast and inventory artifacts and reconstructs the common April opening stock. Its static output contains three teaching SKUs and aggregate results across all 3,049 CA_1 items. It checks the input artifact hashes and includes Python totals for checking the browser simulator. The site data and layout can be deployed from `dist/`.

