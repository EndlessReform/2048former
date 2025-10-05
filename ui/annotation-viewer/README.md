# Annotation Viewer

React + Vite client for browsing annotated 2048 self-play runs served by the Rust annotation server.

## Environment Variables

Set these in a `.env.local` or via your shell before starting the dev server/build:

- `VITE_ANNOTATION_API_BASE` — Base URL for the annotation server (e.g. `http://localhost:8080`). Omit or leave empty to proxy against the same origin the site is served from.

### Serving Over LAN / Custom Hosts

`npm run dev` already binds to `0.0.0.0` so any device on the LAN can connect via `http://<host-ip>:5173`. To change the host or port, pass flags directly to Vite:

```bash
npm run dev -- --host 192.168.1.42 --port 5174
```

If you need to expose the annotation API from a different origin, update `VITE_ANNOTATION_API_BASE` accordingly (e.g. `http://192.168.1.42:8080`). For locked-down environments you can also proxy through Vite by extending `vite.config.ts` with a `server.proxy` entry.

## Installation

```bash
cd ui/annotation-viewer
npm install
```

## Local Development

```bash
# assumes annotation server listening at http://localhost:8080
export VITE_ANNOTATION_API_BASE="http://localhost:8080"
npm run dev
```

`npm run dev` launches Vite with hot-module reload. Visit `http://localhost:5173` or the LAN URL printed in the console.

## Production Build

```bash
npm run build
```

This produces a static bundle under `dist/`. Preview locally with `npm run preview`.

## Serving UI from Annotation Server

To serve the UI directly from the annotation server (recommended for production), build the UI and pass the `--ui-path` flag to the server:

```bash
# Build the UI
npm run build

# Run the annotation server with UI serving
cargo run --bin annotation-server -- --dataset /path/to/dataset --annotations /path/to/annotations --ui-path dist/
```

The server will serve the UI at the root path `/` and assets under `/assets/`. Since the UI and API are on the same origin, no `VITE_ANNOTATION_API_BASE` configuration is needed—the UI will automatically proxy API requests to the same host/port.

If running the UI separately (e.g., for development), ensure `VITE_ANNOTATION_API_BASE` points to the annotation server URL.

## Current Functionality

- Lists annotated runs with max score, step count, and highest tile metadata.
- Fetches run details and shows a summary with a representative board snapshot.
- Highlights teacher move, model argmax, probability, and legal move mask for the sampled step.

Additional panels (timeline, filters, per-branch inspection) are planned but not yet implemented.
