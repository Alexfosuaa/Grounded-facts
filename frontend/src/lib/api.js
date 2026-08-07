import { PUBLIC_API_BASE } from "../config.js";

// Thin, typed-ish wrapper around fetch for the backend JSON API.
//
// Every endpoint lives under the same origin at /api/*, so we only ever pass a
// path (e.g. "/preview?topic=..."). The helper throws on any non-2xx response
// using the backend's `detail` message, which the components surface to the
// user as an error status line.
const BASE = PUBLIC_API_BASE;

export async function api(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  // The API always responds with JSON; fall back to {} if a body is missing.
  const body = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return body;
}
