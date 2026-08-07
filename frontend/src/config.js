// Single source of truth for where the backend API lives.
//
// The SPA is served from the same origin as the API (FastAPI mounts the built
// bundle at "/"), so a relative "/api" prefix is all we need. Kept in its own
// module so it is trivial to point at a different host later if the frontend is
// ever deployed separately from the backend.
export const PUBLIC_API_BASE = "/api";
