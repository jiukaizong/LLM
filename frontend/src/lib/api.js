function trimQuotes(s) {
  return (s || "").trim().replace(/^['"]|['"]$/g, "");
}
function stripTrailingSlash(s) {
  return s.replace(/\/+$/, "");
}
function swapLocalhost(s) {
  if (!s) return s;
  try {
    const u = new URL(s);
    if (u.hostname === "127.0.0.1") u.hostname = "localhost";
    else if (u.hostname === "localhost") u.hostname = "127.0.0.1";
    return u.toString().replace(/\/+$/, "");
  } catch {
    return s;
  }
}

function normalizeBase(v) {
  const raw = trimQuotes(v || "http://127.0.0.1:8000");
  return stripTrailingSlash(raw);
}

let BASE = normalizeBase(import.meta.env.VITE_API_BASE);
if (import.meta.env.DEV) {
  // eslint-disable-next-line no-console
  console.log("[API] BASE =", BASE);
}

function withTimeout(ms, fetcher) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort("timeout"), ms);
  return fetcher(ctrl.signal).finally(() => clearTimeout(id));
}

async function _doFetch(path, init, timeoutMs, baseOverride) {
  const base = baseOverride || BASE;
  const url = `${base}${path}`;
  return withTimeout(timeoutMs, (signal) =>
    fetch(url, {
      mode: "cors",
      ...init,
      signal,
    })
  );
}

async function safeFetch(path, init = {}, timeoutMs = 10000) {
  try {
    const res = await _doFetch(path, init, timeoutMs);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText} ${text}`.trim());
    }
    return await res.json();
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn(`[API] primary fetch failed (${BASE}${path}):`, e?.message || e);

    const altBase = swapLocalhost(BASE);
    if (altBase && altBase !== BASE) {
      try {
        // eslint-disable-next-line no-console
        console.log("[API] retry with alt base:", altBase);
        const res2 = await _doFetch(path, init, timeoutMs, altBase);
        if (!res2.ok) {
          const text = await res2.text().catch(() => "");
          throw new Error(`${res2.status} ${res2.statusText} ${text}`.trim());
        }
        BASE = altBase;
        // eslint-disable-next-line no-console
        console.log("[API] BASE switched to:", BASE);
        return await res2.json();
      } catch (e2) {
        // eslint-disable-next-line no-console
        console.error(`[API] alt fetch failed (${altBase}${path}):`, e2?.message || e2);
      }
    }
    throw new Error(`Failed to fetch ${path}: ${e.message || e}`);
  }
}

//API methods
export async function getEmployeeMetrics() {
  return safeFetch("/api/employee/metrics");
}

export async function getCreditMetrics() {
  return safeFetch("/api/credit/metrics");
}

export async function postInsights(body) {
  return safeFetch(
    "/api/insights",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    },
    30000
  );
}

export async function getEmployeeData({ offset = 0, limit = 50 } = {}) {
  return safeFetch(`/api/employee/data?offset=${offset}&limit=${limit}`);
}

export async function getCreditData({ offset = 0, limit = 50 } = {}) {
  return safeFetch(`/api/credit/data?offset=${offset}&limit=${limit}`);
}
