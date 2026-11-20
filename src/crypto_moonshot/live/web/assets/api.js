// The API client. The token lives in sessionStorage only (gone when the tab closes) and goes
// out as an Authorization header, never in a URL.

const KEY = "moonshot.token";
const PREFIX = "/api/v1";

export class AuthError extends Error {}
export class NetworkError extends Error {}
export class ApiError extends Error {}

export const token = {
  get: () => sessionStorage.getItem(KEY),
  set: (value) => sessionStorage.setItem(KEY, value),
  clear: () => sessionStorage.removeItem(KEY),
};

/** One API call; throws AuthError on 401, NetworkError when the API is unreachable. */
export async function api(path, { method = "GET", params } = {}) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== null && value !== undefined && value !== "") query.set(key, String(value));
  }
  const suffix = query.size ? `?${query}` : "";
  let response;
  try {
    response = await fetch(PREFIX + path + suffix, {
      method,
      headers: { Authorization: `Bearer ${token.get() ?? ""}` },
      cache: "no-store",
    });
  } catch (error) {
    throw new NetworkError(String(error));
  }
  if (response.status === 401) throw new AuthError("the API token was rejected");
  if (!response.ok) {
    let reason = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      reason = body.reason ?? body.detail ?? reason;
    } catch {
      // not JSON: keep the status
    }
    throw new ApiError(reason);
  }
  return response.json();
}
