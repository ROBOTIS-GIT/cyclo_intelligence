import readApiResponse from "../../../utils/readApiResponse";
export async function autonomyRequest(path, method = "GET", body) {
  const response = await fetch(`/api${path}`, { method,
    ...(body !== undefined ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}) });
  return readApiResponse(response, "Autonomy request failed");
}
export const OBSERVATION_PLAN_CHANGED = "observation-spots-changed";
