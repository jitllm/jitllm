import { createClient, ConnectError, Code } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { ModelService } from "./gen/jitllm/v1/model_pb";
import { DeviceService } from "./gen/jitllm/v1/device_pb";
import { InferenceService } from "./gen/jitllm/v1/inference_pb";
import { TelemetryService } from "./gen/jitllm/v1/telemetry_pb";
import { PlacementService } from "./gen/jitllm/v1/placement_pb";
import { SessionService } from "./gen/jitllm/v1/session_pb";
import { takeKey } from "./chat.ts";

// Same-origin by default; ?server=http://host:port points at another jitllmd
// (it must list this page's origin in -allow-origin). The API key is moved out
// of the address into sessionStorage, never localStorage.
const KEY = "jitllm.key";
function stored(): string | null {
  try { return sessionStorage.getItem(KEY); } catch { return null; }
}
export function keepKey(k: string | null) {
  key = k;
  try { k === null ? sessionStorage.removeItem(KEY) : sessionStorage.setItem(KEY, k); } catch { /* private window: the key lasts this page load */ }
}
const taken = takeKey(location.search, stored());
let key: string | null = taken.key;
if (taken.cleaned !== null) {
  keepKey(key);
  history.replaceState(null, "", location.pathname + taken.cleaned + location.hash);
}
export const hasKey = () => key !== null;
const server = new URLSearchParams(location.search).get("server");
const transport = createConnectTransport({
  baseUrl: server ?? location.origin,
  interceptors: [(next) => (req) => { if (key) req.header.set("Authorization", "Bearer " + key); return next(req); }],
});
export const models = createClient(ModelService, transport);
export const devices = createClient(DeviceService, transport);
export const infer = createClient(InferenceService, transport);
export const telemetry = createClient(TelemetryService, transport);
export const placement = createClient(PlacementService, transport);
export const sessions = createClient(SessionService, transport);

export function errText(e: unknown): string {
  return e instanceof ConnectError ? e.rawMessage || Code[e.code] : (e as Error).message;
}
export const isUnauth = (e: unknown) => e instanceof ConnectError && e.code === Code.Unauthenticated;
export const isCancel = (e: unknown) => e instanceof ConnectError && e.code === Code.Canceled;

// For the one request a page makes as it unloads, where a fetch with
// keepalive outlives the page and the Connect client cannot.
export function closeOnUnload(path: string, body: unknown) {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (key) headers.authorization = "Bearer " + key;
  try { fetch((server ?? location.origin) + path, { method: "POST", headers, body: JSON.stringify(body), keepalive: true }); } catch { /* the server drops an idle session itself */ }
}
