import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { create } from "@bufbuild/protobuf";
import { ModelService } from "./gen/jitllm/v1/model_pb";
import { DeviceService } from "./gen/jitllm/v1/device_pb";
import { InferenceService, GenerateRequestSchema } from "./gen/jitllm/v1/inference_pb";
import { FinishReason } from "./gen/jitllm/v1/common_pb";

// Same-origin by default; ?server=http://host:port points at another jitllmd
// (it must list this page's origin in -allow-origin), and ?key= is its API key.
const q = new URLSearchParams(location.search);
const key = q.get("key");
const transport = createConnectTransport({
  baseUrl: q.get("server") ?? location.origin,
  interceptors: key
    ? [(next) => (req) => { req.header.set("Authorization", "Bearer " + key); return next(req); }]
    : [],
});
const models = createClient(ModelService, transport);
const devices = createClient(DeviceService, transport);
const infer = createClient(InferenceService, transport);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

async function refresh() {
  try {
    const [m, d] = await Promise.all([models.listModels({}), devices.listDevices({})]);
    $("devices").innerHTML = d.devices
      .map((x) => `<tr><td>${esc(x.ref?.id ?? "")}</td><td>${esc(x.name)}</td><td class="n">${esc(x.freeMemory?.human ?? "")} / ${esc(x.totalMemory?.human ?? "")}</td></tr>`)
      .join("");
    $("loaded").innerHTML = m.loaded
      .map((x) => `<tr><td>${esc(x.modelId)}</td><td>${esc(x.architecture)}</td><td class="n">${x.blockCount} blocks, ctx ${x.contextLength}</td></tr>`)
      .join("") || `<tr><td colspan="3" class="dim">nothing loaded</td></tr>`;
    const sel = $<HTMLSelectElement>("model");
    const prev = sel.value;
    const ids = [...m.loaded.map((x) => x.modelId), ...m.files.filter((f) => f.isContainer).map((f) => f.name)];
    sel.innerHTML = ids.map((i) => `<option>${esc(i)}</option>`).join("");
    if (ids.includes(prev)) sel.value = prev;
    $("status").textContent = "connected";
  } catch (e) {
    $("status").textContent = "unreachable: " + (e as Error).message;
  }
}

let busy: AbortController | null = null;
async function send() {
  const box = $<HTMLTextAreaElement>("prompt");
  const text = box.value.trim();
  if (!text || busy) return;
  box.value = "";
  const log = $("log");
  log.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(text)}</div><div class="msg bot" id="cur"></div>`);
  const cur = $("cur");
  busy = new AbortController();
  const req = create(GenerateRequestSchema, {
    modelId: $<HTMLSelectElement>("model").value,
    maxTokens: 256,
    prompt: { input: { case: "chat", value: { addGenerationPrompt: true, messages: [{ role: "user", content: text }] } } },
  });
  try {
    for await (const ev of infer.generate(req, { signal: busy.signal })) {
      if (ev.event.case === "token") cur.textContent += ev.event.value.text;
      if (ev.event.case === "finished") cur.insertAdjacentHTML("beforeend", `<span class="dim"> [${FinishReason[ev.event.value.reason]}]</span>`);
    }
  } catch (e) {
    cur.insertAdjacentHTML("beforeend", `<span class="err"> ${esc((e as Error).message)}</span>`);
  } finally {
    cur.removeAttribute("id");
    busy = null;
    log.scrollTop = log.scrollHeight;
  }
}
$("send").addEventListener("click", send);
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("stop").addEventListener("click", () => busy?.abort());
refresh();
setInterval(refresh, 5000);
