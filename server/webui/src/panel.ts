// The side panel: the engine's live numbers, the devices, the models and the
// sampling settings. Nothing here talks to the page's chat; it reads `live`.
import { h, svg, clear } from "./dom.ts";
import { live, subscribe, refreshFiles, refreshPlacement, firstSessionOf, changed, Series } from "./live.ts";
import { models, placement, sessions, errText } from "./api.ts";
import { BlockLocation, BlockKind, DeviceKind, ExecutionMode } from "./gen/jitllm/v1/common_pb";
import type { Device } from "./gen/jitllm/v1/device_pb";
import { settings, saveSettings } from "./store.ts";
import { human } from "./chat.ts";

export interface Ctx {
  // The model and the session of the chat in view: what the panels show unless
  // told to look at another loaded model.
  chatModel(): string;
  chatSession(): string;
  chatId(): string;
  toast(msg: string, bad?: boolean): void;
  newChatOn(model: string): void;
  ensureSession(): Promise<string>;
}

const n = (x: bigint | number | undefined) => Number(x ?? 0);
const fmt = (x: number, d = 1) => (x >= 100 ? x.toFixed(0) : x.toFixed(d));

function kv(k: string, v: string | Node, tip = ""): HTMLElement {
  return h("div", { class: "kv", title: tip }, h("span", { class: "k", text: k }), h("span", { class: "v" }, v));
}
function card(title: string, ...kids: (Node | false | null)[]): HTMLElement {
  return h("section", { class: "card" }, h("h3", { text: title }), ...kids);
}
function meter(frac: number, cls = ""): HTMLElement {
  const f = Math.max(0, Math.min(1, isFinite(frac) ? frac : 0));
  return h("div", { class: "meter " + cls }, h("i", { style: { width: (f * 100).toFixed(1) + "%" } }));
}

export function spark(s: Series, w = 320, hgt = 64): SVGElement {
  const el = svg("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "spark", preserveAspectRatio: "none" });
  const v = s.v;
  if (v.length < 2) {
    el.append(svg("line", { x1: 0, y1: hgt - 1, x2: w, y2: hgt - 1, class: "base" }));
    return el;
  }
  const peak = Math.max(s.peak, 1e-9);
  const step = w / (Math.max(v.length, 40) - 1);
  const x0 = 0;
  const pts = v.map((y, i) => [x0 + i * step, hgt - 3 - (y / peak) * (hgt - 8)] as const);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  el.append(
    svg("path", { d: `${line} L${pts[pts.length - 1][0].toFixed(1)} ${hgt} L${x0.toFixed(1)} ${hgt} Z`, class: "area" }),
    svg("path", { d: line, class: "line" }),
    svg("circle", { cx: pts[pts.length - 1][0], cy: pts[pts.length - 1][1], r: 3, class: "dot" }),
  );
  return el;
}

let tab = settings.tab;
let root: HTMLElement;
let body: HTMLElement;
let ctx: Ctx;
let modelsSig = "";
const conv = new Map<string, string>(); // path -> conversion progress text

const TABS: [string, string][] = [["engine", "Engine"], ["placement", "Placement"], ["devices", "Devices"], ["models", "Models"], ["settings", "Sampling"]];

export function mount(el: HTMLElement, c: Ctx) {
  root = el; ctx = c;
  const bar = h("nav", { class: "tabs", attrs: { role: "tablist" } });
  body = h("div", { class: "pbody" });
  const draw = () => {
    clear(bar);
    for (const [id, label] of TABS) bar.append(h("button", { class: "tab" + (id === tab ? " on" : ""), text: label, attrs: { role: "tab", "aria-selected": String(id === tab) }, on: { click: () => { tab = id; settings.tab = id; saveSettings(); modelsSig = ""; draw(); if (id === "models") refreshFiles(); render(true); } } }));
  };
  el.append(bar, body);
  draw();
  subscribe(() => render(false));
  render(true);
}
export function showTab(id: string) { if (id === "models") refreshFiles(); tab = id; settings.tab = id; saveSettings(); modelsSig = ""; root.querySelectorAll(".tab").forEach((b, i) => b.classList.toggle("on", TABS[i][0] === id)); render(true); }

// The panels show the chat's model unless the viewer picked another loaded one.
let viewing: string | null = null;
const loadedNow = () => live.list?.loaded.map((m) => m.modelId) ?? live.stats?.models.map((m) => m.modelId) ?? [];
export const viewModel = () => (viewing && loadedNow().includes(viewing) ? viewing : ctx.chatModel());
export function followChat() { viewing = null; if (root) render(true); }

function modelPicker(): HTMLElement {
  const ids = loadedNow();
  const sel = h("select", { on: { change: (e: any) => { viewing = e.target.value === ctx.chatModel() ? null : e.target.value; render(true); } } },
    ...ids.map((id) => h("option", { value: id, text: id + (id === ctx.chatModel() ? "  (this chat)" : "") })));
  sel.value = viewModel();
  return h("label", { class: "viewer" }, "Showing", sel);
}

function render(force: boolean) {
  if (tab === "engine") engine(force);
  else if (tab === "placement") placementTab(force);
  else if (tab === "devices") deviceTab();
  else if (tab === "models") modelsTab(force);
  else if (force) settingsTab();
}

// ---- engine ---------------------------------------------------------------

let engineBuilt = false;
let dyn: HTMLElement;

function engine(force: boolean) {
  if (force || !engineBuilt || !body.contains(dyn)) {
    clear(body);
    dyn = h("div", { class: "stack" });
    body.append(dyn);
    engineBuilt = true;
  }
  const id = viewModel();
  const ms = live.stats?.models.find((m) => m.modelId === id);
  const info = live.list?.loaded.find((m) => m.modelId === id);
  const L = live.last;
  const rate = live.streaming ? live.decode.last : L.tps;
  const bw = rate && L.bytesPerToken ? (rate * L.bytesPerToken) / 1e9 : L.gbps;

  const hero = card("Throughput",
    h("div", { class: "hero" }, h("b", { text: rate ? fmt(rate) : "0.0" }), h("span", { text: " tok/s decode" }), live.streaming ? h("em", { class: "live", text: "live" }) : null),
    spark(live.decode),
    h("div", { class: "kvs" },
      kv("peak", live.decode.peak ? fmt(live.decode.peak) : "–"), kv("mean", live.decode.mean ? fmt(live.decode.mean) : "–"),
      kv("prompt", L.promptTps ? fmt(L.promptTps, 0) + " tok/s" : "–", "tokens read per second while prefilling"),
      kv("TTFT", L.ttft ? Math.round(L.ttft) + " ms" : "–", "time to first token, measured by this page"),
      kv("bandwidth", bw ? bw.toFixed(1) + " GB/s" : "–", "decode rate times bytes read per token"),
      kv("per token", L.bytesPerToken ? human(L.bytesPerToken) : "–")));

  const batch = ms?.batch;
  const sess = live.stats?.sessions.filter((s) => s.modelId === id) ?? [];
  const mine = id === ctx.chatModel() ? sess.find((s) => s.sessionId === ctx.chatSession()) : undefined;
  const pos = mine ? mine.position : Math.max(...sess.map((s) => s.position), 0);
  const ctxLen = info?.contextLength ?? 0;
  const model = card("Model " + (id || ""),
    info ? h("div", { class: "kvs" },
      kv("architecture", info.architecture), kv("blocks", String(info.blockCount)),
      kv("width", String(info.embeddingDim)), kv("vocab", String(info.vocabSize)),
      info.expertCount ? kv("experts", `${info.expertsUsed} of ${info.expertCount}`) : null,
      info.linearBlocks ? kv("attn / linear", `${info.attentionBlocks} / ${info.linearBlocks}`) : null) : h("p", { class: "dim", text: id ? "Reading the model's details…" : "No model selected." }),
    ctxLen ? h("div", { class: "ctxrow" }, kv("context", `${pos} / ${ctxLen}`), meter(pos / ctxLen)) : null,
    ms ? h("div", { class: "kvs" },
      kv("generated", n(ms.tokensGenerated).toLocaleString()), kv("prefilled", n(ms.tokensPrefilled).toLocaleString()),
      kv("sessions", String(ms.sessions)),
      batch ? kv("batch rows", `${batch.liveRows} live · ${batch.waiting} waiting`, "rows stepping together, requests queued") : null,
      L.cached ? kv("prefix cache", `${L.cached} of ${L.prompt} tokens`, "prompt tokens answered from cached keys and values") : null) : null);

  const r = ms?.residency;
  const pager = r ? card("Pager",
    h("div", { class: "ctxrow" },
      kv("resident", `${r.resident?.human ?? "0"} / ${r.budget?.human ?? "0"}`), meter(n(r.resident?.bytes) / Math.max(1, n(r.budget?.bytes)), r.fits ? "" : "warn")),
    h("div", { class: "kvs" },
      kv("blocks", `${r.residentBlocks} of ${r.blocks}`), kv("frames", String(r.frames)),
      kv("page-ins", n(r.pageIns).toLocaleString()), kv("evictions", n(r.evictions).toLocaleString()),
      kv("read", `${r.bytesRead?.human ?? "0"} in ${n(r.readRequests)} reads`)),
    h("p", { class: r.fits ? "dim small" : "warn small", text: r.fits ? "The model fits in its page budget; nothing is evicted." : "Over budget: blocks are evicted and read back as needed." })) : null;

  dyn.replaceChildren(modelPicker(), hero, model, ...(pager ? [pager] : []));
}

// ---- placement --------------------------------------------------------------

let placeBuilt = false, placeBox: HTMLElement, placeTimer = 0, placeSession = "";
const targetPick = { id: "" };
let lastMove = "";
// The layers selected on the map: one contiguous range, picked by click,
// shift-click or drag. It belongs to the session on show.
let selRange: [number, number] | null = null, selAnchor = -1, dragging = false, selFor = "";
const inSel = (i: number) => !!selRange && i >= selRange[0] && i <= selRange[1];
function paintSel() {
  placeBox.querySelectorAll<HTMLElement>(".blk[data-i]").forEach((c) => c.classList.toggle("sel", inSel(Number(c.dataset.i))));
  const note = placeBox.querySelector(".pick + p");
  if (note) note.textContent = selRange ? `Selected layers ${selRange[0]}–${selRange[1]} (${selRange[1] - selRange[0] + 1})` : "Click a layer, shift-click or drag to select a range.";
}
document.addEventListener("mouseup", () => { if (dragging) { dragging = false; drawPlacement(); } });

async function placementSession(): Promise<string> {
  const id = viewModel();
  if (id === ctx.chatModel()) return ctx.chatSession();
  return firstSessionOf(id);
}

function placementTab(force: boolean) {
  if (force || !placeBuilt || !body.contains(placeBox)) {
    clear(body);
    placeBox = h("div", { class: "stack" });
    body.append(placeBox);
    placeBuilt = true;
    const poll = async () => {
      if (tab !== "placement") return;
      placeSession = await placementSession();
      refreshPlacement(placeSession);
    };
    poll();
    clearInterval(placeTimer);
    placeTimer = window.setInterval(poll, 10000);
  }
  drawPlacement();
}

const hostName = "the CPU";

function drawPlacement() {
  const p = live.place;
  const id = viewModel();
  const kids: Node[] = [modelPicker()];
  const mine = id === ctx.chatModel();
  if (!p || live.placeFor !== placeSession) {
    kids.push(card("Layers", h("p", { class: "dim small", text: mine ? "Opening this chat's session…" : "No conversation is using this model right now, so it has no layers placed to show." })));
    placeBox.replaceChildren(...kids);
    return;
  }
  if (selFor !== placeSession) { selFor = placeSession; selRange = null; selAnchor = -1; }
  const nb = p.blocks.length;
  const per = new Map<string, { n: number; bytes: number }>();
  for (const b of p.blocks) {
    const k = b.location === BlockLocation.DEVICE ? b.deviceId : "host";
    const e = per.get(k) ?? { n: 0, bytes: 0 };
    e.n++; e.bytes += n(b.bytes?.bytes);
    per.set(k, e);
  }
  const sentence = [...per.entries()].sort((a, b) => (a[0] === "host" ? 1 : 0) - (b[0] === "host" ? 1 : 0))
    .map(([k, e]) => `${e.n} on ${k === "host" ? hostName : k}`).join(", ");
  // Any device the server has can take layers: a model's starting placement
  // does not limit where its session can go. The default is the first card not
  // sharing host memory.
  const order = gpuIds();
  const gpus = live.devs.filter(accelerator).sort((a, b) => (order.indexOf(a.ref?.id ?? "") + 1 || 99) - (order.indexOf(b.ref?.id ?? "") + 1 || 99));
  const target = gpus.find((d) => d.ref?.id === targetPick.id) ?? gpus[0];
  const nd = p.deviceBlockCount;
  const hostIdx = p.blocks.filter((b) => b.location !== BlockLocation.DEVICE).map((b) => b.index);
  const devIdx = p.blocks.filter((b) => b.location === BlockLocation.DEVICE).map((b) => b.index);
  const no = target ? "" : "no GPU on this server";
  const move = async (first: number, last: number, toHost: boolean, what: string) => {
    try {
      const r = await placement.relocateBlocks({ sessionId: placeSession, firstBlock: first, lastBlock: last, toHost, toDeviceId: toHost ? "" : target!.ref!.id });
      lastMove = r.declined[0] ? `Declined: ${r.declined[0].reason}` : `${what}: ${r.blocksMoved} layer(s) moved in ${r.millis} ms`;
      ctx.toast(lastMove, !!r.declined[0]);
    } catch (e) { lastMove = errText(e); ctx.toast(lastMove, true); }
    refreshPlacement(placeSession);
  };
  const btn = (label: string, ok: boolean, tip: string, f: () => void) => h("button", { class: "btn sm", text: label, title: no || tip, disabled: !ok || !!no, on: { click: f } });
  const devCards = [...per.keys(), ...gpus.map((g) => g.ref?.id ?? "")].filter((k, i, a) => a.indexOf(k) === i && k !== "host").map((k) => {
    const d = live.devs.find((x) => x.ref?.id === k);
    if (!d) return null;
    const total = n(d.totalMemory?.bytes), free = n(d.freeMemory?.bytes), mine_ = per.get(k)?.bytes ?? 0;
    return h("div", { class: "ctxrow" }, kv(d.name || k, `${human(total - free)} used of ${d.totalMemory?.human ?? "?"}` + (mine_ ? ` · this model ${human(mine_)}` : "")), meter(total ? (total - free) / total : 0));
  });
  kids.push(
    card("Where this model runs",
      h("p", { class: "plain" }, `A model is a stack of ${nb} layers. Each layer runs either on the CPU or on a graphics card; a card is faster when it has room.`),
      h("div", { class: "seam" }, h("b", { text: String(nb) }), ` layers: ${sentence}`),
      h("div", { class: "blocks pick", attrs: { role: "group", "aria-label": "Layers: click, shift-click or drag to select" } }, ...p.blocks.map((b) => h("i", {
        class: "blk " + (b.location === BlockLocation.DEVICE ? "dev" : "host") + (b.kind === BlockKind.LINEAR ? " lin" : "") + (b.kind === BlockKind.VISION ? " vis" : "") + (b.location !== BlockLocation.DEVICE && !b.hostResident ? " cold" : "") + (inSel(b.index) ? " sel" : ""),
        title: `layer ${b.index} · ${BlockKind[b.kind].toLowerCase()} · ${b.location === BlockLocation.DEVICE ? b.deviceId : "CPU"} · ${b.bytes?.human ?? ""}`,
        attrs: { "data-i": String(b.index) },
        on: {
          mousedown: (e: MouseEvent) => {
            e.preventDefault();
            if (e.shiftKey && selAnchor >= 0) { selRange = [Math.min(selAnchor, b.index), Math.max(selAnchor, b.index)]; drawPlacement(); return; }
            selAnchor = b.index; selRange = [b.index, b.index]; dragging = true;
            paintSel();
          },
          mouseenter: () => { if (dragging) { selRange = [Math.min(selAnchor, b.index), Math.max(selAnchor, b.index)]; paintSel(); } },
        },
      }))),
      h("p", { class: "dim small", text: selRange ? `Selected layers ${selRange[0]}–${selRange[1]} (${selRange[1] - selRange[0] + 1})` : "Click a layer, shift-click or drag to select a range." }),
      h("div", { class: "legend" }, h("span", {}, h("i", { class: "blk host" }), " CPU"), h("span", {}, h("i", { class: "blk dev" }), " graphics card"), h("span", {}, h("i", { class: "blk lin" }), " linear layer")),
      ...p.declines.slice(0, 3).map((d) => h("p", { class: "warn small", text: `layer ${d.blockIndex}: ${d.reason}` }))),
    ...(devCards.length ? [card("Device memory", ...devCards)] : []),
    card("Move layers" + (mine ? "" : " (another chat's session)"),
      gpus.length > 1 ? h("label", {}, "Graphics card", (() => {
        const s = h("select", { on: { change: (e: any) => { targetPick.id = e.target.value; drawPlacement(); } } }, ...gpus.map((g) => h("option", { value: g.ref?.id ?? "", text: `${g.ref?.id} · ${g.name}` })));
        s.value = target?.ref?.id ?? ""; return s;
      })()) : null,
      h("div", { class: "ctl" },
        btn("Send selection to GPU", !!selRange, "run the selected layers on the card", () => selRange && move(selRange[0], selRange[1], false, `layers ${selRange[0]}–${selRange[1]} to GPU`)),
        btn("Send selection to CPU", !!selRange, "run the selected layers on the CPU", () => selRange && move(selRange[0], selRange[1], true, `layers ${selRange[0]}–${selRange[1]} to CPU`)),
        h("button", { class: "btn sm ghost", text: "Select all", on: { click: () => { selRange = [0, nb - 1]; selAnchor = 0; drawPlacement(); } } }),
        h("button", { class: "btn sm ghost", text: "Clear", disabled: !selRange, on: { click: () => { selRange = null; selAnchor = -1; drawPlacement(); } } })),
      h("div", { class: "ctl" },
        btn("All on GPU", hostIdx.length > 0, "run every layer on the card", () => move(0, nb - 1, false, "all to GPU")),
        btn("All on CPU", nd > 0, "run every layer on the CPU", () => move(0, nb - 1, true, "all to CPU"))),
      no ? h("p", { class: "dim small", text: no }) : null,
      lastMove ? h("p", { class: "small", text: lastMove }) : null,
      h("p", { class: "dim small", text: "Moves apply to this conversation's own session; its history moves with the layers." })));
  placeBox.replaceChildren(...kids.filter((k) => k));
}

// ---- devices --------------------------------------------------------------

function deviceCard(d: Device): HTMLElement {
  const total = n(d.totalMemory?.bytes), free = n(d.freeMemory?.bytes);
  const kind = d.kind === DeviceKind.HOST ? "host" : d.kind === DeviceKind.INTEGRATED ? "integrated" : "discrete";
  return h("section", { class: "card dev" + (d.available ? "" : " off") },
    h("h3", {}, d.name || d.ref?.id || "device", h("span", { class: "chip", text: kind })),
    h("div", { class: "id", text: d.ref?.id ?? "" }),
    h("div", { class: "ctxrow" }, kv("memory", `${d.freeMemory?.human ?? "?"} free of ${d.totalMemory?.human ?? "?"}`), meter(total ? 1 - free / total : 0)),
    h("div", { class: "kvs" },
      kv("sessions", `${d.attachedSessions} attached · ${d.runningSessions} running`), kv("queued", String(d.queuedRequests)),
      kv("execution", d.execution === ExecutionMode.PARALLEL ? "parallel" : d.execution === ExecutionMode.SERIALISED ? "serialised" : "–", d.executionNote)),
    d.sameDeviceAs ? h("p", { class: "dim small", text: "Same physical device as " + d.sameDeviceAs }) : null,
    d.available ? null : h("p", { class: "warn small", text: d.unavailableReason || "unavailable" }),
    d.countsTowardHostBudget ? h("p", { class: "dim small", text: "Shares the host's memory; not a second pool." }) : null);
}

function deviceTab() {
  const host = live.stats?.host;
  const i = live.info;
  clear(body);
  body.append(h("div", { class: "stack" },
    host ? card("Host",
      h("div", { class: "ctxrow" }, kv("available", `${host.available?.human ?? "?"} of ${host.total?.human ?? "?"}`), meter(1 - n(host.available?.bytes) / Math.max(1, n(host.total?.bytes)))),
      h("div", { class: "kvs" },
        kv("weight budget", host.weightBudget?.human ?? "?"), kv("decode cores", String(host.decodeCores)),
        kv("performance", String(host.performanceCores)), kv("efficiency", String(host.efficiencyCores)))) : null,
    ...live.devs.map(deviceCard),
    live.devs.length ? null : h("p", { class: "dim", text: "No devices reported." }),
    i ? card("Server", h("div", { class: "kvs" },
      kv("version", i.version + (i.modified ? "+" : "")), kv("platform", `${i.os}/${i.arch}`),
      kv("backends", i.availableBackends.join(", ") || "host"), kv("uptime", uptime(n(i.uptimeMillis))),
      kv("models", String(i.loadedModels)), kv("sessions", String(i.openSessions))),
      h("div", { class: "id", text: i.modelDirectory })) : null));
}
function uptime(ms: number) {
  const s = Math.floor(ms / 1000);
  return s < 90 ? s + " s" : s < 5400 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h";
}

// ---- models ---------------------------------------------------------------

let filter = "";
let loadOpts = { device: "", budgetGiB: "", maxBlocks: "" };

// The devices a load offers its layers to: every available accelerator, CUDA
// and Metal before Vulkan, leaving out a Vulkan entry that is the same card as
// one already listed. "cpu" asks for the host alone.
// A software renderer is no accelerator, and a device sharing the host's memory
// is not a second pool; both stay pickable by name but are not in the default.
const accelerator = (d: Device) => d.kind !== DeviceKind.HOST && d.available && !d.sameDeviceAs && !/llvmpipe|swiftshader|software/i.test(d.name);
function gpuIds(): string[] {
  const rank = (id: string) => (id.startsWith("vulkan") ? 1 : 0);
  return live.devs.filter((d) => accelerator(d) && !d.countsTowardHostBudget).map((d) => d.ref?.id ?? "").filter(Boolean).sort((a, b) => rank(a) - rank(b));
}
const loadDevices = () => (loadOpts.device === "cpu" ? ["cpu"] : loadOpts.device ? [loadOpts.device] : gpuIds());

function modelsTab(force: boolean) {
  const m = live.files;
  const sig = JSON.stringify([live.scanning, live.list?.loaded.map((x) => x.modelId), m?.files.map((f) => [f.path, f.modelId, f.isContainer]), live.devs.map((d) => d.ref?.id), [...conv.entries()], live.stats?.models.map((s) => s.sessions)]);
  if (!force && sig === modelsSig) return;
  if (!force && body.contains(document.activeElement) && document.activeElement instanceof HTMLInputElement) return; // do not rebuild under typing
  modelsSig = sig;
  clear(body);
  if (!m) { body.append(h("p", { class: "dim pad", text: live.scanning ? "Reading the model folder (this can take a moment on a busy machine)…" : "The model folder has not been read yet." }), h("button", { class: "btn", text: "Read it", on: { click: () => refreshFiles() } })); return; }
  const act = async (b: HTMLButtonElement, label: string, run: () => Promise<unknown>) => {
    b.disabled = true; b.textContent = label;
    try { await run(); } catch (e) { ctx.toast(errText(e), true); }
    await refreshFiles();
    modelsSig = "";
    render(true);
  };
  const loaded = (live.list?.loaded ?? m.loaded).map((x) => {
    const un = h("button", { class: "btn sm ghost", text: "Unload", title: "Free this model's memory" });
    un.addEventListener("click", () => act(un, "Unloading", async () => {
      const open = live.stats?.models.find((s) => s.modelId === x.modelId)?.sessions ?? 0;
      if (open > 0 && !confirm(`${x.modelId} has ${open} open session${open === 1 ? "" : "s"}. Close them and unload?`)) return;
      await models.unloadModel({ modelId: x.modelId, force: open > 0 });
    }));
    const chat = h("button", { class: "btn sm", text: "Chat", title: "Start a new conversation on this model", on: { click: () => ctx.newChatOn(x.modelId) } });
    const users = live.stats?.models.find((s) => s.modelId === x.modelId)?.sessions ?? 0;
    return h("div", { class: "row" },
      h("div", { class: "grow" }, h("b", { text: x.modelId }), h("div", { class: "dim small", text: `${x.architecture} · ${x.blockCount} layers · ${x.defaultDeviceIds.length ? x.defaultDeviceIds.join(", ") : "CPU"} · ctx ${x.contextLength} · ${users} session${users === 1 ? "" : "s"}` })),
      chat, un);
  });
  const files = m.files.filter((f) => f.modelId === "" && !f.name.startsWith("mmproj-") && f.name.toLowerCase().includes(filter.toLowerCase()));
  const rows = files.map((f) => {
    const b = h("button", { class: "btn sm", text: f.isContainer ? "Load" : "Convert" });
    const prog = conv.get(f.path);
    if (f.isContainer) b.addEventListener("click", () => act(b, "Loading", () => models.loadModel({
      path: f.path,
      modelId: f.name,
      deviceIds: loadDevices(),
      pageBudgetBytes: loadOpts.budgetGiB ? BigInt(Math.round(Number(loadOpts.budgetGiB) * 2 ** 30)) : 0n,
      ...(loadOpts.maxBlocks !== "" ? { maxDeviceBlocks: Number(loadOpts.maxBlocks) } : {}),
    }).then((r) => ctx.toast(`Loaded ${r.model?.modelId ?? f.name}: ${loadDevices().join(", ") || "CPU"} (${r.loadMillis} ms)`))));
    else b.addEventListener("click", async () => {
      const out = f.path.replace(/\.[^./]+$/, "") + ".jlm";
      conv.set(f.path, "starting"); modelsSig = ""; render(true);
      try {
        for await (const r of models.convert({ sourcePath: f.path, outputPath: out })) { conv.set(f.path, `${r.phase} ${Math.round(r.fraction * 100)}%`); modelsSig = ""; render(true); }
        ctx.toast("Converted " + f.name);
      } catch (e) { ctx.toast(errText(e), true); }
      conv.delete(f.path); await refreshFiles(); modelsSig = ""; render(true);
    });
    return h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "fname", text: f.name }), h("div", { class: "dim small", text: [f.size?.human, f.isContainer ? "container" : "source: converts first"].filter(Boolean).join(" · ") })),
      prog ? h("span", { class: "chip", text: prog }) : b);
  });
  const g = gpuIds();
  const sel = h("select", { on: { change: (e: any) => (loadOpts.device = e.target.value) } },
    h("option", { value: "", text: g.length ? `All graphics cards (${g.join(", ")})` : "CPU (no graphics card found)" }),
    h("option", { value: "cpu", text: "CPU only" }),
    ...live.devs.filter(accelerator).map((d) => h("option", { value: d.ref?.id ?? "", text: `${d.ref?.id} · ${d.name}` })));
  sel.value = loadOpts.device;
  body.append(h("div", { class: "stack" },
    card("Loaded", ...(loaded.length ? loaded : [h("p", { class: "dim small", text: "Nothing loaded. Pick a model file below." })])),
    card("Load options",
      h("label", {}, "Run on", sel),
      h("div", { class: "two" },
        h("label", {}, "Page budget (GiB)", h("input", { value: loadOpts.budgetGiB, placeholder: "auto", inputMode: "decimal", on: { input: (e: any) => (loadOpts.budgetGiB = e.target.value) } })),
        h("label", {}, "Max GPU layers", h("input", { value: loadOpts.maxBlocks, placeholder: "auto", inputMode: "numeric", on: { input: (e: any) => (loadOpts.maxBlocks = e.target.value) } }))),
      h("p", { class: "dim small", text: "Applied to the next load. By default a model gets every graphics card, as many layers as fit; the rest run on the CPU." })),
    card("Model files", h("div", { class: "filehead" }, h("input", { class: "search", placeholder: "Filter", value: filter, on: { input: (e: any) => { filter = e.target.value; modelsSig = ""; render(true); (body.querySelector(".search") as HTMLInputElement)?.focus(); } } }), h("button", { class: "btn sm", text: live.scanning ? "Reading…" : "Refresh", title: "Read the model folder again", disabled: live.scanning, on: { click: () => refreshFiles() } })),
      ...(rows.length ? rows : [h("p", { class: "dim small", text: "No files here. " + (m.directory ? "Folder: " + m.directory : "") })]))));
}

// ---- sampling -------------------------------------------------------------

type Key = keyof typeof settings;
function field(label: string, k: Key, ph: string, tip = ""): HTMLElement {
  return h("label", { title: tip }, label, h("input", { value: String(settings[k]), placeholder: ph, inputMode: "decimal", on: { input: (e: any) => { (settings as any)[k] = e.target.value; saveSettings(); } } }));
}
function toggle(label: string, k: Key, tip = ""): HTMLElement {
  return h("label", { class: "tog", title: tip }, h("input", { type: "checkbox", checked: settings[k] as boolean, on: { change: (e: any) => { (settings as any)[k] = e.target.checked; saveSettings(); document.dispatchEvent(new CustomEvent("jitllm-settings")); } } }), label);
}
function settingsTab() {
  clear(body);
  body.append(h("div", { class: "stack" },
    card("Chat",
      toggle("Chat template", "chat", "Off sends the prompt as a raw completion, the way the benchmarks run."),
      toggle("Show thinking", "showThink", "Open <think> blocks while a reply streams."),
      toggle("Larger text", "bigText"),
      h("label", {}, "System prompt", h("textarea", { rows: 4, value: settings.system, placeholder: "Optional", on: { input: (e: any) => { settings.system = e.target.value; saveSettings(); } } }))),
    card("Sampling", h("p", { class: "dim small", text: "Blank keeps the engine's default. Temperature 0 is greedy." }),
      h("div", { class: "two" }, field("Temperature", "temperature", "default"), field("Top p", "topP", "off"), field("Top k", "topK", "off"), field("Min p", "minP", "off"),
        field("Repeat penalty", "repeatPenalty", "off"), field("Presence", "presence", "off"), field("Frequency", "frequency", "off"), field("Seed", "seed", "random"))),
    card("Output", field("Max tokens", "maxTokens", "1024"),
      h("label", {}, "Stop strings, one per line", h("textarea", { rows: 2, value: settings.stop, on: { input: (e: any) => { settings.stop = e.target.value; saveSettings(); } } })))));
}
