// What the page remembers: settings and saved chats, in localStorage. Both
// reads and writes are guarded, since a private window can refuse either; the
// page works without them and just forgets.
import { titleOf } from "./chat.ts";

export interface Settings {
  chat: boolean; // the chat template, or a raw completion of the prompt
  showThink: boolean;
  system: string;
  temperature: string; topP: string; topK: string; minP: string;
  repeatPenalty: string; presence: string; frequency: string; seed: string;
  maxTokens: string; stop: string;
  panel: boolean; tab: string; chats: boolean; model: string;
  bigText: boolean;
}
export const DEFAULTS: Settings = {
  chat: true, showThink: true, system: "",
  temperature: "", topP: "", topK: "", minP: "", repeatPenalty: "", presence: "", frequency: "", seed: "",
  maxTokens: "1024", stop: "", panel: true, tab: "engine", chats: true, model: "", bigText: false,
};

function read<T>(k: string, d: T): T {
  try { const v = localStorage.getItem(k); return v === null ? d : { ...d, ...JSON.parse(v) }; } catch { return d; }
}
function write(k: string, v: unknown) {
  try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage refused: the page forgets on reload */ }
}

export const settings: Settings = read("jitllm.settings", DEFAULTS);
export const saveSettings = () => write("jitllm.settings", settings);

export interface Msg {
  role: "user" | "assistant";
  content: string;
  pics?: number; // pictures attached; the bytes are not stored
  model?: string;
  meta?: Meta;
}
export interface Meta {
  reason?: string; tokens?: number; tps?: number; ttft?: number; prompt?: number; cached?: number; gbps?: number; error?: string; stopped?: boolean; dev?: number; host?: number;
}
export interface Chat { id: string; title: string; model: string; at: number; msgs: Msg[]; }

let chats: Chat[] = read<{ v: Chat[] }>("jitllm.chats", { v: [] }).v;
export const allChats = () => chats;
export const saveChats = () => write("jitllm.chats", { v: chats.slice(0, 200) });
export function newChat(model: string): Chat {
  const c: Chat = { id: Math.random().toString(36).slice(2, 10), title: "New chat", model, at: Date.now(), msgs: [] };
  chats.unshift(c);
  return c;
}
export function touch(c: Chat) {
  const first = c.msgs.find((m) => m.role === "user");
  if (first) c.title = titleOf(first.content || "Picture");
  c.at = Date.now();
  chats = [c, ...chats.filter((x) => x !== c)];
  saveChats();
}
export function dropChat(c: Chat) { chats = chats.filter((x) => x !== c); saveChats(); }
