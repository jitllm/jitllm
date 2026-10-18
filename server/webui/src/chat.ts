// Pure helpers behind the chat view, kept free of the DOM and the generated
// client so node's test runner can run them.

export interface Turn {
  role: "user" | "assistant";
  content: string;
  // Media types of the pictures attached to a user turn, for display; the
  // bytes live beside the turn in the page.
  images?: { data: Uint8Array; mediaType: string }[];
}

// Stop strings are typed one per line; blank lines mean nothing.
export function parseStop(text: string): string[] {
  return text.split("\n").map((s) => s.trim()).filter((s) => s !== "");
}

// A number field left empty is "unset" (the server's default), not zero:
// temperature 0 is greedy and an absent one is the engine's own default.
export function optNumber(text: string): number | undefined {
  const t = text.trim();
  if (t === "") return undefined;
  const n = Number(t);
  return Number.isFinite(n) ? n : undefined;
}

// The messages a request carries: the whole conversation, since the UI sends
// every turn as a stateless chat and the server re-prefills (or answers from
// its prefix cache). An assistant turn that was stopped before any token came
// back is dropped so the next request does not carry an empty reply.
export function requestMessages(history: Turn[]) {
  return history
    .map((t) => ({ role: t.role, content: t.role === "assistant" ? withoutThink(t.content) : t.content, images: t.images ?? [] }))
    .filter((t) => t.role === "user" || t.content !== "");
}

export function rate(tokens: number, millis: number): string {
  if (millis <= 0 || tokens <= 0) return "";
  return (tokens / (millis / 1000)).toFixed(1) + " tok/s";
}

// The key travels in the URL only to be moved out of it: it is read once,
// kept in sessionStorage (never localStorage) and the address bar is cleaned.
export function takeKey(search: string, stored: string | null): { key: string | null; cleaned: string | null } {
  const q = new URLSearchParams(search);
  const k = q.get("key");
  if (k === null) return { key: stored, cleaned: null };
  q.delete("key");
  const rest = q.toString();
  return { key: k, cleaned: rest === "" ? "" : "?" + rest };
}

// A reply split into the model's reasoning and its answer. Reasoning models
// write <think>...</think> first; while it is still streaming the closing tag
// has not arrived, and everything after the opening one is thought.
export interface Parts { think: string; answer: string; thinking: boolean; }
export function splitThink(text: string): Parts {
  const open = text.indexOf("<think>");
  if (open === -1) {
    // Some templates put the opening tag in the prompt, so the reply starts
    // mid-thought and only the closing tag shows.
    const c = text.indexOf("</think>");
    if (c !== -1) return { think: text.slice(0, c).trim(), answer: text.slice(c + 8).replace(/^\s+/, ""), thinking: false };
    return { think: "", answer: text, thinking: false };
  }
  const head = text.slice(0, open);
  const close = text.indexOf("</think>", open);
  if (close === -1) return { think: text.slice(open + 7).replace(/^\s+/, ""), answer: head, thinking: true };
  return { think: text.slice(open + 7, close).trim(), answer: (head + text.slice(close + 8)).replace(/^\s+/, ""), thinking: false };
}

// The history sent to the server carries the answer only: a model is not
// shown its own earlier reasoning.
export function withoutThink(text: string): string {
  return splitThink(text).answer;
}

export function titleOf(text: string): string {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 48 ? t.slice(0, 47) + "…" : t || "New chat";
}

export function human(bytes: number): string {
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v.toFixed(0) : v.toFixed(v < 10 ? 2 : 1)) + " " + u[i];
}
