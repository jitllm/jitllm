// One long-lived server session per chat: opened when the chat starts (or is
// first shown), reused for every turn so follow-ups keep their KV cache, and
// closed when the chat is deleted or the page unloads. Sessions are not kept
// across a reload; a reopened chat gets a new one and its first turn
// re-reads the history.
import { sessions, closeOnUnload } from "./api.ts";
import { changed } from "./live.ts";

interface Entry { id: string; model: string; }
const open = new Map<string, Entry>();
const opening = new Map<string, Promise<string>>();

export const sessionOf = (chatId: string, model: string): string => {
  const e = open.get(chatId);
  return e && e.model === model ? e.id : "";
};

// ensure opens the chat's session on its model, replacing one opened on another
// model. Concurrent callers share one create.
export function ensure(chatId: string, model: string): Promise<string> {
  const have = open.get(chatId);
  if (have && have.model === model) return Promise.resolve(have.id);
  const pending = opening.get(chatId);
  if (pending) return pending;
  const p = (async () => {
    if (have) await close(chatId);
    const r = await sessions.createSession({ modelId: model });
    const id = r.session?.sessionId ?? "";
    if (id) open.set(chatId, { id, model });
    changed();
    return id;
  })().finally(() => opening.delete(chatId));
  opening.set(chatId, p);
  return p;
}

export async function close(chatId: string) {
  const e = open.get(chatId);
  open.delete(chatId);
  if (!e) return;
  try { await sessions.closeSession({ sessionId: e.id }); } catch { /* already gone */ }
  changed();
}

// forget drops a session the server no longer has.
export const forget = (chatId: string) => { open.delete(chatId); };

addEventListener("pagehide", () => {
  for (const e of open.values()) closeOnUnload("/jitllm.v1.SessionService/CloseSession", { sessionId: e.id });
});
