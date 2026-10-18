// A tiny element builder. Every string goes in as a text node: model output,
// file names and server messages are untrusted and never reach innerHTML.

type Child = Node | string | number | null | undefined | false;
type Props = {
  class?: string;
  text?: string;
  title?: string;
  on?: Record<string, (e: any) => void>;
  attrs?: Record<string, string>;
  style?: Partial<CSSStyleDeclaration> | Record<string, string>;
  [k: string]: unknown;
};

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Props = {}, ...kids: Child[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null) continue;
    if (k === "class") e.className = String(v);
    else if (k === "text") e.textContent = String(v);
    else if (k === "on") for (const [ev, fn] of Object.entries(v as Record<string, EventListener>)) e.addEventListener(ev, fn);
    else if (k === "attrs") for (const [a, val] of Object.entries(v as Record<string, string>)) e.setAttribute(a, val);
    else if (k === "style") for (const [s, val] of Object.entries(v as Record<string, string>)) (e.style as any).setProperty(s, val);
    else if (k in e) (e as any)[k] = v;
    else e.setAttribute(k, String(v));
  }
  for (const c of kids) if (c !== null && c !== undefined && c !== false) e.append(typeof c === "object" ? c : String(c));
  return e;
}

export function svg(tag: string, attrs: Record<string, string | number> = {}, ...kids: Node[]): SVGElement {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  e.append(...kids);
  return e;
}

export function clear(e: Element) { e.replaceChildren(); }
