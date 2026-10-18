// Markdown to DOM, and a small code highlighter. No innerHTML anywhere: text
// goes in as text nodes and links are limited to http(s) and mailto.
import { h } from "./dom.ts";

const SAFE_URL = /^(https?:\/\/|mailto:)/i;

// ---- inline -------------------------------------------------------------

const INLINE = /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)|\*\*([^\s*][\s\S]*?)\*\*|__([^\s_][\s\S]*?)__|\*([^\s*][^*]*?)\*|(?<![A-Za-z0-9])_([^\s_][^_]*?)_(?![A-Za-z0-9])|~~([\s\S]+?)~~|\[([^\]]+)\]\(([^)\s]+)\)|(https?:\/\/[^\s<>)\]]+)/g;

export function inline(text: string): Node[] {
  const out: Node[] = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    const i = m.index ?? 0;
    if (i > last) out.push(...breaks(text.slice(last, i)));
    if (m[2] !== undefined) out.push(h("code", { class: "ic", text: m[2].trim() }));
    else if (m[3] !== undefined || m[4] !== undefined) out.push(h("strong", {}, ...inline(m[3] ?? m[4])));
    else if (m[5] !== undefined || m[6] !== undefined) out.push(h("em", {}, ...inline(m[5] ?? m[6])));
    else if (m[7] !== undefined) out.push(h("del", {}, ...inline(m[7])));
    else if (m[8] !== undefined) {
      if (SAFE_URL.test(m[9])) out.push(h("a", { href: m[9], target: "_blank", rel: "noopener noreferrer" }, ...inline(m[8])));
      else out.push(...breaks(m[0]));
    } else if (m[10] !== undefined) {
      const url = m[10].replace(/[.,;:!?]+$/, "");
      out.push(h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: url }));
      if (url.length < m[10].length) out.push(document.createTextNode(m[10].slice(url.length)));
    }
    last = i + m[0].length;
  }
  if (last < text.length) out.push(...breaks(text.slice(last)));
  return out;
}

function breaks(s: string): Node[] {
  const parts = s.split("\n");
  const out: Node[] = [];
  parts.forEach((p, i) => {
    if (i > 0) out.push(document.createElement("br"));
    if (p) out.push(document.createTextNode(p));
  });
  return out;
}

// ---- code highlighting --------------------------------------------------

const KW = new Set(("abstract as async await break case catch class const continue crate def default defer del do elif else enum except export extends " +
  "false final finally fn for from func function go goto if impl implements import in interface is lambda let loop match mod move mut namespace new nil none null " +
  "of package pass print private protected pub public raise range return self static struct super switch this throw trait true try type typeof unsafe use using var void " +
  "where while with yield select chan map and or not int string bool float double char long short byte uint u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 usize str SELECT FROM WHERE INSERT UPDATE DELETE JOIN").split(" "));

const TOKEN = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/|#[^\n]*|--[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b0x[0-9a-fA-F]+\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|([A-Za-z_$][\w$]*)(?=\s*\()|([A-Za-z_$][\w$]*)/g;

export function highlight(code: string, lang: string): Node[] {
  const out: Node[] = [];
  const hashComment = /^(py|python|sh|bash|shell|zsh|yaml|yml|toml|ruby|rb|r|perl|make|dockerfile|ini|conf)$/i.test(lang);
  const slashComment = !hashComment;
  let last = 0;
  const add = (cls: string, s: string) => out.push(h("span", { class: cls, text: s }));
  for (const m of code.matchAll(TOKEN)) {
    const i = m.index ?? 0;
    if (m[1] !== undefined) {
      const isHash = m[1].startsWith("#");
      const isSlash = m[1].startsWith("/");
      const isDash = m[1].startsWith("--");
      // A '#' opens a comment only in languages that say so; elsewhere it is code (#include, #[derive]).
      if ((isHash && !hashComment) || (isSlash && !slashComment && !m[1].startsWith("/*")) || (isDash && /^(js|ts|go|c|cpp|java|rust|rs|json)/i.test(lang))) continue;
    }
    if (i > last) out.push(document.createTextNode(code.slice(last, i)));
    if (m[1] !== undefined) add("tk-c", m[1]);
    else if (m[2] !== undefined) add("tk-s", m[2]);
    else if (m[3] !== undefined) add("tk-n", m[3]);
    else if (m[4] !== undefined) add(KW.has(m[4]) ? "tk-k" : "tk-f", m[4]);
    else if (KW.has(m[5])) add("tk-k", m[5]);
    else if (/^[A-Z][A-Za-z0-9]*$/.test(m[5]) && m[5].length > 1) add("tk-t", m[5]);
    else out.push(document.createTextNode(m[5]));
    last = i + m[0].length;
  }
  if (last < code.length) out.push(document.createTextNode(code.slice(last)));
  return out;
}

function codeBlock(code: string, lang: string): HTMLElement {
  const copy = h("button", { class: "copy", title: "Copy code", text: "Copy", on: { click: async () => {
    try { await navigator.clipboard.writeText(code); copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy"), 1200); } catch { copy.textContent = "No clipboard"; }
  } } });
  return h("div", { class: "code" },
    h("div", { class: "code-head" }, h("span", { text: lang || "text" }), copy),
    h("pre", {}, h("code", {}, ...highlight(code, lang))));
}

// ---- blocks -------------------------------------------------------------

const isTableSep = (s: string) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(s);
const cells = (s: string) => s.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
const LIST = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

export function markdown(src: string): HTMLElement {
  const root = h("div", { class: "md" });
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === "") { i++; continue; }

    const fence = line.match(/^\s*(`{3,}|~{3,})\s*([\w+#.-]*)/);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++]);
      i++; // the closing fence, or the end of a stream still arriving
      root.append(codeBlock(body.join("\n"), fence[2]));
      continue;
    }
    const hd = line.match(/^(#{1,6})\s+(.*?)\s*#*$/);
    if (hd) { root.append(h(("h" + Math.min(hd[1].length + 1, 6)) as "h2", {}, ...inline(hd[2]))); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { root.append(h("hr")); i++; continue; }
    if (line.startsWith(">")) {
      const q: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) q.push(lines[i++].replace(/^>\s?/, ""));
      root.append(h("blockquote", {}, markdown(q.join("\n"))));
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") rows.push(cells(lines[i++]));
      root.append(h("div", { class: "tbl" }, h("table", {},
        h("thead", {}, h("tr", {}, ...head.map((c) => h("th", {}, ...inline(c))))),
        h("tbody", {}, ...rows.map((r) => h("tr", {}, ...head.map((_, k) => h("td", {}, ...inline(r[k] ?? ""))))))))); 
      continue;
    }
    if (LIST.test(line)) {
      const base = line.match(LIST)![1].length;
      const ordered = /\d/.test(line.match(LIST)![2]);
      const list = h(ordered ? "ol" : "ul");
      let li: HTMLElement | null = null;
      while (i < lines.length) {
        const m = lines[i].match(LIST);
        if (m && m[1].length <= base + 1) {
          li = h("li", {}, ...inline(m[3]));
          list.append(li);
          i++;
        } else if (m && li) {
          // a deeper item: collect its run and nest it
          const sub: string[] = [];
          while (i < lines.length && (LIST.test(lines[i]) && lines[i].match(LIST)![1].length > base + 1)) sub.push(lines[i++].slice(base + 2));
          li.append(markdown(sub.join("\n")));
        } else if (li && lines[i].trim() !== "" && /^\s{2,}\S/.test(lines[i])) {
          li.append(...inline(" " + lines[i].trim()));
          i++;
        } else break;
      }
      root.append(list);
      continue;
    }
    // paragraph: runs until a blank line or the start of another block
    const p: string[] = [];
    while (i < lines.length && lines[i].trim() !== "" && !/^\s*(`{3,}|~{3,})/.test(lines[i]) && !/^#{1,6}\s/.test(lines[i]) && !lines[i].startsWith(">") && !(p.length && LIST.test(lines[i]))) p.push(lines[i++]);
    root.append(h("p", {}, ...inline(p.join("\n"))));
  }
  return root;
}
