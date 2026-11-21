# jitllm website

An Astro site: a hand-built landing page and Starlight docs under `/docs`.

- `src/pages/index.astro`: the landing page, plain HTML; its CSS and scripts are
  in `public/` and load as-is.
- `public/themes.css`: the seven colour themes, shared by the landing page
  (`data-theme`) and the docs (`data-palette`).
- `public/styles.css`: the landing page.
- `public/app.js`: the hero's pixel field, the theme picker and small behaviours.
- `public/machine.js`: "Watch it schedule", an isometric simulation of placement,
  paging and the KV cache. It illustrates the policy; it is not telemetry.
- `public/numbers.js`: the benchmark boards, transcribed from `.notes/perf/current.md`.
- `src/content/docs/docs/`: the documentation, one Markdown file a page, served
  under `/docs`: start-here pages at the top, then `guides/`, `concepts/` and
  `reference/`. The sidebar is in `astro.config.mjs`.
- `src/styles/docs.css` maps the theme tokens onto Starlight's; the components in
  `src/components/` replace Starlight's title (the pixel wordmark) and its
  light/dark switch (the same theme list as the landing page, remembered in the
  same `localStorage` key).

Run it locally from this folder (Node 22 or newer):

```sh
npm install
npm run dev      # http://localhost:4321
npm run build    # static site in dist/, search index included
```

The visual system (pixel-field hero, banded pixel wordmark, square controls,
Geist + JetBrains Mono, Tokyo Night default with a theme picker) takes Omarchy's
site as its reference. The copy, wordmark, icons and code are original to jitllm.

- Themes live as CSS custom properties in `public/themes.css`; add one there, to
  `THEMES` in `app.js` and to the list in `src/components/ThemeSelect.astro`
  (and to `LIGHT` in the two theme components if it is a light theme). Click the
  hero wordmark to cycle them.
- The hero is the "compile" scene over the "flow" noise. `?dev` (or `?hero=` / `?noise=`) shows a bar
  for switching to the other scenes and noise fields.
- The docs quote the engine: flags, endpoints and numbers come from
  `cmd/jitllm`, `server/`, `library/models.go` and `.notes/perf/current.md`. When
  those change, the pages that quote them need the same change.
- `prefers-reduced-motion` draws the field once, skips the wordmark sweep, the
  typewriter and every transition; the page stays fully usable.
- Fonts load from Google Fonts; everything else is local.
