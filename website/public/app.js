// jitllm site. No dependencies: a pixel field on canvas, a theme picker and a
// few small behaviours. Everything reads its colours from CSS custom
// properties, so a theme change is one attribute on <html>.
;(() => {
  const root = document.documentElement
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches
  const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v))

  /* ------------------------------------------------------------ themes */

  // [id, name, bg, brand, field-dim] — the last three only paint the swatch.
  const THEMES = [
    ['tokyo-night', 'Tokyo Night', '#1a1b26', '#9ece6a', '#39482e'],
    ['catppuccin', 'Catppuccin', '#1e1e2e', '#89b4fa', '#34415c'],
    ['gruvbox', 'Gruvbox', '#282828', '#7daea3', '#354440'],
    ['nord', 'Nord', '#2e3440', '#81a1c1', '#384452'],
    ['everforest', 'Everforest', '#2d353b', '#7fbbb3', '#374c4c'],
    ['kanagawa', 'Kanagawa', '#1f1f28', '#dcd7ba', '#4e4c47'],
    ['rose-pine-dawn', 'Rosé Pine Dawn', '#faf4ed', '#56949f', '#b7c6c5'],
  ]
  const THEME_EVENT = 'jitllm:theme'
  const currentTheme = () =>
    THEMES.some(([id]) => id === root.dataset.theme) ? root.dataset.theme : THEMES[0][0]

  function setTheme(id) {
    root.dataset.theme = id
    try {
      localStorage.setItem('jitllm-theme', id)
    } catch {}
    const meta = document.querySelector('meta[name="theme-color"]')
    meta?.setAttribute('content', getComputedStyle(root).getPropertyValue('--f-bg').trim())
    dispatchEvent(new CustomEvent(THEME_EVENT))
  }
  const nextTheme = () => {
    const i = THEMES.findIndex(([id]) => id === currentTheme())
    setTheme(THEMES[(i + 1) % THEMES.length][0])
  }

  /* ------------------------------------------------------------ bitmaps */

  // The wordmark, drawn in rectangles on a 53 x 21 cell grid: [x, y, w, h].
  // Stroke is three cells; letters sit three cells apart.
  const WORD_W = 53
  const WORD_H = 21
  const WORD_RECTS = [
    // j
    [3, 0, 3, 3], [3, 5, 3, 15], [0, 18, 5, 3],
    // i
    [9, 0, 3, 3], [9, 5, 3, 12],
    // t
    [17, 1, 3, 16], [15, 5, 8, 3], [20, 14, 3, 3],
    // l l
    [26, 0, 3, 17], [32, 0, 3, 17],
    // m, the last shoulder cut so the corner reads as a curve
    [38, 5, 14, 3], [38, 5, 3, 12], [44, 5, 3, 12], [50, 6, 3, 11],
  ]
  const WORD_ROWS = Array.from({ length: WORD_H }, () => Array(WORD_W).fill('0'))
  for (const [x, y, w, h] of WORD_RECTS)
    for (let r = y; r < y + h; r++) for (let c = x; c < x + w; c++) WORD_ROWS[r][c] = '1'
  const WORD = WORD_ROWS.map((r) => r.join(''))

  // Row bands, top to bottom: the wordmark cools from crest to dim.
  const BANDS = [['crest', 5], ['hover', 3], ['lit', 5], ['mid', 4], ['dim', 4]]
  const bandOf = (row, height) => {
    const total = BANDS.reduce((s, [, n]) => s + n, 0)
    let at = (row / height) * total
    for (const [name, n] of BANDS) {
      if (at < n) return name
      at -= n
    }
    return 'dim'
  }

  const ICONS = {
    chip: [
      '00010101000',
      '00010101000',
      '00111111100',
      '11100000111',
      '00101110100',
      '11101110111',
      '00101110100',
      '11100000111',
      '00111111100',
      '00010101000',
      '00010101000',
    ],
    probe: [
      '00001110000',
      '00110001100',
      '01000100010',
      '01000100010',
      '10000000001',
      '11110101111',
      '10000000001',
      '01000100010',
      '01000100010',
      '00110001100',
      '00001110000',
    ],
    bolt: [
      '00000011110',
      '00000111100',
      '00001111000',
      '00011110000',
      '00111111110',
      '01111111100',
      '00000111000',
      '00001110000',
      '00011100000',
      '00111000000',
      '01100000000',
    ],
    pages: [
      '00001111111',
      '00001000001',
      '00111111101',
      '00100000101',
      '11111110101',
      '10000010101',
      '10111010111',
      '10000010100',
      '10111011100',
      '10000010000',
      '11111110000',
    ],
    box: [
      '11111111111',
      '10000000001',
      '11111111111',
      '10000000001',
      '10011111001',
      '10000000001',
      '10000000001',
      '10000000001',
      '10000000001',
      '10000000001',
      '11111111111',
    ],
  }

  /** A bitmap as crisp SVG, banded by row. Colours are theme variables. */
  function pixelSVG(rows) {
    const h = rows.length
    const w = rows[0].length
    let rects = ''
    rows.forEach((bits, y) => {
      const fill = `fill:var(--f-${bandOf(y, h)})`
      for (let x = 0; x < w; x++) {
        if (bits[x] !== '1') continue
        let run = 1
        while (bits[x + run] === '1') run++
        rects += `<rect x="${x}" y="${y}" width="${run}" height="1" style="${fill}"/>`
        x += run - 1
      }
    })
    return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true">${rects}</svg>`
  }
  for (const el of document.querySelectorAll('[data-pixel]')) {
    const name = el.dataset.pixel
    el.innerHTML = pixelSVG(name === 'word' ? WORD : ICONS[name])
  }

  /* ------------------------------------------------------------ pixel field */

  // Ordered-dither thresholds, 8 x 8, values 0..63.
  const BAYER = [
    0, 48, 12, 60, 3, 51, 15, 63, 32, 16, 44, 28, 35, 19, 47, 31, 8, 56, 4, 52,
    11, 59, 7, 55, 40, 24, 36, 20, 43, 27, 39, 23, 2, 50, 14, 62, 1, 49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29, 10, 58, 6, 54, 9, 57, 5, 53, 42, 26, 38, 22,
    41, 25, 37, 21,
  ]

  function lcg(seed) {
    let s = seed >>> 0
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296)
  }

  // Smooth tileable value noise: white noise softened by two box blurs.
  const N = 96
  const noise = (() => {
    const rand = lcg(0x6a17)
    let f = Float32Array.from({ length: N * N }, rand)
    for (let pass = 0; pass < 2; pass++) {
      const g = new Float32Array(N * N)
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          let s = 0
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) s += f[((y + dy + N) % N) * N + ((x + dx + N) % N)]
          g[y * N + x] = s / 9
        }
      f = g
    }
    let lo = Infinity
    let hi = -Infinity
    for (const v of f) (lo = Math.min(lo, v)), (hi = Math.max(hi, v))
    return f.map((v) => (v - lo) / (hi - lo))
  })()
  const jitter = Float32Array.from({ length: 64 * 64 }, lcg(0xc0de))

  function sampleNoise(x, y) {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const fx = x - xi
    const fy = y - yi
    const x0 = ((xi % N) + N) % N
    const y0 = ((yi % N) + N) % N
    const x1 = (x0 + 1) % N
    const y1 = (y0 + 1) % N
    const sx = fx * fx * (3 - 2 * fx)
    const sy = fy * fy * (3 - 2 * fy)
    const top = noise[y0 * N + x0] * (1 - sx) + noise[y0 * N + x1] * sx
    const bot = noise[y1 * N + x0] * (1 - sx) + noise[y1 * N + x1] * sx
    return top * (1 - sy) + bot * sy
  }

  /* ------------------------------------------------------------ hero: your machine */

  // What the browser says about the machine it runs on. Nothing leaves the page.
  function cleanGpu(raw) {
    let s = String(raw || '')
    const angle = /^ANGLE \((.*)\)$/.exec(s)
    if (angle) {
      const parts = angle[1].split(', ')
      s = parts[1] || parts[0]
    }
    s = s
      .replace(/ANGLE Metal Renderer:\s*/i, '')
      .replace(/\(0x[0-9a-f]+\)/gi, '')
      .replace(/\s*Direct3D.*$/i, '')
      .replace(/\/PCIe\/SSE2|\/SSE2/gi, '')
      .replace(/\((R|TM)\)/gi, '')
      .replace(/\b(Mesa|Corporation)\b/gi, '')
      .replace(/,? or similar/i, '')
      .replace(/OpenGL.*$/i, '')
      .replace(/\s*(Laptop )?GPU\s*$/i, '')
      .replace(/\s+/g, ' ')
      .trim()
    if (/SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(s)) return 'software GPU'
    return s.length > 30 ? s.slice(0, 29) + '…' : s
  }
  function readMachine() {
    let gpu = ''
    try {
      const gl = document.createElement('canvas').getContext('webgl')
      const ext = gl && gl.getExtension('WEBGL_debug_renderer_info')
      gpu = cleanGpu(gl && gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER))
    } catch {}
    const ua = navigator.userAgent
    const os = /iPhone|iPad/.test(ua) ? 'iOS' : /Android/.test(ua) ? 'Android' : /Mac/.test(ua) ? 'macOS' : /Linux|X11/.test(ua) ? 'Linux' : /Win/.test(ua) ? 'Windows' : ''
    const arch = /arm|aarch64/i.test(ua) || /^Apple/.test(gpu) ? 'ARM64' : /x86_64|Win64|x64|WOW64|Intel/.test(ua) ? 'x86\u201164' : ''
    return { threads: navigator.hardwareConcurrency || 0, gpu, os, arch }
  }

  // The field condenses into a pixel schematic of that machine: a CPU package
  // with a core per thread, a GPU card, and the .jlm on disk below the copy.
  // Model blocks stream out of the disk, the fastest device first, a token
  // runs through what they filled, and it all dithers back into the field.
  function makeMachine(info) {
    const T = { rest: 2600, read: 1600, place: 3800, run: 2600, fade: 1400 }
    const CYCLE = T.rest + T.read + T.place + T.run + T.fade
    let parts = [] // { x, y, kind }
    let slots = [] // { x, y, w, h, dev, path, filled }
    let labels = []
    let boxes = []
    let lanes = {}
    let born = performance.now() - T.rest + 700 // the first read starts almost at once
    let key = ''

    const add = (x, y, kind) => parts.push({ x, y, kind })
    const rect = (x0, y0, w, h, kind) => {
      for (let x = x0; x < x0 + w; x++) add(x, y0, kind), add(x, y0 + h - 1, kind)
      for (let y = y0 + 1; y < y0 + h - 1; y++) add(x0, y, kind), add(x0 + w - 1, y, kind)
    }

    function cpu(x0, y0, room) {
      const n = Math.max(2, Math.min(info.threads || 8, 64))
      const gc = Math.ceil(Math.sqrt(n))
      const gr = Math.ceil(n / gc)
      const s = gc * 3 + 3 <= room ? 2 : 1 // core cells 2x2 when there is room
      const inW = gc * (s + 1) - 1
      const inH = gr * (s + 1) - 1
      const w = inW + 4
      const h = inH + 4
      const bx = x0 + Math.floor((room - w) / 2)
      rect(bx, y0, w, h, 'body')
      for (let x = bx + 2; x < bx + w - 2; x += 2) add(x, y0 - 1, 'pin'), add(x, y0 + h, 'pin')
      for (let y = y0 + 2; y < y0 + h - 2; y += 2) add(bx - 1, y, 'pin'), add(bx + w, y, 'pin')
      for (let i = 0; i < n; i++) {
        const cx = bx + 2 + (i % gc) * (s + 1)
        const cy = y0 + 2 + Math.floor(i / gc) * (s + 1)
        slots.push({ x: cx, y: cy, w: s, h: s, dev: 'cpu', filled: false })
        for (let k = 0; k < s * s; k++) add(cx + (k % s), cy + Math.floor(k / s), 'slot')
      }
      labels.push({ x: bx + w / 2, y: y0 + h + 3, text: `${info.threads || '?'} threads` })
      lanes.cpu = bx - 3
      boxes.push([bx - 2, y0 - 2, bx + w + 2, y0 + h + 4])
      return { x: bx, y: y0, w, h }
    }

    function gpu(x0, y0, room) {
      const w = Math.min(room - 2, 26)
      const h = 12
      const bx = x0 + Math.floor((room - w) / 2)
      rect(bx, y0, w, h, 'body')
      for (let y = y0; y < y0 + h; y++) add(bx - 1, y, 'pin') // the bracket
      for (let x = bx + 4; x < bx + w - 6; x += 2) add(x, y0 + h, 'pin') // the edge connector
      const die = { x: bx + 3, y: y0 + 3, w: 6, h: 6 }
      for (let k = 0; k < 36; k++) add(die.x + (k % 6), die.y + Math.floor(k / 6), 'die')
      for (let r = 0; r < 2; r++)
        for (let c = 0; bx + 11 + c * 3 + 2 <= bx + w - 2; c++) {
          const vx = bx + 11 + c * 3
          const vy = y0 + 3 + r * 4
          slots.push({ x: vx, y: vy, w: 2, h: 2, dev: 'gpu', filled: false })
          for (let k = 0; k < 4; k++) add(vx + (k % 2), vy + Math.floor(k / 2), 'slot')
        }
      labels.push({ x: bx + w / 2, y: y0 + h + 3, text: info.gpu || 'GPU' })
      lanes.gpu = bx + w + 2
      boxes.push([bx - 2, y0 - 1, bx + w + 1, y0 + h + 4])
      return { x: bx, y: y0, w, h, die }
    }

    return {
      layout(g) {
        const k = [g.visC0, g.visC1, g.barRow, g.visR1, g.below, g.above].join(',')
        if (k === key) return
        key = k
        parts = []
        slots = []
        labels = []
        boxes = []
        lanes = {}
        const room = -5 - g.visC0 // columns left of the wordmark
        let disk
        if (room >= 16) {
          const mid = Math.round(WORD_H / 2)
          cpu(g.visC0 + 2, Math.max(g.barRow + 3, mid - 8), room - 2)
          gpu(WORD_W + 4, Math.max(g.barRow + 3, mid - 6), g.visC1 - WORD_W - 5)
          disk = { x: Math.round(WORD_W / 2) - 7, y: g.below + 3, w: 14, h: 4 }
          if (disk.y + disk.h > g.visR1) disk.y = g.visR1 - disk.h
        } else {
          // Narrow screens: the two devices side by side, below the copy if
          // there is room, else in the space above the callout.
          const half = Math.floor((g.visC1 - g.visC0) / 2)
          const top = g.visR1 - g.below - 3 >= 16 ? g.below + 3 : g.above - g.barRow >= 17 ? g.barRow + 2 : -1
          if (top < 0) return
          cpu(g.visC0 + 1, top, half - 1)
          gpu(g.visC0 + half + 1, top + 1, half - 1)
        }
        if (disk) {
          rect(disk.x, disk.y, disk.w, disk.h, 'body')
          labels.push({ x: disk.x + disk.w / 2, y: disk.y + disk.h / 2 + 0.3, text: '.jlm', inside: true })
          boxes.push([disk.x - 1, disk.y - 1, disk.x + disk.w + 1, disk.y + disk.h + 1])
        }
        // Each slot's route: along the band below the copy, up the side, and in.
        const sx = disk ? disk.x + disk.w / 2 : Math.round((g.visC0 + g.visC1) / 2)
        const lane = disk ? disk.y + disk.h + 1 : g.visR1
        for (const s of slots) {
          const path = []
          let x = Math.round(sx)
          let y = lane
          const side = lanes[s.dev]
          const tx = disk ? side : s.x
          if (!disk) y = s.y < g.below ? g.barRow : g.visR1
          while (x !== tx) path.push([x, y]), (x += Math.sign(tx - x))
          while (y !== s.y) path.push([x, y]), (y += Math.sign(s.y - y))
          while (x !== s.x) path.push([x, y]), (x += Math.sign(s.x - x))
          path.push([s.x, s.y])
          s.path = path
        }
        // Fastest device first: the card's slots are filled before the cores.
        slots.sort((a, b) => (a.dev === b.dev ? 0 : a.dev === 'gpu' ? -1 : 1))
      },

      // How much of the field to keep under the schematic: it steps back while
      // the machine is showing, so the shapes read.
      hush(col, row, now) {
        const v = this.shown(now)
        if (!v) return 1
        for (const [a, b, c, d] of boxes) if (col >= a && col <= c && row >= b && row <= d) return 1 - 0.85 * v
        return 1
      },
      shown(now) {
        const t = (now - born) % CYCLE
        if (t < T.rest) return 0
        if (t < T.rest + T.read) return (t - T.rest) / T.read
        if (t < CYCLE - T.fade) return 1
        return 1 - (t - (CYCLE - T.fade)) / T.fade
      },

      draw(ctx, g, pal, now, reduced) {
        if (!parts.length) return
        const t = reduced ? T.rest + T.read + T.place + 400 : (now - born) % CYCLE
        if (t < T.rest) return
        const tRead = t - T.rest
        const tPlace = tRead - T.read
        const tRun = tPlace - T.place
        const fading = t > CYCLE - T.fade
        const kFade = fading ? (t - (CYCLE - T.fade)) / T.fade : 0
        const scan = tRead < T.read ? g.visC0 + (g.visC1 - g.visC0) * (tRead / T.read) : null
        const cell = (x, y, ink) => {
          const px = Math.round(g.ox + x * g.cw)
          const py = Math.round(g.oy + y * g.ch)
          ctx.fillStyle = ink
          ctx.fillRect(px, py, Math.round(g.ox + (x + 1) * g.cw) - px, Math.round(g.oy + (y + 1) * g.ch) - py)
        }
        const seen = (x, y) => {
          const j = jitter[(y & 63) * 64 + (x & 63)]
          if (scan !== null && x > scan) return false
          if (fading && j < kFade) return false
          return true
        }
        // Blocks in flight, and which slots they have filled.
        const each = T.place * 0.7 / Math.max(1, slots.length)
        const filledBy = (i) => tPlace >= 0 && tPlace - i * each >= slots[i].path.length * 20
        const busy = tRun >= 0 && !fading ? Math.floor(tRun / 55) % Math.max(1, slots.length) : -1
        for (const p of parts) {
          if (!seen(p.x, p.y)) continue
          let ink = p.kind === 'pin' ? pal.dim : p.kind === 'body' ? pal.mid : p.kind === 'die' ? pal.mid : pal.dim
          if (scan !== null && Math.abs(p.x - scan) < 1.5) ink = pal.crest
          cell(p.x, p.y, ink)
        }
        slots.forEach((s, i) => {
          const full = filledBy(i)
          if (!full) return
          const hot = i === busy || (s.dev === 'gpu' && tRun >= 0 && !fading && Math.floor(tRun / 120) % 2 === 0)
          for (let k = 0; k < s.w * s.h; k++) {
            const x = s.x + (k % s.w)
            const y = s.y + Math.floor(k / s.w)
            if (seen(x, y)) cell(x, y, hot ? pal.crest : pal.lit)
          }
        })
        if (tPlace >= 0 && !fading)
          slots.forEach((s, i) => {
            const at = Math.floor((tPlace - i * each) / 20)
            if (at < 0 || at >= s.path.length) return
            for (let tail = 0; tail < 4; tail++) {
              const q = s.path[at - tail]
              if (q) cell(q[0], q[1], tail ? pal.mid : pal.hover)
            }
          })
        // Labels, in the page's mono, fading with the shapes.
        const v = this.shown(now)
        ctx.globalAlpha = reduced ? 1 : Math.max(0, Math.min(1, v * 1.2 - 0.1))
        ctx.fillStyle = pal.lit
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.font = `${Math.round(11 * g.dpr)}px "JetBrains Mono", ui-monospace, monospace`
        for (const l of labels) ctx.fillText(l.text, g.ox + l.x * g.cw, g.oy + l.y * g.ch)
        ctx.globalAlpha = 1
      },
    }
  }

  /* ------------------------------------------------------------ hero: forward pass */

  // A forward pass, one token at a time, around the copy. The model's layers
  // are bars of field pixels: the first run on the GPU (the right stack), the
  // rest on the CPU (the left). A token drops through the GPU layers fast,
  // crosses the seam along the bottom, climbs the CPU layers slower, comes out
  // at the top as the next token and goes round again.
  function makePass(info) {
    const GPU_MS = 45 // per layer
    const CPU_MS = 120
    const LANE_MS = 9 // per cell of wire
    let bars = [] // { x, y, w, dev, hot }
    let route = [] // [{ x, y, t, bar }] in order, t = arrival ms from the token's start
    let span = 0
    let labels = []
    let boxes = []
    let key = ''
    let tokens = 0
    let lastLap = -1
    const born = performance.now()

    return {
      layout(g) {
        const k = [g.visC0, g.visC1, g.barRow, g.visR1, g.below].join(',')
        if (k === key) return
        key = k
        bars = []
        route = []
        labels = []
        boxes = []
        const room = -5 - g.visC0
        if (room < 14) return
        const top = g.barRow + 4
        const bottom = g.visR1 - 3
        const n = Math.min(14, Math.floor((bottom - top) / 3))
        const w = Math.min(room - 6, 22)
        const lx = g.visC0 + 2 + Math.floor((room - 4 - w) / 2)
        const rx = WORD_W + 4 + Math.floor((g.visC1 - WORD_W - 5 - w) / 2)
        const gap = (bottom - top) / n
        const gpuN = Math.ceil(n * 0.55)
        const cpuN = n
        for (let i = 0; i < gpuN; i++) bars.push({ x: rx, y: Math.round(top + i * gap), w, dev: 'gpu', hot: -1e9 })
        for (let i = 0; i < cpuN; i++) bars.push({ x: lx, y: Math.round(bottom - 1 - i * gap), w, dev: 'cpu', hot: -1e9 })
        // The token's route: down the right stack, along the bottom, up the
        // left stack, and back along the top.
        let t = 0
        const wire = (x0, y0, x1, y1) => {
          let x = x0
          let y = y0
          while (x !== x1 || y !== y1) {
            if (x !== x1) x += Math.sign(x1 - x)
            else y += Math.sign(y1 - y)
            t += LANE_MS
            route.push({ x, y, t, bar: -1 })
          }
        }
        const inR = rx - 2
        const inL = lx + w + 1
        route.push({ x: inR, y: top - 2, t: 0, bar: -1 })
        let at = { x: inR, y: top - 2 }
        bars.forEach((b, i) => {
          const lane = b.dev === 'gpu' ? inR : inL
          if (i === gpuN) {
            // The seam: out of the card, across the bottom, into the host.
            wire(at.x, at.y, at.x, bottom + 1)
            wire(inR, bottom + 1, inL, bottom + 1)
            at = { x: inL, y: bottom + 1 }
          }
          wire(at.x, at.y, lane, b.y)
          t += b.dev === 'gpu' ? GPU_MS : CPU_MS
          route.push({ x: lane, y: b.y, t, bar: i })
          at = { x: lane, y: b.y }
        })
        wire(at.x, at.y, at.x, top - 2)
        const out = route.length
        wire(inL, top - 2, inR, top - 2)
        span = t + 400
        route.out = out
        labels.push({ x: rx + w / 2, y: top - 4, text: `${info.gpu || 'GPU'} · layers 0–${gpuN - 1}` })
        labels.push({ x: lx + w / 2, y: top - 4, text: `cpu · ${info.threads || '?'} threads · layers ${gpuN}–${gpuN + cpuN - 1}` })
        labels.push({ x: (inL + inR) / 2, y: bottom + 3, text: 'PCIe' })
        boxes.push([lx - 1, top - 3, lx + w + 2, bottom + 2], [rx - 3, top - 3, rx + w + 1, bottom + 2])
      },
      shown: () => 1,
      hush(col, row) {
        for (const [a, b, c, d] of boxes) if (col >= a && col <= c && row >= b && row <= d) return 0.3
        return 1
      },
      draw(ctx, g, pal, now, reduced) {
        if (!bars.length) return
        const t = reduced ? 0 : (now - born) % span
        const lap = Math.floor((now - born) / span)
        if (lap !== lastLap) (lastLap = lap), tokens++
        const cell = (x, y, ink) => {
          const px = Math.round(g.ox + x * g.cw)
          const py = Math.round(g.oy + y * g.ch)
          ctx.fillStyle = ink
          ctx.fillRect(px, py, Math.round(g.ox + (x + 1) * g.cw) - px, Math.round(g.oy + (y + 1) * g.ch) - py)
        }
        // Where the token is, and which bars it has just run.
        let head = 0
        while (head < route.length - 1 && route[head + 1].t <= t) head++
        for (let i = 0; i <= head; i++) if (route[i].bar >= 0 && bars[route[i].bar].hot < now - t + route[i].t - 1) bars[route[i].bar].hot = now - t + route[i].t
        // The bars: resident weights as a dithered rule; a bar the token is
        // running lights across, then cools.
        bars.forEach((b) => {
          const since = now - b.hot
          const run = b.dev === 'gpu' ? GPU_MS : CPU_MS
          for (let x = 0; x < b.w; x++) {
            const col = b.x + x
            const j = jitter[(b.y & 63) * 64 + (col & 63)]
            const sweep = since >= 0 && since < run ? x / b.w < since / run : false
            const warm = since >= run && since < run + 700 ? 1 - (since - run) / 700 : 0
            if (sweep) cell(col, b.y, pal.crest)
            else if (warm > 0.5) cell(col, b.y, pal.lit)
            else if (warm > 0 || j > 0.35) cell(col, b.y, warm > 0 ? pal.mid : j > 0.8 ? pal.mid : pal.dim)
          }
        })
        // The wire it travels, faint, and the token with a short tail.
        for (let i = Math.max(0, head - 6); i <= head; i++) {
          const r = route[i]
          cell(r.x, r.y, i === head ? pal.crest : i > head - 3 ? pal.hover : pal.mid)
        }
        ctx.fillStyle = pal.lit
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.font = `${Math.round(11 * g.dpr)}px "JetBrains Mono", ui-monospace, monospace`
        for (const l of labels) ctx.fillText(l.text, g.ox + l.x * g.cw, g.oy + l.y * g.ch)
        // Each lap is a token out of the model.
        const o = route[route.out - 1]
        ctx.fillStyle = head >= route.out - 1 && head < route.out + 12 ? pal.crest : pal.mid
        ctx.fillText(`token ${tokens}`, g.ox + (o.x + 7) * g.cw, g.oy + (o.y - 2) * g.ch)
      },
    }
  }

  /* ------------------------------------------------------------ hero: defrag */

  // The dense ring of the field as a memory map: every pixel is a page, free
  // or holding one of three models, told apart by shade. A head sweeps round
  // the ring swapping misplaced pages until each model sits in one contiguous
  // arc; then demand shifts, pages come and go, the map fragments, and the
  // compaction starts again.
  function makeDefrag() {
    const MODELS = ['llama-3.2-1b', 'qwen3-moe', 'gemma-2b']
    const SHARE = [0.22, 0.24, 0.2] // the rest is free
    let cells = [] // [{ x, y }] in sweep order
    let map = new Uint8Array(0) // 0 free, 1-3 a model
    let want = new Uint8Array(0)
    let flash = new Float64Array(0)
    let index = new Map()
    let head = 0
    let phase = 'defrag'
    let phaseAt = performance.now()
    let key = ''
    let legendAt = null

    const plan = () => {
      const count = [0, 0, 0, 0]
      for (const v of map) count[v]++
      let i = 0
      for (const m of [1, 2, 3, 0]) for (let k = 0; k < count[m]; k++) want[i++] = m
      head = 0
    }
    const contiguous = () => {
      let ok = 0
      for (let i = 0; i < map.length; i++) if (map[i] === want[i]) ok++
      return map.length ? ok / map.length : 1
    }

    return {
      layout(g) {
        const k = [g.visC0, g.visC1, g.barRow, g.visR1].join(',')
        if (k === key) return
        key = k
        const cx = WORD_W / 2
        const cy = WORD_H / 2 + 6
        const list = []
        for (let y = g.barRow + 1; y <= g.visR1; y++)
          for (let x = g.visC0; x <= g.visC1; x++) {
            if (x >= -2 && x <= WORD_W + 1 && y >= -2 && y <= WORD_H + 1) continue
            if (g.shade(x, y) < 0.3) continue
            // Sweep order: round the ring by angle, then outward.
            list.push({ x, y, a: Math.atan2(y - cy, (x - cx) * 0.6), d: Math.hypot(x - cx, y - cy) })
          }
        list.sort((p, q) => Math.round(p.a * 60) - Math.round(q.a * 60) || p.d - q.d)
        cells = list
        index = new Map(cells.map((c, i) => [c.y * 100000 + c.x, i]))
        map = new Uint8Array(cells.length)
        want = new Uint8Array(cells.length)
        flash = new Float64Array(cells.length)
        const r = lcg(0xdf)
        for (let i = 0; i < map.length; i++) {
          const u = r()
          map[i] = u < SHARE[0] ? 1 : u < SHARE[0] + SHARE[1] ? 2 : u < SHARE[0] + SHARE[1] + SHARE[2] ? 3 : 0
        }
        plan()
        phase = 'defrag'
        phaseAt = performance.now()
        legendAt = { x: g.visC0 + 2, y: g.visR1 - 1 }
      },
      hush: (col, row) => (index.has(row * 100000 + col) ? 0 : 1),
      shown: () => 1,

      draw(ctx, g, pal, now, reduced) {
        if (!cells.length) return
        if (!reduced) {
          if (phase === 'defrag') {
            // A few hundred page moves a second: find the next page out of
            // place, fetch the right one from further round, swap.
            for (let n = 0; n < 24 && head < map.length; n++) {
              while (head < map.length && map[head] === want[head]) head++
              if (head >= map.length) break
              let j = head + 1
              while (j < map.length && map[j] !== want[head]) j++
              if (j >= map.length) {
                head++
                continue
              }
              ;[map[head], map[j]] = [map[j], map[head]]
              flash[head] = flash[j] = now
              head++
            }
            if (head >= map.length && now - phaseAt > 1500) (phase = 'hold'), (phaseAt = now)
          } else if (phase === 'hold' && now - phaseAt > 2600) (phase = 'churn'), (phaseAt = now)
          else if (phase === 'churn') {
            // Demand shifts: pages evicted and loaded anywhere.
            for (let n = 0; n < 14; n++) {
              const i = Math.floor(Math.random() * map.length)
              map[i] = Math.random() < 0.4 ? 0 : 1 + Math.floor(Math.random() * 3)
              flash[i] = now
            }
            if (now - phaseAt > 2600) (plan(), (phase = 'defrag'), (phaseAt = now))
          }
        }
        const ink = [null, pal.lit, pal.mid, pal.dim]
        const seam = Math.max(1, Math.round(g.dpr))
        for (let i = 0; i < cells.length; i++) {
          const v = map[i]
          const hot = now - flash[i] < 160
          if (!v && !hot) continue
          const { x, y } = cells[i]
          const px = Math.round(g.ox + x * g.cw)
          const py = Math.round(g.oy + y * g.ch)
          ctx.fillStyle = hot ? pal.crest : ink[v]
          // Pages keep a seam, like the blocks of a disk map.
          ctx.fillRect(px, py, Math.round(g.ox + (x + 1) * g.cw) - px - seam, Math.round(g.oy + (y + 1) * g.ch) - py - seam)
        }
        // The head, while it works.
        if (phase === 'defrag' && head < cells.length) {
          const { x, y } = cells[head]
          ctx.strokeStyle = pal.crest
          ctx.lineWidth = Math.max(1, g.dpr)
          ctx.strokeRect(Math.round(g.ox + (x - 1) * g.cw) + 0.5, Math.round(g.oy + (y - 1) * g.ch) + 0.5, Math.round(g.cw * 3), Math.round(g.ch * 3))
        }
        // Legend and progress, in the page's mono.
        const fs = Math.round(11 * g.dpr)
        ctx.font = `${fs}px "JetBrains Mono", ui-monospace, monospace`
        ctx.textBaseline = 'middle'
        ctx.textAlign = 'left'
        let lx = g.ox + legendAt.x * g.cw
        const ly = g.oy + (legendAt.y + 0.5) * g.ch
        const status = phase === 'churn' ? 'demand shifting' : phase === 'hold' ? 'compacted' : `compacting · ${Math.round(contiguous() * 100)}% in place`
        // A backing so the legend reads over the map.
        let width = fs * 2
        for (const name of MODELS) width += ctx.measureText(name).width + fs * 2.2
        width += ctx.measureText('compacting · 100% in place').width
        ctx.fillStyle = pal.bg
        ctx.fillRect(lx - fs * 0.6, ly - fs, width, fs * 2)
        for (const [m, name] of MODELS.entries()) {
          ctx.fillStyle = ink[m + 1]
          ctx.fillRect(lx, ly - fs * 0.35, fs * 0.7, fs * 0.7)
          ctx.fillStyle = pal.lit
          ctx.fillText(name, lx + fs, ly)
          lx += ctx.measureText(name).width + fs * 2.2
        }
        ctx.fillStyle = phase === 'defrag' ? pal.crest : pal.mid
        ctx.fillText(status, lx + fs, ly)
      },
    }
  }

  /* ------------------------------------------------------------ hero: shared */

  const painter = (ctx, g) => (x, y, ink, seam = 0) => {
    const px = Math.round(g.ox + x * g.cw)
    const py = Math.round(g.oy + y * g.ch)
    ctx.fillStyle = ink
    ctx.fillRect(px, py, Math.round(g.ox + (x + 1) * g.cw) - px - seam, Math.round(g.oy + (y + 1) * g.ch) - py - seam)
  }
  const mono = (ctx, g, px = 11, align = 'left') => {
    ctx.font = `${Math.round(px * g.dpr)}px "JetBrains Mono", ui-monospace, monospace`
    ctx.textAlign = align
    ctx.textBaseline = 'middle'
  }
  const say = (ctx, g, x, y, s, ink) => {
    ctx.fillStyle = ink
    ctx.fillText(s, g.ox + x * g.cw, g.oy + (y + 0.5) * g.ch)
  }
  const within = (boxes, col, row) => boxes.some(([a, b, c, d]) => col >= a && col <= c && row >= b && row <= d)
  const clipTo = (ctx, g, p) => {
    ctx.save()
    ctx.beginPath()
    ctx.rect(g.ox + p.x * g.cw, g.oy + p.y * g.ch, p.w * g.cw, p.h * g.ch)
    ctx.clip()
  }
  // Cells per character of the mono labels, so text can sit on the lattice.
  const charCells = (g, px) => (px * 0.6 * g.dpr) / g.cw

  // The two columns beside the wordmark, or on a narrow screen two halves of
  // the space below the copy. Null when neither has room.
  function sides(g, minW = 16, minH = 12) {
    const room = -5 - g.visC0
    const top = g.barRow + 3
    if (room >= minW && g.visR1 - 1 - top >= minH)
      return [
        { x: g.visC0 + 2, y: top, w: room - 2, h: g.visR1 - 1 - top },
        { x: WORD_W + 4, y: top, w: g.visC1 - WORD_W - 5, h: g.visR1 - 1 - top },
      ]
    const y = g.below + 3
    const h = g.visR1 - 1 - y
    if (h < minH) return null
    const w = Math.floor((g.visC1 - g.visC0 - 4) / 2)
    return [{ x: g.visC0 + 1, y, w, h }, { x: g.visC0 + w + 3, y, w, h }]
  }
  const layoutKey = (g) => [g.visC0, g.visC1, g.barRow, g.visR1, g.below, g.above].join(',')

  /* ------------------------------------------------------------ hero: sand */

  // Memory as sand. Weights pour from the .jlm through pixel pipes into two
  // vessels, RAM and VRAM, and settle in strata, a shade to a model. When a
  // vessel is nearly full a drain opens in its floor and the oldest pages run
  // out of the bottom while the pour goes on.
  function makeSand(info) {
    const STEP = 34
    let vessels = []
    let disk = null
    let boxes = []
    let key = ''
    let last = 0

    function vessel(p, outer, lane, name, rate, offset) {
      const w = Math.max(6, Math.min(p.w - 8, 24))
      const h = Math.max(6, Math.min(p.h - 12, 26))
      const left = outer === p.x
      const x = left ? p.x + p.w - 1 - w : p.x + 1
      const y = lane - 2 - h
      const mouth = x + Math.floor(w / 2)
      const pipe = []
      const run = (x0, y0, x1, y1) => {
        let cx = x0
        let cy = y0
        pipe.push([cx, cy])
        while (cx !== x1 || cy !== y1) {
          if (cy !== y1) cy += Math.sign(y1 - cy)
          else cx += Math.sign(x1 - cx)
          pipe.push([cx, cy])
        }
      }
      if (disk) {
        const sx = left ? disk.x + 2 : disk.x + disk.w - 3
        run(sx, disk.y + disk.h, sx, lane)
        const at = pipe.pop()
        run(at[0], at[1], outer, lane)
        const at2 = pipe.pop()
        run(at2[0], at2[1], outer, y - 3)
        const at3 = pipe.pop()
        run(at3[0], at3[1], mouth, y - 3)
      } else run(mouth, p.y, mouth, y - 1)
      boxes.push([x - 2, y - 5, x + w + 1, lane])
      const v = { x, y, w, h, mouth, pipe, name, rate, offset, grid: new Uint8Array(w * h), flow: [], falling: [], next: 0, n: 0, count: 0, drain: false }
      if (reduced)
        for (let r = Math.floor(h / 2); r < h; r++) for (let c = 0; c < w; c++) (v.grid[r * w + c] = 1 + (Math.floor((h - r) / 3) + offset) % 3), v.count++
      return v
    }

    function settle(v) {
      const { w, h, grid } = v
      for (let y = h - 2; y >= 0; y--) {
        const flip = Math.random() < 0.5
        for (let i = 0; i < w; i++) {
          const x = flip ? i : w - 1 - i
          const c = grid[y * w + x]
          if (!c) continue
          const below = (y + 1) * w
          if (!grid[below + x]) {
            grid[below + x] = c
            grid[y * w + x] = 0
            continue
          }
          const s = Math.random() < 0.5 ? -1 : 1
          for (const dx of [s, -s]) {
            const nx = x + dx
            if (nx < 0 || nx >= w || grid[below + nx] || grid[y * w + nx]) continue
            grid[below + nx] = c
            grid[y * w + x] = 0
            break
          }
        }
      }
      // Eviction: past 80% full a drain opens under the middle of the floor.
      if (v.count > w * h * 0.8) v.drain = true
      if (v.count < w * h * 0.4) v.drain = false
      if (v.drain) {
        const hx = Math.floor(w / 2) - 1
        for (const x of [hx, hx + 1]) {
          const i = (h - 1) * w + x
          if (!grid[i]) continue
          v.falling.push({ x: v.x + x, y: v.y + h, c: grid[i] })
          grid[i] = 0
          v.count--
        }
      }
      for (const f of v.falling) f.y++
      v.falling = v.falling.filter((f) => f.y < v.y + h + 3)
    }

    return {
      layout(g) {
        const k = layoutKey(g)
        if (k === key) return
        key = k
        vessels = []
        boxes = []
        const ps = sides(g, 18, 14)
        if (!ps) return
        const wide = ps[0].y < g.below
        const lane = g.visR1 - 1
        disk = wide && g.below + 8 < lane ? { x: Math.round(WORD_W / 2) - 7, y: g.below + 3, w: 14, h: 4 } : null
        if (disk) boxes.push([disk.x - 1, disk.y - 1, disk.x + disk.w, disk.y + disk.h])
        vessels.push(vessel(ps[0], ps[0].x, wide ? lane : ps[0].y + ps[0].h, `ram · ${info.threads || '?'} threads`, 70, 0))
        vessels.push(vessel(ps[1], ps[1].x + ps[1].w - 1, wide ? lane : ps[1].y + ps[1].h, `vram · ${info.gpu || 'gpu'}`, 42, 1))
        last = performance.now()
      },
      hush: (col, row) => (within(boxes, col, row) ? 0.12 : 1),
      shown: () => 1,

      draw(ctx, g, pal, now) {
        if (!vessels.length) return
        const cell = painter(ctx, g)
        const ink = [null, pal.lit, pal.mid, pal.dim]
        if (!reduced) {
          for (const v of vessels) {
            // The pour: a grain enters the pipe every `rate` ms, and a model
            // is a run of grains of one shade.
            if (!v.next) v.next = now
            while (v.next <= now) {
              if (v.flow.length < 24) v.flow.push({ born: v.next, c: 1 + ((Math.floor(v.n++ / 40) + v.offset) % 3) })
              v.next += v.rate
            }
            v.flow = v.flow.filter((f) => {
              if ((now - f.born) / 9 < v.pipe.length) return true
              const x = v.mouth - v.x + Math.round((Math.random() - 0.5) * 2)
              if (x < 0 || x >= v.w || v.grid[x]) return true
              v.grid[x] = f.c
              v.count++
              return false
            })
          }
          let steps = 0
          while (now - last > STEP && steps++ < 3) {
            last += STEP
            for (const v of vessels) settle(v)
          }
          if (now - last > STEP) last = now
        }
        if (disk) {
          for (let x = disk.x; x < disk.x + disk.w; x++) cell(x, disk.y, pal.mid), cell(x, disk.y + disk.h - 1, pal.mid)
          for (let y = disk.y; y < disk.y + disk.h; y++) cell(disk.x, y, pal.mid), cell(disk.x + disk.w - 1, y, pal.mid)
        }
        for (const v of vessels) {
          for (const [x, y] of v.pipe) cell(x, y, pal.dim)
          for (let y = v.y; y <= v.y + v.h; y++) cell(v.x - 1, y, pal.mid), cell(v.x + v.w, y, pal.mid)
          const hx = v.x + Math.floor(v.w / 2) - 1
          for (let x = v.x; x < v.x + v.w; x++) if (!v.drain || (x !== hx && x !== hx + 1)) cell(x, v.y + v.h, pal.mid)
          for (let i = 0; i < v.grid.length; i++) if (v.grid[i]) cell(v.x + (i % v.w), v.y + Math.floor(i / v.w), ink[v.grid[i]])
          for (const f of v.flow) {
            const at = Math.floor((now - f.born) / 9)
            const q = v.pipe[Math.min(at, v.pipe.length - 1)]
            cell(q[0], q[1], pal.crest)
          }
          for (const f of v.falling) cell(f.x, f.y, ink[f.c])
        }
        mono(ctx, g, 11, 'center')
        if (disk) say(ctx, g, disk.x + disk.w / 2, disk.y + 1.5, '.jlm', pal.lit)
        for (const v of vessels) {
          const pct = Math.round((v.count / v.grid.length) * 100)
          say(ctx, g, v.x + v.w / 2, v.y + v.h + 1.2, `${v.name} · ${pct}%${v.drain ? ' · evicting' : ''}`, v.drain ? pal.crest : pal.lit)
        }
      },
    }
  }

  /* ------------------------------------------------------------ hero: circuit */

  // The field settles into a board: traces etched between a CPU, a GPU and
  // the .jlm, with vias and stubs where the noise used to be. Tokens pulse
  // along the buses, and every few seconds one bus is torn up and re-routed,
  // the way placement moves a layer.
  function makePcb(info) {
    let pads = []
    let deco = [] // [{ x, y, via }]
    let buses = []
    let boxes = []
    let key = ''
    let born = 0
    let nextRoute = 0
    let note = { text: '', at: -1e9 }

    // 45-degree chamfers at every corner, then walk the polyline cell by cell.
    function trace(points) {
      const pts = [points[0]]
      for (let i = 1; i < points.length - 1; i++) {
        const [a, b, c] = [points[i - 1], points[i], points[i + 1]]
        const d1 = [Math.sign(b[0] - a[0]), Math.sign(b[1] - a[1])]
        const d2 = [Math.sign(c[0] - b[0]), Math.sign(c[1] - b[1])]
        const room = Math.min(Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1]), Math.abs(c[0] - b[0]) + Math.abs(c[1] - b[1]))
        const k = Math.min(2, Math.floor(room / 2))
        pts.push([b[0] - d1[0] * k, b[1] - d1[1] * k], [b[0] + d2[0] * k, b[1] + d2[1] * k])
      }
      pts.push(points[points.length - 1])
      const out = []
      let [x, y] = pts[0]
      out.push([x, y])
      for (const [tx, ty] of pts.slice(1)) {
        while (x !== tx || y !== ty) {
          x += Math.sign(tx - x)
          y += Math.sign(ty - y)
          out.push([x, y])
        }
      }
      return out
    }

    function reroute(bus, now) {
      bus.old = bus.path
      bus.oldAt = now
      bus.path = [0, 2].map((k) => trace(bus.route(k)))
      bus.at = now
    }

    return {
      layout(g) {
        const k = layoutKey(g)
        if (k === key) return
        key = k
        pads = []
        deco = []
        buses = []
        boxes = []
        const ps = sides(g, 18, 14)
        if (!ps) return
        const wide = ps[0].y < g.below
        const pad = (p, label) => {
          const w = Math.min(12, p.w - 6)
          const h = 8
          const x = p.x + Math.floor((p.w - w) / 2)
          const y = p.y + Math.floor((p.h - h) / 2)
          const o = { x, y, w, h, label, cx: x + Math.floor(w / 2), cy: y + Math.floor(h / 2) }
          pads.push(o)
          boxes.push([x - 2, y - 2, x + w + 1, y + h + 3])
          return o
        }
        const cpu = pad(ps[0], `cpu · ${info.threads || '?'} threads`)
        const gpu = pad(ps[1], info.gpu || 'gpu')
        const disk = wide && g.below + 9 < g.visR1 ? pad({ x: Math.round(WORD_W / 2) - 8, y: g.below + 2, w: 16, h: 8 }, '.jlm') : null
        const rnd = (a, b) => a + Math.floor(Math.random() * Math.max(1, b - a + 1))
        const lowest = g.visR1 - 2
        if (disk) {
          // .jlm to the GPU: out of the disk's side or bottom, round to the card.
          buses.push({
            name: '.jlm → gpu',
            route: (k) => {
              const lx = rnd(WORD_W + 3, gpu.x - 5) + k
              if (Math.random() < 0.5) return [[disk.x + disk.w, disk.cy - 1 + k], [lx, disk.cy - 1 + k], [lx, gpu.cy - 1 + k], [gpu.x - 1, gpu.cy - 1 + k]]
              const ly = rnd(disk.y + disk.h + 1, lowest - 2) + k
              return [[disk.x + disk.w - 3 - k, disk.y + disk.h], [disk.x + disk.w - 3 - k, ly], [lx, ly], [lx, gpu.cy - 1 + k], [gpu.x - 1, gpu.cy - 1 + k]]
            },
          })
          buses.push({
            name: '.jlm → cpu',
            route: (k) => {
              const lx = rnd(cpu.x + cpu.w + 4, -5) - k
              if (Math.random() < 0.5) return [[disk.x - 1, disk.cy - 1 + k], [lx, disk.cy - 1 + k], [lx, cpu.cy - 1 + k], [cpu.x + cpu.w, cpu.cy - 1 + k]]
              const ly = rnd(disk.y + disk.h + 1, lowest - 2) + k
              return [[disk.x + 2 + k, disk.y + disk.h], [disk.x + 2 + k, ly], [lx, ly], [lx, cpu.cy - 1 + k], [cpu.x + cpu.w, cpu.cy - 1 + k]]
            },
          })
        }
        // The seam between the two devices: over the top if there is room
        // above the callout, else along the bottom.
        const topRoom = g.above - g.barRow >= 7
        buses.push({
          name: 'cpu ↔ gpu',
          route: (k) => {
            if (topRoom && wide) {
              const ly = rnd(g.barRow + 1, g.above - 6) + k
              return [[cpu.cx - 2 + k, cpu.y - 1], [cpu.cx - 2 + k, ly], [gpu.cx - 2 + k, ly], [gpu.cx - 2 + k, gpu.y - 1]]
            }
            const ly = Math.min(lowest, Math.max(cpu.y + cpu.h + 3, rnd(lowest - 3, lowest))) - k
            return [[cpu.cx - 2 + k, cpu.y + cpu.h], [cpu.cx - 2 + k, ly], [gpu.cx - 2 + k, ly], [gpu.cx - 2 + k, gpu.y + gpu.h]]
          },
        })
        born = performance.now()
        for (const b of buses) (b.path = [0, 2].map((k) => trace(b.route(k)))), (b.at = born), (b.old = null)
        // The rest of the ring becomes copper: short random walks at 45-degree
        // turns, each ending in a via.
        const taken = new Set()
        const id = (x, y) => y * 4096 + x
        for (const b of buses) for (const t of b.path) for (const [x, y] of t) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) taken.add(id(x + dx, y + dy))
        const r = lcg(0xb0a4d)
        const dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        for (let n = 0; n < 900; n++) {
          let x = g.visC0 + Math.floor(r() * (g.visC1 - g.visC0))
          let y = g.barRow + 1 + Math.floor(r() * (g.visR1 - g.barRow))
          if (g.shade(x, y) < 0.3 || taken.has(id(x, y)) || within(boxes, x, y)) continue
          let d = Math.floor(r() * 4) * 2
          const len = 4 + Math.floor(r() * 16)
          const walk = []
          for (let s = 0; s < len; s++) {
            walk.push([x, y])
            if (r() < 0.14) d = (d + (r() < 0.5 ? 1 : 7)) % 8
            const nx = x + dirs[d][0]
            const ny = y + dirs[d][1]
            if (g.shade(nx, ny) < 0.2 || taken.has(id(nx, ny)) || within(boxes, nx, ny)) break
            x = nx
            y = ny
          }
          if (walk.length < 3) continue
          for (const [wx, wy] of walk) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) taken.add(id(wx + dx, wy + dy))
          walk.forEach(([wx, wy], i) => deco.push({ x: wx, y: wy, via: i === walk.length - 1 || i === 0, order: deco.length }))
        }
        nextRoute = born + 3200
      },
      hush: (col, row) => (within(boxes, col, row) ? 0.08 : 0.4),
      shown: () => 1,

      draw(ctx, g, pal, now) {
        if (!buses.length) return
        const cell = painter(ctx, g)
        if (!reduced && now > nextRoute) {
          const b = buses[Math.floor(Math.random() * buses.length)]
          reroute(b, now)
          note = { text: `re-routing ${b.name}`, at: now }
          nextRoute = now + 3600 + Math.random() * 1800
        }
        const etch = (at, i) => reduced || now - at > i * 5
        deco.forEach((d) => {
          if (!etch(born, d.order * 0.35)) return
          cell(d.x, d.y, d.via ? pal.mid : pal.dim)
        })
        for (const p of pads) {
          for (let y = p.y; y < p.y + p.h; y++)
            for (let x = p.x; x < p.x + p.w; x++) {
              const edge = x === p.x || y === p.y || x === p.x + p.w - 1 || y === p.y + p.h - 1
              if (edge) cell(x, y, pal.mid)
              else if ((x + y) % 2 === 0 && x > p.x + 1 && y > p.y + 1 && x < p.x + p.w - 2 && y < p.y + p.h - 2) cell(x, y, pal.dim)
            }
          for (let x = p.x + 1; x < p.x + p.w - 1; x += 2) cell(x, p.y - 1, pal.dim), cell(x, p.y + p.h, pal.dim)
          for (let y = p.y + 1; y < p.y + p.h - 1; y += 2) cell(p.x - 1, y, pal.dim), cell(p.x + p.w, y, pal.dim)
        }
        for (const b of buses) {
          if (b.old)
            for (const t of b.old)
              t.forEach(([x, y], i) => {
                if (now - b.oldAt < i * 4) cell(x, y, now - b.oldAt > i * 4 - 60 ? pal.crest : pal.lit)
              })
          for (const t of b.path) {
            t.forEach(([x, y], i) => {
              if (!etch(b.at, i)) return
              cell(x, y, now - b.at - i * 5 < 90 ? pal.crest : pal.lit)
            })
            for (const end of [t[0], t[t.length - 1]]) cell(end[0], end[1], pal.hover)
          }
          // Tokens on the bus, once it is fully etched.
          const len = b.path[0].length
          if (!reduced && now - b.at > len * 5)
            for (const [ti, t] of b.path.entries()) {
              const head = Math.floor(((now - b.at) / 16 + ti * len * 0.5) % (len + 30))
              for (let tail = 0; tail < 4; tail++) {
                const q = t[head - tail]
                if (q) cell(q[0], q[1], tail ? pal.hover : pal.crest)
              }
            }
        }
        mono(ctx, g, 11, 'center')
        for (const p of pads) say(ctx, g, p.cx, p.y + p.h + 1.6, p.label, pal.lit)
        if (now - note.at < 2200) {
          ctx.globalAlpha = 1 - Math.max(0, (now - note.at - 1600) / 600)
          mono(ctx, g, 11, 'left')
          say(ctx, g, g.visC0 + 2, g.visR1 - 1, note.text, pal.crest)
          ctx.globalAlpha = 1
        }
      },
    }
  }

  /* ------------------------------------------------------------ hero: it writes the compute */

  const LISTINGS = {
    amd64: [
      ['q4_K matvec', ['vpbroadcastd ymm15, [rip+nibble]', '.loop:', 'vmovdqu     ymm4, [rsi+rax]', 'vpand       ymm5, ymm4, ymm15', 'vpsrlw      ymm4, ymm4, 4', 'vpand       ymm4, ymm4, ymm15', 'vpdpbusd    ymm0, ymm5, [rdx]', 'vpdpbusd    ymm1, ymm4, [rdx+32]', 'vcvtdq2ps   ymm0, ymm0', 'vfmadd231ps ymm8, ymm0, ymm12', 'add         rax, 144', 'dec         rcx', 'jnz         .loop']],
      ['rmsnorm', ['vxorps      ymm0, ymm0, ymm0', '.loop:', 'vmovups     ymm1, [rsi+rax*4]', 'vfmadd231ps ymm0, ymm1, ymm1', 'add         rax, 8', 'cmp         rax, rcx', 'jb          .loop', 'vhaddps     ymm0, ymm0, ymm0', 'vrsqrtps    xmm0, xmm0', 'vbroadcastss ymm0, xmm0', 'vmulps      ymm1, ymm0, [rdx]']],
      ['attention scores', ['vxorps      ymm7, ymm7, ymm7', '.loop:', 'vcvtph2ps   ymm2, [r8+rax*2]', 'vfmadd231ps ymm0, ymm2, [rsi+rax*4]', 'vmaxps      ymm7, ymm7, ymm0', 'add         rax, 8', 'cmp         rax, r9', 'jb          .loop', 'vmulps      ymm0, ymm0, ymm14']],
    ],
    arm64: [
      ['q4_K matvec', ['movi    v15.16b, #0x0f', '.loop:', 'ldr     q4, [x1], #16', 'and     v5.16b, v4.16b, v15.16b', 'ushr    v4.16b, v4.16b, #4', 'sdot    v0.4s, v5.16b, v8.16b', 'sdot    v1.4s, v4.16b, v9.16b', 'scvtf   v0.4s, v0.4s', 'fmla    v16.4s, v0.4s, v12.4s', 'subs    x3, x3, #1', 'b.ne    .loop']],
      ['rmsnorm', ['movi    v0.4s, #0', '.loop:', 'ld1     {v1.4s}, [x1], #16', 'fmla    v0.4s, v1.4s, v1.4s', 'subs    x2, x2, #4', 'b.ne    .loop', 'faddp   v0.4s, v0.4s, v0.4s', 'frsqrte s0, s0', 'fmul    v1.4s, v1.4s, v0.s[0]']],
    ],
    ptx: [
      ['q4_K matvec', ['.loop:', 'ld.global.nc.v4.u32 {%r1,%r2,%r3,%r4}, [%rd4];', 'and.b32   %r5, %r1, 0x0f0f0f0f;', 'shr.u32   %r6, %r1, 4;', 'dp4a.s32.s32 %r9, %r5, %r7, %r9;', 'dp4a.s32.s32 %r9, %r6, %r8, %r9;', 'cvt.rn.f32.s32 %f2, %r9;', 'fma.rn.f32 %f1, %f2, %f3, %f1;', 'add.s64   %rd4, %rd4, 16;', '@%p1 bra  .loop;', 'shfl.sync.bfly.b32 %f5, %f1, 16, 31, -1;', 'st.global.f32 [%rd8], %f1;']],
      ['prefill gemm', ['.loop:', 'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1,%r2,%r3,%r4}, [%r20];', 'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32', '    {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};', 'add.s32   %r20, %r20, 32;', '@%p2 bra  .loop;', 'st.global.v4.f32 [%rd9], {%f1,%f2,%f3,%f4};']],
    ],
    metal: [
      ['attention', ['simdgroup_half8x8  q, k;', 'simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0);', 'for (uint i = 0; i < n; i += 8) {', '  simdgroup_load(q, Q + i, 64);', '  simdgroup_load(k, K + i, 64, 0, true);', '  simdgroup_multiply_accumulate(acc, q, k, acc);', '}', 'simdgroup_store(acc, S, 64);']],
      ['q4_K matvec', ['uchar4 b = W[row * nb + j];', 'int4 lo = int4(b & 0xF), hi = int4(b >> 4);', 'acc += dot(float4(lo), a0) * d;', 'acc += dot(float4(hi), a1) * d;', 'acc = simd_sum(acc);']],
    ],
  }
  const hash = (s) => {
    let h = 2166136261
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
    return h >>> 0
  }

  // The JIT at work: each side prints the code it is emitting for a device,
  // a line at a time, with the instruction's bytes as a row of pixels. When a
  // kernel is done an execution pointer runs its loop, then the next begins.
  function makeAsm(info) {
    let panels = []
    let key = ''
    let boxes = []
    const host = info.arch === 'ARM64' ? 'arm64' : 'amd64'
    const dev = /^Apple/.test(info.gpu) ? 'metal' : 'ptx'

    function panel(p, isa, title, pace) {
      const s = { p, isa, title, pace, lines: [], k: 0, queue: [], at: performance.now(), phase: 'emit', run: 0 }
      begin(s, s.at)
      return s
    }
    function begin(s, now) {
      const [name, body] = LISTINGS[s.isa][s.k++ % LISTINGS[s.isa].length]
      const lines = body.map((text) => {
        const h = hash(text)
        const n = text.endsWith(':') || text.startsWith('  ') || text === '}' ? 0 : 2 + (h % 6)
        return { text, bytes: Array.from({ length: n }, (_, i) => (h >>> (i * 4)) & 255), label: text.endsWith(':') }
      })
      const size = lines.reduce((a, l) => a + l.bytes.length, 0)
      if (s.lines.length) s.queue.push({ text: '', bytes: [] })
      s.queue.push({ text: `; ${name} · ${size} B`, bytes: [], note: true }, ...lines)
      s.phase = 'emit'
      s.at = now
    }

    return {
      layout(g) {
        const k = layoutKey(g)
        if (k === key) return
        key = k
        const ps = sides(g, 20, 12)
        boxes = ps ? ps.map((p) => [p.x - 1, p.y - 1, p.x + p.w, p.y + p.h]) : []
        panels = ps ? [panel(ps[0], host, `cpu · ${host}`, 95), panel(ps[1], dev, `gpu · ${dev}`, 80)] : []
        if (reduced) for (const s of panels) s.lines.push(...s.queue.splice(0))
      },
      hush: (col, row) => (within(boxes, col, row) ? 0.05 : 1),
      shown: () => 1,

      draw(ctx, g, pal, now) {
        if (!panels.length) return
        const cell = painter(ctx, g)
        const seam = Math.max(1, Math.round(g.dpr))
        const px = Math.min(11, (g.ch / g.dpr) * 0.82)
        for (const s of panels) {
          const { p } = s
          if (!reduced) {
            if (s.phase === 'emit') {
              while (s.queue.length && now - s.at > s.pace) (s.lines.push(s.queue.shift())), (s.at += s.pace)
              if (!s.queue.length) (s.phase = 'run'), (s.at = now)
            } else if (s.phase === 'run' && now - s.at > 2600) begin(s, now)
          }
          const vis = p.h - 2
          const shown = s.lines.slice(-vis)
          const base = s.lines.length - shown.length
          // The loop the pointer runs: from the kernel's label to its end.
          let from = s.lines.length
          while (from > 0 && !s.lines[from - 1].note) from--
          const lab = s.lines.findIndex((l, i) => i >= from && l.label)
          const loop0 = lab >= 0 ? lab + 1 : from
          const loopN = s.lines.length - loop0
          const pc = s.phase === 'run' && !reduced ? loop0 + (Math.floor((now - s.at) / 55) % Math.max(1, loopN)) : -1
          const typing = s.phase === 'emit' && !reduced ? s.lines.length - 1 : -1
          clipTo(ctx, g, p)
          mono(ctx, g, px, 'left')
          say(ctx, g, p.x, p.y, `${s.title} · ${s.phase === 'emit' ? 'emitting' : 'running'}`, s.phase === 'emit' ? pal.crest : pal.lit)
          shown.forEach((l, i) => {
            const idx = base + i
            const y = p.y + 2 + i
            const fresh = idx === typing && now - s.at < s.pace
            l.bytes.forEach((b, j) => {
              if (j >= 8) return
              cell(p.x + j, y, fresh || idx === pc ? pal.crest : b > 170 ? pal.lit : b > 85 ? pal.mid : pal.dim, seam)
            })
            const chars = fresh ? Math.ceil((l.text.length * (now - s.at)) / s.pace) : l.text.length
            const ink = idx === pc ? pal.crest : l.note ? pal.dim : idx === typing ? pal.lit : l.label ? pal.hover : pal.mid
            say(ctx, g, p.x + 9, y, l.text.slice(0, chars), ink)
            if (idx === pc) say(ctx, g, p.x + 8.1, y, '›', pal.crest)
          })
          ctx.restore()
        }
      },
    }
  }

  /* ------------------------------------------------------------ hero: top */

  // htop for models: the processes an inference OS is running on the left,
  // its cores and devices as pixel meters on the right. A meter's shade says
  // which model holds that core; residency bars show each model's blocks on
  // the card, in RAM, or paged out to the .jlm.
  function makeTop(info) {
    const gname = 'gpu0'
    const PROCS = [
      { pid: 4120, name: 'llama-3.2-1b', dev: gname, tps: 212, res: [1, 0, 0] },
      { pid: 4133, name: 'qwen3-30b-a3b', dev: `${gname}+cpu`, tps: 41, res: [0.35, 0.55, 0.1] },
      { pid: 4151, name: 'gemma-3-27b', dev: gname, tps: 36, res: [0.8, 0.2, 0] },
      { pid: 4172, name: 'nomic-embed', dev: 'cpu', tps: 880, res: [0, 1, 0] },
      { pid: 4190, name: 'qwen3-next-80b', dev: 'paged', tps: 3.4, res: [0.05, 0.35, 0.6] },
    ]
    let procs = []
    let cores = []
    let dev = []
    let L = null
    let R = null
    let key = ''
    let tick = 0
    let boxes = []

    function update() {
      for (const c of cores) {
        c.goal = Math.min(1, Math.max(0.05, c.goal + (Math.random() - 0.5) * 0.35))
        if (Math.random() < 0.08) c.model = Math.floor(Math.random() * 4)
      }
      for (const p of procs) {
        p.tps = p.base * (0.9 + Math.random() * 0.2)
        if (p.base > 3 && p.res[1] > 0.05 && Math.random() < 0.25) {
          // A block relocates onto the card.
          const d = Math.min(p.res[1], 0.05)
          p.res[1] -= d
          p.res[0] += d
          p.moved = performance.now()
        } else if (Math.random() < 0.15) {
          const d = Math.min(p.res[0], 0.04)
          p.res[0] -= d
          p.res[1] += d
        }
      }
      for (const d of dev) d.goal = Math.min(1, Math.max(0.1, d.goal + (Math.random() - 0.5) * 0.2))
    }

    return {
      layout(g) {
        const k = layoutKey(g)
        if (k === key) return
        key = k
        const ps = sides(g, 22, 12)
        L = ps && ps[0]
        R = ps && ps[1]
        boxes = ps ? ps.map((p) => [p.x - 1, p.y - 1, p.x + p.w, p.y + p.h]) : []
        procs = PROCS.map((p) => ({ ...p, base: p.tps, res: [...p.res], moved: -1e9 }))
        const n = Math.max(2, Math.min(info.threads || 8, 64))
        cores = Array.from({ length: n }, (_, i) => ({ id: i, load: 0.4, goal: 0.3 + Math.random() * 0.6, model: i % 4 }))
        dev = [
          { name: gname, load: 0.5, goal: 0.8 },
          { name: 'vram', load: 0.5, goal: 0.86 },
          { name: 'mem', load: 0.5, goal: 0.62 },
        ]
        tick = 0
      },
      hush: (col, row) => (within(boxes, col, row) ? 0.06 : 1),
      shown: () => 1,

      draw(ctx, g, pal, now) {
        if (!L) return
        if (!reduced && now - tick > 480) (tick = now), update()
        const cell = painter(ctx, g)
        const seam = Math.max(1, Math.round(g.dpr))
        const px = Math.min(11, (g.ch / g.dpr) * 0.82)
        const cc = charCells(g, px)
        const shade = [pal.hover, pal.lit, pal.mid, pal.dim]
        mono(ctx, g, px, 'left')

        // Processes.
        clipTo(ctx, g, L)
        const load = cores.reduce((a, c) => a + c.load, 0) / 2
        say(ctx, g, L.x, L.y, `models  ${procs.length} loaded · load ${load.toFixed(1)}`, pal.lit)
        say(ctx, g, L.x, L.y + 2, 'PID   MODEL           DEV        TOK/S', pal.dim)
        const pitch = Math.max(3, Math.min(5, Math.floor((L.h - 6) / procs.length)))
        procs.forEach((p, i) => {
          const y = L.y + 3 + i * pitch
          if (y + 1 >= L.y + L.h) return
          const tps = p.tps >= 100 ? p.tps.toFixed(0) : p.tps.toFixed(1)
          say(ctx, g, L.x, y, `${p.pid}  ${p.name.padEnd(15)} ${p.dev.padEnd(10)} ${tps.padStart(5)}`, i === 0 ? pal.hover : pal.mid)
          const bw = Math.min(L.w - 1, Math.floor(40 * cc))
          const a = Math.round(p.res[0] * bw)
          const b = Math.round((p.res[0] + p.res[1]) * bw)
          for (let x = 0; x < bw; x++) {
            const hot = x >= a - 1 && x <= a && now - p.moved < 300
            cell(L.x + x, y + 1, hot ? pal.crest : x < a ? pal.lit : x < b ? pal.mid : pal.dim, seam)
          }
        })
        const ly = Math.min(L.y + L.h - 1, L.y + 3 + procs.length * pitch)
        say(ctx, g, L.x, ly, 'blocks:', pal.dim)
        let lx = L.x + 8 * cc
        for (const [ink, s] of [[pal.lit, 'card'], [pal.mid, 'ram'], [pal.dim, 'paged']]) {
          cell(Math.round(lx), ly, ink, seam)
          say(ctx, g, Math.round(lx) + 1.5, ly, s, pal.dim)
          lx += 2.5 + (s.length + 2) * cc
        }
        ctx.restore()

        // Meters: cores in one or two columns, then the devices.
        clipTo(ctx, g, R)
        say(ctx, g, R.x, R.y, `cpu  ${cores.length} threads`, pal.lit)
        const rows = R.h - 7
        const colsN = cores.length > rows ? 2 : 1
        const per = Math.ceil(cores.length / colsN)
        const cp = rows >= per * 2 ? 2 : 1
        const colW = Math.floor((R.w - (colsN - 1)) / colsN)
        const meter = (x, y, w, label, v, ink) => {
          const lw = Math.ceil(label.length * cc) + 1
          say(ctx, g, x, y, label, pal.dim)
          const bw = w - lw - 1
          const n = Math.round(v * bw)
          for (let i = 0; i < bw; i++) if (i < n || i % 2 === 0) cell(x + lw + i, y, i < n ? ink : pal.dim, seam)
        }
        cores.forEach((c, i) => {
          if (!reduced) c.load += (c.goal - c.load) * 0.18
          const col = Math.floor(i / per)
          const y = R.y + 2 + (i % per) * cp
          if (y >= R.y + 2 + rows) return
          meter(R.x + col * (colW + 1), y, colW, String(c.id).padStart(2), c.load, shade[c.model])
        })
        const dy = R.y + 2 + Math.min(per * cp, rows) + 1
        dev.forEach((d, i) => {
          if (!reduced) d.load += (d.goal - d.load) * 0.18
          meter(R.x, dy + i * cp, R.w, d.name.padEnd(4), d.load, i === 0 ? pal.crest : pal.lit)
        })
        ctx.restore()
      },
    }
  }

  /* ------------------------------------------------------------ hero: worker swarm */

  // Pixels as threads. On the left a CPU pool: a handful of workers fork onto
  // a region, each sweeps its rows at its own pace, and they wait at the
  // barrier for the straggler. On the right a GPU: many lanes moving as one.
  function makeSwarm(info) {
    let teams = []
    let lit = new Map()
    let key = ''
    let tick = 0
    let boxes = []

    function team(p, n, lock, name) {
      const r = lcg(n * 31 + 7)
      const ws = Array.from({ length: n }, () => ({
        x: p.x + Math.floor(r() * p.w),
        y: p.y + 2 + Math.floor(r() * (p.h - 2)),
        tx: 0,
        ty: 0,
        pace: lock ? 1 : 0.35 + r() * 0.65,
        trail: [],
        state: 'go',
      }))
      const t = { p, ws, lock, name, phase: 'fork', at: 0, count: 0, region: null }
      region(t)
      return t
    }
    function region(t) {
      const { p, ws, lock } = t
      const h = Math.min(lock ? 8 : ws.length, p.h - 5)
      const w = Math.max(6, Math.min(p.w - 5, 8 + Math.floor(Math.random() * Math.max(1, p.w - 12))))
      const x = p.x + Math.floor(Math.random() * Math.max(1, p.w - w - 3))
      const y = p.y + 3 + Math.floor(Math.random() * Math.max(1, p.h - h - 4))
      t.region = { x, y, w, h, bar: x + w + 1 }
      const segs = Math.ceil(ws.length / h)
      ws.forEach((wk, i) => {
        const s = Math.floor(i / h)
        wk.row = y + (i % h)
        wk.x0 = x + Math.floor((s * w) / segs)
        wk.x1 = x + Math.floor(((s + 1) * w) / segs) - 1
        wk.tx = wk.x0
        wk.ty = wk.row
        wk.state = 'go'
      })
      t.phase = 'fork'
      t.count++
    }
    const arrived = (wk) => wk.x === wk.tx && wk.y === wk.ty

    function step(t, now) {
      const { ws, lock } = t
      for (const wk of ws) {
        const moves = lock || Math.random() < wk.pace
        if (t.phase === 'work' && wk.state === 'sweep') {
          if (!moves) continue
          lit.set(wk.row * 4096 + wk.x, now)
          if (wk.x < wk.x1) wk.trail = [[wk.x, wk.y], ...wk.trail].slice(0, 2), wk.x++, (wk.tx = wk.x)
          else (wk.state = 'wait'), (wk.tx = t.region.bar), (wk.ty = wk.row)
          continue
        }
        if (!arrived(wk)) {
          wk.trail = [[wk.x, wk.y], ...wk.trail].slice(0, 3)
          wk.x += Math.sign(wk.tx - wk.x)
          wk.y += Math.sign(wk.ty - wk.y)
        } else if (arrived(wk)) wk.trail = wk.trail.slice(0, -1)
      }
      if (t.phase === 'fork' && ws.every(arrived)) {
        for (const wk of ws) wk.state = 'sweep'
        t.phase = 'work'
      } else if (t.phase === 'work' && ws.every((wk) => wk.state === 'wait' && arrived(wk))) (t.phase = 'release'), (t.at = now)
      else if (t.phase === 'release' && now - t.at > 280) {
        for (const wk of ws) (wk.tx = t.p.x + Math.floor(Math.random() * t.p.w)), (wk.ty = t.p.y + 2 + Math.floor(Math.random() * (t.p.h - 2))), (wk.state = 'go')
        t.phase = 'scatter'
        t.at = now
      } else if (t.phase === 'scatter' && (ws.every(arrived) || now - t.at > 900)) region(t)
    }

    return {
      layout(g) {
        const k = layoutKey(g)
        if (k === key) return
        key = k
        lit = new Map()
        const ps = sides(g, 16, 12)
        boxes = ps ? ps.map((p) => [p.x - 1, p.y - 1, p.x + p.w, p.y + p.h]) : []
        if (!ps) return (teams = [])
        const n = Math.max(4, Math.min(info.threads || 8, 16, ps[0].h - 5))
        teams = [team(ps[0], n, false, 'cpu pool'), team(ps[1], 24, true, 'gpu')]
      },
      hush: (col, row) => (within(boxes, col, row) ? 0.3 : 1),
      shown: () => 1,

      draw(ctx, g, pal, now) {
        if (!teams.length) return
        if (!reduced) while (now - tick > 45) (tick = now - tick > 500 ? now : tick + 45), teams.forEach((t) => step(t, now))
        const cell = painter(ctx, g)
        const seam = Math.max(1, Math.round(g.dpr))
        for (const [k, at] of lit) {
          const age = now - at
          if (age > 4200) {
            lit.delete(k)
            continue
          }
          cell(k % 4096, Math.floor(k / 4096), age < 160 ? pal.hover : age < 1400 ? pal.lit : age < 2800 ? pal.mid : pal.dim, seam)
        }
        mono(ctx, g, 11, 'left')
        for (const t of teams) {
          const { region: r } = t
          if (t.phase === 'work' || t.phase === 'release')
            for (let y = r.y; y < r.y + r.h; y++) if (t.phase === 'release' || y % 2 === 0) cell(r.bar, y, t.phase === 'release' ? pal.crest : pal.dim)
          for (const wk of t.ws) {
            wk.trail.forEach(([x, y], i) => cell(x, y, i ? pal.mid : pal.hover))
            cell(wk.x, wk.y, pal.crest)
          }
          const what = t.phase === 'work' ? (t.lock ? 'lockstep' : 'working') : t.phase === 'release' ? 'barrier' : t.phase === 'fork' ? 'fork' : 'scatter'
          const who = t.lock ? `${t.ws.length} lanes` : `${t.ws.length} workers`
          say(ctx, g, t.p.x, t.p.y, `${t.name} · ${who} · region ${t.count} · ${what}`, t.phase === 'release' ? pal.crest : pal.lit)
        }
      },
    }
  }

  /* ------------------------------------------------------------ hero: the wordmark compiles */

  // The wordmark is emitted, not drawn. Each of its cells is taken from the
  // field: a pixel of the noise that is lit at that moment lifts out of it,
  // flies in and locks into place, left to right. It holds, then every cell
  // flies back to the spot it came from and settles into the noise again.
  function makeCompile() {
    const T = { build: 3200, hold: 6500, unbuild: 2400, rest: 900 }
    const P = T.build + T.hold + T.unbuild + T.rest
    let parts = []
    let byWord = new Map()
    let bySrc = new Map() // a claimed field cell -> the part that took it
    let src = [] // fallback sources, when nothing near is lit
    let born = performance.now()
    let key = ''
    let caption = null
    const phaseOf = (t) => (t < T.build ? 'build' : t < T.build + T.hold ? 'hold' : t < T.build + T.hold + T.unbuild ? 'unbuild' : 'rest')

    return {
      layout(g) {
        const k = layoutKey(g)
        if (k === key) return
        key = k
        parts = []
        byWord = new Map()
        bySrc = new Map()
        src = []
        for (let y = g.barRow + 1; y <= g.visR1; y++)
          for (let x = g.visC0; x <= g.visC1; x++) {
            if (x >= -2 && x <= WORD_W + 1 && y >= -2 && y <= WORD_H + 1) continue
            if (g.shade(x, y) >= 0.3) src.push([x, y])
          }
        if (!src.length) return
        const r = lcg(0x5eed)
        for (let i = src.length - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1))
          ;[src[i], src[j]] = [src[j], src[i]]
        }
        let n = 0
        for (let x = 0; x < WORD_W; x++)
          for (let y = 0; y < WORD_H; y++) {
            if (WORD[y][x] !== '1') continue
            const [sx, sy] = src[n++ % src.length]
            const j = jitter[y * 64 + x]
            const p = { sx, sy, tx: x, ty: y, go: (x / WORD_W) * 1700 + j * 400, dur: 900, back: T.build + T.hold + ((WORD_W - x) / WORD_W) * 1000 + j * 300, c: -1 }
            parts.push(p)
            byWord.set(y * WORD_W + x, p)
          }
        caption = { x: g.visC0 + 2, y: g.visR1 - 1 }
        born = performance.now()
      },
      hush(col, row, now) {
        if (reduced) return 1
        const p = bySrc.get(row * 4096 + col)
        if (!p || p.c !== Math.floor((now - born) / P)) return 1
        const t = (now - born) % P
        return t >= p.go && t < p.back + p.dur ? 0 : 1
      },
      shown: () => 1,
      // The wordmark hook: null hides a cell, 'crest' flashes it.
      word(row, col, now) {
        if (reduced) return undefined
        const p = byWord.get(row * WORD_W + col)
        if (!p) return undefined
        const t = (now - born) % P
        if (t < p.go + p.dur || t >= p.back) return null
        return t - p.go - p.dur < 200 ? 'crest' : undefined
      },

      draw(ctx, g, pal, now) {
        // The wordmark's outline stays put, so the empty slot reads as jitllm
        // while it compiles and while it is evicted.
        const s = Math.max(1, Math.round(g.dpr))
        const X = (x) => Math.round(g.ox + x * g.cw)
        const Y = (y) => Math.round(g.oy + y * g.ch)
        const on = (x, y) => y >= 0 && y < WORD_H && x >= 0 && x < WORD_W && WORD[y][x] === '1'
        ctx.fillStyle = pal.outline
        for (let y = 0; y < WORD_H; y++)
          for (let x = 0; x < WORD_W; x++) {
            if (!on(x, y)) continue
            if (!on(x - 1, y)) ctx.fillRect(X(x) - s, Y(y) - s, s, Y(y + 1) - Y(y) + 2 * s)
            if (!on(x + 1, y)) ctx.fillRect(X(x + 1), Y(y) - s, s, Y(y + 1) - Y(y) + 2 * s)
            if (!on(x, y - 1)) ctx.fillRect(X(x) - s, Y(y) - s, X(x + 1) - X(x) + 2 * s, s)
            if (!on(x, y + 1)) ctx.fillRect(X(x) - s, Y(y + 1), X(x + 1) - X(x) + 2 * s, s)
          }
        if (!parts.length || reduced) return
        const cell = painter(ctx, g)
        const c = Math.floor((now - born) / P)
        const t = (now - born) % P
        const ease = (u) => u * u * (3 - 2 * u)
        // A part leaving now takes a pixel the field is showing this frame:
        // of a handful of lit cells, the closest to where it is going.
        const lit = g.lit
        const claim = (p) => {
          if (p.c >= 0 && bySrc.get(p.sy * 4096 + p.sx) === p) bySrc.delete(p.sy * 4096 + p.sx)
          let best = null
          let bestD = Infinity
          for (let k = 0; k < 16 && lit.length; k++) {
            const i = 2 * Math.floor(Math.random() * (lit.length / 2))
            const x = lit[i]
            const y = lit[i + 1]
            if (bySrc.has(y * 4096 + x) || (x >= -2 && x <= WORD_W + 1 && y >= -2 && y <= WORD_H + 1)) continue
            const d = Math.hypot(x - p.tx, (y - p.ty) * 1.6)
            if (d < bestD) (bestD = d), (best = [x, y])
          }
          if (!best) best = src[Math.floor(Math.random() * src.length)]
          ;[p.sx, p.sy] = best
          p.dur = Math.min(1100, 380 + Math.hypot(p.sx - p.tx, p.sy - p.ty) * 10)
          p.c = c
          bySrc.set(p.sy * 4096 + p.sx, p)
        }
        let placed = 0
        for (const p of parts) {
          if (p.c !== c && t >= p.go) claim(p)
          if (p.c === c && t >= p.back + p.dur && t < p.back + p.dur + 1400) {
            // Home again: it settles back into the noise.
            const k = (t - p.back - p.dur) / 1400
            if (jitter[(p.sy & 63) * 64 + (p.sx & 63)] > k) cell(p.sx, p.sy, k < 0.3 ? pal.mid : pal.dim)
          }
          let u = -1
          let from = [p.sx, p.sy]
          let to = [p.tx, p.ty]
          if (t >= p.go && t < p.go + p.dur) u = (t - p.go) / p.dur
          else if (t >= p.back && t < p.back + p.dur) (u = (t - p.back) / p.dur), (from = [p.tx, p.ty]), (to = [p.sx, p.sy])
          if (t >= p.go + p.dur && t < p.back) placed++
          if (u < 0) continue
          // It leaves as the dim pixel it was and heats as it travels, and
          // cools the other way on its way home.
          const heat = from[0] === p.sx && from[1] === p.sy ? u : 1 - u
          const head = heat < 0.2 ? pal.dim : heat < 0.45 ? pal.mid : heat < 0.75 ? pal.lit : pal.hover
          if (u > 0.06) {
            const e = ease(u - 0.06)
            cell(Math.round(from[0] + (to[0] - from[0]) * e), Math.round(from[1] + (to[1] - from[1]) * e), heat < 0.45 ? pal.dim : pal.mid)
          }
          const e = ease(u)
          cell(Math.round(from[0] + (to[0] - from[0]) * e), Math.round(from[1] + (to[1] - from[1]) * e), head)
        }
        const phase = phaseOf(t)
        const text =
          phase === 'build' ? `jit · emitting jitllm · ${placed}/${parts.length} cells` : phase === 'hold' ? `resident · ${parts.length} cells` : phase === 'unbuild' ? 'evicting to the field' : 'idle'
        mono(ctx, g, 11, 'left')
        const fs = Math.round(11 * g.dpr)
        const x = g.ox + caption.x * g.cw
        const y = g.oy + (caption.y + 0.5) * g.ch
        ctx.fillStyle = pal.bg
        ctx.fillRect(x - fs * 0.5, y - fs, ctx.measureText(text).width + fs, fs * 2)
        say(ctx, g, caption.x, caption.y, text, phase === 'build' ? pal.crest : pal.mid)
      },
    }
  }

  /* ------------------------------------------------------------ field textures */

  // How much ink a cell of the footer's field carries, before the ramp and the
  // dither: a twinkling value noise. The hero draws its warp field instead
  // (see stepWarp in pixelField).
  function texture(u, v, t, jit) {
    const base = 0.62 * sampleNoise(u + t * 0.12, v - t * 0.05) + 0.38 * sampleNoise(u * 0.5 - t * 0.07, v * 0.5 + t * 0.05)
    const flicker = 0.5 + 0.5 * Math.sin(t * 1.25 + jit * 6.283)
    return 0.28 + 0.55 * base * base + 0.17 * flicker
  }

  const STAMP = ICONS.chip
  const STAMP_N = STAMP.length
  let sharedCell = 10 // css px; the hero publishes its cell so the footer matches

  function pixelField(canvas, isHero) {
    const ctx = canvas.getContext('2d', { alpha: false })
    const host = canvas.parentElement
    const slot = isHero ? host.querySelector('[data-wordmark]') : null
    const quiet = isHero
      ? [
          ...document.querySelectorAll('.bar a, .bar button, .hero-switch button'),
          ...host.querySelectorAll('.callout, .hero-copy > *'),
        ]
      : [...host.querySelectorAll('[data-quiet]')]

    let pal = {}
    const readPalette = () => {
      const cs = getComputedStyle(root)
      for (const k of ['bg', 'dim', 'mid', 'lit', 'hover', 'crest'])
        pal[k] = cs.getPropertyValue(`--f-${k}`).trim()
      pal.outline = cs.getPropertyValue('--line-strong').trim()
    }
    readPalette()

    let dpr = 1, W = 0, H = 0
    let ox = 0, oy = 0, cw = 10, ch = 10
    let c0 = 0, r0 = 0, cols = 0, rows = 0
    let ramp = new Float32Array(0)
    let visible = true
    const pointer = { x: -1e4, y: -1e4, s: 0, goal: 0 }
    let tokens = []
    let nextToken = 0
    // The hero reads the machine it runs on and draws it in the field.
    let machine = isHero ? makeScene() : null
    let stamps = []
    let open = new Float32Array(0) // 0 beside the copy, 1 clear of it
    // warp's stars: a position on the screen plane and a depth.
    let stars = []
    let warpGrid = new Float32Array(0)
    let warpHead = new Float32Array(0)
    let warpClock = 0
    let warpBoost = 0
    const vanish = { c: 0, r: 0 }
    // The wordmark "compiles" in: a left-to-right sweep, replayed on each theme.
    let sweep = reduced || !isHero || machine.word ? -Infinity : performance.now() + 300

    const distToQuiet = (clientX, clientY) => {
      let best = Infinity
      for (const el of quiet) {
        const r = el.getBoundingClientRect()
        if (r.width < 1) continue
        const dx = Math.max(r.left - clientX, 0, clientX - r.right)
        const dy = Math.max(r.top - clientY, 0, clientY - r.bottom)
        best = Math.min(best, Math.hypot(dx, dy))
      }
      return best
    }
    // Glows fade out near text and controls so nothing lights up behind them.
    const hush = (clientX, clientY) => clamp(distToQuiet(clientX, clientY) / 96) ** 3

    function measure() {
      const box = host.getBoundingClientRect()
      if (box.width < 1 || box.height < 1) return false
      dpr = Math.min(devicePixelRatio || 1, 2)
      const w = Math.round(box.width * dpr)
      const h = Math.round(box.height * dpr)
      if (w !== W || h !== H) (canvas.width = W = w), (canvas.height = H = h)

      if (slot) {
        const s = slot.getBoundingClientRect()
        ox = (s.left - box.left) * dpr
        oy = (s.top - box.top) * dpr
        cw = (s.width * dpr) / WORD_W
        ch = (s.height * dpr) / WORD_H
        sharedCell = s.width / WORD_W
      } else {
        cw = ch = Math.max(6, Math.round(sharedCell)) * dpr
        ox = oy = 0
      }
      c0 = -Math.ceil(ox / cw) - 1
      r0 = -Math.ceil(oy / ch) - 1
      cols = Math.ceil((W - ox) / cw) - c0 + 1
      rows = Math.ceil((H - oy) / ch) - r0 + 1

      // How much field each cell may carry: dense at the hero's edges,
      // thinning toward its centre and under the bar; an even haze elsewhere,
      // cleared around readable blocks.
      const boxes = (isHero ? [...host.querySelectorAll('.callout, .hero-copy > *'), ...document.querySelectorAll('header a, header button')] : quiet).map((el) => {
            const r = el.getBoundingClientRect()
        return [(r.left - box.left) * dpr, (r.top - box.top) * dpr, (r.right - box.left) * dpr, (r.bottom - box.top) * dpr]
      })
      const reach = (isHero ? 64 : 140) * dpr
      const clearOf = (x, y) => {
        let near = Infinity
        for (const [l, t, rr, b] of boxes)
          near = Math.min(near, Math.hypot(Math.max(l - x, 0, x - rr), Math.max(t - y, 0, y - b)))
        return clamp(near / reach) ** 3
      }
      ramp = new Float32Array(cols * rows)
      open = new Float32Array(cols * rows)
      for (let r = 0; r < rows; r++) {
        const y = oy + (r0 + r + 0.5) * ch
        for (let c = 0; c < cols; c++) {
          const x = ox + (c0 + c + 0.5) * cw
          let v
          if (isHero) {
            const nx = (x / W) * 2 - 1
            const ny = (y / H) * 2 - 1
            const edge = clamp((Math.hypot(nx, ny * 0.9) - 0.34) / 0.7)
            v = edge * edge * clamp((y / dpr - 28) / 150, 0.14, 1) * clearOf(x, y)
          } else v = 0.32 * clearOf(x, y)
          ramp[r * cols + c] = v
          open[r * cols + c] = clearOf(x, y)
        }
      }
      if (isHero) {
        warpGrid = new Float32Array(cols * rows)
        warpHead = new Float32Array(cols * rows)
        stars = []
      }
      if (isHero) {
        const below = Math.max(...[...host.querySelectorAll('.hero-copy > *')].map((el) => Math.ceil(((el.getBoundingClientRect().bottom - box.top) * dpr - oy) / ch)))
        const above = Math.floor(((host.querySelector('.callout').getBoundingClientRect().top - box.top) * dpr - oy) / ch)
        machine.layout({
          shade: (col, row) => {
            const c = col - c0
            const r = row - r0
            return c < 0 || r < 0 || c >= cols || r >= rows ? 0 : ramp[r * cols + c]
          },
          above,
          visC0: Math.ceil(-ox / cw),
          visC1: Math.floor((W - ox) / cw) - 1,
          barRow: Math.ceil(((56 + 10) * dpr - oy) / ch),
          visR1: Math.floor((H - oy) / ch) - 1,
          below,
        })
      }
      return true
    }

    const onWord = (x, y) =>
      isHero && x >= ox && y >= oy && x < ox + WORD_W * cw && y < oy + WORD_H * ch

    function stampAt(x, y, list) {
      let amp = 0
      for (const s of list) {
        const lx = Math.floor((x - s.x) / s.px + STAMP_N / 2)
        const ly = Math.floor((y - s.y) / s.px + STAMP_N / 2)
        if (lx < 0 || ly < 0 || lx >= STAMP_N || ly >= STAMP_N) continue
        if (STAMP[ly][lx] === '1') amp = Math.max(amp, s.amp)
      }
      return amp
    }

    // Warp speed: stars at a depth z fly toward the viewer, projected from a
    // vanishing point behind the wordmark that leans toward the pointer. Each
    // frame a star is drawn as a streak from where it was to where it is, so
    // the streak stretches as it speeds up toward the edge, and brightens as
    // it comes closer. A click is a jump: everything runs six times as fast
    // for a moment.
    function stepWarp(now) {
      const dt = warpClock ? Math.min(0.05, (now - warpClock) / 1000) : 0
      warpClock = now
      warpBoost *= Math.pow(0.25, dt)
      // The vanishing point: the wordmark's centre, eased toward the pointer.
      const homeC = WORD_W / 2 - c0
      const homeR = WORD_H / 2 - r0
      const lean = pointer.s > 0.01 ? 0.18 : 0
      const goalC = homeC + lean * ((pointer.x - ox) / cw - c0 - homeC)
      const goalR = homeR + lean * ((pointer.y - oy) / ch - r0 - homeR)
      vanish.c = vanish.c ? vanish.c + (goalC - vanish.c) * 0.08 : homeC
      vanish.r = vanish.r ? vanish.r + (goalR - vanish.r) * 0.08 : homeR
      const F = Math.max(cols, rows) * 0.42
      const want = Math.min(260, Math.floor((cols * rows) / 28))
      const spawn = (st, z) => {
        let x, y
        do (x = Math.random() * 2 - 1), (y = Math.random() * 2 - 1)
        while (Math.abs(x) + Math.abs(y) < 0.08)
        st.x = x
        st.y = y
        st.z = z
        st.v = 0.28 + Math.random() * 0.3
        return st
      }
      while (stars.length < want) stars.push(spawn({}, 0.1 + Math.random() * 0.9))
      warpGrid.fill(0)
      warpHead.fill(0)
      const speed = 1 + 6 * warpBoost
      const at = (x, y, z) => [vanish.c + (x / z) * F, vanish.r + (y / z) * F * 0.9]
      for (const st of stars) {
        st.z -= st.v * speed * dt
        const [c, r] = at(st.x, st.y, st.z)
        if (st.z <= 0.03 || c < -2 || r < -2 || c > cols + 1 || r > rows + 1) {
          spawn(st, 1)
          continue
        }
        // The streak is where the star was a short depth ago: long when it
        // is close and fast, a point when it is far, as a long exposure sees
        // it. A jump stretches every streak.
        const [tc, tr] = at(st.x, st.y, Math.min(1.2, st.z + 0.09 * speed))
        const b = clamp((1 - st.z) * 2)
        const n = Math.min(60, Math.round(Math.max(Math.abs(c - tc), Math.abs(r - tr))))
        for (let k = 0; k <= n; k++) {
          const f = n ? k / n : 1
          const cc = Math.round(tc + (c - tc) * f)
          const rr = Math.round(tr + (r - tr) * f)
          if (cc < 0 || rr < 0 || cc >= cols || rr >= rows) continue
          const i = rr * cols + cc
          warpGrid[i] = Math.max(warpGrid[i], b * (0.45 + 0.55 * f))
        }
        const hc = Math.round(c)
        const hr = Math.round(r)
        if (hc >= 0 && hr >= 0 && hc < cols && hr < rows) warpHead[hr * cols + hc] = Math.max(warpHead[hr * cols + hc], b)
      }
    }

    function draw(now) {
      const t = reduced ? 0 : now / 1000
      pointer.s += (pointer.goal - pointer.s) * 0.3

      // Tokens: packets that run along a row of the field, left to right, each
      // with a fading tail. They only show where the field has ink.
      if (!isHero && !reduced && now >= nextToken) {
        const row = r0 + Math.floor(Math.random() * rows)
        const speed = (isHero ? 38 : 26) + Math.random() * 30
        tokens.push({ row, born: now, speed, len: 6 + Math.random() * 10 })
        nextToken = now + (isHero ? 90 : 260) + Math.random() * (isHero ? 220 : 500)
      }
      const span = cols + 20
      tokens = tokens.filter((k) => ((now - k.born) / 1000) * k.speed < span + k.len)
      const glows = []
      const reach = (s) => 11 * cw * (0.5 + 0.5 * s)
      if (pointer.s > 0.01) glows.push([pointer.x, pointer.y, pointer.s, reach(pointer.s)])
      const glowAt = (x, y) => {
        let g = 0
        for (const [gx, gy, s, rr] of glows) {
          const d = Math.hypot(x - gx, y - gy)
          if (d < rr) g = Math.max(g, (1 - d / rr) ** 2 * s)
        }
        return g
      }

      stamps = stamps.filter((s) => now - s.born < s.life)
      const live = stamps.map((s) => {
        const age = (now - s.born) / s.life
        return { x: s.x, y: s.y, px: cw * (0.5 + 2.2 * (1 - (1 - age) ** 3)), amp: (1 - age) ** 1.6 }
      })

      ctx.fillStyle = pal.bg
      ctx.fillRect(0, 0, W, H)
      const lit = [] // the noise's lit cells this frame, for a scene to take from
      if (isHero && !reduced) stepWarp(now)

      for (let r = 0; r < rows; r++) {
        const row = r0 + r
        const yTop = oy + row * ch
        const y = Math.round(yTop)
        const hgt = Math.round(yTop + ch) - y
        const cy = yTop + ch / 2
        const inWordRows = isHero && row >= 0 && row < WORD_H
        const heads = []
        for (const k of tokens) if (k.row === row) heads.push([c0 - 10 + ((now - k.born) / 1000) * k.speed, k.len])
        for (let c = 0; c < cols; c++) {
          const col = c0 + c
          if (inWordRows && col >= 0 && col < WORD_W && WORD[row][col] === '1') continue
          const xLeft = ox + col * cw
          const cx = xLeft + cw / 2
          const shade = ramp[r * cols + c]
          const jit = jitter[(row & 63) * 64 + (col & 63)]
          let lum = 0
          if (!isHero && shade > 0.002) {
            const u = col / 9
            const v = row / 9
            lum = shade * texture(u, v, t, jit) * 0.82
          }
          if (isHero && warpGrid[r * cols + c] > 0) {
            // A streak is drawn solid, bright at the head and dimming down its
            // tail; the dither would break it into dots.
            const o = (0.2 + 0.8 * open[r * cols + c]) * machine.hush(col, row, now)
            const w = warpGrid[r * cols + c] * o
            const wh = warpHead[r * cols + c] * o
            if (w > 0.14) {
              ctx.fillStyle = wh > 0.6 ? pal.crest : wh > 0.3 || w > 0.7 ? pal.hover : w > 0.5 ? pal.lit : w > 0.3 ? pal.mid : pal.dim
              const x = Math.round(xLeft)
              ctx.fillRect(x, y, Math.round(xLeft + cw) - x, hgt)
              continue
            }
          }
          if (isHero && lum > 0) lum *= machine.hush(col, row, now)
          const g = glows.length ? glowAt(cx, cy) : 0
          const st = live.length ? stampAt(cx, cy, live) : 0
          let tk = 0
          for (const [head, len] of heads) {
            const d = head - col
            if (d >= 0 && d < len) tk = Math.max(tk, (1 - d / len) ** 1.4)
          }
          tk *= clamp(shade * 5)
          lum += g * 0.6 + st * 1.1 + tk * 1.2
          const threshold = 0.76 * ((BAYER[(row & 7) * 8 + (col & 7)] + 0.5) / 64) + 0.24 * jit
          if (lum <= threshold) continue
          const heat = Math.max(g, st, tk)
          if (isHero && heat < 0.1) lit.push(col, row)
          const deep = isHero && lum - threshold > 0.3
          ctx.fillStyle = heat > 0.7 ? pal.hover : heat > 0.34 ? pal.lit : heat > 0.1 || deep ? pal.mid : pal.dim
          const x = Math.round(xLeft)
          ctx.fillRect(x, y, Math.round(xLeft + cw) - x, hgt)
        }
      }

      if (!isHero) return
      machine.draw(ctx, { lit, ox, oy, cw, ch, dpr, visC0: Math.ceil(-ox / cw), visC1: Math.floor((W - ox) / cw) - 1 }, pal, reduced ? 0 : now, reduced)
      // The wordmark on the same lattice, cell for cell.
      for (let row = 0; row < WORD_H; row++) {
        const bits = WORD[row]
        const yTop = oy + row * ch
        const y = Math.round(yTop)
        const hgt = Math.round(yTop + ch) - y
        const rest = pal[bandOf(row, WORD_H)]
        for (let col = 0; col < WORD_W; col++) {
          if (bits[col] !== '1') continue
          const xLeft = ox + col * cw
          let ink = rest
          const at = sweep + (col / WORD_W) * 850 + jitter[row * 64 + col] * 140
          const own = machine.word?.(row, col, now)
          if (now < at || own === null) continue
          if (now - at < 180 || own === 'crest') ink = pal.crest
          else {
            const cx = xLeft + cw / 2
            const cy = yTop + ch / 2
            const hit = Math.max(glows.length ? glowAt(cx, cy) : 0, live.length ? stampAt(cx, cy, live) : 0)
            if (hit > 0.45) ink = pal.crest
            else if (hit > 0.12) ink = pal.hover
          }
          ctx.fillStyle = ink
          const x = Math.round(xLeft)
          ctx.fillRect(x, y, Math.round(xLeft + cw) - x, hgt)
        }
      }
    }

    let frame = 0
    let last = 0
    const loop = (now) => {
      frame = requestAnimationFrame(loop)
      if (now - last < 28) return
      last = now
      draw(now)
    }
    const redraw = () => draw(reduced ? 0 : performance.now())
    const start = () => {
      if (reduced) redraw()
      else if (!frame && visible) frame = requestAnimationFrame(loop)
    }

    const locate = (e) => {
      const box = host.getBoundingClientRect()
      return {
        inside: e.clientX >= box.left && e.clientX <= box.right && e.clientY >= box.top && e.clientY <= box.bottom,
        x: (e.clientX - box.left) * dpr,
        y: (e.clientY - box.top) * dpr,
      }
    }
    const isControl = (el) => el instanceof Element && el.closest('a, button, input, header, [data-quiet]') !== null

    if (finePointer)
      addEventListener('pointermove', (e) => {
        if (!visible) return
        const p = locate(e)
        pointer.goal = p.inside ? hush(e.clientX, e.clientY) : 0
        if (!p.inside) return
        pointer.x = p.x
        pointer.y = p.y
        if (slot) host.style.cursor = onWord(p.x, p.y) ? 'pointer' : ''
        if (reduced) redraw()
      }, { passive: true })

    let pressedWord = false
    addEventListener('pointerdown', (e) => {
      const p = locate(e)
      pressedWord = p.inside && onWord(p.x, p.y)
      if (!p.inside || pressedWord || reduced || isControl(e.target)) return
      stamps = [...stamps.slice(-3), { x: p.x, y: p.y, born: performance.now(), life: 900 + Math.random() * 250 }]
      if (isHero) warpBoost = 1
    }, { passive: true })
    addEventListener('pointerup', (e) => {
      const p = locate(e)
      if (pressedWord && p.inside && onWord(p.x, p.y)) nextTheme()
      pressedWord = false
    }, { passive: true })

    addEventListener(THEME_EVENT, () => {
      readPalette()
      if (isHero && !reduced && !machine.word) sweep = performance.now()
      redraw()
    })
    if (isHero)
      addEventListener(HERO_EVENT, () => {
        machine = makeScene()
        if (measure()) redraw()
      })

    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      if (reduced) return
      if (visible) start()
      else if (frame) cancelAnimationFrame(frame), (frame = 0)
    }, { rootMargin: '64px' }).observe(host)

    const ro = new ResizeObserver(() => {
      if (measure()) redraw(), start()
    })
    ro.observe(host)
    if (slot) ro.observe(slot)
    // Fonts shift the text boxes the footer field keeps clear of.
    document.fonts?.ready.then(() => measure() && redraw())
  }

  const MACHINE = readMachine()
  // Hero scenes under comparison: a switcher for choosing one; delete
  // it and the losers once the hero is decided.
  const SCENES = {
    machine: () => makeMachine(MACHINE),
    pass: () => makePass(MACHINE),
    defrag: () => makeDefrag(),
    sand: () => makeSand(MACHINE),
    pcb: () => makePcb(MACHINE),
    asm: () => makeAsm(MACHINE),
    top: () => makeTop(MACHINE),
    swarm: () => makeSwarm(MACHINE),
    compile: () => makeCompile(),
  }
  // The scene named now, built; a name that is not one of SCENES' own builds compile.
  const makeScene = () => {
    const make = Object.hasOwn(SCENES, scene) ? SCENES[scene] : null
    return typeof make === 'function' ? make() : SCENES.compile()
  }
  const HERO_EVENT = 'jitllm:hero'
  const heroQuery = new URLSearchParams(location.search)
  let scene = heroQuery.get('hero')
  if (!Object.hasOwn(SCENES, scene)) scene = 'compile'
  // The scene and noise switcher is for comparing heroes, not for visitors: it
  // shows only when the URL already names one (?hero=) or asks (?dev).
  if (['hero', 'dev'].some((k) => heroQuery.has(k))) {
    const heroEl = document.querySelector('[data-hero]')
    const nav = document.createElement('div')
    nav.className = 'hero-switch'
    nav.setAttribute('role', 'radiogroup')
    nav.setAttribute('aria-label', 'Hero scene')
    nav.innerHTML =
      Object.keys(SCENES).map((id) => `<button type="button" role="radio" data-scene="${id}">${id}</button>`).join('')
    const mark = () =>
      nav.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.scene === scene)))
    const remember = () => {
      const q = new URLSearchParams()
      q.set('hero', scene)
      history.replaceState(null, '', q.size ? `?${q}` : location.pathname)
    }
    mark()
    nav.addEventListener('click', (e) => {
      const b = e.target.closest('button')
      if (!b) return
      if (Object.hasOwn(SCENES, b.dataset.scene) && b.dataset.scene !== scene) (scene = b.dataset.scene), dispatchEvent(new Event(HERO_EVENT))
      remember()
      mark()
    })
    heroEl?.append(nav)
  }
  {
    const el = document.querySelector('[data-detected]')
    const say = () => {
      const bits = [MACHINE.threads && `${MACHINE.threads} threads`, MACHINE.gpu, [MACHINE.os, MACHINE.arch].filter(Boolean).join(' ')].filter(Boolean)
      if (el && bits.length) el.textContent = `Your machine, as this browser reports it: ${bits.join(' · ')}. Read locally; nothing is sent.`
    }
    say()
    navigator.userAgentData?.getHighEntropyValues?.(['architecture', 'bitness']).then((v) => {
      if (v.architecture === 'arm') MACHINE.arch = 'ARM64'
      else if (v.architecture === 'x86') MACHINE.arch = v.bitness === '64' ? 'x86\u201164' : 'x86'
      say()
    }, () => {})
  }
  const heroCanvas = document.querySelector('[data-field="hero"]')
  const groundCanvas = document.querySelector('[data-field="ground"]')
  if (heroCanvas) pixelField(heroCanvas, true)
  if (groundCanvas) pixelField(groundCanvas, false)

  /* ------------------------------------------------------------ header */

  const bar = document.querySelector('[data-bar]')
  const hero = document.querySelector('[data-hero]')
  const menuBtn = document.querySelector('[data-menu]')
  const menu = document.getElementById('menu')
  let pastHero = false
  const paintBar = () => bar.toggleAttribute('data-solid', pastHero || !menu.hidden)
  new IntersectionObserver(([e]) => {
    pastHero = !e.isIntersecting
    paintBar()
  }, { rootMargin: '-57px 0px 0px 0px' }).observe(hero)

  const setMenu = (open) => {
    menu.hidden = !open
    menuBtn.setAttribute('aria-expanded', String(open))
    paintBar()
  }
  menuBtn.addEventListener('click', () => setMenu(menu.hidden))
  menu.addEventListener('click', (e) => e.target.closest('a') && setMenu(false))

  /* ------------------------------------------------------------ theme picker */

  const picker = document.querySelector('[data-picker]')
  const list = picker.querySelector('[data-picker-list]')
  list.setAttribute('role', 'radiogroup')
  list.setAttribute('aria-label', 'Theme')
  list.innerHTML = THEMES.map(
    ([id, name, bg, brand, dim]) =>
      `<button class="swatch" type="button" role="radio" data-id="${id}">` +
      `<span class="chips" aria-hidden="true"><i style="background:${bg}"></i><i style="background:${dim}"></i><i style="background:${brand}"></i></span>${name}</button>`,
  ).join('')
  let opener = null
  const closePicker = () => {
    picker.hidden = true
    opener?.focus()
  }
  const openPicker = (from) => {
    opener = from
    setMenu(false)
    for (const b of list.children) b.setAttribute('aria-checked', String(b.dataset.id === currentTheme()))
    picker.hidden = false
    list.querySelector('[aria-checked="true"]').focus()
  }
  document.querySelectorAll('[data-theme-open]').forEach((b) => b.addEventListener('click', () => openPicker(b)))
  picker.querySelectorAll('[data-picker-close]').forEach((b) => b.addEventListener('click', closePicker))
  list.addEventListener('click', (e) => {
    const b = e.target.closest('[data-id]')
    if (!b) return
    setTheme(b.dataset.id)
    closePicker()
  })
  addEventListener('keydown', (e) => {
    if (picker.hidden) return
    if (e.key === 'Escape') return closePicker()
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    e.preventDefault()
    const items = [...list.children]
    const i = items.indexOf(document.activeElement)
    items[(i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length].focus()
  })

  /* ------------------------------------------------------------ small things */

  // Phone-width card rail: a progress bar instead of a scrollbar.
  const rail = document.querySelector('[data-rail]')
  const thumb = document.querySelector('[data-rail-thumb]')
  if (rail && thumb) {
    const sync = () => {
      const reach = rail.scrollWidth - rail.clientWidth
      const ratio = Math.min(1, rail.clientWidth / rail.scrollWidth)
      thumb.style.width = `${ratio * 100}%`
      thumb.style.transform = `translateX(${reach > 0 ? ((rail.scrollLeft / reach) * (1 - ratio) * 100) / ratio : 0}%)`
    }
    rail.addEventListener('scroll', sync, { passive: true })
    new ResizeObserver(sync).observe(rail)
  }

  // Copy a command block without its prompts.
  for (const btn of document.querySelectorAll('[data-copy]')) {
    btn.addEventListener('click', async () => {
      const text = btn.closest('.cmd')
        .querySelector('code')
        .innerText.split('\n')
        .map((l) => l.replace(/^\$ /, ''))
        .join('\n')
      try {
        await navigator.clipboard.writeText(text)
        btn.textContent = 'Copied'
        btn.setAttribute('data-done', '')
      } catch {
        btn.textContent = 'Select'
      }
      setTimeout(() => ((btn.textContent = 'Copy'), btn.removeAttribute('data-done')), 1600)
    })
  }
})()
