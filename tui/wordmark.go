package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// The website's pixel wordmark (website/public/app.js, WORD_RECTS): a 53 x 21
// grid of cells, stroke three cells, drawn here two pixels to a character with
// half blocks.
const (
	wordW = 53
	wordH = 21
)

func wordRects() [][4]int {
	return [][4]int{
		{3, 0, 3, 3}, {3, 5, 3, 15}, {0, 18, 5, 3}, // j
		{9, 0, 3, 3}, {9, 5, 3, 12}, // i
		{17, 1, 3, 16}, {15, 5, 8, 3}, {20, 14, 3, 3}, // t
		{26, 0, 3, 17}, {32, 0, 3, 17}, // l l
		{38, 5, 14, 3}, {38, 5, 3, 12}, {44, 5, 3, 12}, {50, 6, 3, 11}, // m
	}
}

func wordOn() [wordH][wordW]bool {
	var on [wordH][wordW]bool
	for _, r := range wordRects() {
		for y := r[1]; y < r[1]+r[3]; y++ {
			for x := r[0]; x < r[0]+r[2]; x++ {
				on[y][x] = true
			}
		}
	}
	return on
}

// The website's green theme (themes.css): the wordmark cools from crest to
// dim, top to bottom, in bands of rows.
type rgb struct{ r, g, b uint8 }

func fCrest() rgb { return rgb{0xda, 0xec, 0xc6} }
func fHover() rgb { return rgb{0xbb, 0xdd, 0x97} }
func fLit() rgb   { return rgb{0x9e, 0xce, 0x6a} }
func fMid() rgb   { return rgb{0x67, 0x85, 0x49} }
func fDim() rgb   { return rgb{0x39, 0x48, 0x2e} }

func bands() []struct {
	c    rgb
	rows int
} {
	return []struct {
		c    rgb
		rows int
	}{{fCrest(), 5}, {fHover(), 3}, {fLit(), 5}, {fMid(), 4}, {fDim(), 4}}
}

func bandOf(row int) rgb {
	for _, b := range bands() {
		if row < b.rows {
			return b.c
		}
		row -= b.rows
	}
	return fDim()
}

func mix(a, b rgb, t float64) rgb {
	t = min(max(t, 0), 1)
	f := func(x, y uint8) uint8 { return uint8(float64(x) + (float64(y)-float64(x))*t + 0.5) }
	return rgb{f(a.r, b.r), f(a.g, b.g), f(a.b, b.b)}
}

// canvas is pixels two to a character row; a zero alpha pixel is empty and
// shows the terminal's ground.
type canvas struct {
	w, h int
	px   []rgb
	on   []bool
}

func newCanvas(w, h int) *canvas {
	return &canvas{w: w, h: h, px: make([]rgb, w*h), on: make([]bool, w*h)}
}

func (c *canvas) set(x, y int, col rgb) {
	if x >= 0 && y >= 0 && x < c.w && y < c.h {
		c.px[y*c.w+x], c.on[y*c.w+x] = col, true
	}
}

// String draws the canvas with half blocks: the upper pixel as the
// foreground of ▀, the lower as its background.
func (c *canvas) String() string {
	var b strings.Builder
	fg := func(p rgb) string { return fmt.Sprintf("\x1b[38;2;%d;%d;%dm", p.r, p.g, p.b) }
	bg := func(p rgb) string { return fmt.Sprintf("\x1b[48;2;%d;%d;%dm", p.r, p.g, p.b) }
	for y := 0; y < c.h; y += 2 {
		if y > 0 {
			b.WriteByte('\n')
		}
		for x := 0; x < c.w; x++ {
			ti, bi := y*c.w+x, (y+1)*c.w+x
			top := c.on[ti]
			bot := bi < len(c.on) && c.on[bi]
			switch {
			case top && bot:
				b.WriteString(fg(c.px[ti]) + bg(c.px[bi]) + "▀\x1b[0m")
			case top:
				b.WriteString(fg(c.px[ti]) + "▀\x1b[0m")
			case bot:
				b.WriteString(fg(c.px[bi]) + "▄\x1b[0m")
			default:
				b.WriteByte(' ')
			}
		}
	}
	return b.String()
}

// hash is a cheap stable noise in [0, 1).
func hash(x, y, z int) float64 {
	h := uint32(x)*374761393 + uint32(y)*668265263 + uint32(z)*2147483647
	h = (h ^ (h >> 13)) * 1274126177
	return float64(h^(h>>16)) / float64(1<<32)
}

// wordmark is the still mark, scale pixels to a cell, with the website's
// scan line passing over it at phase.
func wordmark(scale, phase int) string {
	c := newCanvas(wordW*scale, wordH*scale)
	scan := float64(phase%(wordW+60)) - 10
	on := wordOn()
	for y := 0; y < wordH; y++ {
		for x := 0; x < wordW; x++ {
			if !on[y][x] {
				continue
			}
			col := bandOf(y)
			if d := math.Abs(float64(x) - scan); d < 1.5 {
				col = mix(col, fCrest(), 1-d/1.5)
			}
			for dy := 0; dy < scale; dy++ {
				for dx := 0; dx < scale; dx++ {
					c.set(x*scale+dx, y*scale+dy, col)
				}
			}
		}
	}
	return c.String()
}

// compile is the website's hero: the wordmark is emitted, not drawn. Each of
// its cells lifts out of a field of noise, flies in and locks into place,
// left to right, flashing as it lands; then it holds.
type compile struct {
	start  time.Time
	w, h   int // the canvas, in pixels
	scale  int
	ox, oy int
	parts  []part
}

type part struct {
	sx, sy, tx, ty float64
	band           rgb
	go_, dur       time.Duration
}

const (
	compileBuild = 2300 * time.Millisecond
	compileHold  = 900 * time.Millisecond
	compileTotal = compileBuild + compileHold
)

func newCompile(cols, rows int) *compile {
	w, h := cols, rows*2
	scale := max(1, min((w-4)/wordW, (h-10)/wordH, 2))
	c := &compile{start: time.Now(), w: w, h: h, scale: scale,
		ox: (w - wordW*scale) / 2, oy: (h-wordH*scale)/2 - 2}
	n := 0
	on := wordOn()
	for x := 0; x < wordW; x++ {
		for y := 0; y < wordH; y++ {
			if !on[y][x] {
				continue
			}
			for dy := 0; dy < scale; dy++ {
				for dx := 0; dx < scale; dx++ {
					// A source somewhere in the field, away from the mark.
					sx := hash(n, 1, 7) * float64(w)
					sy := hash(n, 2, 7) * float64(h)
					tx := float64(c.ox + x*scale + dx)
					ty := float64(c.oy + y*scale + dy)
					dist := math.Hypot(sx-tx, (sy-ty)*1.6)
					j := hash(x, y, 3)
					c.parts = append(c.parts, part{
						sx: sx, sy: sy, tx: tx, ty: ty, band: bandOf(y),
						go_: time.Duration((float64(x)/wordW*1100 + j*250) * float64(time.Millisecond)),
						dur: time.Duration(min(900, 380+dist*6) * float64(time.Millisecond)),
					})
					n++
				}
			}
		}
	}
	return c
}

func (c *compile) done() bool { return time.Since(c.start) >= compileTotal }

// built is true once every cell has landed.
func (c *compile) built() bool { return time.Since(c.start) >= compileBuild }

func (c *compile) String() string {
	t := time.Since(c.start)
	cv := newCanvas(c.w, c.h)
	// The field: a sparse noise that twinkles, dimmer as the word fills.
	fade := 1 - min(1, float64(t)/float64(compileBuild))
	tick := int(t / (180 * time.Millisecond))
	for y := 0; y < c.h; y++ {
		for x := 0; x < c.w; x++ {
			if v := hash(x, y, tick); v < 0.035*fade+0.004 {
				cv.set(x, y, mix(fDim(), fMid(), hash(x, y, tick+1)))
			}
		}
	}
	ease := func(u float64) float64 { return u * u * (3 - 2*u) }
	for i, p := range c.parts {
		switch {
		case t < p.go_:
			// Waiting in the field: a few of the waiting cells are lit at a
			// time, so the field reads as noise rather than a crowd.
			if hash(i, tick, 5) < 0.12 {
				cv.set(int(p.sx), int(p.sy), fMid())
			}
		case t < p.go_+p.dur:
			u := ease(float64(t-p.go_) / float64(p.dur))
			cv.set(int(p.sx+(p.tx-p.sx)*u+0.5), int(p.sy+(p.ty-p.sy)*u+0.5), mix(fLit(), fHover(), u))
		default:
			col := p.band
			if since := t - p.go_ - p.dur; since < 200*time.Millisecond {
				col = mix(fCrest(), p.band, float64(since)/float64(200*time.Millisecond))
			}
			cv.set(int(p.tx), int(p.ty), col)
		}
	}
	return cv.String()
}
