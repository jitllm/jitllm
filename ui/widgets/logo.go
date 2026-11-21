package widgets

import (
	"image"
	"image/color"
	"math"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"
)

// The jitllm mark, as the website draws it: the word in pixels, banded from a bright
// crest at the top to dim at the bottom. Its first glyph alone is the "j" of
// the favicon.
const (
	markW = 53
	markH = 21
	// glyphW is how wide the "j" is: the rectangles that start left of it.
	glyphW = 6
)

// markRects() are x, y, width, height in mark pixels.
func markRects() [][4]int {
	return [][4]int{
		{3, 0, 3, 3}, {3, 5, 3, 15}, {0, 18, 5, 3}, {9, 0, 3, 3}, {9, 5, 3, 12},
		{17, 1, 3, 16}, {15, 5, 8, 3}, {20, 14, 3, 3}, {26, 0, 3, 17}, {32, 0, 3, 17},
		{38, 5, 14, 3}, {38, 5, 3, 12}, {44, 5, 3, 12}, {50, 6, 3, 11},
	}
}

// markBand is the band a row of the mark is painted in: 0 the crest, 4 dim.
func markBand(y int) int {
	for b, n := range [...]int{5, 3, 5, 4} {
		if y < n {
			return b
		}
		y -= n
	}
	return 4
}

// markRuns calls f once per horizontal run of the mark, glyph only or whole.
func markRuns(glyph bool, f func(x, y, n, band int)) {
	w := markW
	if glyph {
		w = glyphW
	}
	var on [markH][markW]bool
	for _, r := range markRects() {
		for y := r[1]; y < r[1]+r[3]; y++ {
			for x := r[0]; x < r[0]+r[2] && x < w; x++ {
				on[y][x] = true
			}
		}
	}
	for y := range markH {
		for x := 0; x < w; x++ {
			if !on[y][x] {
				continue
			}
			n := 1
			for x+n < w && on[y][x+n] {
				n++
			}
			f(x, y, n, markBand(y))
			x += n - 1
		}
	}
}

// Logo is the mark at the head of the sidebar: the word on a wide sidebar,
// the "j" on a rail too narrow for it.
type Logo struct {
	widget.WidgetBase
	bands [5]widget.Color
	px    float32
}

// NewLogo draws the mark at px screen pixels per mark pixel, in bands from
// crest to dim.
func NewLogo(bands [5]widget.Color, px float32) *Logo {
	l := &Logo{bands: bands, px: px}
	l.SetVisible(true)
	return l
}

func (l *Logo) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	sz := c.Constrain(geometry.Sz(c.MaxWidth, markH*l.px))
	l.SetBounds(geometry.FromPointSize(l.Position(), sz))
	return sz
}

func (l *Logo) Draw(_ widget.Context, cv widget.Canvas) {
	r := l.Bounds()
	glyph := r.Width() < navCompactWidth
	// Whole pixels, so every run lands on the screen grid and stays crisp.
	x0 := float32(math.Round(float64(r.Min.X + 12)))
	if glyph {
		x0 = float32(math.Round(float64(r.Min.X + (r.Width()-glyphW*l.px)/2)))
	}
	y0 := float32(math.Round(float64(r.Min.Y)))
	markRuns(glyph, func(x, y, n, band int) {
		cv.DrawRect(geometry.NewRect(x0+float32(x)*l.px, y0+float32(y)*l.px, float32(n)*l.px, l.px), l.bands[band])
	})
}

func (l *Logo) Event(widget.Context, event.Event) bool { return false }
func (l *Logo) Children() []widget.Widget              { return nil }

// LogoIcon is the window icon: the favicon's "j" on its square, size pixels
// a side. The favicon is 23 mark pixels square with the glyph at (8, 1).
func LogoIcon(size int, bg color.Color, bands [5]color.Color) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			img.Set(x, y, bg)
		}
	}
	at := func(v int) int { return v * size / 23 }
	markRuns(true, func(x, y, n, band int) {
		for py := at(y + 1); py < at(y+2); py++ {
			for px := at(x + 8); px < at(x+8+n); px++ {
				img.Set(px, py, bands[band])
			}
		}
	})
	return img
}
