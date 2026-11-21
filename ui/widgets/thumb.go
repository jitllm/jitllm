package widgets

import (
	"fmt"
	"image"
	_ "image/jpeg" // the two formats model.Preprocess reads
	_ "image/png"
	"os"
	"path/filepath"
	"sync"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"
	xdraw "golang.org/x/image/draw"
)

// Thumb is a picture scaled to fit a square box, drawn with the canvas's real
// DrawImage. primitives.Image only draws a placeholder (UPSTREAM.md #12), and
// DrawImage takes a point and blits at native size, so the picture is scaled
// once on the CPU when the widget is built.
type Thumb struct {
	widget.WidgetBase
	img *image.RGBA // nil when the file could not be read
	err string
	box float32
}

// New reads path and scales it to fit a box of edge px. A file that does
// not decode becomes a labelled placeholder rather than an empty space, because
// an attachment that silently shows nothing reads as one that was never made.
func (c *Thumbs) New(path string, px int) *Thumb {
	t := &Thumb{box: float32(px)}
	t.SetVisible(true)
	t.SetEnabled(true)
	t.img, t.err = c.of(path, px)
	return t
}

// Err is why the picture could not be shown, or "".
func (t *Thumb) Err() string { return t.err }

// Layout takes the scaled picture's size, or the whole box for a placeholder.
func (t *Thumb) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	w, h := t.box, t.box
	if t.img != nil {
		b := t.img.Bounds()
		w, h = float32(b.Dx()), float32(b.Dy())
	}
	sz := c.Constrain(geometry.Sz(w, h))
	t.SetBounds(geometry.FromPointSize(t.Position(), sz))
	return sz
}

// Draw blits the picture at the widget's origin.
func (t *Thumb) Draw(_ widget.Context, canvas widget.Canvas) {
	if !t.IsVisible() {
		return
	}
	r := t.Bounds()
	if t.img == nil {
		canvas.DrawRect(r, widget.RGBA(0.5, 0.5, 0.5, 0.25))
		canvas.DrawText(t.err, r, 11, widget.RGBA(0.5, 0.5, 0.5, 1), false, widget.TextAlignCenter)
		return
	}
	canvas.DrawImage(t.img, r.Min)
}

// Event is false: a thumbnail takes no input.
func (t *Thumb) Event(widget.Context, event.Event) bool { return false }

// Children is nil: a thumbnail is a leaf.
func (t *Thumb) Children() []widget.Widget { return nil }

// Thumbs caches scaled pictures by (path, edge), since the transcript rebuilds
// every bubble when the turn count moves. Decoding happens on the UI goroutine
// the first time a picture is shown.
// Unbounded and keyed by path; a session attaches a handful. Evict by
// size, and decode off the UI goroutine, if large photos ever stutter.
type Thumbs struct{ m sync.Map }

type thumbKey struct {
	path string
	px   int
}

type thumbVal struct {
	img *image.RGBA
	err string
}

func (c *Thumbs) of(path string, px int) (*image.RGBA, string) {
	k := thumbKey{path, px}
	if v, ok := c.m.Load(k); ok {
		tv := v.(thumbVal)
		return tv.img, tv.err
	}
	img, err := scaleFile(path, px)
	tv := thumbVal{img: img}
	if err != nil {
		tv.err = fmt.Sprintf("%s: %v", filepath.Base(path), err)
	}
	c.m.Store(k, tv)
	return tv.img, tv.err
}

// scaleFile decodes a picture and scales it so its longer side is px, never
// enlarging one that is already smaller.
func scaleFile(path string, px int) (*image.RGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("empty image")
	}
	if m := max(w, h); m > px {
		w, h = max(1, w*px/m), max(1, h*px/m)
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), src, b, xdraw.Src, nil)
	return dst, nil
}
