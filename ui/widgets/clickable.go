package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// Clickable is a row a person picks: its child on a rounded surface that
// lights on hover, holds a tint while selected, and calls onClick on a press.
type Clickable struct {
	widget.WidgetBase
	child    widget.Widget
	selected state.ReadonlySignal[bool]
	onClick  func()
	hoverBg  widget.Color
	selBg    widget.Color
	selLine  widget.Color
	pressState
}

// NewClickable wraps child. selected may be nil.
func NewClickable(child widget.Widget, selected state.ReadonlySignal[bool], onClick func(), hoverBg, selBg, selLine widget.Color) *Clickable {
	var _ widget.Lifecycle = (*Clickable)(nil)
	c := &Clickable{child: child, selected: selected, onClick: onClick, hoverBg: hoverBg, selBg: selBg, selLine: selLine}
	if s, ok := child.(interface{ SetParent(widget.Widget) }); ok {
		s.SetParent(c)
	}
	c.SetVisible(true)
	c.SetEnabled(true)
	return c
}

func (c *Clickable) on() bool { return c.selected != nil && c.selected.Get() }

func (c *Clickable) Layout(ctx widget.Context, cons geometry.Constraints) geometry.Size {
	sz := widget.LayoutChild(c.child, ctx, cons)
	c.SetBounds(geometry.FromPointSize(c.Position(), sz))
	return sz
}

func (c *Clickable) Draw(ctx widget.Context, cv widget.Canvas) {
	r := c.Bounds()
	switch {
	case c.on():
		cv.DrawRoundRect(r, c.selBg, 10)
		cv.StrokeRoundRect(r, c.selLine, 10, 1)
	case c.hover:
		cv.DrawRoundRect(r, c.hoverBg, 10)
	}
	if s, ok := c.child.(interface{ SetBounds(geometry.Rect) }); ok {
		s.SetBounds(r)
	}
	widget.StampScreenOrigin(c.child, cv)
	c.child.Draw(ctx, cv)
}

func (c *Clickable) Event(ctx widget.Context, e event.Event) bool {
	me, ok := e.(*event.MouseEvent)
	if !ok {
		return false
	}
	return c.pressState.handle(c, ctx, me, true, c.onClick)
}

func (c *Clickable) Children() []widget.Widget { return []widget.Widget{c.child} }

func (c *Clickable) Mount(ctx widget.Context) {
	widget.MountTree(c.child, ctx)
	if sched := ctx.Scheduler(); sched != nil && c.selected != nil {
		c.AddBinding(state.BindToScheduler(c.selected, c, sched))
	}
}

func (c *Clickable) Unmount() { widget.UnmountTree(c.child) }

// Tile is a rounded square holding an icon or a letter: a model's maker as
// its initial, the machine as a chip.
type Tile struct {
	widget.WidgetBase
	icon   Icon
	letter string
	size   float32
	bg, fg widget.Color
	// look, when bound, supplies the letter and background per frame: a tile
	// in a row whose model moves when the ranking changes.
	look state.ReadonlySignal[TileLook]
}

// TileLook is a bound tile's letter and background.
type TileLook struct {
	Letter string
	Bg     widget.Color
}

// Bind makes the tile draw look's letter and background.
func (t *Tile) Bind(look state.ReadonlySignal[TileLook]) *Tile { t.look = look; return t }

// Mount repaints a bound tile when its look changes.
func (t *Tile) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil && t.look != nil {
		t.AddBinding(state.BindToScheduler(t.look, t, sched))
	}
}

func (t *Tile) Unmount() {}

// NewIconTile is a tile showing icon.
func NewIconTile(icon Icon, size float32, bg, fg widget.Color) *Tile {
	t := &Tile{icon: icon, size: size, bg: bg, fg: fg}
	t.SetVisible(true)
	return t
}

// NewLetterTile is a tile showing one letter.
func NewLetterTile(letter string, size float32, bg, fg widget.Color) *Tile {
	t := &Tile{letter: letter, size: size, bg: bg, fg: fg}
	t.SetVisible(true)
	return t
}

func (t *Tile) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	sz := c.Constrain(geometry.Sz(t.size, t.size))
	t.SetBounds(geometry.FromPointSize(t.Position(), sz))
	return sz
}

func (t *Tile) Draw(_ widget.Context, cv widget.Canvas) {
	r := t.Bounds()
	letter, bg := t.letter, t.bg
	if t.look != nil {
		l := t.look.Get()
		letter, bg = l.Letter, l.Bg
	}
	cv.DrawRoundRect(r, bg, r.Width()*0.25)
	if letter != "" {
		cv.DrawText(letter, r, r.Width()*0.45, t.fg, true, widget.TextAlignCenter)
		return
	}
	in := r.Width() * 0.5
	if svg, ok := cv.(widget.SVGRenderer); ok {
		svg.RenderSVG(t.icon, geometry.NewRect(r.Min.X+(r.Width()-in)/2, r.Min.Y+(r.Height()-in)/2, in, in), t.fg)
	}
}

func (t *Tile) Event(widget.Context, event.Event) bool { return false }
func (t *Tile) Children() []widget.Widget              { return nil }
