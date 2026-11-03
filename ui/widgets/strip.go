package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"
)

// Strip is a row that shows as many of its children as fit, in order, and
// drops the rest whole. An HBox lays out past its width and paints the
// overflow over whatever is beside it; a row of chips must never do that.
type Strip struct {
	widget.WidgetBase
	gap   float32
	all   []widget.Widget
	shown []widget.Widget
}

// NewStrip builds a strip.
func NewStrip(gap float32, children ...widget.Widget) *Strip {
	var _ widget.Lifecycle = (*Strip)(nil)
	s := &Strip{gap: gap, all: children}
	for _, ch := range children {
		if p, ok := ch.(interface{ SetParent(widget.Widget) }); ok {
			p.SetParent(s)
		}
	}
	s.SetVisible(true)
	s.SetEnabled(true)
	return s
}

func (s *Strip) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	s.shown = s.shown[:0]
	var x, h float32
	full := false
	for _, ch := range s.all {
		left := c.MaxWidth - x
		sz := widget.LayoutChild(ch, ctx, geometry.Constraints{MaxWidth: max(left, 0), MaxHeight: c.MaxHeight})
		if sz.Width == 0 {
			continue
		}
		// Measured against what is left, a child that does not fit comes back
		// squeezed to it; anything at the edge, or after one that was dropped,
		// goes.
		if full || sz.Width >= left-0.5 {
			full = true
			continue
		}
		if p, ok := ch.(interface{ SetPosition(geometry.Point) }); ok {
			p.SetPosition(geometry.Pt(x, 0))
		}
		if b, ok := ch.(interface{ SetBounds(geometry.Rect) }); ok {
			b.SetBounds(geometry.FromPointSize(geometry.Pt(x, 0), sz))
		}
		s.shown = append(s.shown, ch)
		x += sz.Width + s.gap
		h = max(h, sz.Height)
	}
	if len(s.shown) > 0 {
		x -= s.gap
	}
	sz := c.Constrain(geometry.Sz(x, h))
	s.SetBounds(geometry.FromPointSize(s.Position(), sz))
	return sz
}

func (s *Strip) Draw(ctx widget.Context, cv widget.Canvas) {
	cv.PushTransform(s.Bounds().Min)
	for _, ch := range s.shown {
		widget.StampScreenOrigin(ch, cv)
		widget.DrawChild(ch, ctx, cv)
	}
	cv.PopTransform()
}

func (s *Strip) Event(ctx widget.Context, e event.Event) bool {
	me, ok := e.(*event.MouseEvent)
	if !ok {
		return false
	}
	local := *me
	local.Position = me.Position.Sub(s.Bounds().Min)
	for _, ch := range s.shown {
		if ch.Event(ctx, &local) {
			return true
		}
	}
	return false
}

// Children is what is shown: a dropped chip is not there to hit or draw.
func (s *Strip) Children() []widget.Widget { return s.shown }

// Mount and Unmount reach every child, shown or not, so a dropped one keeps
// its bindings and can come back when there is room.
func (s *Strip) Mount(ctx widget.Context) {
	for _, ch := range s.all {
		widget.MountTree(ch, ctx)
	}
}

func (s *Strip) Unmount() {
	for _, ch := range s.all {
		widget.UnmountTree(ch)
	}
}
