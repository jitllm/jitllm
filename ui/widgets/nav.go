package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// NavColors is a navigation row's palette.
type NavColors struct {
	Text, Muted, Active, ActiveBg, HoverBg widget.Color
}

// NavItem is one row of the sidebar: an icon and a label, highlighted while
// its page is the one shown, and a click away from it.
type NavItem struct {
	widget.WidgetBase

	icon     Icon
	label    string
	selected state.ReadonlySignal[bool]
	onClick  func()
	colors   NavColors
	size     float32
	pressState
}

// NewNavItem builds a row. selected may be nil for an action row.
func NewNavItem(icon Icon, label string, selected state.ReadonlySignal[bool], onClick func(), colors NavColors, size float32) *NavItem {
	var _ widget.Lifecycle = (*NavItem)(nil)
	n := &NavItem{icon: icon, label: label, selected: selected, onClick: onClick, colors: colors, size: size}
	n.SetVisible(true)
	n.SetEnabled(true)
	return n
}

const navRowH = 36

// Layout takes the sidebar's width and one row's height.
func (n *NavItem) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	sz := c.Constrain(geometry.Sz(c.MaxWidth, navRowH))
	n.SetBounds(geometry.FromPointSize(n.Position(), sz))
	return sz
}

func (n *NavItem) on() bool { return n.selected != nil && n.selected.Get() }

// Draw paints the highlight, the icon and the label.
func (n *NavItem) Draw(_ widget.Context, cv widget.Canvas) {
	r := n.Bounds()
	fg := n.colors.Muted
	switch {
	case n.on():
		cv.DrawRoundRect(r, n.colors.ActiveBg, 8)
		fg = n.colors.Active
	case n.hover:
		cv.DrawRoundRect(r, n.colors.HoverBg, 8)
		fg = n.colors.Text
	}
	const icon = 18
	// A rail too narrow for the label is a column of centred icons.
	compact := r.Width() < navCompactWidth
	ix := r.Min.X + 12
	if compact {
		ix = r.Min.X + (r.Width()-icon)/2
	}
	if svg, ok := cv.(widget.SVGRenderer); ok {
		svg.RenderSVG(n.icon, geometry.NewRect(ix, r.Min.Y+(r.Height()-icon)/2, icon, icon), fg)
	}
	if compact {
		return
	}
	lx := r.Min.X + 12 + icon + 12
	cv.DrawText(n.label, geometry.NewRect(lx, r.Min.Y, r.Max.X-lx-8, r.Height()), n.size, fg, n.on(), widget.TextAlignLeft)
}

// navCompactWidth is the row width below which the label is dropped.
const navCompactWidth = 100

// Event selects on a press inside the row and tracks the hover.
func (n *NavItem) Event(ctx widget.Context, e event.Event) bool {
	me, ok := e.(*event.MouseEvent)
	if !ok {
		return false
	}
	return n.pressState.handle(n, ctx, me, true, n.onClick)
}

// Children returns nil.
func (n *NavItem) Children() []widget.Widget { return nil }

// Mount repaints the row when its page is selected or left.
func (n *NavItem) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil && n.selected != nil {
		n.AddBinding(state.BindToScheduler(n.selected, n, sched))
	}
}

// Unmount has nothing to release; see MemoryMap.Unmount for why it exists.
func (n *NavItem) Unmount() {}

// ChatRow is one conversation in the sidebar: its title, highlighted while it
// is the one on screen. A rail too narrow for a title has no room for the
// list at all, so there the row takes no space.
type ChatRow struct {
	widget.WidgetBase

	label    state.ReadonlySignal[string]
	selected state.ReadonlySignal[bool]
	onClick  func()
	colors   NavColors
	size     float32
	pressState
}

// NewChatRow builds a row.
func NewChatRow(label state.ReadonlySignal[string], selected state.ReadonlySignal[bool], onClick func(), colors NavColors, size float32) *ChatRow {
	var _ widget.Lifecycle = (*ChatRow)(nil)
	r := &ChatRow{label: label, selected: selected, onClick: onClick, colors: colors, size: size}
	r.SetVisible(true)
	r.SetEnabled(true)
	return r
}

const chatRowH = 32

func (r *ChatRow) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	h := float32(chatRowH)
	if c.MaxWidth < navCompactWidth {
		h = 0
	}
	sz := c.Constrain(geometry.Sz(c.MaxWidth, h))
	r.SetBounds(geometry.FromPointSize(r.Position(), sz))
	return sz
}

func (r *ChatRow) Draw(_ widget.Context, cv widget.Canvas) {
	b := r.Bounds()
	if b.Height() == 0 {
		return
	}
	on := r.selected != nil && r.selected.Get()
	fg := r.colors.Muted
	switch {
	case on:
		cv.DrawRoundRect(b, r.colors.ActiveBg, 8)
		fg = r.colors.Text
	case r.hover:
		cv.DrawRoundRect(b, r.colors.HoverBg, 8)
		fg = r.colors.Text
	}
	text := r.label.Get()
	if text == "" {
		text = "New chat"
	}
	w := b.Width() - 24
	text = Elide(cv, text, r.size, w)
	cv.DrawText(text, geometry.NewRect(b.Min.X+12, b.Min.Y, w, b.Height()), r.size, fg, false, widget.TextAlignLeft)
}

// Elide cuts s with an ellipsis until it measures w or less.
func Elide(cv widget.Canvas, s string, size, w float32) string {
	if cv.MeasureText(s, size, false) <= w {
		return s
	}
	rs := []rune(s)
	for len(rs) > 0 {
		rs = rs[:len(rs)-1]
		if t := string(rs) + "\u2026"; cv.MeasureText(t, size, false) <= w {
			return t
		}
	}
	return ""
}

func (r *ChatRow) Event(ctx widget.Context, e event.Event) bool {
	me, ok := e.(*event.MouseEvent)
	if !ok || r.Bounds().Height() == 0 {
		return false
	}
	return r.pressState.handle(r, ctx, me, true, r.onClick)
}

func (r *ChatRow) Children() []widget.Widget { return nil }

// Mount repaints the row when its title or selection moves.
func (r *ChatRow) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil {
		r.AddBinding(state.BindToScheduler(r.label, r, sched))
		if r.selected != nil {
			r.AddBinding(state.BindToScheduler(r.selected, r, sched))
		}
	}
}

func (r *ChatRow) Unmount() {}

// NavHeading is a small muted label over a group of rows, absent in a rail.
type NavHeading struct {
	widget.WidgetBase
	text  string
	size  float32
	color widget.Color
}

// NewNavHeading builds a heading.
func NewNavHeading(text string, size float32, color widget.Color) *NavHeading {
	h := &NavHeading{text: text, size: size, color: color}
	h.SetVisible(true)
	return h
}

func (h *NavHeading) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	ht := h.size * 2
	if c.MaxWidth < navCompactWidth {
		ht = 0
	}
	sz := c.Constrain(geometry.Sz(c.MaxWidth, ht))
	h.SetBounds(geometry.FromPointSize(h.Position(), sz))
	return sz
}

func (h *NavHeading) Draw(_ widget.Context, cv widget.Canvas) {
	b := h.Bounds()
	if b.Height() == 0 {
		return
	}
	cv.DrawText(h.text, geometry.NewRect(b.Min.X+12, b.Min.Y, b.Width()-12, b.Height()), h.size, h.color, false, widget.TextAlignLeft)
}

func (h *NavHeading) Event(widget.Context, event.Event) bool { return false }
func (h *NavHeading) Children() []widget.Widget              { return nil }
