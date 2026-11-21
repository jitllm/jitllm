package widgets

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// IconButtonColors is an icon button's palette. Fill is the disc a filled
// button sits on, and On the icon drawn on it.
type IconButtonColors struct {
	Icon, Hover, Disabled, Fill, On widget.Color
}

// IconButton is a square, borderless button showing one icon: a ghost that
// lights on hover, or -- Filled -- a disc in the accent colour, for the one
// action a region is for (Send).
type IconButton struct {
	widget.WidgetBase

	icon     Icon
	onClick  func()
	disabled state.ReadonlySignal[bool]
	colors   IconButtonColors
	filled   bool
	size     float32
	pressState
	a11y string
}

// NewIconButton builds a ghost button; label names it for assistive tools.
func NewIconButton(icon Icon, label string, onClick func(), colors IconButtonColors) *IconButton {
	var _ widget.Lifecycle = (*IconButton)(nil)
	b := &IconButton{icon: icon, onClick: onClick, colors: colors, size: 32, a11y: label}
	b.SetVisible(true)
	b.SetEnabled(true)
	return b
}

// Filled makes it the region's primary action: a disc in the accent colour.
func (b *IconButton) Filled() *IconButton { b.filled = true; return b }

// Size sets the square's side.
func (b *IconButton) Size(px float32) *IconButton { b.size = px; return b }

// DisabledSignal greys the button out and ignores clicks while it is true.
func (b *IconButton) DisabledSignal(sig state.ReadonlySignal[bool]) *IconButton {
	b.disabled = sig
	return b
}

// Label is what the button does, for assistive tools.
func (b *IconButton) Label() string { return b.a11y }

func (b *IconButton) off() bool { return b.disabled != nil && b.disabled.Get() }

func (b *IconButton) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	sz := c.Constrain(geometry.Sz(b.size, b.size))
	b.SetBounds(geometry.FromPointSize(b.Position(), sz))
	return sz
}

func (b *IconButton) Draw(_ widget.Context, cv widget.Canvas) {
	r := b.Bounds()
	fg := b.colors.Icon
	switch {
	case b.filled && b.off():
		cv.DrawRoundRect(r, b.colors.Hover, r.Width()/2)
		fg = b.colors.Disabled
	case b.filled:
		cv.DrawRoundRect(r, b.colors.Fill, r.Width()/2)
		fg = b.colors.On
	case b.off():
		fg = b.colors.Disabled
	case b.hover:
		cv.DrawRoundRect(r, b.colors.Hover, 8)
	}
	icon := r.Width() * 0.55
	if svg, ok := cv.(widget.SVGRenderer); ok {
		svg.RenderSVG(b.icon, geometry.NewRect(r.Min.X+(r.Width()-icon)/2, r.Min.Y+(r.Height()-icon)/2, icon, icon), fg)
	}
}

func (b *IconButton) Event(ctx widget.Context, e event.Event) bool {
	me, ok := e.(*event.MouseEvent)
	if !ok {
		return false
	}
	return b.pressState.handle(b, ctx, me, !b.off(), b.onClick)
}

func (b *IconButton) Children() []widget.Widget { return nil }

// Mount repaints the button when it is enabled or disabled.
func (b *IconButton) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil && b.disabled != nil {
		b.AddBinding(state.BindToScheduler(b.disabled, b, sched))
	}
}

func (b *IconButton) Unmount() {}

// Status is what a status pill says: a label and the colour of its dot.
type Status struct {
	Label string
	Dot   widget.Color
}

// StatusPill is a dot and a word: what the engine is doing, at a glance.
type StatusPill struct {
	widget.WidgetBase
	sig  state.ReadonlySignal[Status]
	text widget.Color
	size float32
}

// NewStatusPill builds a pill bound to sig.
func NewStatusPill(sig state.ReadonlySignal[Status], text widget.Color, size float32) *StatusPill {
	var _ widget.Lifecycle = (*StatusPill)(nil)
	p := &StatusPill{sig: sig, text: text, size: size}
	p.SetVisible(true)
	return p
}

func (p *StatusPill) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	// The label's width is not measurable here; the pill takes a fixed slot
	// wide enough for the longest state, and draws from its left.
	sz := c.Constrain(geometry.Sz(statusPillWidth, 28))
	p.SetBounds(geometry.FromPointSize(p.Position(), sz))
	return sz
}

const statusPillWidth = 170

func (p *StatusPill) Draw(_ widget.Context, cv widget.Canvas) {
	st := p.sig.Get()
	if st.Label == "" {
		return
	}
	r := p.Bounds()
	cy := r.Min.Y + r.Height()/2
	cv.DrawCircle(geometry.Pt(r.Min.X+5, cy), 4, st.Dot)
	cv.DrawText(st.Label, geometry.NewRect(r.Min.X+16, r.Min.Y, r.Width()-16, r.Height()), p.size, p.text, false, widget.TextAlignLeft)
}

func (p *StatusPill) Event(widget.Context, event.Event) bool { return false }
func (p *StatusPill) Children() []widget.Widget              { return nil }

func (p *StatusPill) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil {
		p.AddBinding(state.BindToScheduler(p.sig, p, sched))
	}
}

func (p *StatusPill) Unmount() {}
