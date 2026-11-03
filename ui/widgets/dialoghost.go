package widgets

import (
	"sync/atomic"

	"github.com/gogpu/ui/core/dialog"
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// DialogKind selects which shape of dialog a request raises.
type DialogKind uint8

// The dialog shapes. DialogNone is the zero value and raises nothing.
const (
	DialogNone DialogKind = iota
	DialogAlert
	DialogConfirm
)

// DialogReq is a request to raise a modal dialog.
//
// Seq is the trigger, not a boolean: a monotonic sequence can raise the same
// alert twice and needs nobody to clear it. Use [DialogSeq.Next] to fill it.
type DialogReq struct {
	Seq      uint64
	Kind     DialogKind
	Title    string
	Message  string
	OK       string // button label; "OK" when empty
	Cancel   string // button label; "Cancel" when empty
	OnOK     func()
	OnCancel func()
}

// DialogSeq numbers the requests one owner raises.
type DialogSeq struct{ n atomic.Uint64 }

// Next returns a fresh sequence number for a [DialogReq]. It is safe to call
// from any goroutine.
func (d *DialogSeq) Next() uint64 { return d.n.Add(1) }

// DialogHost wraps the whole root and raises a dialog whenever the request
// signal carries a sequence it has not shown yet.
//
// dialog.Widget.Show needs a widget.Context and button.OnClick has none;
// wrapping the root lets any goroutine raise a dialog with one signal Set.
// TickAnimation raises it every frame; Layout also pumps, for the offscreen
// renderer, which lays out and never ticks.
type DialogHost struct {
	widget.WidgetBase

	child   widget.Widget
	req     state.ReadonlySignal[DialogReq]
	painter dialog.Painter

	dlg   *dialog.Widget
	shown uint64
}

// NewDialogHost wraps child and watches req. painter may be nil, in which case
// the dialog package's default painter is used.
func NewDialogHost(child widget.Widget, req state.ReadonlySignal[DialogReq], painter dialog.Painter) *DialogHost {
	h := &DialogHost{child: child, req: req, painter: painter}
	if s, ok := child.(interface{ SetParent(widget.Widget) }); ok {
		s.SetParent(h)
	}
	h.SetVisible(true)
	h.SetEnabled(true)
	return h
}

// Layout measures the child and, first, raises any pending dialog.
func (h *DialogHost) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	h.pump(ctx)
	return widget.LayoutChild(h.child, ctx, c)
}

// Draw renders the child at the host's own bounds. The dialog itself is drawn
// by the overlay stack, not from here.
func (h *DialogHost) Draw(ctx widget.Context, canvas widget.Canvas) {
	setBounds(h.child, h.Bounds())
	widget.StampScreenOrigin(h.child, canvas)
	h.child.Draw(ctx, canvas)
}

// Event forwards to the child.
func (h *DialogHost) Event(ctx widget.Context, e event.Event) bool {
	return h.child.Event(ctx, e)
}

// Children returns the wrapped child. The dialog is deliberately absent: while
// open it lives in the window's overlay stack, and dialog.Widget.Draw is a
// no-op, so including it would add a zero-sized child for nothing.
func (h *DialogHost) Children() []widget.Widget { return []widget.Widget{h.child} }

// Mount mounts the wrapped subtree.
func (h *DialogHost) Mount(ctx widget.Context) { widget.MountTree(h.child, ctx) }

// TickAnimation raises any pending dialog. The window calls it on the UI
// goroutine at the start of every frame, before layout.
//
// Layout alone is not enough: a cached root is not laid out again until
// something under it changes. A frame hook rather than a subscription reads
// the signal on the UI goroutine (a subscriber runs on the setter's) and also
// raises a request that waited behind an open dialog once that one closes.
func (h *DialogHost) TickAnimation(ctx widget.Context) { h.pump(ctx) }

// Unmount unmounts the wrapped subtree.
func (h *DialogHost) Unmount() { widget.UnmountTree(h.child) }

func (h *DialogHost) pump(ctx widget.Context) {
	if h.req == nil {
		return
	}
	r := h.req.Get()
	if r.Seq == 0 || r.Seq == h.shown || r.Kind == DialogNone {
		return
	}
	if h.dlg != nil && h.dlg.IsOpen() {
		return
	}
	h.shown = r.Seq
	h.dlg = h.build(ctx, r)
	h.dlg.Show(ctx)
}

// dialogBodyLines is what core/dialog has room for: its height is a constant
// (title + actions + 2*padding, a 48 px band for content) and it never lays its
// content out, so a third line would be drawn over the buttons.
//
// Longer messages are ellipsised; UPSTREAM.md #13 is the real fix.
const dialogBodyLines = 2

// dialogWidth is the dialog's MaxWidth for a message: 520, or wide when that
// would cut the message. core/dialog grows sideways (to 90% of the window) and
// never down, so width is the only room there is.
func dialogWidth(msg string) float32 {
	cols := func(w float32) int { return int((w - 2*24) / (14 * charWidthRatio)) } // 24: core/dialog's contentPadding; 14: Paragraph's default size
	if len(WrapText(msg, cols(520))) <= dialogBodyLines {
		return 520
	}
	return 1000
}

func (h *DialogHost) build(ctx widget.Context, r DialogReq) *dialog.Widget {
	okLabel, cancelLabel := r.OK, r.Cancel
	if okLabel == "" {
		okLabel = "OK"
	}
	if cancelLabel == "" {
		cancelLabel = "Cancel"
	}

	actions := make([]dialog.Action, 0, 2)
	if r.Kind == DialogConfirm {
		actions = append(actions, dialog.Action{Label: cancelLabel, OnClick: r.OnCancel})
	}
	actions = append(actions, dialog.Action{Label: okLabel, OnClick: r.OnOK})

	// A Paragraph, because primitives.Text measures as if it wrapped and
	// paints one line.
	// Selectable, so an error can be copied out of the dialog.
	body := NewParagraph(r.Message).MaxLines(dialogBodyLines).Selectable()
	if tp := ctx.ThemeProvider(); tp != nil {
		body = body.Color(tp.OnSurface())
	}

	opts := []dialog.Option{
		dialog.Title(r.Title),
		dialog.Content(body),
		dialog.Actions(actions...),
		dialog.MaxWidth(dialogWidth(r.Message)),
	}
	if h.painter != nil {
		opts = append(opts, dialog.PainterOpt(h.painter))
	}
	return dialog.New(opts...)
}

// setBounds positions a child that embeds widget.WidgetBase. A child that does
// not is left where it is rather than silently mispositioned.
func setBounds(w widget.Widget, r geometry.Rect) {
	if s, ok := w.(interface{ SetBounds(geometry.Rect) }); ok {
		s.SetBounds(r)
	}
}
