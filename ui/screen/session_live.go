package screen

import (
	"github.com/gogpu/ui/a11y"
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// Widgets a streaming view needs that gogpu/ui does not ship. They live here
// rather than in widgets/ until a second consumer exists.

// ---------------------------------------------------------------------------

// LiveText is a text widget bound to a string signal that re-measures when
// the string changes. primitives.Text.ContentSignal binds with the paint-only
// state.BindToScheduler, and widget.LayoutChild caches the measured size, so a
// bound Text repaints inside the bounds it had when empty and clips every
// token after the first. LiveText binds with state.BindToSchedulerLayout on a
// widget that has a cached layout, so the invalidation reaches the root and a
// real layout pass runs.
type LiveText struct {
	widget.WidgetBase

	sig   state.ReadonlySignal[string]
	child *primitives.TextWidget
}

// NewLiveText wraps a text widget already bound to sig. Build the child with
// primitives.Text("").ContentSignal(sig); the signal is passed twice because
// the wrapper invalidates layout on it and the child reads it.
func NewLiveText(sig state.ReadonlySignal[string], child *primitives.TextWidget) *LiveText {
	var _ widget.Lifecycle = (*LiveText)(nil)
	w := &LiveText{sig: sig, child: child}
	w.SetVisible(true)
	w.SetEnabled(true)
	if child != nil {
		child.SetParent(w)
	}
	return w
}

// Layout measures the child through the layout cache and adopts its size.
func (w *LiveText) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	if w.child == nil {
		return geometry.Size{}
	}
	size := widget.LayoutChild(w.child, ctx, c)
	w.child.SetBounds(geometry.FromPointSize(geometry.Point{}, size))
	w.SetBounds(geometry.FromPointSize(w.Position(), size))
	return size
}

// Draw delegates to the child at this widget's origin.
func (w *LiveText) Draw(ctx widget.Context, canvas widget.Canvas) {
	if !w.IsVisible() || w.child == nil {
		return
	}
	b := w.Bounds()
	canvas.PushTransform(b.Min)
	widget.StampScreenOrigin(w.child, canvas)
	w.child.Draw(ctx, canvas)
	canvas.PopTransform()
}

// Event ignores input; a readout takes none.
func (w *LiveText) Event(_ widget.Context, _ event.Event) bool { return false }

// Children returns the wrapped text.
func (w *LiveText) Children() []widget.Widget {
	if w.child == nil {
		return nil
	}
	return []widget.Widget{w.child}
}

// Mount binds the signal so a change invalidates the layout and not only the
// paint; see the type comment.
func (w *LiveText) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil {
		w.AddBinding(state.BindToSchedulerLayout(w.sig, w, sched))
	}
}

// Unmount must exist: widget.MountTree calls Mount only on a full Lifecycle.
func (w *LiveText) Unmount() {}

// AccessibilityRole reports the wrapper as plain text.
func (w *LiveText) AccessibilityRole() a11y.Role { return a11y.RoleStaticText }

// AccessibilityLabel returns the live string.
func (w *LiveText) AccessibilityLabel() string { return w.sig.Get() }

// ---------------------------------------------------------------------------
// TailBox
// ---------------------------------------------------------------------------

// TailBox shows the bottom of a single child that is taller than the space
// available, clipping the top. It anchors at layout time instead of using a
// scroll controller: scrollview.Draw reads its offset unclamped, and
// listview.ScrollToIndex is a widget mutator the engine goroutine may not
// call. The finished turn goes into the scrollable transcript, so nothing is
// lost by clipping the top here.
type TailBox struct {
	widget.WidgetBase

	child widget.Widget
	max   float32

	content geometry.Size
}

// NewTailBox anchors child to the bottom, capped at maxHeight logical pixels.
// A maxHeight of 0 means "as tall as the parent allows".
func NewTailBox(child widget.Widget, maxHeight float32) *TailBox {
	t := &TailBox{child: child, max: maxHeight}
	t.SetVisible(true)
	t.SetEnabled(true)
	if ps, ok := child.(interface{ SetParent(widget.Widget) }); ok {
		ps.SetParent(t)
	}
	return t
}

// Layout measures the child with unbounded height and reports the capped one,
// so the box collapses to nothing when the child is empty.
func (t *TailBox) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	if t.child == nil {
		t.content = geometry.Size{}
		t.SetBounds(geometry.FromPointSize(t.Position(), geometry.Size{}))
		return geometry.Size{}
	}

	cc := geometry.Constraints{
		MinWidth:  c.MinWidth,
		MaxWidth:  c.MaxWidth,
		MinHeight: 0,
		MaxHeight: geometry.Infinity,
	}
	t.content = widget.LayoutChild(t.child, ctx, cc)
	if setter, ok := t.child.(interface{ SetBounds(geometry.Rect) }); ok {
		setter.SetBounds(geometry.NewRect(0, 0, t.content.Width, t.content.Height))
	}

	h := t.content.Height
	if t.max > 0 && h > t.max {
		h = t.max
	}
	size := c.Constrain(geometry.Sz(t.content.Width, h))
	t.SetBounds(geometry.FromPointSize(t.Position(), size))
	return size
}

// Draw clips to the box and offsets the child up by whatever it overflows by.
func (t *TailBox) Draw(ctx widget.Context, canvas widget.Canvas) {
	if !t.IsVisible() || t.child == nil {
		return
	}
	b := t.Bounds()
	if b.IsEmpty() {
		return
	}
	// A positive overflow scrolls the child up, which shows its tail.
	off := t.content.Height - b.Height()
	if off < 0 {
		off = 0
	}
	canvas.PushClip(b)
	canvas.PushTransform(geometry.Pt(b.Min.X, b.Min.Y-off))
	widget.StampScreenOrigin(t.child, canvas)
	t.child.Draw(ctx, canvas)
	canvas.PopTransform()
	canvas.PopClip()
}

// TailOffset reports how far the child is scrolled up, which is 0 until it
// overflows. It exists so the anchoring can be asserted rather than eyeballed.
func (t *TailBox) TailOffset() float32 {
	off := t.content.Height - t.Bounds().Height()
	if off < 0 {
		return 0
	}
	return off
}

// Event translates mouse and wheel positions into the child's space.
func (t *TailBox) Event(ctx widget.Context, e event.Event) bool {
	if !t.IsVisible() || !t.IsEnabled() || t.child == nil {
		return false
	}
	shift := geometry.Pt(t.Bounds().Min.X, t.Bounds().Min.Y-t.TailOffset())
	switch ev := e.(type) {
	case *event.MouseEvent:
		local := *ev
		local.Position = ev.Position.Sub(shift)
		return t.child.Event(ctx, &local)
	case *event.WheelEvent:
		local := *ev
		local.Position = ev.Position.Sub(shift)
		return t.child.Event(ctx, &local)
	}
	return t.child.Event(ctx, e)
}

// Children returns the single child.
func (t *TailBox) Children() []widget.Widget {
	if t.child == nil {
		return nil
	}
	return []widget.Widget{t.child}
}

// ---------------------------------------------------------------------------
// signalWatcher
// ---------------------------------------------------------------------------

// signalWatcher runs fn whenever sig changes. It draws nothing and takes no
// space. Mounting the subscription on a widget ties it to the tree's
// lifetime; a bare SubscribeForever in a ScreenFunc would leak one callback
// per theme swap. fn runs on whichever goroutine wrote the signal (often the
// engine worker), so it must only hand work to Shell.Post.
type signalWatcher[T any] struct {
	widget.WidgetBase

	sig   state.ReadonlySignal[T]
	fn    func(T)
	unsub state.Unsubscribe
}

func newSignalWatcher[T any](sig state.ReadonlySignal[T], fn func(T)) *signalWatcher[T] {
	w := &signalWatcher[T]{sig: sig, fn: fn}
	w.SetVisible(false)
	w.SetEnabled(false)
	return w
}

func (w *signalWatcher[T]) Layout(_ widget.Context, _ geometry.Constraints) geometry.Size {
	return geometry.Size{}
}
func (w *signalWatcher[T]) Draw(_ widget.Context, _ widget.Canvas)     {}
func (w *signalWatcher[T]) Event(_ widget.Context, _ event.Event) bool { return false }
func (w *signalWatcher[T]) Children() []widget.Widget                  { return nil }

func (w *signalWatcher[T]) Mount(_ widget.Context) {
	if w.fn == nil || w.unsub != nil {
		return
	}
	w.unsub = w.sig.SubscribeForever(func(v T) { w.fn(v) })
}

// Unmount drops the subscription. state.Binding cannot carry it (no
// constructor from an Unsubscribe), so this is done by hand.
func (w *signalWatcher[T]) Unmount() {
	if w.unsub != nil {
		w.unsub()
		w.unsub = nil
	}
}

// keyWatcher is a zero-size widget that claims one key. It works because
// primitives.Box broadcasts non-mouse events to every child in reverse order,
// so a watcher placed last is offered the key before the composer's text
// field; gogpu/ui has no global key handler. It consumes the key only when it
// acts, so Escape still reaches dialogs and menus when nothing is generating.
type keyWatcher struct {
	widget.WidgetBase
	key event.Key
	fn  func() bool
}

func newKeyWatcher(k event.Key, fn func() bool) *keyWatcher {
	w := &keyWatcher{key: k, fn: fn}
	w.SetVisible(false)
	w.SetEnabled(false)
	return w
}

func (w *keyWatcher) Layout(_ widget.Context, _ geometry.Constraints) geometry.Size {
	return geometry.Size{}
}
func (w *keyWatcher) Draw(_ widget.Context, _ widget.Canvas) {}
func (w *keyWatcher) Children() []widget.Widget              { return nil }

func (w *keyWatcher) Event(_ widget.Context, e event.Event) bool {
	ke, ok := e.(*event.KeyEvent)
	if !ok || ke.KeyType != event.KeyPress || ke.Key != w.key || w.fn == nil {
		return false
	}
	return w.fn()
}
