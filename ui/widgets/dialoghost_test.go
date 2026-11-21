package widgets

import (
	"strings"
	"sync"
	"testing"

	uiapp "github.com/gogpu/ui/app"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

// recorder counts what reached the overlay stack. The DialogHost's whole claim
// is that a signal Set from any goroutine ends in exactly one PushOverlay, and
// nothing smaller than this can check it.
type recorder struct {
	mu     sync.Mutex
	pushed int
	last   widget.Widget
}

func (r *recorder) PushOverlay(w widget.Widget, _ func()) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pushed++
	r.last = w
}
func (r *recorder) PopOverlay()                   {}
func (r *recorder) RemoveOverlay(_ widget.Widget) {}
func (r *recorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pushed
}

func hostAndCtx(sig state.Signal[DialogReq]) (*DialogHost, *uitest.MockContext, *recorder) {
	h := NewDialogHost(primitives.Text("body"), sig, nil)
	ctx := uitest.NewMockContext()
	rec := &recorder{}
	ctx.OverlayVal = rec
	return h, ctx, rec
}

func layout(h *DialogHost, ctx widget.Context) {
	h.Layout(ctx, geometry.Constraints{MaxWidth: 800, MaxHeight: 600})
}

func TestDialogHostRaisesADialogSetFromAnotherGoroutine(t *testing.T) {
	var seq DialogSeq
	sig := state.NewSignal(DialogReq{})
	h, ctx, rec := hostAndCtx(sig)

	layout(h, ctx)
	if rec.count() != 0 {
		t.Fatal("an empty request must raise nothing")
	}

	// The point of the whole widget: a worker raises a dialog with one Set.
	done := make(chan struct{})
	go func() {
		sig.Set(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "load failed", Message: "no such file"})
		close(done)
	}()
	<-done

	layout(h, ctx)
	if got := rec.count(); got != 1 {
		t.Fatalf("PushOverlay called %d times, want 1", got)
	}
}

func TestDialogHostDoesNotReRaiseTheSameRequestEveryFrame(t *testing.T) {
	var seq DialogSeq
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "t"})
	h, ctx, rec := hostAndCtx(sig)

	for i := 0; i < 10; i++ {
		layout(h, ctx)
	}
	if got := rec.count(); got != 1 {
		t.Fatalf("ten frames raised %d dialogs, want 1 -- Layout runs every frame", got)
	}
}

func TestDialogHostIgnoresASeqOfZeroAndKindNone(t *testing.T) {
	var seq DialogSeq
	// Seq 0 is the zero value: "nothing has ever been requested".
	sig := state.NewSignal(DialogReq{Kind: DialogAlert, Title: "t"})
	h, ctx, rec := hostAndCtx(sig)
	layout(h, ctx)
	if rec.count() != 0 {
		t.Fatal("a request with Seq 0 must not raise: the zero value is not a request")
	}

	// A sequence with no kind is equally not a request.
	sig.Set(DialogReq{Seq: seq.Next()})
	layout(h, ctx)
	if rec.count() != 0 {
		t.Fatal("DialogNone must not raise")
	}
}

func TestDialogHostWillNotStackASecondDialogOverAnOpenOne(t *testing.T) {
	var seq DialogSeq
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "first"})
	h, ctx, rec := hostAndCtx(sig)
	layout(h, ctx)

	sig.Set(DialogReq{Seq: seq.Next(), Kind: DialogConfirm, Title: "second"})
	layout(h, ctx)

	if got := rec.count(); got != 1 {
		t.Fatalf("a second request while one is open raised %d dialogs, want 1", got)
	}
}

func TestDialogHostPassesThroughToItsChild(t *testing.T) {
	sig := state.NewSignal(DialogReq{})
	h, ctx, _ := hostAndCtx(sig)
	size := h.Layout(ctx, geometry.Constraints{MaxWidth: 800, MaxHeight: 600})
	if size.Width <= 0 || size.Height <= 0 {
		t.Fatalf("the host must measure its child, got %v", size)
	}
	if len(h.Children()) != 1 {
		t.Fatal("the host wraps exactly one child")
	}
}

func TestNextDialogSeqIsMonotonic(t *testing.T) {
	var seq DialogSeq
	a, b := seq.Next(), seq.Next()
	if a == 0 || b <= a {
		t.Fatalf("sequence must be non-zero and increasing, got %d then %d", a, b)
	}
}

func TestDialogHostDoesNotReRaiseARequestTHEUSERHASDISMISSED(t *testing.T) {
	var seq DialogSeq
	// This is what the Seq guard is for (the "ten frames" test above passes
	// without it, because IsOpen suppresses a re-raise): once the dialog is
	// closed, the same request must not be raised again on the next frame.
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "t"})
	h, ctx, rec := hostAndCtx(sig)

	layout(h, ctx)
	if rec.count() != 1 {
		t.Fatalf("setup: want 1 push, got %d", rec.count())
	}

	h.dlg.Close(ctx) // the user dismisses it
	if h.dlg.IsOpen() {
		t.Fatal("setup: Close must close it")
	}

	for i := 0; i < 5; i++ {
		layout(h, ctx)
	}
	if got := rec.count(); got != 1 {
		t.Fatalf("a dismissed request raised again %d times; it must stay dismissed", got)
	}
}

// windowHost puts the host in a real toolkit window, headless, as the app runs
// it: through widget.LayoutChild's cache, ticking the tree before layout. The
// tests above call Layout directly and cannot see a cached root.
func windowHost(sig state.Signal[DialogReq]) (*DialogHost, *uiapp.Window) {
	h := NewDialogHost(primitives.Text("body"), sig, nil)
	w := uiapp.New().Window()
	w.SetRoot(h)
	w.Frame()
	w.Frame() // the layout cache is warm, as it is on an idle window
	return h, w
}

// A request must be raised on the next frame even when nothing else changed.
// Against the violation (no TickAnimation) this reads 0 overlays.
func TestDialogHostRaisesARequestOnAnIdleWindow(t *testing.T) {
	var seq DialogSeq
	sig := state.NewSignal(DialogReq{})
	_, w := windowHost(sig)

	sig.Set(DialogReq{Seq: seq.Next(), Kind: DialogConfirm, Title: "Clear the conversation?"})
	w.Frame()
	if got := w.OverlayCount(); got != 1 {
		t.Fatalf("a request on an idle window left %d overlays, want 1", got)
	}
}

// A request that arrives while a dialog is open must be raised when it closes.
func TestDialogHostRaisesARequestThatWaitedBehindAnOpenOne(t *testing.T) {
	var seq DialogSeq
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "first"})
	h, w := windowHost(sig)
	if w.OverlayCount() != 1 {
		t.Fatalf("setup: %d overlays, want the first dialog", w.OverlayCount())
	}

	second := seq.Next()
	sig.Set(DialogReq{Seq: second, Kind: DialogAlert, Title: "second"})
	w.Frame()
	if got := w.OverlayCount(); got != 1 {
		t.Fatalf("a second dialog was stacked over the first: %d overlays", got)
	}

	h.dlg.Close(w.Context()) // the user dismisses the first
	w.Frame()
	if got := w.OverlayCount(); got != 1 || h.shown != second {
		t.Fatalf("the request queued behind the first dialog was not raised when it closed: %d overlays, showing seq %d, want %d", got, h.shown, second)
	}
}

// rec gives the wrap gates a recorder over the same raise path, since they
// need the surface widget to draw.
func mountedHost(sig state.Signal[DialogReq]) (*DialogHost, *uitest.MockContext, *recorder, func()) {
	h, ctx, rec := hostAndCtx(sig)
	widget.MountTree(h, ctx)
	frame := func() {
		h.TickAnimation(ctx)
		widget.LayoutChild(h, ctx, geometry.Constraints{MaxWidth: 800, MaxHeight: 600})
	}
	return h, ctx, rec, frame
}

// A message must wrap inside the dialog rather than run past its edge. It
// measures every drawn line, since the canvas does not clip to the rect.
func TestDialogHostWrapsTheMessageInsideTheDialog(t *testing.T) {
	var seq DialogSeq
	msg := "Removes the conversation, the draft and any attached pictures. The model stays loaded."
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogConfirm, Title: "Clear the conversation?", Message: msg})
	_, ctx, rec, frame := mountedHost(sig)
	frame()
	if rec.count() != 1 {
		t.Fatal("setup: the dialog was not raised")
	}

	c := &uitest.MockCanvas{}
	rec.last.Layout(ctx, geometry.Constraints{MaxWidth: 800, MaxHeight: 600})
	rec.last.Draw(ctx, c)

	var lines []drawnLine
	for _, x := range c.Texts {
		lines = append(lines, drawnLine{x.Text, x.FontSize})
	}
	for _, x := range c.StyledTexts {
		lines = append(lines, drawnLine{x.Text, x.Style.FontSize})
	}
	band := float32(520 - 2*24) // MaxWidth minus core/dialog's contentPadding
	var body []string
	for _, l := range lines {
		if t := strings.TrimSpace(l.Text); t == "" || !strings.Contains(msg, t) {
			continue
		}
		body = append(body, l.Text)
		if w := float32(len([]rune(l.Text))) * l.Size * charWidthRatio; w > band {
			t.Errorf("line %q is ~%.0f px in a %.0f px dialog", l.Text, w, band)
		}
	}
	if len(body) < 2 || len(body) > dialogBodyLines {
		t.Errorf("the message drew as %d line(s), want 2..%d: %q", len(body), dialogBodyLines, body)
	}
}

// A message longer than the dialog has room for is cut to the band, not drawn
// over the buttons: core/dialog is 160 px tall whatever its content. This
// message is six lines, where the one above is exactly two.
func TestDialogHostCapsALongMessageToTheBand(t *testing.T) {
	var seq DialogSeq
	msg := strings.Repeat("jlm: container is version 17 and this build reads 21; re-run convert. ", 6)
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "Cannot open model", Message: msg})
	_, ctx, rec, frame := mountedHost(sig)
	frame()
	c := &uitest.MockCanvas{}
	rec.last.Layout(ctx, geometry.Constraints{MaxWidth: 800, MaxHeight: 600})
	rec.last.Draw(ctx, c)
	n := 0
	for _, x := range c.StyledTexts {
		if strings.Contains(x.Text, "container") || strings.Contains(x.Text, "re-run") {
			n++
		}
	}
	for _, x := range c.Texts {
		if strings.Contains(x.Text, "container") || strings.Contains(x.Text, "re-run") {
			n++
		}
	}
	if n == 0 || n > dialogBodyLines {
		t.Errorf("a six-line message drew %d line(s); the dialog has room for %d", n, dialogBodyLines)
	}
}

// drawnLine is one text draw, for the wrap gate.
type drawnLine struct {
	Text string
	Size float32
}

// A real error that does not fit two lines at 520 must get the width it needs
// rather than lose its reason. Against the violation (a fixed 520) the last
// words are never drawn.
func TestDialogHostWidensForAMessageThatWouldBeCut(t *testing.T) {
	var seq DialogSeq
	msg := "jlm: container is version 17 and this build reads 21; " +
		"re-run `jitllm convert` (there is no backward compatibility, on purpose)"
	sig := state.NewSignal(DialogReq{Seq: seq.Next(), Kind: DialogAlert, Title: "Cannot open model", Message: msg})
	_, ctx, rec, frame := mountedHost(sig)
	ctx.WindowSizeVal = geometry.Sz(1440, 900)
	frame()
	c := &uitest.MockCanvas{}
	rec.last.Layout(ctx, geometry.Constraints{MaxWidth: 1440, MaxHeight: 900})
	rec.last.Draw(ctx, c)
	var drawn strings.Builder
	for _, x := range c.StyledTexts {
		drawn.WriteString(x.Text + "\n")
	}
	for _, x := range c.Texts {
		drawn.WriteString(x.Text + "\n")
	}
	if !strings.Contains(drawn.String(), "on purpose)") {
		t.Errorf("the message lost its end; drew:\n%s", drawn.String())
	}
}
