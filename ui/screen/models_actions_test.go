package screen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	uiapp "github.com/gogpu/ui/app"
	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/offscreen"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/app"
)

// --- a real window over the detail pane --------------------------------------

// paneWindow puts the Models detail pane in a real toolkit window, headless,
// so a gate presses the button through the gesture pipeline and its disabled
// check instead of calling m.load() directly.
func paneWindow(m *modelsScreen, width int) *uiapp.Window {
	w := uiapp.New().Window()
	w.HandleResize(width, 1400) // tall enough that nothing in the pane scrolls
	w.SetRoot(primitives.Box(m.detailSide()))
	paneFrame(w)
	return w
}

// paneFrame lays out and draws, which is what stamps each widget's screen
// origin -- a press is hit-tested against it.
func paneFrame(w *uiapp.Window) {
	w.Frame()
	w.DrawTo(&uitest.MockCanvas{})
}

// paneDraw draws a fresh detail pane and returns every text it drew. It is
// fresh because the window replays a retained scene while nothing is dirty,
// so a mock canvas handed to it would see no DrawText.
func paneDraw(m *modelsScreen, width float32) *uitest.MockCanvas {
	w := m.detailSide()
	ctx := mockCtx()
	widget.MountTree(w, ctx)
	w.Layout(ctx, geometry.Constraints{MaxWidth: width, MaxHeight: 1400})
	c := &uitest.MockCanvas{}
	w.Draw(ctx, c)
	return c
}

// paneText is what the pane says, one line after another joined by single
// spaces, so a sentence the paragraph wrapped still reads as one.
func paneText(m *modelsScreen) string {
	return strings.Join(strings.Fields(drawnText(paneDraw(m, 560))), " ")
}

// formDraw draws a fresh Convert form at width and returns every text it drew.
func formDraw(c *convertScreen, width float32) *uitest.MockCanvas {
	w := c.formSide()
	ctx := mockCtx()
	widget.MountTree(w, ctx)
	w.Layout(ctx, geometry.Constraints{MaxWidth: width, MaxHeight: 1400})
	cv := &uitest.MockCanvas{}
	w.Draw(ctx, cv)
	return cv
}

// formText is what the Convert form says, as paneText is for the Models pane.
func formText(c *convertScreen) string {
	return strings.Join(strings.Fields(drawnText(formDraw(c, 560))), " ")
}

// buttonLabelled finds the button that draws label.
func buttonLabelled(t *testing.T, root widget.Widget, label string) *button.Widget {
	t.Helper()
	var found *button.Widget
	var walk func(widget.Widget)
	walk = func(w widget.Widget) {
		if b, ok := w.(*button.Widget); ok && found == nil {
			c := &uitest.MockCanvas{}
			b.Draw(mockCtx(), c)
			if strings.TrimSpace(drawnText(c)) == label {
				found = b
			}
		}
		for _, ch := range w.Children() {
			walk(ch)
		}
	}
	walk(root)
	if found == nil {
		t.Fatalf("no button labelled %q in the pane", label)
	}
	return found
}

// press clicks the middle of b the way a mouse does.
func press(w *uiapp.Window, b *button.Widget) {
	sb := b.ScreenBounds()
	at := geometry.Pt(sb.Min.X+sb.Width()/2, sb.Min.Y+sb.Height()/2)
	w.HandleEvent(event.NewMouseEvent(event.MousePress, event.ButtonLeft, event.ButtonStateLeft, at, at, 0))
	w.HandleEvent(event.NewMouseEvent(event.MouseRelease, event.ButtonLeft, 0, at, at, 0))
	paneFrame(w)
}

// modelsRows is one of each kind of row the table shows.
func modelsRows(dir string) (ready, stale, tower, gguf catalog.Entry) {
	ready = catalog.Entry{Path: filepath.Join(dir, "ready.jlm"), Name: "ready.jlm",
		Kind: catalog.KindContainer, Probed: true, Version: catalog.CurrentVersion, NLayer: 16, NVocab: 32000}
	stale = catalog.Entry{Path: filepath.Join(dir, "old.jlm"), Name: "old.jlm",
		Kind: catalog.KindContainer, Probed: true, Stale: true, Version: 17, ProbeErr: "container is v17"}
	tower = catalog.Entry{Path: filepath.Join(dir, "mmproj-x.jlm"), Name: "mmproj-x.jlm",
		Kind: catalog.KindContainer, Probed: true, Version: catalog.CurrentVersion, HasVision: true}
	gguf = catalog.Entry{Path: filepath.Join(dir, "weights.gguf"), Name: "weights.gguf", Kind: catalog.KindGGUF}
	return
}

// modelsReady is a shell whose startup scan has finished, with rows published.
func modelsReady(t *testing.T, rows ...catalog.Entry) (*app.Shell, *modelsScreen) {
	t.Helper()
	sh := dropShell(t)
	m := modelsFor(sh)
	waitForScan(t, m)
	m.publish(rows)
	return sh, m
}

func selectRow(t *testing.T, m *modelsScreen, e catalog.Entry) {
	t.Helper()
	if !m.selectPath(e.Path) {
		t.Fatalf("%s is not in the table", e.Name)
	}
}

// --- Load ---------------------------------------------------------------------

// Pressing Load must hand the model to the engine and take you to the
// Session, and the rows that cannot load (a stale file, a lone vision tower,
// a GGUF) must not load from the same button.
func TestLoadTakesYouToTheSession(t *testing.T) {
	ready, stale, tower, gguf := modelsRows(t.TempDir())
	sh, m := modelsReady(t, ready, stale, tower, gguf)
	var loaded []string
	Attach(sh, Deps{Engine: &fakeEngine{load: func(p string) { loaded = append(loaded, p) }}})

	w := paneWindow(m, 560)
	load := buttonLabelled(t, w.Root(), "Load")

	// The GGUF is not in this table at all: it is the Convert tab's.
	if m.selectPath(gguf.Path) {
		t.Fatalf("%s is listed on the Models tab beside the containers", gguf.Name)
	}
	if rowOf(sh.Store.Sources.Get(), gguf.Path) < 0 {
		t.Fatalf("%s is not on the Convert tab's list", gguf.Name)
	}
	for _, e := range []catalog.Entry{stale, tower} {
		selectRow(t, m, e)
		sh.Store.Tab.Set(app.TabModels)
		press(w, load)
		if len(loaded) != 0 {
			t.Fatalf("%s is not loadable and Load sent %v to the engine", e.Name, loaded)
		}
		if sh.Store.Tab.Get() != app.TabModels {
			t.Fatalf("pressing a disabled Load on %s switched tabs", e.Name)
		}
	}

	selectRow(t, m, ready)
	sh.Store.Tab.Set(app.TabModels)
	press(w, load)
	if len(loaded) != 1 || loaded[0] != ready.Path {
		t.Fatalf("Load sent %v, want [%s]", loaded, ready.Path)
	}
	if got := sh.Store.Tab.Get(); got != app.TabSession {
		t.Fatalf("after Load the tab is %d, want the Session (%d)", got, app.TabSession)
	}
}

// --- Reconvert's enabled state and the pane's one-sentence reason --------------

// Reconvert is for containers, and a lone vision tower is not one worth
// rebuilding. With nothing selected it must be off too. IsFocusable reads the
// button's bound disabled state, so this fails if the binding is dropped.
func TestReconvertIsOffWithoutAContainer(t *testing.T) {
	ready, stale, tower, gguf := modelsRows(t.TempDir())
	sh, m := modelsReady(t, ready, stale, tower, gguf)
	w := paneWindow(m, 560)
	rc := buttonLabelled(t, w.Root(), "Reconvert")

	m.store.CatalogSel.Set(-1)
	if rc.IsFocusable() {
		t.Error("Reconvert is on with nothing selected")
	}
	press(w, rc)
	if c := convertFor(sh); c.src.Get() != "" || c.dst.Get() != "" {
		t.Errorf("pressing Reconvert with nothing selected filled the form: src %q dst %q", c.src.Get(), c.dst.Get())
	}
	for _, e := range []catalog.Entry{tower} {
		selectRow(t, m, e)
		if rc.IsFocusable() {
			t.Errorf("Reconvert is on for %s", e.Name)
		}
	}
	for _, e := range []catalog.Entry{ready, stale} {
		selectRow(t, m, e)
		if !rc.IsFocusable() {
			t.Errorf("Reconvert is off for the container %s", e.Name)
		}
	}
}

// The pane must say, in one sentence, why a row will not load -- and say
// nothing for one that will.
func TestThePaneSaysWhyARowWillNotLoad(t *testing.T) {
	ready, stale, tower, gguf := modelsRows(t.TempDir())
	_, m := modelsReady(t, ready, stale, tower, gguf)

	for _, e := range []catalog.Entry{stale, tower} {
		selectRow(t, m, e)
		got := paneText(m)
		if want := entryHint(e); want == "" || !strings.Contains(got, want) {
			t.Errorf("%s: the pane does not draw %q; it drew:\n%s", e.Name, want, got)
		}
	}
	selectRow(t, m, ready)
	got := paneText(m)
	for _, e := range []catalog.Entry{stale, tower} {
		if strings.Contains(got, entryHint(e)) {
			t.Errorf("a loadable row still shows %q", entryHint(e))
		}
	}
}

// --- one-click Reconvert --------------------------------------------------------

// queueRuns records conversion jobs instead of running them, so a gate can see
// that a conversion started without converting anything.
func queueRuns(sh *app.Shell) *[]func() {
	var jobs []func()
	Attach(sh, Deps{Engine: &fakeEngine{run: func(job func()) bool { jobs = append(jobs, job); return true }}})
	return &jobs
}

// A stale container with its GGUF beside it is rebuilt in one click, through
// the same runner and progress as the Convert button.
func TestReconvertStartsAtOnceWhenTheSourceIsBeside(t *testing.T) {
	dir := t.TempDir()
	_, stale, _, _ := modelsRows(dir)
	writeStaleContainer(t, stale.Path, 17)
	writeZeros(t, strings.TrimSuffix(stale.Path, ".jlm")+".gguf", 64)
	sh, m := modelsReady(t, stale)
	jobs := queueRuns(sh)

	selectRow(t, m, stale)
	w := paneWindow(m, 560)
	press(w, buttonLabelled(t, w.Root(), "Reconvert"))
	c := convertFor(sh)

	if sh.Store.Tab.Get() != app.TabConvert {
		t.Error("Reconvert did not take you to the Convert tab, where the form and its progress are")
	}
	if len(*jobs) != 1 {
		t.Fatalf("Reconvert queued %d conversion(s); with the GGUF beside a stale file it must start one", len(*jobs))
	}
	if !c.converting.Get() {
		t.Error("a started conversion does not show as running")
	}
	if want := strings.TrimSuffix(stale.Path, ".jlm") + ".gguf"; c.src.Get() != want || c.dst.Get() != stale.Path {
		t.Errorf("form src %q dst %q, want %q -> %q", c.src.Get(), c.dst.Get(), want, stale.Path)
	}
	if c.note.Get() != "" {
		t.Errorf("a started conversion left a note: %q", c.note.Get())
	}

	// The source is garbage, so the job fails -- and the file it was meant to
	// replace must still be there.
	(*jobs)[0]()
	if c.converting.Get() {
		t.Error("a failed conversion still shows as running")
	}
	if _, err := os.Stat(stale.Path); err != nil {
		t.Fatalf("a failed reconvert deleted the container it was replacing: %v", err)
	}
}

// Without the GGUF beside it, Reconvert fills the form with what it looked
// for and says so, and starts nothing.
func TestReconvertSaysWhatIsMissing(t *testing.T) {
	dir := t.TempDir()
	vlm := catalog.Entry{Path: filepath.Join(dir, "qwen2vl-vlm.jlm"), Name: "qwen2vl-vlm.jlm",
		Kind: catalog.KindContainer, Probed: true, Stale: true, Version: 17}
	writeHeader(t, vlm.Path, 17, true) // its header says it carries a tower
	sh, m := modelsReady(t, vlm)
	jobs := queueRuns(sh)

	selectRow(t, m, vlm)
	w := paneWindow(m, 560)
	press(w, buttonLabelled(t, w.Root(), "Reconvert"))

	if len(*jobs) != 0 {
		t.Fatalf("Reconvert started a conversion with no source on disk")
	}
	c := convertFor(sh)
	guess := filepath.Join(dir, "qwen2vl-vlm.gguf")
	if c.src.Get() != guess || c.dst.Get() != vlm.Path {
		t.Errorf("form src %q dst %q, want %q -> %q", c.src.Get(), c.dst.Get(), guess, vlm.Path)
	}
	note := c.note.Get()
	for _, want := range []string{"qwen2vl-vlm.gguf", "Pick the GGUF", "mmproj"} {
		if !strings.Contains(note, want) {
			t.Errorf("the note %q does not say %q", note, want)
		}
	}
	if got := formText(c); !strings.Contains(got, "is not beside this model") {
		t.Errorf("the note is not on the Convert form; it drew:\n%s", got)
	}
}

// A container that loads is not replaced by one click: the form is filled and
// Convert is still the user's press.
func TestReconvertOfALoadableFileWaitsForConvert(t *testing.T) {
	dir := t.TempDir()
	ready, _, _, _ := modelsRows(dir)
	writeStaleContainer(t, ready.Path, catalog.CurrentVersion) // a current version word
	writeZeros(t, strings.TrimSuffix(ready.Path, ".jlm")+".gguf", 64)
	sh, m := modelsReady(t, ready)
	jobs := queueRuns(sh)

	selectRow(t, m, ready)
	w := paneWindow(m, 560)
	press(w, buttonLabelled(t, w.Root(), "Reconvert"))

	if len(*jobs) != 0 {
		t.Fatal("one click rewrote a container that loads")
	}
	if c := convertFor(sh); !strings.Contains(c.note.Get(), "Press Convert") {
		t.Errorf("the note %q does not say how to go on", c.note.Get())
	}
}

// --- the cross-screen offers ------------------------------------------------------

// OfferConvert is how the Session sends a GGUF here: it lands on the Convert
// tab with the form filled and the file's row selected.
func TestOfferConvertFillsTheFormOnTheConvertTab(t *testing.T) {
	ready, _, _, gguf := modelsRows(t.TempDir())
	sh, _ := modelsReady(t, ready, gguf)
	c := convertFor(sh)
	sh.Store.Tab.Set(app.TabSession)
	c.note.Set("left over from before")

	OfferConvert(sh, gguf.Path)

	if sh.Store.Tab.Get() != app.TabConvert {
		t.Error("OfferConvert did not switch to the Convert tab")
	}
	if e, ok := sh.Store.SelectedSource(); !ok || e.Path != gguf.Path {
		t.Errorf("selected %+v, want %s", e, gguf.Name)
	}
	if c.src.Get() != gguf.Path || c.dst.Get() != DestFor(gguf.Path, primaryDir(sh.Cfg.ModelDirs)) || c.mmproj.Get() != "" {
		t.Errorf("form src %q mmproj %q dst %q", c.src.Get(), c.mmproj.Get(), c.dst.Get())
	}
	if c.note.Get() != "" {
		t.Errorf("a filled form kept an old note: %q", c.note.Get())
	}
}

// OfferReconvert is how the Session sends a stale container here: it lands on
// the Convert tab and runs the one-click flow.
func TestOfferReconvertRunsTheOneClickFlow(t *testing.T) {
	dir := t.TempDir()
	ready, stale, _, _ := modelsRows(dir)
	writeStaleContainer(t, stale.Path, 17)
	writeZeros(t, strings.TrimSuffix(stale.Path, ".jlm")+".gguf", 64)
	sh, _ := modelsReady(t, ready, stale)
	c := convertFor(sh)
	jobs := queueRuns(sh)
	sh.Store.Tab.Set(app.TabSession)

	OfferReconvert(sh, stale.Path)

	if sh.Store.Tab.Get() != app.TabConvert {
		t.Error("OfferReconvert did not switch to the Convert tab")
	}
	if len(*jobs) != 1 {
		t.Fatalf("OfferReconvert queued %d conversion(s), want the one-click one", len(*jobs))
	}

	// A second offer while it runs must not start another, and says why.
	OfferReconvert(sh, stale.Path)
	if len(*jobs) != 1 {
		t.Fatalf("a second offer during a conversion queued another: %d", len(*jobs))
	}
	if note := c.note.Get(); !strings.Contains(note, "Another conversion is running") {
		t.Errorf("a second offer during a conversion says %q", note)
	}
}

// --- the convert form's placeholders ----------------------------------------------

// Each placeholder must fit its field at the form's narrowest, since the
// field does not cut it (UPSTREAM.md #8). Widths come from the real
// rasteriser, against the field's own content rect.
func TestConvertPlaceholdersFitTheirFields(t *testing.T) {
	sh := dropShell(t)
	waitForScan(t, modelsFor(sh))

	const paneW = 380 // the form's minimum width (splitview.MinSecond)
	c := formDraw(convertFor(sh), paneW)

	fits := 0
	for _, ph := range []string{placeholderSource, placeholderTower, placeholderDest} {
		var call *uitest.DrawTextCall
		for i := range c.Texts {
			if c.Texts[i].Text == ph {
				call = &c.Texts[i]
			}
		}
		if call == nil {
			t.Errorf("placeholder %q was not drawn", ph)
			continue
		}
		width := realTextWidth(t, ph, call.FontSize)
		if width <= 0 {
			t.Fatalf("the real renderer measured %q as %.1f px", ph, width)
		}
		t.Logf("%q: %.0f px in a %.0f px field", ph, width, call.Bounds.Width())
		if width > call.Bounds.Width() {
			t.Errorf("placeholder %q is %.0f px in a %.0f px field", ph, width, call.Bounds.Width())
			continue
		}
		fits++
	}
	if fits != 3 {
		t.Fatalf("%d of 3 placeholders fit", fits)
	}
}

// Every word on the right of a model row fits that column.
func TestEveryRowStateFitsItsColumn(t *testing.T) {
	ready, stale, tower, gguf := modelsRows(t.TempDir())
	sh, m := modelsReady(t, ready, stale, tower, gguf)
	size := sh.P.Type.BodySmall.FontSize
	unprobed := catalog.Entry{Kind: catalog.KindContainer}
	broken := catalog.Entry{Kind: catalog.KindContainer, Probed: true, ProbeErr: "truncated"}
	words := []string{"Loading\u2026", "Loaded"}
	for _, e := range []catalog.Entry{ready, stale, tower, gguf, unprobed, broken} {
		words = append(words, m.rowState(e))
	}
	for _, f := range []string{"GPU", "RAM", "paged"} {
		words = append(words, discover.FitWordsOf(f))
	}
	for _, s := range words {
		if w := realTextWidth(t, s, size); w > rowSideWidth {
			t.Errorf("%q is %.0f px; the column is %d", s, w, rowSideWidth)
		}
	}
}

// realTextWidth measures s with the canvas the offscreen renderer draws with.
func realTextWidth(t *testing.T, s string, size float32) float32 {
	t.Helper()
	(&app.Fonts{}).Load() // what main and cmd/shots do before the first frame
	p := &measureProbe{text: s, size: size}
	p.SetVisible(true)
	offscreen.NewRenderer(400, 40).Render(p)
	return p.width
}

// measureProbe records a MeasureText from inside a real draw.
type measureProbe struct {
	widget.WidgetBase
	text  string
	size  float32
	width float32
}

func (p *measureProbe) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	sz := c.Constrain(geometry.Sz(400, 40))
	p.SetBounds(geometry.FromPointSize(p.Position(), sz))
	return sz
}
func (p *measureProbe) Draw(_ widget.Context, c widget.Canvas) {
	p.width = c.MeasureText(p.text, p.size, false)
}
func (p *measureProbe) Event(widget.Context, event.Event) bool { return false }
func (p *measureProbe) Children() []widget.Widget              { return nil }

// A conversion the engine could not queue must not leave the screen waiting:
// the dropped job's defer never runs.
func TestAConversionTheEngineDroppedDoesNotStickTheScreen(t *testing.T) {
	sh := dropShell(t)
	c := convertFor(sh)
	Attach(sh, Deps{Engine: &fakeEngine{run: func(func()) bool { return false }}})
	src := filepath.Join(t.TempDir(), "x.gguf")
	writeZeros(t, src, 64)
	c.src.Set(src)
	c.dst.Set(DestFor(src, ""))
	c.convert()
	if c.converting.Get() {
		t.Error("a job the engine never took left the screen converting")
	}
}

// A stale vision model must not be rebuilt without its tower in one click:
// nothing records which mmproj a container came from, so the stale header's
// tower flag is what holds the click back.
func TestReconvertKeepsAVisionModelsTower(t *testing.T) {
	dir := t.TempDir()
	_, stale, _, _ := modelsRows(dir)
	writeHeader(t, stale.Path, 17, true)
	writeZeros(t, strings.TrimSuffix(stale.Path, ".jlm")+".gguf", 64)
	sh, _ := modelsReady(t, stale)
	jobs := queueRuns(sh)
	c := convertFor(sh)

	c.reconvertPath(stale.Path)
	if len(*jobs) != 0 {
		t.Fatalf("one click started a text-only rebuild of a model with a vision tower")
	}
	if !strings.Contains(c.note.Get(), "mmproj") {
		t.Errorf("the note does not ask for the tower: %q", c.note.Get())
	}
}

// A container from a newer build must never be downgraded in one click.
func TestReconvertNeverDowngradesANewerFile(t *testing.T) {
	dir := t.TempDir()
	_, stale, _, _ := modelsRows(dir)
	writeHeader(t, stale.Path, catalog.CurrentVersion+1, false)
	writeZeros(t, strings.TrimSuffix(stale.Path, ".jlm")+".gguf", 64)
	sh, _ := modelsReady(t, stale)
	jobs := queueRuns(sh)
	c := convertFor(sh)

	c.reconvertPath(stale.Path)
	if len(*jobs) != 0 {
		t.Fatalf("one click rewrote a newer container in this build's older format")
	}
	if !strings.Contains(c.note.Get(), "newer") {
		t.Errorf("the note does not say the file is newer: %q", c.note.Get())
	}
}

// Two loaded models are both "Loaded": neither is ranked "in use" over an
// "open" one, and the one the chat is not talking to offers a chat, not a
// switch.
func TestTwoLoadedModelsAreBothLoaded(t *testing.T) {
	dir := t.TempDir()
	ready, _, _, _ := modelsRows(dir)
	other := ready
	other.Path, other.Name = filepath.Join(dir, "other.jlm"), "other.jlm"
	sh, m := modelsReady(t, ready, other)
	sh.Store.Models.Set([]session.LoadedModel{
		{Path: ready.Path, Name: ready.Name},
		{Path: other.Path, Name: other.Name, Colour: 1, Active: true},
	})
	sh.Store.Active.Set(other.Path)
	for _, e := range []catalog.Entry{ready, other} {
		if got := m.rowState(e); got != "Loaded" {
			t.Errorf("%s reads %q, want Loaded", e.Name, got)
		}
	}
	selectRow(t, m, ready)
	if got := m.dStatus.Get(); got != "loaded" {
		t.Errorf("the panel says %q for a loaded model", got)
	}
	if got := m.loadLabel.Get(); got != "Chat with it" {
		t.Errorf("the loaded model the chat is not on offers %q", got)
	}
	selectRow(t, m, other)
	if got := m.loadLabel.Get(); got != "Go to chat" {
		t.Errorf("the chat's own model offers %q", got)
	}
}
