package screen

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/ui/app"
)

// writeStaleContainer writes the twelve bytes that make a file look like a
// container of the given version: the magic, then a little-endian version.
func writeStaleContainer(t *testing.T, path string, version uint32) {
	t.Helper()
	writeHeader(t, path, version, false)
}

// writeHeader writes a container header: the magic, the version and, when
// tower is set, a non-zero vision section where catalog.HasTower reads it.
func writeHeader(t *testing.T, path string, version uint32, tower bool) {
	t.Helper()
	b := make([]byte, 256)
	copy(b, []byte{'J', 'I', 'T', 'L', 'L', 'M', 0, 0})
	binary.LittleEndian.PutUint32(b[8:], version)
	if tower {
		binary.LittleEndian.PutUint64(b[128:], 7<<20) // VisLen
		binary.LittleEndian.PutUint32(b[152:], 12)    // NVisBlocks
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeZeros(t *testing.T, path string, n int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, n), 0o600); err != nil {
		t.Fatal(err)
	}
}

// A row that has only been stat'd must not read as loadable: its zero
// Version and empty error look exactly like a healthy container.
func TestLoadableRefusesEverythingButAReadableContainer(t *testing.T) {
	good := catalog.Entry{Kind: catalog.KindContainer, Probed: true, Version: catalog.CurrentVersion}
	if !ModelLoadable(good, true) {
		t.Fatal("a probed, current container must be loadable")
	}
	if ModelLoadable(good, false) {
		t.Fatal("nothing selected must not be loadable")
	}

	unprobed := catalog.Entry{Kind: catalog.KindContainer}
	if ModelLoadable(unprobed, true) {
		t.Fatal("a container nobody has read yet must not be loadable")
	}
	stale := catalog.Entry{Kind: catalog.KindContainer, Probed: true, Stale: true, Version: 19}
	if ModelLoadable(stale, true) {
		t.Fatal("a stale container must not be loadable -- model.Open refuses it")
	}
	broken := catalog.Entry{Kind: catalog.KindContainer, Probed: true, ProbeErr: "truncated"}
	if ModelLoadable(broken, true) {
		t.Fatal("a container whose probe failed must not be loadable")
	}
	gguf := catalog.Entry{Kind: catalog.KindGGUF, Probed: true}
	if ModelLoadable(gguf, true) {
		t.Fatal("a GGUF must not be loadable -- model.Open reads one format")
	}
	if !ModelConvertible(gguf, true) {
		t.Fatal("a GGUF is exactly what the convert form wants")
	}
	if ModelConvertible(good, true) {
		t.Fatal("a container is not a conversion source")
	}
}

// A stale row says what it needs, not which versions differ; the table cuts
// long text and the detail pane keeps the versions.
func TestAStaleRowSaysWhatItNeeds(t *testing.T) {
	e := catalog.Entry{Kind: catalog.KindContainer, Probed: true, Stale: true, Version: 19,
		ProbeErr: "container is v19"}
	if got := entryShortStatus(e); got != "reconvert it" {
		t.Fatalf("shortStatus = %q, want %q", got, "reconvert it")
	}
	st := entryStatus(e)
	for _, want := range []string{"needs reconverting", "v19", "v" + modelsItoa(int(catalog.CurrentVersion))} {
		if !strings.Contains(st, want) {
			t.Errorf("the detail status %q does not carry %q", st, want)
		}
	}
	if entryHint(e) == "" {
		t.Error("a stale row gets no sentence saying what reconverting is for")
	}
}

// A lone vision tower reads cleanly and cannot be loaded, so the row must say
// so.
func TestAVisionTowerOnItsOwnIsFlaggedAndNotLoadable(t *testing.T) {
	tower := catalog.Entry{Kind: catalog.KindContainer, Probed: true, Version: catalog.CurrentVersion,
		HasVision: true}
	if got := entryShortStatus(tower); got != "vision tower" {
		t.Errorf("shortStatus = %q, want %q", got, "vision tower")
	}
	if ModelLoadable(tower, true) {
		t.Error("a vision tower on its own must not be loadable")
	}
	if ModelReconvertible(tower, true) {
		t.Error("rebuilding a lone vision tower gives a lone vision tower; Reconvert must not offer it")
	}
	if entryHint(tower) == "" {
		t.Error("a lone vision tower gets no sentence saying why Load is off")
	}

	model := tower
	model.NLayer, model.NVocab = 16, 32000 // a model WITH a tower is the normal case, and it loads
	if !ModelLoadable(model, true) || entryHint(model) != "" {
		t.Errorf("a model carrying a tower must load with no hint: loadable=%v hint=%q",
			ModelLoadable(model, true), entryHint(model))
	}
}

func modelsItoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b []byte
	for v > 0 {
		b = append([]byte{byte('0' + v%10)}, b...)
		v /= 10
	}
	return string(b)
}

func TestSortEntriesIsStableAndReversible(t *testing.T) {
	in := []catalog.Entry{
		{Name: "b.jlm", Kind: catalog.KindContainer, Size: 30},
		{Name: "a.gguf", Kind: catalog.KindGGUF, Size: 10},
		{Name: "c.jlm", Kind: catalog.KindContainer, Size: 20},
	}

	asc := append([]catalog.Entry(nil), in...)
	SortModels(asc, "size", true)
	if asc[0].Size != 10 || asc[2].Size != 30 {
		t.Fatalf("ascending size = %v", []int64{asc[0].Size, asc[1].Size, asc[2].Size})
	}

	desc := append([]catalog.Entry(nil), in...)
	SortModels(desc, "size", false)
	if desc[0].Size != 30 || desc[2].Size != 10 {
		t.Fatalf("descending size = %v", []int64{desc[0].Size, desc[1].Size, desc[2].Size})
	}

	// An unknown column leaves the scan's own order rather than inventing one.
	same := append([]catalog.Entry(nil), in...)
	SortModels(same, "page", true)
	for i := range same {
		if same[i].Name != in[i].Name {
			t.Fatalf("an unsortable column must not reorder: got %q at %d", same[i].Name, i)
		}
	}
}

// A screen is rebuilt on a theme swap: build twice, and the convert form the
// user typed into is still there.
func TestScreenStateIsPerShellAndSurvivesARebuild(t *testing.T) {
	cfg := app.DefaultConfig(app.Env{})
	cfg.ModelDirs = []string{t.TempDir()}

	a := app.NewShell(nil, cfg)
	b := app.NewShell(nil, cfg)

	sa := modelsFor(a)
	if modelsFor(a) != sa {
		t.Fatal("one shell must get ONE screen state, however often it rebuilds")
	}
	if modelsFor(b) == sa {
		t.Fatal("two shells must not share a screen state")
	}
	ca := convertFor(a)
	if convertFor(a) != ca || convertFor(b) == ca {
		t.Fatal("the convert form must be one per shell and not shared")
	}

	ca.src.Set("/models/x.gguf")
	for _, build := range []func(*app.Shell) widget.Widget{Models, Convert} {
		if build(a) == nil || build(a) == nil { // the second is the theme swap
			t.Fatal("a screen returned no widget")
		}
	}
	if got := convertFor(a).src.Get(); got != "/models/x.gguf" {
		t.Fatalf("the convert form was lost across a rebuild: src = %q", got)
	}
}

// The table fills in two passes and both are published: names and sizes at
// once, architecture and page geometry behind them, one probe at a time.
func TestRescanPublishesEveryFileAndMarksAStaleContainer(t *testing.T) {
	dir := t.TempDir()
	writeZeros(t, filepath.Join(dir, "weights.gguf"), 2048)
	writeStaleContainer(t, filepath.Join(dir, "old.jlm"), 19)

	cfg := app.DefaultConfig(app.Env{})
	cfg.ModelDirs = []string{dir}
	sh := app.NewShell(nil, cfg)

	m := modelsFor(sh) // the first touch starts the scan

	// The scan splits: containers to the Models table, the GGUF to the Convert
	// table.
	deadline := time.Now().Add(5 * time.Second)
	var got []catalog.Entry
	for time.Now().Before(deadline) {
		sh.DrainQueue(0) // the scan publishes on the UI goroutine, as OnUpdate drains it
		got = sh.Store.Catalog.Get()
		if len(got) == 1 && got[0].Probed {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(got) != 1 || got[0].Kind != catalog.KindContainer {
		t.Fatalf("the Models list holds %v, want the one container", got)
	}
	src := sh.Store.Sources.Get()
	if len(src) != 1 || src[0].Kind != catalog.KindGGUF || sh.Store.SourcesRows.Get() != 1 {
		t.Fatalf("the Convert list holds %v, want the one GGUF", src)
	}
	if n := sh.Store.CatalogRows.Get(); n != len(got) {
		t.Fatalf("row count %d does not match the published slice of %d -- a table "+
			"bound to the count would read past it", n, len(got))
	}

	var container *catalog.Entry
	for i := range got {
		if got[i].Kind == catalog.KindContainer {
			container = &got[i]
		}
	}
	if container == nil {
		t.Fatal("the .jlm was not classified as a container")
	}
	if !container.Probed {
		t.Fatal("a container in the catalog must be probed, not merely stat'd")
	}
	if !container.Stale {
		t.Fatalf("a v19 file against v%d must read as stale", catalog.CurrentVersion)
	}
	if container.ProbeErr == "" {
		t.Fatal("a stale container needs a reason a person can act on")
	}

	// The GGUF is listed and deliberately not probed: it is something to
	// convert, not something to describe.
	if src[0].NLayer != 0 {
		t.Fatal("a GGUF row must not claim a config it was never read for")
	}

	// A second rescan while none is running must be allowed and must not
	// duplicate either list.
	m.rescan()
	time.Sleep(200 * time.Millisecond)
	if n, k := len(sh.Store.Catalog.Get()), len(sh.Store.Sources.Get()); n != 1 || k != 1 {
		t.Fatalf("a rescan duplicated the lists: %d containers, %d sources", n, k)
	}
}

// A row says what someone choosing a model asks; the engine's numbers are
// the panel's.
func TestAModelRowSaysWhatAChooserAsks(t *testing.T) {
	e := catalog.Entry{Kind: catalog.KindContainer, Probed: true, NLayer: 48,
		Quant: "Q4_K_M", Size: 18 << 30, HasVision: true, PageSize: 1 << 20, ChunkBytes: 1 << 20}
	got := entryFacts(e)
	for _, want := range []string{"48 layers", "Q4_K_M", "18.00 GiB", "sees images"} {
		if !strings.Contains(got, want) {
			t.Errorf("the row does not say %q: %s", want, got)
		}
	}
	for _, gone := range []string{"page", "chunk", "stream"} {
		if strings.Contains(strings.ToLower(got), gone) {
			t.Errorf("the engine's %q is on the row: %s", gone, got)
		}
	}
	if got := entryFacts(catalog.Entry{Kind: catalog.KindContainer}); !strings.Contains(got, "reading") {
		t.Errorf("an unread row claims facts: %q", got)
	}
}
