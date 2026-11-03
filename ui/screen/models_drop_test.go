package screen

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/ui/app"
)

func TestClassifyDropTakesOneFileOfEachRole(t *testing.T) {
	d := ClassifyDrop([]string{
		"/m/first.gguf",
		"/m/second.gguf",
		"/m/first.jlm",
		"/m/second.jlm",
	})
	if d.Source != "/m/first.gguf" {
		t.Errorf("Source = %q; the first source wins", d.Source)
	}
	if d.Container != "/m/first.jlm" {
		t.Errorf("Container = %q; the first container wins", d.Container)
	}
	if len(d.Ignored) != 0 {
		t.Errorf("Ignored = %v; every path was a model", d.Ignored)
	}
}

func TestClassifyDropNamesWhatItCannotRead(t *testing.T) {
	d := ClassifyDrop([]string{"/m/notes.txt", "/m/weights.bin", "  ", ""})
	if !d.Empty() {
		t.Fatalf("a drop of unreadable files must be empty: %+v", d)
	}
	if len(d.Ignored) != 2 {
		t.Fatalf("Ignored = %v; want the two named files and neither blank", d.Ignored)
	}
	if d.Ignored[0] != "notes.txt" || d.Ignored[1] != "weights.bin" {
		t.Errorf("Ignored = %v; the base name is what a status line can show", d.Ignored)
	}
}

func TestClassifyDropReadsSafetensorsAsASource(t *testing.T) {
	d := ClassifyDrop([]string{"/m/model.SafeTensors"})
	if d.Source != "/m/model.SafeTensors" {
		t.Errorf("Source = %q; the extension match is case-insensitive", d.Source)
	}
}

// A dropped GGUF must arrive in the convert form with its destination already
// derived.
func TestDropFilesFillsTheConvertForm(t *testing.T) {
	sh := dropShell(t)
	DropFiles(sh, []string{"/models/tinyllama.gguf"})

	c := convertFor(sh)
	if got := c.src.Get(); got != "/models/tinyllama.gguf" {
		t.Errorf("src = %q", got)
	}
	if want := DestFor("/models/tinyllama.gguf", primaryDir(sh.Cfg.ModelDirs)); c.dst.Get() != want {
		t.Errorf("dst = %q, want %q", c.dst.Get(), want)
	}
	if sh.Store.Tab.Get() != app.TabConvert {
		t.Error("a dropped GGUF must land the user on the Convert tab")
	}
}

// A dropped container that the scan has already seen is selected, not converted.
func TestDropFilesSelectsAContainerAlreadyInTheCatalog(t *testing.T) {
	sh := dropShell(t)
	waitForScan(t, modelsFor(sh)) // the startup scan publishes too; let it finish
	sh.Store.SetCatalog([]catalog.Entry{
		{Path: "/models/a.jlm", Name: "a.jlm", Kind: catalog.KindContainer},
		{Path: "/models/b.jlm", Name: "b.jlm", Kind: catalog.KindContainer},
	})
	sh.Store.CatalogSel.Set(-1)

	DropFiles(sh, []string{"/models/b.jlm"})

	if got := sh.Store.CatalogSel.Get(); got != 1 {
		t.Fatalf("CatalogSel = %d; want row 1, the dropped container", got)
	}
	if c := convertFor(sh); c.src.Get() != "" {
		t.Errorf("src = %q; a container is not a convert source", c.src.Get())
	}
	if sh.Store.Tab.Get() != app.TabModels {
		t.Error("a dropped container must land the user on the Models tab")
	}
}

// A container outside every configured folder pulls its folder in, so the file
// is reachable after the rescan rather than silently ignored.
func TestDropFilesAdoptsTheFolderOfAnUnknownContainer(t *testing.T) {
	sh := dropShell(t)
	before := len(sh.Cfg.ModelDirs)

	dir := t.TempDir()
	DropFiles(sh, []string{filepath.Join(dir, "elsewhere.jlm")})

	if len(sh.Cfg.ModelDirs) != before+1 {
		t.Fatalf("ModelDirs = %v; the dropped container's folder was not adopted", sh.Cfg.ModelDirs)
	}
	if sh.Cfg.ModelDirs[len(sh.Cfg.ModelDirs)-1] != dir {
		t.Errorf("adopted %q, want %q", sh.Cfg.ModelDirs[len(sh.Cfg.ModelDirs)-1], dir)
	}
}

func TestDropFilesSaysWhatItCannotRead(t *testing.T) {
	sh := dropShell(t)
	sh.Store.Tab.Set(app.TabSession)

	DropFiles(sh, []string{"/models/readme.md"})

	if sh.Store.Tab.Get() != app.TabSession {
		t.Error("an unreadable drop must not switch tabs")
	}
	// Assert the file's name, not a non-empty status: a fresh store seeds
	// Status to "ready".
	if got := sh.Store.Status.Get(); !strings.Contains(got, "readme.md") {
		t.Errorf("status = %q; an unreadable drop must name what it refused", got)
	}
}

// dropShell is a windowless shell whose catalog directory is empty, so the
// scan a Models screen starts on creation finds nothing and finishes at once.
//
// The user config folder is redirected on every OS (XDG_CONFIG_HOME, HOME for
// macOS, AppData for Windows) because adopting a dropped container's folder
// saves the settings, which must not touch the developer's own file.
func dropShell(t *testing.T) *app.Shell {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
	cfg := app.DefaultConfig(app.Env{})
	cfg.ModelDirs = []string{t.TempDir()}
	return app.NewShell(nil, cfg)
}
