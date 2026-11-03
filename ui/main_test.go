package main

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/screen"
)

// TestEveryTabHasAScreen relates the two lists nothing else relates: the tab
// bar's labels and the screens behind them. A screen added without a name, or
// a name added without a screen, is an index panic inside register.
func TestEveryTabHasAScreen(t *testing.T) {
	sc := screen.Screens()
	if len(sc) != len(app.TabNames()) {
		t.Fatalf("%d screen(s) registered against %d tab name(s): %v",
			len(sc), len(app.TabNames()), app.TabNames())
	}
	for i, build := range sc {
		if build == nil {
			t.Errorf("tab %d (%q) has no screen", i, app.TabNames()[i])
		}
	}
}

// TestEveryRegisteredScreenBuilds constructs the whole widget tree the window
// shows.
//
// It is the only gate in this module that builds a widget tree: Shell.Build
// calls every registered ScreenFunc and tabview mounts them all, so a build
// that panics would otherwise pass until the window opens.
//
// Two things are set before building, both deliberately:
//
//   - ModelDirs is a temp dir, so the Models screen's scan walks an empty
//     directory instead of the real model volume.
//   - Loaded is true, so Devices' probe refuses (the shipped behaviour): a
//     unit test must not open and close every backend on the machine.
func TestEveryRegisteredScreenBuilds(t *testing.T) {
	cfg := app.DefaultConfig(app.Env{})
	cfg.ModelDirs = []string{t.TempDir()}

	sh := app.NewShell(nil, cfg)
	sh.Store.Loaded.Set(true)
	register(sh, screen.Deps{})

	root := sh.Build()
	if root == nil {
		t.Fatal("Shell.Build returned no root")
	}
}

// TestDropFilesReachesTheCatalog covers the wiring and not the handler: the
// handler has its own tests in screen, and what is asserted here is that
// register actually installed one.
func TestDropFilesReachesTheCatalog(t *testing.T) {
	cfg := app.DefaultConfig(app.Env{})
	cfg.ModelDirs = []string{t.TempDir()}

	sh := app.NewShell(nil, cfg)
	sh.Store.Loaded.Set(true)
	sh.Store.Tab.Set(app.TabSession)
	register(sh, screen.Deps{})

	sh.DropFiles([]string{"/tmp/does-not-need-to-exist.gguf"}, 0, 0)

	if got := sh.Store.Tab.Get(); got != app.TabConvert {
		t.Errorf("a dropped GGUF left the window on tab %d; want Convert (%d)", got, app.TabConvert)
	}
	if got := sh.Store.Status.Get(); !strings.Contains(got, "convert source") {
		t.Errorf("status = %q; a dropped GGUF must say it became the convert source", got)
	}
}
