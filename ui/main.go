// Command jitllm-ui is the desktop front end for jitllm: an alternative to the
// CLI, not a wrapper around it.
//
// The engine is driven in process: shelling out to the jitllm binary cannot
// hold a model open across turns, and it would turn counters back into text.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"

	_ "github.com/gogpu/gg/gpu" // enable GPU SDF acceleration; required

	"github.com/gogpu/gogpu"
	"github.com/gogpu/ui/desktop"

	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/crash"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/engine"
	"github.com/jitllm/jitllm/ui/mock"
	"github.com/jitllm/jitllm/ui/screen"
)

// register installs every screen in tab order and attaches what the screens
// reach outside the widget tree: the engine, the hardware and the model files.
// Without an engine the Models tab opens the "No engine wired" dialog.
func register(sh *app.Shell, d screen.Deps) {
	screen.Install(sh, d)

	// Drop a .jlm or a .gguf anywhere on the window and it lands in the
	// catalog: a container is revealed in the table, a GGUF fills the convert
	// form. Drop a picture and it is attached to the next message.
	// Shell.OnFilesDropped holds one handler, so DropFiles dispatches the
	// window's whole drag-and-drop behaviour.
	sh.OnFilesDropped(func(paths []string) { screen.DropFiles(sh, paths) })
}

// buildVersion is the release version the module was built at, "dev" for a
// build from an untagged tree.
func buildVersion() string {
	if bi, ok := debug.ReadBuildInfo(); ok && bi.Main.Version != "" && bi.Main.Version != "(devel)" {
		return bi.Main.Version
	}
	return "dev"
}

func main() {
	mockMode := flag.Bool("mock", false, "run against package mock: no model, no GPU, fixtures only")
	showVersion := flag.Bool("version", false, "print the version and exit, opening no window")
	flag.Parse()
	var held stdio
	defer runtime.KeepAlive(&held)
	if *showVersion {
		// Before the log moves to a file: a portable copy is checked by
		// running it with -version, which must answer on standard output.
		// A GUI program started from a console has no standard output of its
		// own; one whose output is redirected keeps the redirection.
		if _, err := os.Stdout.Stat(); err != nil {
			attachParentConsole(&held)
		}
		fmt.Println("jitllm-desktop", buildVersion())
		return
	}
	// The previous run's log is read before this run's truncates it: its last
	// lines go in the report of a crash that ended it.
	noConsole := !stderrIsRead(&held)
	guard := &crash.Guard{}
	var prev *crash.Report
	if p := config.ConfigPath(); p != "" {
		dir := filepath.Dir(p)
		tail := crash.LogTail(filepath.Join(dir, logName), 20)
		logToFileWithoutAConsole(&held, noConsole)
		guard.SetInfo(crash.Info{App: "jitllm-desktop", Version: buildVersion()})
		var err error
		if prev, err = guard.Arm(dir, tail, noConsole); err != nil {
			log.Printf("crash reports: %v", err)
		}
	}
	// After Arm: the GPU it names is written beside crash.log.
	tapAdapter(guard)
	cfg := app.LoadConfig(app.Env{Models: os.Getenv("JITLLM_MODELS"), DataHome: os.Getenv("XDG_DATA_HOME"),
		APIKey: os.Getenv("JITLLM_API_KEY")})

	gpuApp := gogpu.NewApp(gogpu.DefaultConfig().
		WithTitle("jitllm").
		WithSize(cfg.WindowW, cfg.WindowH).
		WithIcon(app.Icon()).
		WithMinSize(960, 640))

	sh := app.NewShell(gpuApp, cfg)
	sh.SetCrashes(guard)
	guard.OnReport(sh.ShowCrash)
	// Before the first frame: a font registered after a widget has measured
	// itself is a font the layout has already decided without.
	sh.Fonts.Load()
	if prev != nil {
		sh.ShowCrash(*prev)
	}
	if *mockMode {
		d, eng := mock.Deps(sh)
		eng.Live = true
		register(sh, d)
	} else {
		// The engine setting decides: under Auto a jitllmd that answers is
		// used and the window runs its own engine only when none does.
		eng := engine.Start(sh)
		register(sh, screen.Deps{Engine: eng.Driver()})
		// The macOS app's first launch offers to put its jitllm and jitllmd
		// on PATH; anywhere else it does nothing.
		screen.OfferCLI(sh)
	}

	sh.UI.SetRoot(sh.Build())

	// The async -> UI bridge. OnUpdate runs inside the frame on the main
	// thread, and it only fires when the loop is awake -- which is why
	// Shell.Post pairs every send with a RequestRedraw.
	gpuApp.OnUpdate(sh.DrainQueue)

	// Shutdown runs after desktop.Run returns, not from OnClose: gogpu's OnClose
	// and OnDragDrop are single-slot setters that desktop.Run overwrites, so a
	// handler set here never ran and settings were never saved. Returning from Run
	// is the shutdown; Shell.Close reads the window size from GPU.Size().

	err := desktop.Run(gpuApp, sh.UI)
	sh.Close()
	if err != nil {
		log.Fatal(err)
	}
}
