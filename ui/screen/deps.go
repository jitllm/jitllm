package screen

import (
	"sync/atomic"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/ui/app"
)

// The screens reach everything outside the widget tree through these
// interfaces: the engine, the hardware and the model files. The real ones are
// engine.Engine, hardware.Host and catalog.Disk; package mock stands in for all
// three, which is how the shots, the stage tests and `-mock` run with no model
// and no GPU.

// Engine is the inference engine as the screens drive it. Every method queues
// and returns; results come back as Store writes posted to the UI goroutine.
type Engine interface {
	// Load opens a model and makes it active.
	Load(path string)
	// Use makes an already-open model the active one.
	Use(path string)
	// Unload closes one open model and gives back what it held; closing the
	// one in chat leaves nothing running.
	Unload(path string)
	// SetPriority switches memory between load order and following the active
	// model.
	SetPriority(on bool)
	// Relocate asks the device to hold n blocks. It is called from a click, so
	// the engine queues the move to happen between tokens.
	Relocate(n int)
	// Run queues a long job that must not overlap a decode, and reports
	// whether it was accepted.
	Run(job func()) bool
	// Send starts one generation. The engine sets Busy and Streaming, appends
	// to Stream at no more than ~30 Hz, and finishes by writing the completed
	// text into the reply turn.
	Send(ChatRequest)
	// Stop cancels the generation in flight and keeps the model loaded. A Stop
	// with nothing running is not an error.
	Stop()
}

// Machine reads the hardware.
type Machine interface {
	// Probe reads the machine; it opens every backend, so it is slow and must
	// not run beside a live device tier.
	Probe() hardware.Report
	// Cached is the last probe, and whether there has been one.
	Cached() (hardware.Report, bool)
}

// Files finds model files and reads their headers.
type Files interface {
	Scan(dirs []string) []catalog.Entry
	Probe(e *catalog.Entry)
}

// Deps is what one shell's screens talk to. A nil Engine leaves the screens
// inert and saying so; a nil Machine or Files is the real one.
type Deps struct {
	Engine  Engine
	Machine Machine
	Files   Files
}

type depsBox struct{ v atomic.Pointer[Deps] }

func boxOf(sh *app.Shell) *depsBox {
	b, _ := app.Local(sh, func() *depsBox { return &depsBox{} })
	return b
}

// Attach binds a shell's screens to their dependencies. Call it before the
// first build; a screen reads its Deps when an action runs, so a later Attach
// reaches the controls already on screen too.
func Attach(sh *app.Shell, d Deps) { boxOf(sh).v.Store(&d) }

func depsOf(sh *app.Shell) Deps {
	var deps Deps
	if d := boxOf(sh).v.Load(); d != nil {
		deps = *d
	}
	if deps.Machine == nil {
		deps.Machine = &hardware.Host{}
	}
	if deps.Files == nil {
		deps.Files = catalog.Disk{}
	}
	return deps
}

// Screens is the window's tabs in registration order, indexed by the tab
// constants so the mapping is written down rather than inferred.
// TestEveryTabHasAScreen relates it to app.TabNames(), which can drift from it.
func Screens() []app.ScreenFunc {
	return []app.ScreenFunc{
		// Sampling belongs beside the conversation it changes.
		app.TabSession:  SessionWith(SessionOptions{Side: Tuning}),
		app.TabModels:   Models,
		app.TabDownload: Discover,
		app.TabConvert:  Convert,
		app.TabMachine:  Devices,
		app.TabSettings: Settings,
	}
}

// Install registers every tab and attaches d: the whole window, ready for
// Shell.Build. main, the shots and the stage tests all build it this way.
func Install(sh *app.Shell, d Deps) {
	for i, build := range Screens() {
		sh.Register(app.TabNames()[i], build)
	}
	Attach(sh, d)
}
