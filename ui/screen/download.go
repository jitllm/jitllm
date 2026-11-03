package screen

import (
	gocontext "context"
	"errors"
	"os"
	"sync"

	"github.com/gogpu/ui/state"

	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/ui/app"
)

// The Download screen: the models jitllm fetches by name, one press from the
// Hub to a container on the Models screen. What is offered and how it is
// fetched is [discover]'s.

// downloadScreen is this screen's state, keyed by shell like modelsScreen's.
type downloadScreen struct {
	sh *app.Shell

	sel      state.Signal[int]
	rows     state.Signal[int]
	have     state.Signal[[]bool] // a current container for each row is on disk
	busy     state.Signal[bool]   // one download at a time
	progress state.Signal[float64]
	note     state.Signal[string]
	pref     state.Signal[int] // the balance, an index into discover.BalanceSteps()
	picked   bool              // a person chose a model; stop following the top
	ord      state.ReadonlySignal[[]int]

	// Seams for the tests: the Hub, the whole-file download a stream falls
	// back to, and the disk.
	endpoint string
	noStream bool
	fetchFn  discover.FetchFunc
	free     discover.FreeFunc

	cancelMu sync.Mutex
	stop     gocontext.CancelFunc // the running download's
}

func downloadsFor(sh *app.Shell) *downloadScreen {
	d, created := app.Local(sh, func() *downloadScreen {
		return &downloadScreen{
			sh:       sh,
			sel:      state.NewSignal(-1),
			rows:     state.NewSignal(len(library.Models())),
			have:     state.NewSignal([]bool(nil)),
			busy:     state.NewSignal(false),
			progress: state.NewSignal(0.0),
			note:     state.NewSignal(""),
			pref:     state.NewSignal(discover.DefaultBalance),
			fetchFn:  discover.HubFetch,
			free:     discover.FreeBytes,
		}
	})
	if created {
		d.refresh()
	}
	return d
}

// dir is where downloads land; see [discover.Dir].
func (d *downloadScreen) dir() string {
	return discover.Dir(d.sh.Cfg.ModelDirs, d.sh.Cfg.DefaultModelDir())
}

// containerFor is where a library model's container is written.
func (d *downloadScreen) containerFor(m library.Model) string {
	return discover.ContainerFor(d.dir(), m)
}

// refresh re-reads which models are already on disk.
func (d *downloadScreen) refresh() { d.have.Set(discover.OnDisk(d.dir())) }

func (d *downloadScreen) selected() (library.Model, int, bool) {
	i := d.sel.Get()
	if i < 0 || i >= len(library.Models()) {
		return library.Model{}, i, false
	}
	return library.Models()[i], i, true
}

func (d *downloadScreen) onDisk(i int) bool {
	have := d.have.Get()
	return i >= 0 && i < len(have) && have[i]
}

// --- the action ----------------------------------------------------------------

// act is the button: Load a model that is on disk, fetch one that is not.
func (d *downloadScreen) act() {
	m, i, ok := d.selected()
	if !ok || d.busy.Get() {
		return
	}
	if d.onDisk(i) {
		if eng := depsOf(d.sh).Engine; eng != nil {
			eng.Load(d.containerFor(m))
			d.sh.GoSession()
		}
		return
	}
	d.fetch(m)
}

// fetch turns m (and its vision tower) into a container, converting the GGUF
// as it streams from the Hub, and says where the model is. The steps are
// [discover.Get]'s; this puts them on the screen.
func (d *downloadScreen) fetch(m library.Model) {
	dir := d.dir()
	if note := discover.SpaceNote(m, dir, d.free); note != "" {
		d.note.Set(note)
		return
	}

	d.busy.Set(true)
	d.progress.Set(0)
	d.note.Set("Downloading " + m.Title + "...")
	d.sh.SetStatus("downloading " + m.Name)

	ctx, cancel := gocontext.WithCancel(gocontext.Background())
	d.cancelMu.Lock()
	d.stop = cancel
	d.cancelMu.Unlock()
	run := modelsFor(d.sh).runner()
	hub := discover.Hub{
		Endpoint: d.endpoint,
		Token:    hf.FindToken(os.Getenv, os.UserHomeDir),
		Fetch:    d.fetchFn,
		NoStream: d.noStream,
		// The runner may return before the job has run; Get waits for it.
		Convert: func(job func()) bool {
			done := make(chan struct{})
			if !run(func() { defer close(done); job() }) {
				return false
			}
			<-done
			return true
		},
	}
	go func() {
		defer d.sh.Crashes().Recover("download")
		defer cancel()
		_, err := discover.Get(ctx, m, dir, hub, func(s discover.Step) {
			d.sh.Post(func() {
				d.progress.Set(s.Fraction)
				switch s.Stage {
				case "streaming":
					d.note.Set("Downloading " + m.Title + " and converting it as it arrives...")
				case "converting":
					d.note.Set("Converting " + m.Title + " to a .jlm (the bar is an estimate)...")
				}
			})
		})
		if errors.Is(err, gocontext.Canceled) {
			d.sh.Post(func() {
				d.busy.Set(false)
				d.note.Set("Stopped. Press Download to carry on where it stopped.")
				d.sh.SetStatus("download stopped: " + m.Name)
			})
			return
		}
		if err != nil {
			d.fail(m, err)
			return
		}
		d.sh.Post(func() {
			d.refresh()
			modelsFor(d.sh).rescan()
			d.busy.Set(false)
			d.note.Set(m.Title + " is ready. Press Load, or find it on the Models tab.")
			d.sh.SetStatus("downloaded " + m.Name)
		})
	}()
}

// cancel stops the running download; what it wrote is kept, and the next
// Download of the same model carries on from it.
func (d *downloadScreen) cancel() {
	d.cancelMu.Lock()
	defer d.cancelMu.Unlock()
	if d.stop != nil {
		d.stop()
	}
}

// fail says what went wrong in words a person can act on.
func (d *downloadScreen) fail(m library.Model, err error) {
	msg := discover.FailMessage(m, err)
	d.sh.Post(func() {
		d.busy.Set(false)
		d.note.Set(msg)
		d.sh.SetStatus("download failed: " + m.Name)
	})
}

// ShowDiscover switches to Discover with the named model selected. Call it
// on the UI goroutine.
func ShowDiscover(sh *app.Shell, name string) {
	d := downloadsFor(sh)
	for i, m := range library.Models() {
		if m.Name == name {
			d.picked = true
			d.sel.Set(i)
		}
	}
	sh.SelectTab(app.TabDownload)
}
