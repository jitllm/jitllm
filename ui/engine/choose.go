package engine

import (
	"context"

	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/screen"
)

// Engines is the window's engine as the engine setting chooses it: a jitllmd
// the window is a client of, or the window's own. Applying the setting again
// swaps it under the screens, which hold the Switch.
type Engines struct {
	sh *app.Shell
	sw core.Switch
	// stopAPI stops the embedded engine's API; nil when connected.
	stopAPI func()
}

// Start chooses the engine for the saved setting (service first, under Auto)
// and starts it. It asks the network before the window's own API starts.
// Shell.ConnectTo is set to switch to a service.
func Start(sh *app.Shell) *Engines {
	var _ screen.Engine = (*core.Switch)(nil)
	e := &Engines{sh: sh}
	sh.ConnectTo = func(addr string) {
		sh.Store.EngineMode.Set(core.ModeService)
		sh.Store.ServiceAddr.Set(addr)
		e.Apply()
	}
	sh.ApplyEngine = e.Apply
	e.install(core.Choose(context.Background(), sh.Store.EngineMode.Get(), sh.Store.ServiceAddr.Get(), localserver.WithAPIKey(sh.Cfg.ServiceAPIKey())))
	sh.OnShutdown(e.Close)
	return e
}

// Driver is what the screens hold.
func (e *Engines) Driver() *core.Switch { return &e.sw }

// Apply re-reads the engine setting and switches to what it chooses. The
// engine in use stops first (its API too, so Auto does not find the window
// itself); the probe runs off the UI goroutine.
func (e *Engines) Apply() {
	e.stop()
	mode, addr := e.sh.Store.EngineMode.Get(), e.sh.Store.ServiceAddr.Get()
	e.sh.SetStatus("choosing the engine…")
	go func() {
		ch := core.Choose(context.Background(), mode, addr, localserver.WithAPIKey(e.sh.Cfg.ServiceAPIKey()))
		e.sh.Post(func() { e.install(ch) })
	}()
}

// stop closes the engine in use and clears what it published.
func (e *Engines) stop() {
	if e.stopAPI != nil {
		e.stopAPI()
		e.stopAPI = nil
	}
	e.sw.Set(nil)
	st := e.sh.Store
	st.Models.Set(nil)
	st.Active.Set("")
	st.Loaded.Set(false)
	st.ModelPath.Set("")
	st.ModelSummary.Set("")
	st.BlockMap.Set(nil)
	st.Alloc.Set(session.Allocation{})
	st.Problem.Set(session.Problem{})
}

// install starts the engine ch chose and says which it is.
func (e *Engines) install(ch core.Choice) {
	sh := e.sh
	f, st := front{sh}, StateOf(sh.Store)
	if ch.Conn != nil {
		e.sw.Set(core.NewRemote(f, st, ch.Conn, ch.Addr, func(w core.Where) { sh.Store.Engine.Set(w) }))
		if ch.Conn.Info != nil {
			sh.SetStatus("connected to jitllmd " + ch.Conn.Info.GetVersion() + " at " + ch.Addr)
		}
		if ch.Warning != "" {
			sh.Report(session.Problem{Title: "The service at " + ch.Addr + " is refused", Detail: ch.Warning})
		}
		return
	}
	eng := core.New(f, st)
	sh.Store.Engine.Set(core.Where{})
	e.sw.Set(eng)
	if ch.Warning != "" {
		p := session.Problem{Title: "Two engines on this computer", Detail: ch.Warning}
		if ch.Beside {
			addr := ch.Addr
			p.Action, p.Do = "Connect to it", func() { sh.ConnectTo(addr) }
		}
		sh.Report(p)
	}
	e.stopAPI = sh.ServeAPI(eng.Server())
}

// Close stops the engine in use.
func (e *Engines) Close() {
	if e.stopAPI != nil {
		e.stopAPI()
		e.stopAPI = nil
	}
	e.sw.Close()
}
