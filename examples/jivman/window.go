package main

import (
	"fmt"
	"time"

	"github.com/gogpu/gg"
	"github.com/gogpu/gg/integration/ggcanvas"
	"github.com/gogpu/gogpu"
	"github.com/gogpu/gpucontext"

	"github.com/jitllm/jitllm/examples/jivman/brain"
	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
	"github.com/jitllm/jitllm/examples/jivman/view"
)

// win is the window's state. Every method runs on the window's thread (draw
// and the input callbacks); a model loads on a goroutine and hands its result
// back through loaded, and decisions run on play.Async's goroutines.
type win struct {
	o      options
	app    *gogpu.App
	canvas *ggcanvas.Canvas
	paint  *view.Painter
	store  *scores.Store
	models []brain.Entry
	brains map[string]*brain.Brain

	g      *game.Game
	sc     view.Scene
	pac    *play.Async
	ghosts *play.Async
	stats  play.Stats
	// tainted is a game where the controls changed hands: not recorded.
	tainted  bool
	recorded bool
	paused   bool
	last     time.Time
	t0       time.Time

	loaded chan loadResult
}

type loadResult struct {
	b   []*brain.Brain
	err error
}

func window(o options) error {
	f, err := view.LoadFonts()
	if err != nil {
		return err
	}
	store, err := scores.Open(o.db)
	if err != nil {
		return err
	}
	defer store.Close()
	w := &win{o: o, paint: &view.Painter{F: f}, store: store, models: brain.List(o.models), brains: map[string]*brain.Brain{},
		loaded: make(chan loadResult, 1), t0: time.Now()}
	defer func() {
		w.stop()
		for _, b := range w.brains {
			b.Close()
		}
	}()
	w.g = game.New(false, false)
	w.sc = view.Scene{G: w.g, PacmanBy: "You", GhostsBy: "Classic", Device: "cpu"}
	d := &view.Dialog{Pacman: -1, Ghosts: -1}
	for i, e := range w.models {
		d.Models = append(d.Models, e.Name)
		if o.model != "" && (e.Name == o.model || e.Readout == o.model) {
			d.Pacman = i
		}
	}
	if len(w.models) == 0 {
		d.Status = "No decision model in " + o.models + " (-models DIR); you can still play."
	} else if d.Pacman < 0 && o.model == "" {
		d.Pacman = 0
	}
	w.sc.Dialog = d
	w.refreshBoard()

	w.app = gogpu.NewApp(gogpu.DefaultConfig().
		WithTitle("jivman").
		WithSize(view.W, view.H).
		WithMinSize(view.W, view.H).
		WithContinuousRender(true))
	ev := w.app.EventSource()
	ev.OnKeyPress(w.key)
	ev.OnMousePress(func(b gpucontext.MouseButton, x, y float64) {
		if b == gpucontext.MouseButtonLeft {
			w.click(x, y)
		}
	})
	w.app.OnDraw(w.draw)
	return w.app.Run()
}

func (w *win) refreshBoard() {
	if b, err := w.store.Board(); err == nil {
		w.sc.Board = b
	}
	if hs, err := w.store.HighScore(); err == nil {
		w.sc.High = hs
	}
}

// offset is where the page sits in a window larger than it.
func (w *win) offset() (float64, float64) {
	if w.canvas == nil {
		return 0, 0
	}
	ww, wh := w.app.Size()
	return max(0, float64(ww-view.W)/2), max(0, float64(wh-view.H)/2)
}

func (w *win) draw(dc *gogpu.Context) {
	if w.canvas == nil {
		p := w.app.GPUContextProvider()
		if p == nil {
			return
		}
		c, err := ggcanvas.New(p, view.W, view.H)
		if err != nil {
			return
		}
		w.canvas = c
		w.last = time.Now()
	}
	now := time.Now()
	dt := min(now.Sub(w.last).Seconds(), 0.1)
	w.last = now
	w.poll()
	if !w.paused && w.sc.Dialog == nil {
		var c game.Controls = w
		if w.pac != nil {
			w.pac.Frame(w.g)
		}
		if w.ghosts != nil && w.ghosts != w.pac {
			w.ghosts.Frame(w.g)
		}
		w.g.Step(dt, c)
	}
	w.finish()
	w.sc.T = now.Sub(w.t0).Seconds()
	w.sc.Paused = w.paused
	w.sc.Stats = w.mergedStats()
	for id := 0; id < game.NActors; id++ {
		w.sc.Driven[id] = w.controller(id) != nil && w.g.Driven(id)
		w.sc.Thinking[id] = w.sc.Driven[id] && w.g.Status == game.Playing && w.g.Actors[id].Waiting
	}
	// Draw marks the canvas dirty, so the frame is uploaded.
	w.canvas.Draw(func(cc *gg.Context) { w.paint.Draw(cc, &w.sc) })
	ox, oy := w.offset()
	dc.Clear(0.043, 0.043, 0.07, 1)
	w.canvas.RenderToPosition(dc.AsTextureDrawer(), float32(ox), float32(oy))
}

// controller is the model playing actor id, or nil.
func (w *win) controller(id int) *play.Async {
	if id == game.Pacman {
		return w.pac
	}
	return w.ghosts
}

// Decide routes each question to the model playing that actor.
func (w *win) Decide(g *game.Game, p *game.Point) game.Dir {
	if a := w.controller(p.Actor); a != nil {
		return a.Decide(g, p)
	}
	return g.Greedy(p, g.Options(p))
}

// mergedStats is the panel's view of both sides' models.
func (w *win) mergedStats() *play.Stats {
	s := &w.stats
	*s = play.Stats{}
	for _, a := range w.players() {
		s.Calls += a.Calls
		s.Tokens += a.Tokens
		s.Decisions += a.Decisions
		s.Fallbacks += a.Fallbacks
		s.Latencies = append(s.Latencies, a.Latencies...)
		if a.Err != nil {
			s.Err = a.Err
		}
		for id := 0; id < game.NActors; id++ {
			if w.controller(id) == a {
				s.Last[id] = a.Last[id]
			}
		}
	}
	return s
}

// players is each model player once (the two sides may share one).
func (w *win) players() []*play.Async {
	var out []*play.Async
	if w.pac != nil {
		out = append(out, w.pac)
	}
	if w.ghosts != nil && w.ghosts != w.pac {
		out = append(out, w.ghosts)
	}
	return out
}

func (w *win) stop() {
	for _, a := range w.players() {
		a.Close()
	}
	w.pac, w.ghosts = nil, nil
}

// start loads the dialog's models (on a goroutine) and then starts a game.
func (w *win) start() {
	d := w.sc.Dialog
	var want []brain.Entry
	for _, i := range []int{d.Pacman, d.Ghosts} {
		if i >= 0 {
			want = append(want, w.models[i])
		}
	}
	d.Busy = true
	d.Status = "Loading…"
	if len(want) > 0 {
		d.Status = "Loading " + want[0].Name + " into jitllm…"
	}
	have := w.brains
	devices := w.o.devices
	go func() {
		var out []*brain.Brain
		for _, e := range want {
			b, ok := have[e.Path]
			if !ok {
				var err error
				if b, err = brain.Open(e, devices); err != nil {
					w.loaded <- loadResult{err: fmt.Errorf("%s: %w", e.Name, err)}
					return
				}
			}
			out = append(out, b)
		}
		w.loaded <- loadResult{b: out}
	}()
}

func (w *win) poll() {
	select {
	case r := <-w.loaded:
		d := w.sc.Dialog
		if d == nil {
			return
		}
		d.Busy = false
		if r.err != nil {
			d.Status = "error: " + r.err.Error()
			return
		}
		for _, b := range r.b {
			w.brains[b.Path] = b
		}
		w.newGame(d.Pacman, d.Ghosts)
	default:
	}
}

func (w *win) brainOf(i int) *brain.Brain {
	if i < 0 {
		return nil
	}
	return w.brains[w.models[i].Path]
}

// newGame starts a game with Pac-Man played by model pac (-1: the keyboard)
// and the ghosts by model ghosts (-1: the classic rule).
func (w *win) newGame(pac, ghosts int) {
	w.stop()
	pb, gb := w.brainOf(pac), w.brainOf(ghosts)
	w.g = game.New(pb != nil, gb != nil)
	w.sc.G = w.g
	w.sc.Summary = ""
	w.sc.PacmanBy, w.sc.GhostsBy, w.sc.Device = "You", "Classic", "cpu"
	const ghostMask = 1<<game.Blinky | 1<<game.Pinky | 1<<game.Inky | 1<<game.Clyde
	switch {
	case pb != nil && pb == gb:
		w.pac = play.NewAsync(pb, w.o.timeout, 1<<game.Pacman|ghostMask)
		w.ghosts = w.pac
	default:
		if pb != nil {
			w.pac = play.NewAsync(pb, w.o.timeout, 1<<game.Pacman)
		}
		if gb != nil {
			w.ghosts = play.NewAsync(gb, w.o.timeout, ghostMask)
		}
	}
	if pb != nil {
		w.sc.PacmanBy, w.sc.Device = pb.Name, pb.Device
	}
	if gb != nil {
		w.sc.GhostsBy, w.sc.Device = gb.Name, gb.Device
	}
	w.tainted, w.recorded, w.paused = false, false, false
	w.sc.Dialog = nil
}

// finish records a game that just ended and writes its summary.
func (w *win) finish() {
	g := w.g
	if g.Status != game.GameOver || w.recorded || w.sc.Dialog != nil {
		return
	}
	w.recorded = true
	s := w.mergedStats()
	r := scores.Run{Score: g.Score, Level: g.Level, Ticks: g.Ticks, Decisions: s.Decisions, Fallbacks: s.Fallbacks,
		AvgMs: ms(s.Mean()), P50Ms: ms(s.P50()), Device: w.sc.Device}
	switch {
	case w.tainted:
	case w.pac != nil && w.ghosts == nil:
		r.Mode, r.Who = scores.ModeAI, w.sc.PacmanBy
	case w.pac == nil && w.ghosts == nil:
		r.Mode, r.Who = scores.ModeHuman, "human"
	case w.pac == nil:
		r.Mode, r.Who = scores.ModeGhosts, w.sc.GhostsBy
	}
	w.sc.Summary = fmt.Sprintf("Game over: %d points, level %d", g.Score, g.Level)
	if s.Decisions > 0 {
		w.sc.Summary += fmt.Sprintf(", %d decisions, mean %.0f ms, p50 %.0f ms", s.Decisions, ms(s.Mean()), ms(s.P50()))
	}
	if r.Mode == "" {
		w.sc.Summary += " (not ranked: the controls changed hands, or models played both sides)"
		return
	}
	if err := w.store.Add(r); err != nil {
		w.sc.Summary += " (not saved: " + err.Error() + ")"
	}
	w.refreshBoard()
}

func (w *win) openDialog() {
	if w.sc.Dialog == nil {
		d := &view.Dialog{Pacman: -1, Ghosts: -1}
		for _, e := range w.models {
			d.Models = append(d.Models, e.Name)
		}
		if len(w.models) > 0 {
			d.Pacman = 0
		}
		w.sc.Dialog = d
	}
}

func (w *win) act(a view.Action) {
	d := w.sc.Dialog
	switch a.Kind {
	case view.ActOpenPlay, view.ActRestart:
		w.openDialog()
	case view.ActPickPacman:
		if d != nil && !d.Busy {
			d.Pacman = a.Index
		}
	case view.ActPickGhosts:
		if d != nil && !d.Busy {
			d.Ghosts = a.Index
		}
	case view.ActStart:
		if d != nil && !d.Busy {
			w.start()
		}
	case view.ActCloseDialog:
		if d != nil && !d.Busy {
			w.sc.Dialog = nil
		}
	case view.ActTab:
		w.sc.Tab = view.Tab(a.Index)
	case view.ActPause:
		w.paused = !w.paused
	case view.ActTakeOver:
		w.takeOver()
	}
}

// takeOver hands Pac-Man between the keyboard and the last model loaded
// (jevman's J). The run is no longer one player's, so it is not ranked.
func (w *win) takeOver() {
	if w.sc.Dialog != nil || w.g.Status == game.GameOver {
		return
	}
	w.tainted = true
	if w.g.PacmanAI {
		w.g.PacmanAI = false
		w.sc.PacmanBy = "You"
		return
	}
	var b *brain.Brain
	if w.ghosts != nil {
		b = w.ghosts.B
	} else {
		for _, x := range w.brains {
			b = x
			break
		}
	}
	if b == nil {
		return
	}
	if w.pac == nil {
		if w.ghosts != nil && w.ghosts.B == b {
			w.pac = w.ghosts
			w.pac.Actors |= 1 << game.Pacman
		} else {
			w.pac = play.NewAsync(b, w.o.timeout, 1<<game.Pacman)
		}
	}
	w.g.PacmanAI = true
	w.sc.PacmanBy = b.Name
}

func (w *win) click(x, y float64) {
	ox, oy := w.offset()
	w.act(w.sc.At(x-ox, y-oy))
}

func (w *win) key(k gpucontext.Key, _ gpucontext.Modifiers) {
	if d := w.sc.Dialog; d != nil {
		switch k {
		case gpucontext.KeyEnter, gpucontext.KeySpace:
			w.act(view.Action{Kind: view.ActStart})
		case gpucontext.KeyEscape:
			w.act(view.Action{Kind: view.ActCloseDialog})
		}
		return
	}
	switch k {
	case gpucontext.KeyUp, gpucontext.KeyW:
		w.g.Key = game.Up
	case gpucontext.KeyDown, gpucontext.KeyS:
		w.g.Key = game.Down
	case gpucontext.KeyLeft, gpucontext.KeyA:
		w.g.Key = game.Left
	case gpucontext.KeyRight, gpucontext.KeyD:
		w.g.Key = game.Right
	case gpucontext.KeyP:
		w.paused = !w.paused
	case gpucontext.KeyR:
		w.openDialog()
	case gpucontext.KeyJ:
		w.takeOver()
	case gpucontext.KeyEnter, gpucontext.KeySpace:
		if w.g.Status == game.GameOver {
			w.openDialog()
		}
	}
}
