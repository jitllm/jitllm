package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/gogpu/gg"

	"github.com/jitllm/jitllm/examples/jivman/brain"
	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
	"github.com/jitllm/jitllm/examples/jivman/view"
)

// record plays one game exactly as the window does -- in real time, the model
// answering on its own goroutine, an actor waiting at its junction for the
// answer and taking the greedy rule's way after -timeout -- and draws every
// frame offscreen. Frames are numbered by wall-clock slot at -fps: a slot the
// drawing missed repeats the frame before it, so the frames played at -fps
// run at the speed the game ran.
func record(o options) error {
	if o.model == "" {
		return fmt.Errorf("-record needs -model")
	}
	e, err := findModel(o.models, o.model)
	if err != nil {
		return err
	}
	b, err := brain.Open(e, o.devices)
	if err != nil {
		return err
	}
	defer b.Close()
	fmt.Printf("%s (%s readout) on %s, %d blocks placed\n", e.Name, e.Readout, b.Device, b.Blocks)
	f, err := view.LoadFonts()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(o.record, 0o755); err != nil {
		return err
	}
	var board []scores.Row
	if st, err := scores.Open(o.db); err == nil {
		board, _ = st.Board()
		st.Close()
	}
	a := play.NewAsync(b, o.timeout, 1<<game.Pacman)
	defer a.Close()
	g := game.New(true, false)
	p := &view.Painter{F: f}
	dc := gg.NewContext(view.W, view.H)
	sc := &view.Scene{G: g, Stats: &a.Stats, PacmanBy: e.Name, GhostsBy: "Classic", Device: b.Device, Board: board,
		Model: e.Name, Placement: b.Placement()}
	for _, r := range board {
		sc.High = max(sc.High, r.Best)
	}
	fps := max(1, o.fps)
	t0 := time.Now()
	last := t0
	slot := -1
	name := func(i int) string { return filepath.Join(o.record, fmt.Sprintf("frame_%05d.png", i)) }
	for {
		now := time.Now()
		el := now.Sub(t0).Seconds()
		if el > o.seconds || g.Status == game.GameOver && el > float64(slot+1)/float64(fps)+2 {
			break
		}
		a.Frame(g)
		g.Step(min(now.Sub(last).Seconds(), 0.1), a)
		last = now
		cur := int(el * float64(fps))
		if cur == slot {
			time.Sleep(2 * time.Millisecond)
			continue
		}
		sc.T = el
		sc.Driven[game.Pacman] = true
		sc.Thinking[game.Pacman] = g.Status == game.Playing && g.Actors[game.Pacman].Waiting
		if g.Status == game.GameOver {
			sc.Summary = fmt.Sprintf("Game over: %d points, %d decisions, mean %.0f ms, p50 %.0f ms",
				g.Score, a.Decisions, ms(a.Mean()), ms(a.P50()))
		}
		p.Draw(dc, sc)
		if err := dc.SavePNG(name(cur)); err != nil {
			return err
		}
		// Slots the drawing took too long for repeat the last frame.
		for i := slot + 1; i < cur && slot >= 0; i++ {
			if err := os.Link(name(slot), name(i)); err != nil {
				return err
			}
		}
		slot = cur
	}
	fmt.Printf("%s: score %d, %d decisions in %d calls, %d fallbacks, mean %.1f ms, p50 %.1f ms, %d frames\n",
		e.Name, g.Score, a.Decisions, a.Calls, a.Fallbacks, ms(a.Mean()), ms(a.P50()), slot+1)
	return nil
}
