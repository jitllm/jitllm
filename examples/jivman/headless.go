package main

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/jitllm/jitllm/examples/jivman/brain"
	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
)

// Greedy is the pseudo-model that plays by the fixed rule: a baseline row on
// the board, and a run that needs no model.
const Greedy = "greedy"

// findModel resolves -model: a path to a .jlm, or a name (or part of one) of
// a decision model in the models folder.
func findModel(dir, name string) (brain.Entry, error) {
	if strings.HasSuffix(name, ".jlm") {
		return brain.EntryOf(name)
	}
	es := brain.List(dir)
	for _, e := range es {
		if strings.EqualFold(e.Name, name) {
			return e, nil
		}
	}
	for _, e := range es {
		if strings.Contains(strings.ToLower(e.Name), strings.ToLower(name)) || strings.EqualFold(e.Readout, name) {
			return e, nil
		}
	}
	names := make([]string, len(es))
	for i, e := range es {
		names[i] = e.Name
	}
	return brain.Entry{}, fmt.Errorf("no decision model %q in %s (found: %s)", name, dir, strings.Join(names, ", "))
}

// headlessRun is one game played without a window: the model plays Pac-Man
// against the classic ghosts, the clock stopping while it thinks.
type headlessRun struct {
	Game   *game.Game
	Stats  *play.Stats
	Who    string
	Device string
	// Placement is where the model ran (brain.Brain.Placement).
	Placement string
	Elapsed   time.Duration
}

func runHeadless(o options, w io.Writer, frame func(*game.Game, *play.Stats)) (*headlessRun, error) {
	g := game.New(true, false)
	var c game.Controls = game.GreedyControls{}
	st := &play.Stats{}
	who, device, placement := Greedy, "cpu", ""
	if o.model != Greedy {
		e, err := findModel(o.models, o.model)
		if err != nil {
			return nil, err
		}
		t := time.Now()
		b, err := brain.Open(e, o.devices)
		if err != nil {
			return nil, err
		}
		defer b.Close()
		fmt.Fprintf(w, "%s (%s readout) on %s (%s), opened in %v\n", e.Name, e.Readout, b.Device, b.Placement(),
			time.Since(t).Round(time.Millisecond))
		s := &play.Sync{B: b}
		c, st, who, device, placement = s, &s.Stats, e.Name, b.Device, b.Placement()
	}
	t := time.Now()
	for g.Ticks < o.ticks && g.Status != game.GameOver {
		g.Step(game.Substep, c)
		if frame != nil {
			frame(g, st)
		}
	}
	return &headlessRun{Game: g, Stats: st, Who: who, Device: device, Placement: placement, Elapsed: time.Since(t)}, nil
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func (r *headlessRun) print(w io.Writer) {
	g, s := r.Game, r.Stats
	status := "playing"
	if g.Status == game.GameOver {
		status = "game over"
	}
	fmt.Fprintf(w, "%s: score %d, level %d, lives %d, %d ticks (%s), %v wall\n", r.Who, g.Score, g.Level, max(g.Lives, 0),
		g.Ticks, status, r.Elapsed.Round(time.Millisecond))
	if r.Who != Greedy {
		fmt.Fprintf(w, "decisions %d in %d calls, %d fallbacks; latency mean %.1f ms, p50 %.1f ms, %d prompt tokens a call, on %s\n",
			s.Decisions, s.Calls, s.Fallbacks, ms(s.Mean()), ms(s.P50()), s.Tokens/max(1, s.Calls), r.Device)
	}
	if p10, p50, p90, n := s.Warm(); n > 0 {
		fmt.Fprintf(w, "warm calls: %d, p50 %.1f ms (p10 %.1f, p90 %.1f)\n", n, ms(p50), ms(p10), ms(p90))
	}
	if s.Err != nil {
		fmt.Fprintf(w, "last model error: %v\n", s.Err)
	}
}

func (r *headlessRun) run() scores.Run {
	return scores.Run{Mode: scores.ModeAI, Who: r.Who, Score: r.Game.Score, Level: r.Game.Level, Ticks: r.Game.Ticks,
		Decisions: r.Stats.Decisions, Fallbacks: r.Stats.Fallbacks, AvgMs: ms(r.Stats.Mean()), P50Ms: ms(r.Stats.P50()),
		Device: r.Device}
}

func printBoard(w io.Writer, b []scores.Row) {
	fmt.Fprintln(w, "\nleaderboard")
	fmt.Fprintf(w, "%-3s %-34s %5s %7s %16s %10s\n", "#", "player", "runs", "best", "mean ± 2 SE", "avg ms")
	for i, r := range b {
		fmt.Fprintf(w, "%-3d %-34s %5d %7d %9.0f ± %-5.0f %10.1f\n", i+1, r.Label(), r.Runs, r.Best, r.Mean, 2*r.SE, r.AvgMs)
	}
}

// firstRequest is the request a model playing Pac-Man is asked at his first
// junction, the greedy rule playing until then.
func firstRequest() []byte {
	g := game.New(true, false)
	var body []byte
	c := &asker{body: &body}
	for body == nil && g.Status != game.GameOver {
		g.Step(game.Substep, c)
	}
	return body
}

type asker struct{ body *[]byte }

func (a *asker) Decide(g *game.Game, p *game.Point) game.Dir {
	if *a.body == nil && !p.Escape {
		*a.body = g.Request(nil, []game.Point{*p})
	}
	return g.Greedy(p, g.Options(p))
}
