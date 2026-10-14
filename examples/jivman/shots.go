package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/gogpu/gg"

	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
	"github.com/jitllm/jitllm/examples/jivman/view"
)

// shooter draws a headless game the way the window would, into PNGs: one at
// the end (-shot), and one every few ticks (-frames) for a video.
type shooter struct {
	o     options
	p     *view.Painter
	dc    *gg.Context
	n     int
	who   string
	board []scores.Row
	err   error
}

func newShooter(o options) *shooter {
	f, err := view.LoadFonts()
	s := &shooter{o: o, err: err, who: o.model}
	if err != nil {
		return s
	}
	s.p = &view.Painter{F: f}
	s.dc = gg.NewContext(view.W, view.H)
	if st, err := scores.Open(o.db); err == nil {
		s.board, _ = st.Board()
		st.Close()
	}
	if o.frames != "" {
		s.err = os.MkdirAll(o.frames, 0o755)
	}
	return s
}

func (s *shooter) scene(g *game.Game, st *play.Stats, device string) *view.Scene {
	sc := &view.Scene{G: g, Stats: st, PacmanBy: s.who, GhostsBy: "Classic", Device: device, Board: s.board,
		T: float64(g.Ticks) / 60}
	if s.o.frames != "" {
		sc.Banner = "game paused while the model thinks · real latency shown"
	}
	sc.Driven[game.Pacman] = true
	if len(s.board) > 0 {
		sc.High = s.board[0].Best
		for _, r := range s.board {
			sc.High = max(sc.High, r.Best)
		}
	}
	return sc
}

func (s *shooter) frame(g *game.Game, st *play.Stats) {
	if s.err != nil || s.o.frames == "" || g.Ticks%max(1, s.o.every) != 0 {
		return
	}
	s.p.Draw(s.dc, s.scene(g, st, ""))
	s.err = s.dc.SavePNG(filepath.Join(s.o.frames, fmt.Sprintf("frame_%05d.png", s.n)))
	s.n++
}

func (s *shooter) finish(r *headlessRun) error {
	if s.err != nil {
		return s.err
	}
	if s.o.shot == "" {
		return nil
	}
	sc := s.scene(r.Game, r.Stats, r.Device)
	sc.Summary = fmt.Sprintf("%s scored %d in %d ticks: %d decisions, mean %.0f ms, p50 %.0f ms",
		r.Who, r.Game.Score, r.Game.Ticks, r.Stats.Decisions, ms(r.Stats.Mean()), ms(r.Stats.P50()))
	s.p.Draw(s.dc, sc)
	return s.dc.SavePNG(s.o.shot)
}
