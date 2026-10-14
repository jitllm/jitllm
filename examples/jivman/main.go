// Command jivman is Pac-Man played by a decision model running inside the
// process: pick a model, press Play and watch it steer Pac-Man, or play him
// yourself, or play him against the model's ghosts. A Go re-implementation of
// jevman (github.com/joch/jevman, see README.md) on jitllm and gogpu.
//
//	jivman                                    the window
//	jivman -headless -ticks 3600 -model laya  one game without a window
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
)

type options struct {
	models   string
	model    string
	devices  string
	headless bool
	ticks    int
	timeout  time.Duration
	db       string
	shot     string
	frames   string
	every    int
	request  string
}

func main() {
	var o options
	// The command reads the environment and hands it over; the libraries
	// read none.
	flag.StringVar(&o.models, "models", catalog.DefaultDir(os.Getenv("JITLLM_MODELS"), os.Getenv("XDG_DATA_HOME")),
		"the folder the decision models (.jlm) are found in")
	flag.StringVar(&o.model, "model", "", "the model to play: a name from the models folder, a .jlm path, or greedy (the fixed rule)")
	flag.StringVar(&o.devices, "devices", "auto", "where the model runs, as jitllm's -devices: auto, cpu, cuda, vulkan, metal ...")
	flag.BoolVar(&o.headless, "headless", false, "play one game without a window and print the score")
	flag.IntVar(&o.ticks, "ticks", 60*180, "headless: the most 60 Hz ticks to play")
	flag.DurationVar(&o.timeout, "timeout", 2*time.Second,
		"window: how long an actor waits at a junction for the model before the greedy rule decides (0 waits for the model)")
	flag.StringVar(&o.db, "db", scores.DefaultPath(), "the scoreboard database")
	flag.StringVar(&o.shot, "shot", "", "headless: write the window's last frame here as PNG")
	flag.StringVar(&o.frames, "frames", "", "headless: write a PNG frame into this folder every -every ticks")
	flag.IntVar(&o.every, "every", 4, "headless: ticks between -frames")
	flag.StringVar(&o.request, "request", "", "write the first request a model would be asked (System One JSON) to this file and exit")
	flag.Parse()
	if flag.NArg() > 0 {
		fmt.Fprintf(os.Stderr, "jivman takes no arguments (got %q); flags go before nothing\n", flag.Args())
		os.Exit(2)
	}
	var err error
	if o.request != "" {
		err = os.WriteFile(o.request, firstRequest(), 0o644)
	} else if o.headless {
		err = headless(o)
	} else {
		err = window(o)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "jivman:", err)
		os.Exit(1)
	}
}

func headless(o options) error {
	if o.model == "" {
		return fmt.Errorf("-headless needs -model (a name, a .jlm, or %s)", Greedy)
	}
	var shots *shooter
	if o.shot != "" || o.frames != "" {
		shots = newShooter(o)
	}
	var frame func(*game.Game, *play.Stats)
	if shots != nil {
		frame = shots.frame
	}
	r, err := runHeadless(o, os.Stdout, frame)
	if err != nil {
		return err
	}
	r.print(os.Stdout)
	if shots != nil {
		if err := shots.finish(r); err != nil {
			return err
		}
	}
	s, err := scores.Open(o.db)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := s.Add(r.run()); err != nil {
		return err
	}
	b, err := s.Board()
	if err != nil {
		return err
	}
	printBoard(os.Stdout, b)
	return nil
}
