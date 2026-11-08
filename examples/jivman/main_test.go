package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/examples/jivman/brain"
	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
)

// smallestModel is the smallest decision model under $JITLLM_MODELS (the
// folder the engine's own tests read), or "" with the reason.
func smallestModel() (string, string) {
	dir := os.Getenv("JITLLM_MODELS")
	if dir == "" {
		return "", "JITLLM_MODELS is not set"
	}
	es := brain.List(dir)
	if len(es) == 0 {
		return "", "no decision model (.jlm with a decision readout) under " + dir
	}
	return es[0].Path, ""
}

// TestHeadlessPlaysWithAModel plays a short game with the smallest decision
// model there is, in process, and checks the model made the decisions.
func TestHeadlessPlaysWithAModel(t *testing.T) {
	path, why := smallestModel()
	if path == "" {
		t.Skip("SKIPPED, no model ran: " + why)
	}
	t.Logf("model %s", path)
	var out bytes.Buffer
	o := options{model: path, devices: "auto", ticks: 600}
	r, err := runHeadless(o, &out, nil)
	if err != nil {
		t.Fatal(err)
	}
	r.print(&out)
	t.Log(out.String())
	if r.Stats.Err != nil {
		t.Fatalf("the model failed: %v", r.Stats.Err)
	}
	if r.Stats.Calls == 0 || r.Stats.Decisions == 0 {
		t.Fatalf("%d calls, %d decisions: the model was never asked", r.Stats.Calls, r.Stats.Decisions)
	}
	if r.Stats.Fallbacks != 0 {
		t.Fatalf("%d fallbacks: headless waits for the model, so none should be needed", r.Stats.Fallbacks)
	}
	if r.Game.Ticks < 600 && r.Game.Lives > 0 {
		t.Fatalf("stopped at tick %d with lives left", r.Game.Ticks)
	}
}

// TestHeadlessGreedyRecordsARun is the headless command end to end with the
// fixed rule: it plays, saves the run and prints the board.
func TestHeadlessGreedyRecordsARun(t *testing.T) {
	db := filepath.Join(t.TempDir(), "scores.musq")
	o := options{model: Greedy, ticks: 1200, db: db}
	var out bytes.Buffer
	r, err := runHeadless(o, &out, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Game.Score == 0 {
		t.Fatal("the greedy rule scored nothing")
	}
	s, err := scores.Open(db)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Add(r.run()); err != nil {
		t.Fatal(err)
	}
	b, err := s.Board()
	if err != nil || len(b) != 1 || b[0].Best != r.Game.Score || b[0].Who != Greedy {
		t.Fatalf("board %+v, %v", b, err)
	}
	printBoard(&out, b)
	if !strings.Contains(out.String(), "greedy") {
		t.Fatalf("the board does not print the run:\n%s", out.String())
	}
}

// TestTheWindowsPlayerNeverBlocksTheFrame drives a game the way the window
// does -- Frame, then Step, once a frame -- with the model answering on its
// own goroutine, and checks that the model's answers steer Pac-Man and that
// no frame waited for the model.
func TestTheWindowsPlayerNeverBlocksTheFrame(t *testing.T) {
	path, why := smallestModel()
	if path == "" {
		t.Skip("SKIPPED, no model ran: " + why)
	}
	e, err := brain.EntryOf(path)
	if err != nil {
		t.Fatal(err)
	}
	b, err := brain.Open(e, "auto")
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	a := play.NewAsync(b, 0, 1<<game.Pacman)
	defer a.Close()
	g := game.New(true, false)
	var worst time.Duration
	deadline := time.Now().Add(20 * time.Second)
	for a.Decisions < 5 && time.Now().Before(deadline) && g.Status != game.GameOver {
		t0 := time.Now()
		a.Frame(g)
		g.Step(1.0/60, a)
		worst = max(worst, time.Since(t0))
		time.Sleep(time.Millisecond)
	}
	if a.Err != nil {
		t.Fatal(a.Err)
	}
	if a.Decisions < 5 {
		t.Fatalf("%d decisions in 20 s", a.Decisions)
	}
	if a.Fallbacks != 0 {
		t.Fatalf("%d fallbacks with no timeout", a.Fallbacks)
	}
	// A frame builds the request and steps the game; the model's time is
	// elsewhere. 50 ms is far above that and far below any model call.
	if worst > 50*time.Millisecond {
		t.Fatalf("a frame took %v: the model ran on the render thread", worst)
	}
	t.Logf("%d decisions, %d calls, mean %v, worst frame %v", a.Decisions, a.Calls, a.Mean(), worst)
}

func TestFindModelByName(t *testing.T) {
	path, why := smallestModel()
	if path == "" {
		t.Skip("SKIPPED: " + why)
	}
	e, err := findModel(os.Getenv("JITLLM_MODELS"), strings.TrimSuffix(filepath.Base(path), ".jlm"))
	if err != nil || e.Path == "" {
		t.Fatalf("%v %+v", err, e)
	}
	if _, err := findModel(os.Getenv("JITLLM_MODELS"), "no-such-model-xyz"); err == nil {
		t.Fatal("an unknown model was found")
	}
}
