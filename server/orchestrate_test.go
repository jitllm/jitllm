package server

import (
	"context"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// orchFixture loads two models on one card whose shared pool cannot hold
// both, and the same two on the CPU alone as the controls.
type orchFixture struct {
	e   *Engine
	c   clients
	ctx context.Context
	t   *testing.T
}

func newOrchFixture(t *testing.T, cfg Config) *orchFixture {
	devs := testdevices.Take(t, 1, "cuda:0")
	cfg.Version, cfg.NoMemCache = "test", true
	e := New(cfg)
	t.Cleanup(e.Close)
	// 1200 MiB of one card: the first model fits, the second does not beside it.
	// The CPU is listed: it is where the orchestrator moves a victim's blocks.
	spec := devs[0] + "=1200M"
	for _, m := range []struct{ id, file string }{
		{"a", "Qwen3-0.6B-Q8_0.jlm"},
		{"b", "Llama-3.2-1B-Instruct-Q4_K_M.jlm"},
	} {
		if _, err := e.LoadModel(LoadOptions{Path: modelPath(t, m.file), ModelID: m.id, DeviceIDs: []string{spec, nn.CPU}}); err != nil {
			t.Fatal(err)
		}
		if _, err := e.LoadModel(LoadOptions{Path: modelPath(t, m.file), ModelID: m.id + "-host", DeviceIDs: []string{"cpu"}}); err != nil {
			t.Fatal(err)
		}
	}
	return &orchFixture{e: e, c: serveEngine(t, e), ctx: context.Background(), t: t}
}

func (f *orchFixture) open(model, id string) {
	f.t.Helper()
	if _, err := f.e.CreateSession(SessionOptions{ModelID: model, SessionID: id, MaxSeq: 160}); err != nil {
		f.t.Fatal(err)
	}
}

func (f *orchFixture) turn(id, prompt string) []int32 {
	f.t.Helper()
	r, err := f.c.inference.Complete(f.ctx, req(&v1.GenerateRequest{SessionId: id, Prompt: text(prompt), MaxTokens: 6}))
	if err != nil {
		f.t.Fatal(err)
	}
	return r.Msg.GetTokenIds()
}

func (f *orchFixture) dev(id string) int {
	f.t.Helper()
	pl, err := f.c.placement.GetPlacement(f.ctx, req(&v1.GetPlacementRequest{SessionId: id}))
	if err != nil {
		f.t.Fatal(err)
	}
	return int(pl.Msg.GetPlacement().GetDeviceBlockCount())
}

func orchTurns() []string {
	return []string{story, " and then", " until"}
}

// TestTheTalkingModelGetsTheCard: two models on a card that holds one. Each
// turn, the model generating is promoted onto the card and the idle one is
// demoted home to make room, alternately, and every reply is the token-for-
// token continuation of an unmoved session (on the host or on the card) at
// each turn's first token: the history travels, nothing is recomputed or
// refused.
//
// VIOLATION SIGNATURE. Without the shared pool (gpu.WithPoolRegistry) the
// second model never gets a block; without promote, a session demoted once
// stays home ("b holds 0 blocks after its turn"); a history dropped on a
// move parts a reply from its control.
func TestTheTalkingModelGetsTheCard(t *testing.T) {
	f := newOrchFixture(t, Config{})
	f.open("a-host", "ah")
	f.open("b-host", "bh")
	var ctlA, ctlB [][]int32
	for _, p := range orchTurns() {
		ctlA = append(ctlA, f.turn("ah", p))
		ctlB = append(ctlB, f.turn("bh", p))
	}
	// Controls that never leave the card: CPU-vs-device replies part in the
	// reduction-order band (RULE 11c). The engine's gates hold a moved history
	// to the unmoved one exactly (TestAnyRangeMovesWithItsHistory); this one
	// holds the orchestrated conversation to coherence with both controls.
	devA, devB := deviceControls(t)
	f.open("a", "a")
	f.open("b", "b")
	// The first token of a turn is sampled straight from the moved history;
	// later ones may part from either control in the band, and do (the card
	// control parts from the host's at the fourth token here).
	same := func(got []int32, ctl ...[]int32) bool {
		for _, c := range ctl {
			if len(got) > 0 && len(c) > 0 && got[0] == c[0] {
				return true
			}
		}
		return false
	}
	for i, p := range orchTurns() {
		if got := f.turn("a", p); !same(got, ctlA[i], devA[i]) {
			t.Fatalf("turn %d: a's reply %v parts from the host's %v and the card's %v", i, got, ctlA[i], devA[i])
		}
		if f.dev("a") == 0 {
			t.Fatalf("turn %d: a holds 0 blocks after its turn", i)
		}
		if got := f.turn("b", p); !same(got, ctlB[i], devB[i]) {
			t.Fatalf("turn %d: b's reply %v parts from the host's %v and the card's %v", i, got, ctlB[i], devB[i])
		}
		if f.dev("b") == 0 {
			pro, dem, up, home := f.e.OrchCounts()
			t.Fatalf("turn %d: b holds 0 blocks after its turn (a %d; %d promotions, %d demotions, %d up, %d home; moves %+v; b declines %+v)",
				i, f.dev("a"), pro, dem, up, home, f.e.OrchMoves(), f.declines("b"))
		}
		t.Logf("turn %d: a %d blocks, b %d blocks on the card", i, f.dev("a"), f.dev("b"))
	}
	pro, dem, up, home := f.e.OrchCounts()
	t.Logf("%d promotions, %d demotions, %d blocks up, %d home", pro, dem, up, home)
	if dem == 0 {
		t.Fatalf("no session was demoted: the card held both models, so the gate saw nothing")
	}
	if len(f.e.OrchMoves()) == 0 {
		t.Fatalf("moves were made and none was recorded")
	}
	// The stats stream carries them.
	st, err := f.c.telemetry.GetStats(f.ctx, req(&v1.GetStatsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if o := st.Msg.GetOrchestrator(); o.GetDemotions() != dem || len(o.GetMoves()) != len(f.e.OrchMoves()) {
		t.Fatalf("the stats carry %d demotions and %d moves; the engine made %d and recorded %d",
			o.GetDemotions(), len(o.GetMoves()), dem, len(f.e.OrchMoves()))
	}
}

// TestAPinnedSessionIsNotMoved: a session the caller placed keeps its blocks
// when another model talks, and with the orchestrator off nothing moves.
//
// VIOLATION SIGNATURE. Drop the pin check from demotionVictim and b loses
// its blocks; ignore NoOrchestrate and a's blocks change.
func TestAPinnedSessionIsNotMoved(t *testing.T) {
	t.Run("pinned", func(t *testing.T) {
		f := newOrchFixture(t, Config{})
		f.open("b", "b")
		f.turn("b", story)
		nb := int32(f.e.mustModel(t, "b").m.Cfg.NLayer)
		dev := f.e.mustModel(t, "b").devIDs()[0]
		if _, err := f.c.placement.RelocateBlocks(f.ctx, req(&v1.RelocateBlocksRequest{
			SessionId: "b", FirstBlock: 0, LastBlock: nb - 1, ToDeviceId: dev,
		})); err != nil {
			t.Fatal(err)
		}
		held := f.dev("b")
		if held == 0 {
			t.Skip("the card took none of b's blocks")
		}
		f.open("a", "a")
		f.turn("a", story)
		if got := f.dev("b"); got != held {
			t.Fatalf("the pinned session went from %d to %d blocks (moves %+v)", held, got, f.e.OrchMoves())
		}
	})
	t.Run("off", func(t *testing.T) {
		f := newOrchFixture(t, Config{NoOrchestrate: true})
		f.open("a", "a")
		before := f.dev("a")
		f.turn("a", story)
		if got := f.dev("a"); got != before {
			t.Fatalf("with the orchestrator off a went from %d to %d blocks", before, got)
		}
		if pro, dem, _, _ := f.e.OrchCounts(); pro+dem != 0 {
			t.Fatalf("with the orchestrator off it made %d promotions and %d demotions", pro, dem)
		}
	})
}

func (e *Engine) mustModel(t *testing.T, id string) *LoadedModel {
	t.Helper()
	lm, err := e.Model(id)
	if err != nil {
		t.Fatal(err)
	}
	return lm
}

func (f *orchFixture) declines(id string) []any {
	s, err := f.e.Session(id)
	if err != nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []any
	for _, d := range s.in().DeviceDeclines() {
		out = append(out, d)
	}
	return out
}

// deviceControls is each model's replies to orchTurns() from a session that
// holds every block on the card from its first token to its last.
func deviceControls(t *testing.T) (a, b [][]int32) {
	devs := testdevices.Take(t, 1, "cuda:0")
	e := New(Config{Version: "test", NoMemCache: true, NoOrchestrate: true})
	defer e.Close()
	f := &orchFixture{e: e, c: serveEngine(t, e), ctx: context.Background(), t: t}
	for _, m := range []struct {
		id, file string
		out      *[][]int32
	}{{"a", "Qwen3-0.6B-Q8_0.jlm", &a}, {"b", "Llama-3.2-1B-Instruct-Q4_K_M.jlm", &b}} {
		if _, err := e.LoadModel(LoadOptions{Path: modelPath(t, m.file), ModelID: m.id, DeviceIDs: []string{devs[0]}}); err != nil {
			t.Fatal(err)
		}
		f.open(m.id, m.id)
		for _, p := range orchTurns() {
			*m.out = append(*m.out, f.turn(m.id, p))
		}
		if n, all := f.dev(m.id), int(e.mustModel(t, m.id).m.Cfg.NLayer); n != all {
			t.Fatalf("the card control of %s ran %d of %d blocks on the card", m.id, n, all)
		}
		if _, err := e.UnloadModel(m.id, true); err != nil {
			t.Fatal(err)
		}
	}
	return a, b
}
