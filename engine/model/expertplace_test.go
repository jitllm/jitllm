//go:build linux

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestParsePlacementExperts: %host and %card name where a block's experts run,
// anything else is refused, and the host cannot take them beside itself.
func TestParsePlacementExperts(t *testing.T) {
	p, err := ParsePlacement("1-3=cuda:0%host,4=cuda:0~%card", false)
	if err != nil {
		t.Fatal(err)
	}
	if p.Blocks[2].Experts != "host" || p.Blocks[4].Experts != "card" || !p.Blocks[4].Stream || p.Blocks[2].On != "cuda:0" {
		t.Fatalf("parsed %+v", p.Blocks)
	}
	for _, bad := range []string{"1=cuda:0%gpu", "1=host%host"} {
		if _, err := ParsePlacement(bad, false); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// runExperts runs ids through m, its blocks on tier g (the host alone when g
// is nil), and returns the logits.
func runExperts(t *testing.T, m *Model, g *tier.GPU, ids []int32) [][]float32 {
	t.Helper()
	s := m.NewState(32)
	defer s.Close()
	if g != nil {
		s.SetDeviceLayers(g, m.Cfg.NLayer)
	}
	var out [][]float32
	for _, id := range ids {
		l, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), l...))
	}
	return out
}

// TestExpertPlacementIsHonoured: WithExperts places every mixture block's
// experts off the card even where the blocks fit -- "host" runs them hybrid,
// "card" streams their sheets -- and each run takes its path alone (counted,
// RULE 10) with the host's greedy tokens.
func TestExpertPlacementIsHonoured(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	ids := []int32{1008, 10484, 318, 15383, 387, 17374}
	host, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	want := runExperts(t, host, nil, ids)
	host.Close()
	for _, where := range []string{"host", "card"} {
		t.Run(where, func(t *testing.T) {
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithExperts(where), WithStreamTrial(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || g == nil {
				noDevice(t, "gpu:0", err)
			}
			defer g.Close()
			got := runExperts(t, m, g, ids)
			st := g.Stats()
			hybrid := where == "host"
			if hybrid && (st.HybridRuns == 0 || st.StreamFills != 0) || !hybrid && (st.StreamFills == 0 || st.HybridRuns != 0) {
				t.Fatalf("experts %s: %d hybrid block-steps, %d fills", where, st.HybridRuns, st.StreamFills)
			}
			for p := range want {
				if argmaxOf(got[p]) != argmaxOf(want[p]) {
					t.Fatalf("pos %d: argmax %d, host %d", p, argmaxOf(got[p]), argmaxOf(want[p]))
				}
			}
			t.Logf("experts %s: %d hybrid block-steps, %d fills", where, st.HybridRuns, st.StreamFills)
		})
	}
}

// TestOffCardExpertsRelocate holds a block whose experts run off the card to
// the relocation principle: after the first token the sequence hops to a
// second tier (blocks, KV pages and all), and later the blocks go home and come
// back; the greedy tokens are the unmoved run's, and each destination counts
// the moved blocks running the same way there -- hybrid block-steps for %host,
// fills for %card -- so a block that arrived unmarked (resident, or refused
// for its bank) fails.
func TestOffCardExpertsRelocate(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	for _, where := range []string{"host", "card"} {
		t.Run(where, func(t *testing.T) {
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithExperts(where), WithStreamTrial(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			open := func() *tier.GPU {
				g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || g == nil {
					noDevice(t, "gpu:0", err)
				}
				return g
			}
			count := func(g *tier.GPU) int {
				if where == "host" {
					return g.Stats().HybridRuns
				}
				return g.Stats().StreamFills
			}
			const ntok = 10
			run := func(move bool) []int32 {
				a, b := open(), open()
				defer a.Close()
				defer b.Close()
				s := m.NewState(ntok + 2)
				defer s.Close()
				s.SetDeviceLayers(a, m.Cfg.NLayer)
				n := s.GPULayers()
				tok, out := int32(1008), []int32{}
				for i := 0; i < ntok; i++ {
					l, err := s.Forward(tok)
					if err != nil {
						t.Fatal(err)
					}
					tok = int32(argmaxOf(l))
					out = append(out, tok)
					if !move {
						continue
					}
					switch i {
					case 1:
						// Hop to the second tier, history and all.
						before := count(b)
						s.SetDeviceLayers(b, m.Cfg.NLayer)
						if s.GPULayers() != n {
							t.Fatalf("the hop placed %d of %d blocks", s.GPULayers(), n)
						}
						defer func(c int) {
							if count(b) <= c {
								t.Errorf("tier b counted no off-card expert runs after the hop")
							}
						}(before)
					case 4:
						// Home, then back.
						s.SetGPULayers(0)
						if s.GPULayers() != 0 {
							t.Fatalf("%d blocks stayed on the card", s.GPULayers())
						}
					case 6:
						c := count(b)
						if got := s.SetGPULayers(n); got != n {
							t.Fatalf("%d of %d blocks came back", got, n)
						}
						defer func() {
							if count(b) <= c {
								t.Errorf("the blocks came back and counted no off-card expert runs")
							}
						}()
					}
				}
				return out
			}
			want, got := run(false), run(true)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("token %d: %d moved, %d unmoved (%v against %v)", i, got[i], want[i], got, want)
				}
			}
		})
	}
}

// TestHybridPrefillMatchesTheHost: a prompt through hybrid blocks runs as one
// batched chunk -- the rows routed on the device, each row's experts on the
// host -- and lands within the device band of the host's prefill, the batched
// hybrid path counted (rows, not just success; RULE 10).
func TestHybridPrefillMatchesTheHost(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	prompt := []int32{1008, 10484, 318, 15383, 387, 17374, 13, 646, 606, 142957}
	prefill := func(m *Model, g *tier.GPU) []float32 {
		s := m.NewState(32)
		defer s.Close()
		if g != nil {
			s.SetDeviceLayers(g, m.Cfg.NLayer)
		}
		l, err := s.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), l...)
	}
	host, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	want := prefill(host, nil)
	host.Close()
	// A streamed block's chunk runs its experts on the host too: its bank
	// holds a token's selection, not a chunk's (emitGroupedMoE).
	for _, where := range []string{"host", "card"} {
		t.Run(where, func(t *testing.T) {
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithExperts(where), WithStreamTrial(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			// Streamed, so the card arm's bank is a token's sheets even where
			// the whole bank would fit.
			g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
				tier.WithConfig(func(c *tier.Config) { c.StreamExperts = true }))
			if err != nil || g == nil {
				noDevice(t, "gpu:0", err)
			}
			defer g.Close()
			got := prefill(m, g)
			if where == "card" && g.Stats().StreamBlocks == 0 {
				t.Fatal("the card arm placed no streamed block")
			}
			st := g.Stats()
			// Kimi-K3 leads with a dense block, so a streamed block's chunk is
			// declined before anything has advanced and runs elsewhere; the
			// chunk that reaches a streamed bank after a linear block has run
			// is Qwen3.8-Flash-Next's (placement.md, "The split").
			if where == "host" && st.HybridRows < len(prompt) {
				t.Fatalf("%d rows ran their experts on the host in a batched chunk, want the prompt's %d",
					st.HybridRows, len(prompt))
			}
			var num, den float64
			for i := range want {
				d := float64(got[i] - want[i])
				num += d * d
				den += float64(want[i]) * float64(want[i])
			}
			if nmse := num / den; nmse > 1e-5 || argmaxOf(got) != argmaxOf(want) {
				t.Fatalf("%s prefill NMSE %.2e, argmax %d against the host's %d", where, nmse, argmaxOf(got), argmaxOf(want))
			}
			t.Logf("%d host rows, %d hybrid block-steps", st.HybridRows, st.HybridRuns)
		})
	}
}

// TestTopOneRouterRunsHybrid: Llama 4's top-1 sigmoid router, whose weight
// scales the expert's input, places hybrid -- no indexed matvec runs, so the
// one-slot bank that kept a top-1 router off the streamed path is no obstacle
// -- with the host's greedy tokens, the hybrid path counted.
func TestTopOneRouterRunsHybrid(t *testing.T) {
	path, ok := existingModel(testmodels.Path("synth-llama4.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: synth-llama4.gguf -- this gate proved nothing")
	}
	ids := []int32{1, 2, 3, 4, 5, 6}
	host, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	want := runExperts(t, host, nil, ids)
	host.Close()
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithExperts("host"), WithStreamTrial(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.NExpertUsed != 1 {
		t.Fatalf("the fixture routes %d experts; this gate is about a top-1 router", m.Cfg.NExpertUsed)
	}
	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "gpu:0", err)
	}
	defer g.Close()
	got := runExperts(t, m, g, ids)
	if st := g.Stats(); st.HybridRuns == 0 {
		t.Fatalf("no hybrid block-step: the top-1 mixture did not run its experts on the host (%s)", g.Err())
	}
	for p := range want {
		if argmaxOf(got[p]) != argmaxOf(want[p]) {
			t.Fatalf("pos %d: argmax %d, host %d", p, argmaxOf(got[p]), argmaxOf(want[p]))
		}
	}
}
