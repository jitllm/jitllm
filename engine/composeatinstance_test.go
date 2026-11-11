package engine

import (
	"cmp"
	"context"
	"maps"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// cardComposed is what g's composer formed afresh and its devices compiled
// so far: counted where the composer and the driver run, whatever a caller
// marked the form as.
func cardComposed(g *gpu.GPU) [2]int {
	s := g.Stats()
	return [2]int{s.ComposerRuns, s.PartsFormed}
}

// composedNothingSince fails t where g's composer formed or its devices
// compiled anything since was: phase is what ran in between.
func composedNothingSince(t testing.TB, g *gpu.GPU, was [2]int, phase string) {
	t.Helper()
	if now := cardComposed(g); now != was {
		t.Errorf("%s: the card's composer formed %d requests and compiled %d parts (%v, %s)", phase,
			now[0]-was[0], now[1]-was[1], g.Stats().UnformedAtStepParts, g.Err())
	}
}

// TestComposesOnlyWithAnInstance is the design's composition rule on the
// CPU: once an instance is made and served, sessions opened on it, their
// prompts, their tokens alone and stepped together, and every way of drawing
// a token ask the composer for nothing (Instance.Account's Asks). Making a second
// instance in the middle must move the count: that is the gate seeing a
// composition at all.
func TestComposesOnlyWithAnInstance(t *testing.T) {
	for _, src := range []string{models()["stories260K"], testmodels.Path("SmolLM2-360M-Instruct-Q8_0.gguf"),
		models()["qwen3moe"], testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf")} {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path, ok := existingModel(src)
			if !ok {
				t.Fatalf("model not present: %s (nor its container)", src)
			}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			// One CPU device for both instances below, so the second's
			// compositions are asked of the forms the count reads.
			cd, err := OpenCPU()
			if err != nil {
				t.Fatal(err)
			}
			defer cd.Close()
			ctx := context.Background()
			in := testInstance(t, m, 64, WithCPU(cd))
			defer in.Close()
			pool := NewPool()
			if err := pool.Serve(in, 0); err != nil {
				t.Fatal(err)
			}
			defer pool.Unserve(in)
			a0 := in.Account(nn.CPU)
			c0 := a0.Asks
			if a0.Composing <= 0 {
				t.Fatalf("%d forms asked in making the instance and no time counted composing them", c0)
			}
			t.Logf("making the instance asked %d forms in %v", c0, a0.Composing)
			var spent time.Duration
			names := slices.Collect(maps.Keys(a0.Passes))
			for _, n := range names {
				spent += a0.Passes[n]
			}
			if spent <= 0 || spent > a0.Composing {
				t.Fatalf("the passes and lowerings took %v of %v composing: the time is not counted where it goes", spent, a0.Composing)
			}
			slices.SortFunc(names, func(x, y string) int { return cmp.Compare(a0.Passes[y], a0.Passes[x]) })
			for _, n := range names[:min(8, len(names))] {
				t.Logf("  %-14s %v", n, a0.Passes[n])
			}
			samplers := []*Sampler{nil, {Temp: 0.8, TopK: 40, Seed: 1}, {Temp: 0.7, TopP: 0.9, Seed: 2},
				{Temp: 1, MinP: 0.05, Seed: 3}, {Temp: 0.9, TopK: 20, RepeatPen: 1.2, Seed: 4}}
			prompts := [][]int32{{1, 45, 79, 33}, {1, 65, 108, 250, 205, 63}}
			sess := make([]*Session, len(prompts))
			last := make([][]float32, len(prompts))
			for i, p := range prompts {
				s, err := pool.NewSession(64)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				sess[i] = s
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
					lg, err := in.Prefill(p)
					last[i] = lg
					return err
				}); err != nil {
					t.Fatal(err)
				}
			}
			for k := range 10 {
				smp := samplers[k%len(samplers)]
				runs := make([]Run, len(sess))
				for i, s := range sess {
					var tok int32
					if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
						if tok, err = in.Sample(smp, last[i]); err != nil {
							return err
						}
						smp.Observe(tok)
						return nil
					}); err != nil {
						t.Fatal(err)
					}
					runs[i] = Run{Session: s, Tokens: []int32{tok}, Logits: true}
				}
				if k%2 == 0 {
					lg, err := StepRuns(runs)
					if err != nil {
						t.Fatal(err)
					}
					copy(last, lg)
					continue
				}
				for i, s := range sess {
					if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
						lg, err := in.Forward(runs[i].Tokens[0])
						last[i] = lg
						return err
					}); err != nil {
						t.Fatal(err)
					}
				}
			}
			a1 := in.Account(nn.CPU)
			if n := a1.Asks - c0; n != 0 {
				t.Errorf("sessions' prompts and tokens asked the composer %d times", n)
			}
			if d := a1.Composing - a0.Composing; d != 0 {
				t.Errorf("sessions' prompts and tokens spent %v composing", d)
			}
			second := testInstance(t, m, 64, WithCPU(cd))
			second.Close()
			if a2 := in.Account(nn.CPU); a2.Asks == a1.Asks || a2.Composing == a1.Composing {
				t.Errorf("making an instance moved the count to %d and the time to %v: the account cannot see a composition", a2.Asks, a2.Composing)
			}
		})
	}
}

// TestComposesOnlyWithAnInstanceOnACard is TestComposesOnlyWithAnInstance
// with the instance on a card: sessions' prompts and tokens, alone and stepped
// together, form no part on the card at a step (gpu.Stats.UnformedAtStep) and
// ask no device -- the CPU or a card the instance runs on, by its account --
// for a form; the card formed its parts when the instance was placed. The
// mixture runs a second time with its experts streamed: a prompt through a
// streamed block is cut to the rows its pages hold (pagedCall), and those
// calls' forms are the placement's too.
func TestComposesOnlyWithAnInstanceOnACard(t *testing.T) {
	streamed := gpu.WithConfig(func(c *gpu.Config) { c.StreamExperts = true; c.NoAutoStream = true })
	for _, arm := range []struct {
		src    string
		stream bool
	}{{testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), false}, {models()["qwen3moe"], false}, {models()["qwen3moe"], true}} {
		src := arm.src
		name := filepath.Base(src)
		if arm.stream {
			name += "/streamed"
		}
		t.Run(name, func(t *testing.T) {
			path, ok := existingModel(src)
			if !ok {
				t.Fatalf("model not present: %s (nor its container)", src)
			}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			var extra []gpu.Option
			if arm.stream {
				extra = append(extra, streamed)
			}
			g := swaTierWith(t, extra...)
			defer g.Close()
			ctx := context.Background()
			in := testInstance(t, m, 128)
			defer in.Close()
			in.SetDevice(g)
			if in.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("placed %d of %d blocks: %q", in.GPULayers(), m.Cfg.NLayer, g.Err())
			}
			if sb := g.Stats().StreamBlocks; arm.stream != (sb > 0) {
				t.Fatalf("streamed arm %v placed %d streamed blocks", arm.stream, sb)
			}
			pool := NewPool()
			if err := pool.Serve(in, 0); err != nil {
				t.Fatal(err)
			}
			defer pool.Unserve(in)
			f0, k0 := g.Stats().UnformedAtStep, cardComposed(g)
			// The placement composed: a count that never moves would pass
			// every step below whatever it formed.
			if k0[0] == 0 || k0[1] == 0 {
				t.Fatalf("the placement formed %d requests and compiled %d parts: the count sees no composition", k0[0], k0[1])
			}
			// Every device the instance runs on, by name, the CPU among them.
			asks := map[string]int64{nn.CPU: in.Account(nn.CPU).Asks}
			for _, at := range in.Locations() {
				if at != "" && at != nn.CPU {
					asks[at] = in.Account(at).Asks
				}
			}
			for dev, n := range asks {
				if dev != nn.CPU && n == 0 {
					t.Fatalf("placing blocks on %s asked it for no form: its count cannot see a composition", dev)
				}
			}
			prompts := [][]int32{{1, 45, 79, 33, 21, 9}, {1, 65, 108, 250, 205, 63}}
			sess := make([]*Session, len(prompts))
			last := make([][]float32, len(prompts))
			for i, p := range prompts {
				s, err := pool.NewSession(128)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				sess[i] = s
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
					lg, err := in.Prefill(p)
					last[i] = lg
					return err
				}); err != nil {
					t.Fatal(err)
				}
			}
			runs := make([]Run, len(sess))
			for k := range 12 {
				for i, s := range sess {
					if err := s.Acquire(ctx, PhaseGenerate); err != nil {
						t.Fatal(err)
					}
					runs[i] = Run{Session: s, Tokens: []int32{int32(argmax(last[i]))}, Logits: true}
				}
				if k%2 == 0 {
					lg, err := StepRuns(runs)
					if err != nil {
						t.Fatal(err)
					}
					copy(last, lg)
					continue
				}
				for i, s := range sess {
					if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
						lg, err := in.Forward(runs[i].Tokens[0])
						last[i] = lg
						return err
					}); err != nil {
						t.Fatal(err)
					}
				}
			}
			if n := g.Stats().UnformedAtStep - f0; n != 0 {
				t.Errorf("the card formed %d parts at a step (%v)", n, g.Stats().UnformedAtStepParts)
			}
			composedNothingSince(t, g, k0, "sessions' prompts and steps")
			for dev, a := range asks {
				if n := in.Account(dev).Asks - a; n != 0 {
					t.Errorf("sessions' prompts and tokens asked %s for %d forms", dev, n)
				}
			}
			if g.Stats().SessionRows == 0 {
				t.Errorf("no step ran the sessions' rows together on the card")
			}
		})
	}
}
