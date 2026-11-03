package model

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
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
			now[0]-was[0], now[1]-was[1], g.Stats().FormedAtStepParts, g.Err())
	}
}

// TestComposesOnlyWithAnInstance is the design's composition rule on the
// CPU: once an instance is made and served, sessions opened on it, their
// prompts, their tokens alone and stepped together, and every way of drawing
// a token ask the composer for nothing (Model.CPUComposes). Making a second
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
			m, err := Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ctx := context.Background()
			in := m.NewInstance(64)
			defer in.Close()
			if err := m.Serve(in, 0); err != nil {
				t.Fatal(err)
			}
			defer m.Unserve(in)
			c0 := m.CPUComposes()
			samplers := []*Sampler{nil, {Temp: 0.8, TopK: 40, Seed: 1}, {Temp: 0.7, TopP: 0.9, Seed: 2},
				{Temp: 1, MinP: 0.05, Seed: 3}, {Temp: 0.9, TopK: 20, RepeatPen: 1.2, Seed: 4}}
			prompts := [][]int32{{1, 45, 79, 33}, {1, 65, 108, 250, 205, 63}}
			sess := make([]*Session, len(prompts))
			last := make([][]float32, len(prompts))
			for i, p := range prompts {
				s, err := m.NewSession(64)
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
						tok = in.Sample(smp, last[i])
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
			if n := m.CPUComposes() - c0; n != 0 {
				t.Errorf("sessions' prompts and tokens asked the composer %d times", n)
			}
			c1 := m.CPUComposes()
			second := m.NewInstance(64)
			second.Close()
			if m.CPUComposes() == c1 {
				t.Errorf("making an instance asked the composer for nothing: the count cannot see a composition")
			}
		})
	}
}

// TestComposesOnlyWithAnInstanceOnACard is TestComposesOnlyWithAnInstance
// with the instance on a card: sessions' prompts and tokens, alone and stepped
// together, form no part on the card at a step (gpu.Stats.FormedAtStep) and
// ask the CPU composer for nothing; the card formed its parts when the
// instance was placed.
func TestComposesOnlyWithAnInstanceOnACard(t *testing.T) {
	for _, src := range []string{testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), models()["qwen3moe"]} {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path, ok := existingModel(src)
			if !ok {
				t.Fatalf("model not present: %s (nor its container)", src)
			}
			m, err := Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			g := swaTier(t)
			defer g.Close()
			ctx := context.Background()
			in := m.NewInstance(128)
			defer in.Close()
			in.SetDevice(g)
			if in.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("placed %d of %d blocks: %q", in.GPULayers(), m.Cfg.NLayer, g.Err())
			}
			if err := m.Serve(in, 0); err != nil {
				t.Fatal(err)
			}
			defer m.Unserve(in)
			c0, f0, k0 := m.CPUComposes(), g.Stats().FormedAtStep, cardComposed(g)
			// The placement composed: a count that never moves would pass
			// every step below whatever it formed.
			if k0[0] == 0 || k0[1] == 0 {
				t.Fatalf("the placement formed %d requests and compiled %d parts: the count sees no composition", k0[0], k0[1])
			}
			prompts := [][]int32{{1, 45, 79, 33, 21, 9}, {1, 65, 108, 250, 205, 63}}
			sess := make([]*Session, len(prompts))
			last := make([][]float32, len(prompts))
			for i, p := range prompts {
				s, err := m.NewSession(128)
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
			if n := g.Stats().FormedAtStep - f0; n != 0 {
				t.Errorf("the card formed %d parts at a step (%v)", n, g.Stats().FormedAtStepParts)
			}
			composedNothingSince(t, g, k0, "sessions' prompts and steps")
			if n := m.CPUComposes() - c0; n != 0 {
				t.Errorf("sessions' prompts and tokens asked the CPU composer %d times", n)
			}
			if g.Stats().SessionRows == 0 {
				t.Errorf("no step ran the sessions' rows together on the card")
			}
		})
	}
}
