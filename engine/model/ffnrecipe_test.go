package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// ffnRecipeDecode decodes toks teacher-forced after ids on one fresh CUDA
// tier, once per arm, the arm's switches set between them on the same tier
// so every arm runs the splits its tuner chose once. An arm is a recipe
// (tier.FFNForm); FFNTwo also turns Gate2 off. It returns each arm's logits
// and the stats after each; nil where the host has no CUDA device.
func ffnRecipeDecode(t *testing.T, m *Model, ids, toks []int32, noGraph bool, layers int, cache string, arms []tier.FFNForm, extra ...tier.Option) ([][][]float32, []tier.Stats) {
	t.Helper()
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda"), tier.WithTuneCache(cache),
		tier.WithConfig(func(c *tier.Config) { c.NoGraph = noGraph })}, extra...)...)
	if err != nil || g == nil {
		return nil, nil
	}
	defer g.Close()
	run := func(f tier.FFNForm) [][]float32 {
		g.GateUp2, g.FFNForm = f != tier.FFNTwo, f
		s := m.NewState(len(ids) + len(toks) + 4)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if layers >= 0 {
			s.SetGPULayers(layers)
		}
		// The placement the arm names, asserted (RULE 10): "every block" is
		// what fits, and another process holding the card's memory leaves a
		// seam the control did not have -- different blocks on the host, a
		// different answer, read as a recipe that is not exact.
		want := layers
		if want < 0 {
			want = m.Cfg.NLayer
		}
		if s.GPULayers() != want {
			t.Skipf("CARD TOO SMALL: %d of %d blocks placed, the arm names %d -- this gate proved nothing",
				s.GPULayers(), m.Cfg.NLayer, want)
		}
		l, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		var out [][]float32
		for _, tok := range toks {
			out = append(out, append([]float32(nil), l...))
			if l, err = s.Forward(tok); err != nil {
				t.Fatal(err)
			}
		}
		s.SetGPULayers(0)
		return out
	}
	// The first pass runs the tuners (split, prologue, FFN) with every recipe
	// offered; the arms after it run their answers.
	run(tier.FFNTuned)
	var outs [][][]float32
	var sts []tier.Stats
	for _, f := range arms {
		outs = append(outs, run(f))
		sts = append(sts, g.Stats())
	}
	return outs, sts
}

// TestFFNRecipesMatchTwoLaunches decodes real dense models under the FFN
// site's recipes (tier.FFNForm) and under the per-card verdict on CUDA, and
// holds each to the two-launch arm's logits bit for bit: gate2in (the FFN
// norm, gate and up in one launch: ActIn's norm in a Gate2 matvec) and
// whatever tuneFFN chose. Every block placed with and without the captured
// graph, and a partial seam. The recipe's own counter is the selection check
// (RULE 10): a forced recipe that ran nowhere fails, as does a tuner that
// decided no key.
func TestFFNRecipesMatchTwoLaunches(t *testing.T) {
	ran := 0
	for _, name := range gateUp2Models {
		m := gateUp2Model(t, name)
		if m == nil {
			t.Logf("MODEL MISSING: %s", name)
			continue
		}
		t.Run(name, func(t *testing.T) {
			defer m.Close()
			ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
			toks := make([]int32, 16)
			for i := range toks {
				toks[i] = int32(100 + 37*i)
			}
			for _, c := range []struct {
				seam    int
				noGraph bool
			}{{-1, false}, {-1, true}, {m.Cfg.NLayer / 2, false}} {
				arms := []tier.FFNForm{tier.FFNTwo, tier.FFNGate2In, tier.FFNTuned}
				outs, sts := ffnRecipeDecode(t, m, ids, toks, c.noGraph, c.seam, t.TempDir(), arms)
				if outs == nil {
					t.Skip("no CUDA device here")
				}
				in := sts[1].Gate2InRuns - sts[0].Gate2InRuns
				if in == 0 {
					t.Fatalf("seam %d: no FFN ran its norm, gate and up as one launch (%s)", c.seam, sts[1].LastErr)
				}
				if sts[2].FFNTimed == 0 {
					t.Fatalf("seam %d: the FFN tuner decided no shape", c.seam)
				}
				for a := 1; a < len(arms); a++ {
					if n, d := gateUp2NMSE(outs[a], outs[0]); d != 0 {
						t.Fatalf("seam %d graph %v recipe %v: %d logits differ from the two launches (NMSE %.3g)",
							c.seam, !c.noGraph, arms[a], d, n)
					}
				}
				t.Logf("seam %d graph %v: gate2in in %d FFNs; tuner timed %d keys, kept %d; every arm bit-identical over %d steps",
					c.seam, !c.noGraph, in, sts[2].FFNTimed, sts[2].FFNKept, len(toks))
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}
