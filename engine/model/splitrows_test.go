//go:build amd64 || arm64

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// splitRowsModels are the packed mixtures the split-rows gate runs: a hybrid
// block's cache holds sheets of a packed bank only. blocks is how many blocks
// go on the card (0: all), band the device's logit NMSE against the host at
// that depth (RULE 11c): a step of rows reduces in another order than
// decode, so Kimi-K3's is wider than TestSplitExpertsMatchTheHost's.
var splitRowsModels = []struct {
	name, path string
	blocks     int
	band       float64
}{
	{"deepseek-v2-lite", "DeepSeek-V2-Lite.Q4_K_M.gguf", 6, 1e-3},
	{"kimik3", "kimik3/Kimi-K3-0.40B.Q8_0.gguf", 0, 1e-4},
}

// splitRowsCase decodes all but the last four of ids one at a time on a
// device running the model's experts as hybrid blocks with an expert cache
// (the cache fills and admits as decode runs), then runs the last four as one
// speculation step (stepRows, every row's logits wanted), and returns the
// worst logit NMSE of those rows against the host's decode of the same ids,
// with the device's counters. noSplit is the control arm: every expert of the
// step on the host.
func splitRowsCase(t *testing.T, m *Model, ids []int32, blocks int, noSplit, noAdmit bool) (float64, tier.Stats, rowLearning) {
	t.Helper()
	c := m.Cfg
	n := len(ids)
	want := make([][]float32, 0, n)
	func() {
		s := m.NewState(n + 2)
		defer s.Close()
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, append([]float32(nil), l...))
		}
	}()
	s := m.NewState(n + 10)
	defer s.Close()
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(cf *tier.Config) {
			cf.StreamExperts, cf.HybridExperts = true, true
			cf.SplitAdmit, cf.NoSplitRows, cf.NoRowAdmits = c.NExpertUsed, noSplit, noAdmit
		}))
	if err != nil || g == nil {
		noDevice(t, "gpu:0", err)
	}
	defer g.Close()
	placed := c.NLayer
	if blocks > 0 {
		placed = blocks
	}
	s.SetDeviceLayers(g, placed)
	if s.GPULayers() != placed {
		t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this arm proved nothing",
			s.GPULayers(), placed, g.Stats().LastErr)
	}
	head := n - 4
	for _, id := range ids[:head] {
		if _, err := s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	if g.Stats().StreamCacheSize == 0 {
		t.Skip("CARD TOO SMALL: placement left no room for an expert cache -- this arm proved nothing")
	}
	defer m.enterPager()()
	rows := ids[head:]
	out := rowsOut{logits: make([]float32, len(rows)*c.NVocab), hidden: make([]float32, len(rows)*c.ResidW())}
	before := g.Stats()
	if err := s.stepRows(embedSrc(s, rows), rows, &out); err != nil {
		t.Fatal(err)
	}
	mid := g.Stats()
	worst := 0.0
	for r := range rows {
		got := out.logits[r*c.NVocab : (r+1)*c.NVocab]
		w := want[head+r]
		var num, den float64
		for i := range w {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("row %d logit %d is not finite", r, i)
			}
			d := float64(got[i] - w[i])
			num += d * d
			den += float64(w[i]) * float64(w[i])
		}
		worst = math.Max(worst, num/den)
	}
	// The same rows again: what the first step admitted is in the cache now,
	// so the second hits where the first could not.
	out2 := rowsOut{logits: make([]float32, len(rows)*c.NVocab), hidden: make([]float32, len(rows)*c.ResidW())}
	if err := s.stepRows(embedSrc(s, rows), rows, &out2); err != nil {
		t.Fatal(err)
	}
	end := g.Stats()
	return worst, end, rowLearning{
		admits:     mid.SplitAdmits - before.SplitAdmits,
		secondHits: end.StreamCacheHits - mid.StreamCacheHits,
		firstHits:  mid.StreamCacheHits - before.StreamCacheHits,
	}
}

// rowLearning is what a step of rows taught the cache: the sheets its misses
// admitted, and the hits of the step and of the identical step after it.
type rowLearning struct{ admits, firstHits, secondHits int }

// TestSplitRowsMatchTheHost holds a hybrid block's step of several rows, its
// experts split between the card's cache and the host (streamBank.splitRows),
// to the host's decode of the same tokens, and demands the split ran: a
// speculation step's verification takes it. The control arm runs every
// expert of the step on the host and must count no split row step.
func TestSplitRowsMatchTheHost(t *testing.T) {
	for _, md := range splitRowsModels {
		t.Run(md.name, func(t *testing.T) {
			path, ok := existingModel(testmodels.Path(md.path))
			if !ok {
				t.Skipf("MODEL MISSING: %s -- this gate proved nothing", md.path)
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is Rome; Spain's", true)
			for _, arm := range []struct {
				name    string
				noSplit bool
			}{{"split", false}, {"host", true}} {
				t.Run(arm.name, func(t *testing.T) {
					worst, st, _ := splitRowsCase(t, m, ids, md.blocks, arm.noSplit, false)
					t.Logf("%s: worst NMSE %.2e (band %.0e); %d split row steps, %d cache hits, %d misses",
						arm.name, worst, md.band, st.SplitRowSteps, st.StreamCacheHits, st.StreamCacheMisses)
					if worst > md.band {
						t.Fatalf("the step's rows part from the host: worst NMSE %.2e against a band of %.0e",
							worst, md.band)
					}
					if arm.noSplit != (st.SplitRowSteps == 0) {
						t.Fatalf("%d split row steps with the split %s", st.SplitRowSteps,
							map[bool]string{false: "on", true: "off"}[arm.noSplit])
					}
				})
			}
		})
	}
}

// TestSplitRowsTeachTheCache holds a speculation step's rows to the same
// cache lessons a decode token takes: its misses are admitted, and the same
// rows run again hit what was admitted. A run that speculates has only steps
// of rows, so a step that does not teach leaves its cache at the decode
// prompt's contents. The control arm (Config.NoRowAdmits) must admit nothing
// and gain no hits, or the arms prove nothing.
func TestSplitRowsTeachTheCache(t *testing.T) {
	for _, md := range splitRowsModels {
		t.Run(md.name, func(t *testing.T) {
			path, ok := existingModel(testmodels.Path(md.path))
			if !ok {
				t.Skipf("MODEL MISSING: %s -- this gate proved nothing", md.path)
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is Rome; Spain's", true)
			var learned, control rowLearning
			for _, arm := range []struct {
				name    string
				noAdmit bool
			}{{"admit", false}, {"control", true}} {
				t.Run(arm.name, func(t *testing.T) {
					worst, st, rl := splitRowsCase(t, m, ids, md.blocks, false, arm.noAdmit)
					t.Logf("%s: worst NMSE %.2e; %d split row steps; admitted %d, hits %d then %d",
						arm.name, worst, st.SplitRowSteps, rl.admits, rl.firstHits, rl.secondHits)
					if worst > md.band {
						t.Fatalf("worst NMSE %.2e against a band of %.0e", worst, md.band)
					}
					if st.SplitRowSteps == 0 {
						t.Fatal("no split row step ran: this arm proved nothing")
					}
					if arm.noAdmit {
						control = rl
					} else {
						learned = rl
					}
				})
			}
			if control.admits != 0 {
				t.Fatalf("the control arm admitted %d sheets", control.admits)
			}
			if learned.admits == 0 {
				t.Fatal("a step of rows admitted nothing into the cache")
			}
			if learned.secondHits <= control.secondHits {
				t.Fatalf("the repeated step hit %d times after rows admitted, %d without: the sheets were not used",
					learned.secondHits, control.secondHits)
			}
		})
	}
}
