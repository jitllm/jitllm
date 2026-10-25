package model

import (
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// laneModels is what the lane gate decodes: heads 48, 64 and 128 wide, one,
// three and two query heads to a key head -- what a split attention's
// workgroups are laid out by.
var laneModels = []string{"stories15M-q8_0.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf"}

// TestComposedBlocksAtEachLaneCount decodes with every block and the head
// the composer's, formed for a subgroup of 1, 8 and 32 lanes and for the
// device's own (Config.ComposedLanes), and with the whole tier at one lane
// (WithLanes(1)), against the tier on the same device: each arm's distance
// from the float reference, formed at the device's lanes, is held to the
// band TestComposedBlocksServe holds the device's own forms to, a poisoned
// arm must change nothing, and Stats.ComposedLanes is the selection check.
// A subgroup wider than the device guarantees is refused, never formed.
func TestComposedBlocksAtEachLaneCount(t *testing.T) {
	const bandRatio, bandWorst = 1.5, 4.0
	parts := tier.ComposedBlock | tier.ComposedHead
	ran := 0
	for _, file := range laneModels {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			n := 140
			if c := int(m.Cfg.NCtx); c > 0 {
				n = min(n, c-len(ids)-4)
			}
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			type devRun struct {
				want, ref [][]float32
				lanes     int
			}
			runs := map[string]*devRun{}
			widest := 0.0
			for _, dev := range stepDevices() {
				t.Run(dev+"/reference", func(t *testing.T) {
					r := &devRun{}
					var st tier.Stats
					r.want, st, _, _ = composedDecode(t, m, dev, ids, toks, 0, true, false)
					r.ref, _, _, _ = composedDecode(t, m, dev, ids, toks, parts, true, true)
					r.lanes = st.Lanes
					for i := 1; i < len(r.ref); i++ {
						widest = max(widest, logitNMSE(r.want[i], r.ref[i]))
					}
					runs[dev] = r
				})
			}
			for _, dev := range stepDevices() {
				r := runs[dev]
				if r == nil {
					continue
				}
				t.Run(dev, func(t *testing.T) {
					type arm struct {
						name  string
						lanes int
						opts  []tier.Option
					}
					arms := []arm{{"the device's own lanes", r.lanes, nil}, {"WithLanes(1)", 1, []tier.Option{tier.WithLanes(1)}}}
					for _, l := range []int{1, 8, 32} {
						if l == r.lanes {
							continue
						}
						if l > r.lanes {
							t.Logf("%d lanes: the device guarantees %d, and a wider subgroup is refused", l, r.lanes)
							continue
						}
						arms = append(arms, arm{fmt.Sprintf("%d lanes", l), l,
							[]tier.Option{tier.WithConfig(func(c *tier.Config) { c.ComposedLanes = l })}})
					}
					for _, a := range arms {
						t.Run(a.name, func(t *testing.T) {
							got, st, headOn, _ := composedDecode(t, m, dev, ids, toks, parts, true, false, a.opts...)
							if st.ComposedLanes != a.lanes {
								t.Fatalf("the composer formed for %d lanes, not %d: the arm did not select its subgroup", st.ComposedLanes, a.lanes)
							}
							if nb, d, _ := st.ServedPart(tier.ComposedBlock); nb != len(toks)*m.Cfg.NLayer || d != 0 {
								t.Fatalf("the composer served %d block steps and declined %d where %d steps of %d blocks are %d (%s)",
									nb, d, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer, st.LastErr)
							}
							if nh, d, _ := st.ServedPart(tier.ComposedHead); headOn && (nh != len(toks) || d != 0) {
								t.Fatalf("the composer served the head %d times and declined it %d in %d steps", nh, d, len(toks))
							}
							var ratios []float64
							worst, worstAt, worstTier, fromTier := 0.0, 0, 0.0, 0.0
							for i := 1; i < len(r.ref); i++ {
								et, eg := logitNMSE(r.want[i], r.ref[i]), logitNMSE(got[i], r.ref[i])
								if math.IsNaN(eg) {
									t.Fatalf("step %d: at %d lanes the logits are not finite", i, a.lanes)
								}
								if math.IsNaN(et) || !(et > 0) {
									t.Fatalf("step %d: the float reference is %.3g from the tier: a degenerate reference proves nothing", i, et)
								}
								ratios = append(ratios, eg/et)
								if eg > worst {
									worst, worstAt = eg, i
								}
								worstTier = max(worstTier, et)
								fromTier = max(fromTier, logitNMSE(got[i], r.want[i]))
							}
							off := 0.0
							for i := 2; i < len(r.ref); i++ {
								off = max(off, logitNMSE(got[i], r.ref[i-1]))
							}
							if !(off > max(1e-9, 10*worst)) {
								t.Errorf("one position off the logits are within %.3g of the reference's, where the arm's worst is %.3g (step %d): the comparison cannot see a wrong part", off, worst, worstAt)
							}
							slices.Sort(ratios)
							med := ratios[len(ratios)/2]
							t.Logf("%d lanes (the tier's %d): against the float reference median ratio to the tier's distance %.3f (range %.3f..%.3f), worst %.3g at step %d against the tier's %.3g here and %.3g on its widest backend; worst from the tier %.3g, one position off %.3g",
								a.lanes, st.Lanes, med, ratios[0], ratios[len(ratios)-1], worst, worstAt, worstTier, widest, fromTier, off)
							if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
								t.Errorf("at %d lanes the composed parts are %.3g at the median of the tier's distance from the float reference and %.3g at their worst, where the tier's backends reach %.3g (bounds %.1f and %.1fx)",
									a.lanes, med, worst, widest, bandRatio, bandWorst)
							}
							poisoned, _, _, _ := composedDecode(t, m, dev, ids, toks, parts, true, false,
								append(a.opts, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))...)
							for i := range got {
								if !slices.Equal(poisoned[i], got[i]) {
									t.Fatalf("step %d: at %d lanes, with every buffer poisoned the logits change (%.3g): a composed kernel reads what nothing wrote",
										i, a.lanes, logitNMSE(poisoned[i], got[i]))
								}
							}
							ran++
						})
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}
