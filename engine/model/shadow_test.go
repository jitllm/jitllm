//go:build jitllmtest

package model

import (
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestShadowDenseDecode holds the composer's slices of a block -- the
// attention's input (the norm and the q, k and v projections) and the
// post-attention half (the output projection, the residual add, the FFN
// norm, the gated MLP and its add) -- to what the tier's own kernels write,
// block by block, at every decoded position, on every device: every block
// of a plain dense model is compared in each slice, the slices must form,
// and the composer's distance from the tier's may be at most a quarter of
// the slice's int8 band, the tier's own distance from the slice with float
// activations (tier.ShadowLog); so is the output head (the output norm and
// the vocabulary projection) at every position. A block that is not a slice
// is refused by name, never compared; each block slice's violation binds two
// of its weights the wrong way round.
func TestShadowDenseDecode(t *testing.T) {
	// The composer's distance from the tier, as a fraction of the slice's
	// int8 band: 0.115 at worst on these models (Qwen3-0.6B's FFN output).
	const ratio = 0.25
	// The largest band a slice may have: the tier's int8 arithmetic is at
	// most this far from the float slice (0.227 at worst, Qwen3-0.6B).
	const maxBand = 0.5
	// The tier's binary16 contractions against the float slice: 1.4e-6 at
	// worst (Llama-3.2-1B prefill, CUDA).
	const floatBound = 1e-5
	slicesOf := []string{"attention in", "post-attention"}
	const head = "head"
	violations := []struct {
		slice string
		swap  [2]jlm.Role
	}{
		{"attention in", [2]jlm.Role{jlm.RoleAttnK, jlm.RoleAttnV}},
		{"post-attention", [2]jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp}},
	}
	// The slices a block's residual makes: a model whose residual is not
	// the plain one has these refused by name, while its query and head,
	// plain ops whatever the residual, are compared.
	residual := []string{"attention in", "post-attention"}
	for _, c := range []struct {
		file   string
		refuse bool // every block's residual slices refused by name
	}{
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", false},
		{"Qwen3-0.6B-Q8_0.gguf", false},
		{"tinyllama-1.1b-q3_K_M.gguf", false},
		{"synth-gemma3n.gguf", true},
	} {
		t.Run(c.file, func(t *testing.T) {
			path := testmodels.Path(c.file)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			opened := map[string]bool{}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					prefill := false
					wrongHistory := false
					run := func(swap [2]jlm.Role, perturb jlm.Role, table string) *tier.ShadowLog {
						log := &tier.ShadowLog{Ratio: ratio, Floor: 1e-9, MaxBand: maxBand, FloatBound: floatBound,
							Swap: swap, Perturb: perturb, PerturbTable: table, WrongHistory: wrongHistory}
						restore := tier.ShadowForTest(log)
						defer restore()
						g, err := tier.OpenWith(slices.Concat([]tier.Option{tier.WithDevices(dev),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t))...)
						if err != nil || g == nil {
							if opened[dev] {
								t.Fatalf("%s opened earlier in this gate and does not now: %v", dev, err)
							}
							noDevice(t, dev, err)
						}
						opened[dev] = true
						defer g.Close()
						st := m.NewState(len(ids) + 1)
						defer st.Close()
						if err := st.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						if st.devCount() != m.Cfg.NLayer {
							t.Fatalf("placed %d of %d blocks: %v %s", st.devCount(), m.Cfg.NLayer, st.DeviceDeclines(), g.Err())
						}
						if prefill {
							if _, err := st.Prefill(ids); err != nil {
								t.Fatal(err)
							}
						} else {
							for _, id := range ids {
								if _, err := st.Forward(id); err != nil {
									t.Fatal(err)
								}
							}
						}
						if len(log.Errors) > 0 {
							t.Fatalf("the shadow's own runs went wrong: %v", log.Errors[:min(4, len(log.Errors))])
						}
						return log
					}
					// caught holds a violation to failing every slice of the
					// kind it touches that was checked, compared or a gap.
					caught := func(what, slice string, bad *tier.ShadowLog) {
						t.Helper()
						checked := bad.Compared[slice] + bad.GapSlices[slice]
						if checked == 0 || bad.FailedSlices[slice] != checked {
							t.Fatalf("%s: %d of %d %s slices failed; a violation must fail every one (worst %.3g)",
								what, bad.FailedSlices[slice], checked, slice, bad.Worst)
						}
						t.Logf("%s: every one of %d %s slices failed", what, checked, slice)
					}
					clean := func(what string, log *tier.ShadowLog) {
						t.Helper()
						t.Logf("%s on %s: compared %v, gaps %v; composer-tier %.3g, tier-float %.3g, composer-float %.3g, band ratio %.3g; refused %v",
							what, dev, log.Compared, log.GapSlices, log.Worst, log.WorstTier, log.WorstComposer, log.WorstRatio, log.Refused)
						if len(log.Unformed) > 0 || len(log.Failed) > 0 {
							for _, f := range log.Failed[:min(12, len(log.Failed))] {
								t.Log(f)
							}
							t.Fatalf("%s: unformed %v; %d failed", what, log.Unformed, len(log.Failed))
						}
					}
					log := run([2]jlm.Role{}, 0, "")
					clean("decode", log)
					if c.refuse {
						if log.Compared[residual[0]]+log.Compared[residual[1]] != 0 || len(log.Refused) == 0 {
							t.Fatalf("this model's residual is not the plain one and its slices must be refused by name")
						}
					} else {
						if len(log.Refused) > 0 {
							t.Fatalf("refused %v", log.Refused)
						}
						for _, name := range append(slicesOf, "key+attend", "pools") {
							if want := m.Cfg.NLayer * len(ids); log.Compared[name] != want {
								t.Fatalf("compared %d %s slices, want every block at every position (%d)", log.Compared[name], name, want)
							}
						}
						t.Logf("the appended keys at most %.3g from the tier's, every value word the same", log.PoolKeys)
					}
					// The query and the head are plain ops whatever the residual.
					if log.Compared["query"] != m.Cfg.NLayer*len(ids) {
						t.Fatalf("compared %d queries, want every block at every position (%d)", log.Compared["query"], m.Cfg.NLayer*len(ids))
					}
					if log.Compared[head] != len(ids) {
						t.Fatalf("compared %d heads, want one at every position (%d)", log.Compared[head], len(ids))
					}
					if !c.refuse {
						for _, v := range violations {
							caught(fmt.Sprintf("%v and %v bound the wrong way round", v.swap[0], v.swap[1]), v.slice, run(v.swap, 0, ""))
						}
					}
					rot := run([2]jlm.Role{}, 0, "rope.cs")
					caught("the rotary table read doubled", "query", rot)
					if !c.refuse {
						// At position 0 the rotation is by 0 and a doubled
						// table doubles k, which attention over one key cannot
						// see; the keys it appends show it at every position.
						caught("the rotary table read doubled", "pools", rot)
						wrongHistory = true
						caught("the keys' pool stated laid out by position", "pools", run([2]jlm.Role{}, 0, ""))
						wrongHistory = false
					}
					if m.Cfg.QKNorm {
						caught("the q norm read doubled", "query", run([2]jlm.Role{}, jlm.RoleAttnQNorm, ""))
					}
					caught("the output norm read doubled", head, run([2]jlm.Role{}, jlm.RoleOutputNorm, ""))
					if c.refuse {
						return
					}

					// The prompt as one step: every block's slices at once over
					// its rows, each compared or a named gap.
					prefill = true
					pre := run([2]jlm.Role{}, 0, "")
					clean("prefill", pre)
					for gap, n := range pre.Gaps {
						t.Logf("gap: %s (%d)", gap, n)
					}
					for _, name := range append(slicesOf, "query") {
						if n := pre.Compared[name] + pre.GapSlices[name]; n != m.Cfg.NLayer {
							t.Fatalf("prefill checked %d %s slices (compared %d, gaps %d), want every block (%d)",
								n, name, pre.Compared[name], pre.GapSlices[name], m.Cfg.NLayer)
						}
					}
					// A prompt chunk's head runs on the row-logits path, which
					// the head's slice does not reach.
					if len(pre.Refused) > 0 || pre.Compared[head] != 0 {
						t.Fatalf("prefill refused %v and compared %d heads, want neither", pre.Refused, pre.Compared[head])
					}
					caught("prefill with gate and up bound the wrong way round", "post-attention", run([2]jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp}, 0, ""))
				})
			}
		})
	}
}
