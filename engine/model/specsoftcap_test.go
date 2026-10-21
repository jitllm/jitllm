//go:build amd64 || arm64

package model

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSpecOverAFinalSoftcap is speculation over a model with a final logit
// softcap (gemma2's), on the host and on every device. No model with a
// prediction block carries one, so the cap is perturbed onto the MTP models
// as TestStepAcrossSessionsAppliesTheFinalSoftcap and
// TestFinalSoftcapReachesTheDeviceHead perturb it.
//
// Two checks, because the first cannot see a missing cap. Greedy speculation
// must emit what plain greedy decode emits (on a device, up to a tie judged by
// devFlip, as TestSpecGreedyOnDevice judges it) -- but a cap is monotone, so
// no argmax moves without one. So the verification's own logits are compared
// too: a speculation step of four rows (stepRows, what Speculator.Next runs),
// teacher-forced along plain decode's tokens, against plain decode's logits at
// the same positions, on the same tier. The verify rows skipping the cap
// (rowsOut.noCap on the host, tier's "nocap" on a device: its tagged twin
// TestSpecSoftcapGateDiscriminates) must land far outside.
//
// The two take different caps. The greedy arm's is half the largest logit,
// where tanh is not yet flat: at an eighth the top logits saturate to one
// binary32 value, greedy breaks the exact tie by index, and on an integrated GPU
// and an arm64 host the chains parted at token 1 on a tie of 0.0000, which ended the
// comparison there. The logit arm's is an eighth, where an uncapped row parts
// by more than its own norm: at half, Qwen3.5-0.8B's uncapped verify rows read
// 3.1e-02-3.3e-02 against a device bound of ~1e-02, 3x and no more.
//
// JITLLM_SPEC_CAP_MODELS names the models, comma-separated; the default is
// the hybrid fixture and the real Qwen3.5-0.8B with its prediction block.
func TestSpecOverAFinalSoftcap(t *testing.T) {
	n := min(specTokens(), 64)
	for _, path := range specCapModels() {
		name := strings.TrimSuffix(path[strings.LastIndex(path, "/")+1:], ".gguf")
		t.Run(name, func(t *testing.T) {
			m, amax := openSpecCapped(t, path)
			defer m.Close()
			text := specPrompts[0]
			prompt := m.Vocab.Encode(text, true)
			greedyCap, logitCap := float32(amax/2), float32(amax/8)
			m.Cfg.FinalSoftcap = greedyCap
			want := plainGreedy(t, m, prompt, n, nil)
			oracle := oracleFor(name, want)

			// The host: exact arithmetic, so speculation is decode token for
			// token and a verified row's logits are decode's.
			got, st := specGreedy(t, m, prompt, n, nil, specFaultNone, oracle, WithSpecDraft(3))
			if i := firstDiff(got, want); i >= 0 {
				t.Fatalf("host: token %d is %d, plain greedy says %d (%d drafted, %d accepted)", i, got[i], want[i],
					st.Drafted, st.Accepted)
			}
			if st.Drafted == 0 || st.Accepted == 0 {
				t.Fatalf("host: %d drafted and %d accepted -- the speculation never ran", st.Drafted, st.Accepted)
			}
			m.Cfg.FinalSoftcap = logitCap
			w := specVerifyCapped(t, m, nil, prompt, want, false)
			bad := specVerifyCapped(t, m, nil, prompt, want, true)
			t.Logf("host: %d tokens equal, %d drafted, %d accepted; verify rows against decode: worst logit NMSE "+
				"%.3e, uncapped %.3e", n, st.Drafted, st.Accepted, w, bad)
			if !(w < 1e-9) {
				t.Fatalf("host: the verify rows part from decode under a cap: NMSE %.3e", w)
			}
			if !(bad > 1e-2) {
				t.Errorf("host: verify rows with the cap skipped read %.3e against decode -- the gate cannot see it", bad)
			}

			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					// The control first, on a tier of its own closed before
					// the gate's opens: one copy of the model on the card at a
					// time. Both read the host's tokens.
					// Each cap on a tier of its own: a tier's head is built
					// for the cap it first saw.
					bound := specCapBound(t, m, spec, prompt, want)
					open := func() *tier.GPU {
						g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
						if err != nil || g == nil {
							noDevice(t, spec, err)
						}
						return g
					}
					g := open()
					w := specVerifyCapped(t, m, g, prompt, want, false)
					g.Close()
					m.Cfg.FinalSoftcap = greedyCap
					defer func() { m.Cfg.FinalSoftcap = logitCap }()
					g = open()
					defer g.Close()
					want := plainGreedy(t, m, prompt, n, g)
					got, st, onDev := specGreedyDev(t, m, prompt, n, g, oracleFor(name, want), WithSpecDraft(3))
					if !onDev {
						t.Fatalf("the trunk and the prediction block are not both on %s: this would gate the host", spec)
					}
					if st.Drafted == 0 || st.Accepted == 0 {
						t.Fatalf("%d drafted and %d accepted on %s: the speculation never ran", st.Drafted, st.Accepted, spec)
					}
					if i := firstDiff(got, want); i >= 0 {
						gap, pert := devFlip(t, m, g, prompt, want[:i], want[i], got[i])
						if gap > pert {
							t.Fatalf("token %d is %d, plain greedy on %s says %d, %.4f apart against a %.4f reach "+
								"of the band -- not a tie", i, got[i], spec, want[i], gap, pert)
						}
						t.Logf("a tie at token %d (%.4f apart, the band reaching %.4f); equal before it", i, gap, pert)
					}
					t.Logf("%s: %d drafted, %d accepted; verify rows against decode on the device: worst logit "+
						"NMSE %.3e, bound %.3e", spec, st.Drafted, st.Accepted, w, bound)
					if !(w < bound) {
						t.Fatalf("%s: the verify rows part from decode under a cap: NMSE %.3e, bound %.3e",
							spec, w, bound)
					}
				})
			}
		})
	}
}

// specCapBound is how far a device's verify rows may sit from its own decode
// under the cap: denseStepBand times the same comparison with no cap, on a
// tier of its own (a ragged step against the decode matvec is the same
// arithmetic in another order, and how far that carries is the model's), and
// never under 1e-9, where an f32 fixture's control of ~1e-13 is no band.
// The fixture's capped rows read ~5e-13 and Qwen3.5-0.8B's ~1e-03; the
// uncapped ones 2 and up.
func specCapBound(t *testing.T, m *Model, spec string, prompt, toks []int32) float64 {
	t.Helper()
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)},
		testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	keep := m.Cfg.FinalSoftcap
	m.Cfg.FinalSoftcap = 0
	defer func() { m.Cfg.FinalSoftcap = keep }()
	ctl := specVerifyCapped(t, m, g, prompt, toks, false)
	t.Logf("%s: the same rows uncapped against uncapped decode: worst logit NMSE %.3e", spec, ctl)
	return max(denseStepBand*ctl, 1e-9)
}

// specCapModels is the models TestSpecOverAFinalSoftcap runs.
func specCapModels() []string {
	if v := os.Getenv("JITLLM_SPEC_CAP_MODELS"); v != "" {
		return strings.Split(v, ",")
	}
	return []string{"synth-qwen35-hybrid-mtp.gguf", "qwen35/Qwen3.5-0.8B-MTP-Q8_0.gguf"}
}

// openSpecCapped opens an MTP model under the exact GEMM, so the host's
// verify rows are decode's bit for bit, with a final softcap of an eighth of
// its largest logit on the first speculation prompt, and returns that logit.
func openSpecCapped(t *testing.T, path string) (*Model, float64) {
	t.Helper()
	m := openSpecModel(t, path, noTune, WithKVF16(false), WithJITOptions(nn.WithGEMMExact(true)))
	amax := 0.0
	for _, l := range teacherForce(t, m, m.Vocab.Encode(specPrompts[0], true)) {
		for _, v := range l {
			amax = max(amax, math.Abs(float64(v)))
		}
	}
	if amax == 0 {
		m.Close()
		t.Fatal("the host's logits are all zero -- a cap would compare nothing")
	}
	m.Cfg.FinalSoftcap = float32(amax / 8)
	return m, amax
}

// specVerifyCapped runs toks after prompt on dev (nil is the host) twice: as
// plain decode, and as speculation steps of four rows each wanting every row's
// logits (stepRows, the verification Speculator.Next runs), and returns the
// worst logit NMSE of a verified row against decode at its position. noCap
// skips the cap in the host's verify head (rowsOut.noCap). On a device the
// steps must run as ragged rows (State.rowsVerifiable), the per-row head.
func specVerifyCapped(t *testing.T, m *Model, dev nn.Device, prompt, toks []int32, noCap bool) float64 {
	t.Helper()
	const k = 4
	toks = toks[:len(toks)/k*k]
	c := m.Cfg
	open := func() *State {
		st := m.NewState(len(prompt) + len(toks) + 2)
		if dev != nil {
			if err := st.SetDevice(dev); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := st.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		return st
	}
	dec := open()
	want := make([][]float32, 0, len(toks))
	for _, id := range toks {
		lg, err := dec.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, slices.Clone(lg))
	}
	dec.Close()
	rs := open()
	defer rs.Close()
	if dev != nil && !rs.rowsVerifiable() {
		t.Fatalf("a speculation step on the device would not run as rows: it would gate the prompt chunk")
	}
	worst := 0.0
	for i := 0; i < len(toks); i += k {
		out := rowsOut{logits: make([]float32, k*c.NVocab), hidden: make([]float32, k*c.NEmbd), noCap: noCap}
		func() {
			defer m.enterPager()()
			rs.kv.note(rs.seqPos(0), toks[i:i+k]...)
			if err := rs.stepRows(embedSrc(rs, toks[i:i+k]), toks[i:i+k], &out); err != nil {
				t.Fatal(err)
			}
		}()
		for j := range k {
			e := logitNMSE(out.logits[j*c.NVocab:(j+1)*c.NVocab], want[i+j])
			if math.IsNaN(e) || math.IsInf(e, 0) {
				return math.Inf(1)
			}
			worst = max(worst, e)
		}
	}
	return worst
}
