//go:build amd64 || arm64

package model

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The prompt-lookup gates. Greedy decoding through a lookup Speculator must be
// greedy decoding, token for token, on a dense model, a hybrid (a recurrent
// state to roll back) and a state-space model; with drafts actually drafted
// and accepted, which on these models' own outputs is the selection check.

// lookupModels are the models prompt lookup is gated on: two real dense ones,
// the hybrid fixture (its prediction block ignored: lookup is forced) and a
// state-space fixture.
var lookupModels = []string{
	"stories15M-q8_0.gguf",
	"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
	"synth-qwen35-hybrid-mtp.gguf",
	"synth-mamba2.gguf",
	"synth-qwen4exp.gguf",
}

// copyPrompt asks for a paragraph again: what prompt lookup exists for.
const copyPrompt = "Repeat this paragraph exactly.\n\nThe old lighthouse keeper climbed the spiral stairs " +
	"every evening at dusk, carrying a small brass lantern and a worn leather notebook. He wrote down the " +
	"colour of the sky, the direction of the wind and the names of the ships that passed.\n\n" +
	"The old lighthouse keeper climbed the spiral stairs every evening"

var lookupPrompts = append([]string{copyPrompt}, specPrompts...)

func openLookupModel(t *testing.T, name string, opts ...Option) *Model {
	t.Helper()
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), opts...)
	if err != nil {
		t.Fatal(err)
	}
	if m.Vocab == nil {
		m.Close()
		t.Fatalf("%s: no tokenizer: %v", name, m.TokErr)
	}
	return m
}

// lookupGreedy is n tokens through a lookup Speculator, on dev when it is not
// nil.
func lookupGreedy(t *testing.T, m *Model, prompt []int32, n int, dev nn.Device, fault specFault,
	oracle []int32, opts ...SpecOption) ([]int32, SpecStats) {
	t.Helper()
	size := len(prompt) + 5*n + 8
	if c := m.Cfg.NCtx; c > 0 {
		size = min(size, c)
	}
	st := m.NewState(size)
	defer st.Close()
	if dev != nil {
		if err := st.SetDevice(dev); err != nil {
			t.Fatal(err)
		}
	}
	sp, err := st.Speculate(append([]SpecOption{WithSpecDrafter(SpecDraftLookup)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Close()
	if sp.Draft() != nil || sp.Stats().Drafter != SpecDraftLookup {
		t.Fatalf("a lookup Speculator ran a draft model (drafter %v)", sp.Stats().Drafter)
	}
	sp.fault = fault
	sp.oracle, sp.corrupt = oracle, oracleCorrupt
	y, err := sp.Start(prompt, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := []int32{y}
	for len(out) < n {
		ys, err := sp.Next(nil)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ys...)
	}
	return out[:n], sp.Stats()
}

// lookupTokens is n, or what the model's context leaves after prompt.
func lookupTokens(m *Model, prompt []int32, n int) int {
	if c := m.Cfg.NCtx; c > 0 {
		n = min(n, c-len(prompt)-8)
	}
	return n
}

func lookupArms(m *Model) []struct {
	name string
	opts []SpecOption
} {
	arms := []struct {
		name string
		opts []SpecOption
	}{
		{"default", nil},
		{"k2-ngram1", []SpecOption{WithSpecDraft(2), WithSpecNGram(1, 3)}},
		{"schedule", []SpecOption{withSpecSchedule([]int{0, 4, 1, 0, 6})}},
	}
	if m.Cfg.Hybrid() {
		arms = append(arms,
			struct {
				name string
				opts []SpecOption
			}{"replay", []SpecOption{WithSpecRollback(SpecRollbackReplay)}},
			struct {
				name string
				opts []SpecOption
			}{"rows", []SpecOption{WithSpecRollback(SpecRollbackRows)}})
	}
	return arms
}

// TestSpecLookupMatchesDecode is the contract on the host, exact arithmetic,
// and the acceptance report: drafted, accepted and tokens a pass on a copying
// prompt against plain ones.
func TestSpecLookupMatchesDecode(t *testing.T) {
	n := specTokens()
	for _, name := range lookupModels {
		t.Run(strings.TrimSuffix(name, ".gguf"), func(t *testing.T) {
			m := openLookupModel(t, name, noTune, WithJITOptions(nn.WithGEMMExact(true)))
			defer m.Close()
			var drafted, accepted int64
			for _, text := range lookupPrompts {
				prompt := m.Vocab.Encode(text, true)
				n := lookupTokens(m, prompt, n)
				want := plainGreedy(t, m, prompt, n, nil)
				for _, arm := range lookupArms(m) {
					got, st := lookupGreedy(t, m, prompt, n, nil, specFaultNone, nil, arm.opts...)
					if i := firstDiff(got, want); i >= 0 {
						t.Fatalf("%s, prompt %q: token %d is %d, plain greedy says %d (%d drafted, %d accepted, "+
							"rollback %v)", arm.name, text, i, got[i], want[i], st.Drafted, st.Accepted, st.Rollback)
					}
					drafted += st.Drafted
					accepted += st.Accepted
					t.Logf("%-9s %-24q: %3d rounds, %4d drafted, %4d accepted, %.2f tokens a pass, rollback %v",
						arm.name, text[:min(len(text), 24)], st.Rounds, st.Drafted, st.Accepted,
						st.TokensPerRound(), st.Rollback)
				}
			}
			// A random-weight fixture rarely repeats itself, so its own drafts
			// are rarely accepted; the accepting and rejecting paths on it are
			// TestSpecLookupGateDiscriminates's forced arm, which demands both.
			if strings.HasPrefix(name, "synth") {
				t.Logf("%d drafted, %d accepted on a random-weight fixture", drafted, accepted)
			} else if drafted == 0 || accepted == 0 {
				t.Fatalf("%d drafted and %d accepted: the speculation never ran", drafted, accepted)
			}
		})
	}
}

// TestSpecLookupGateDiscriminates runs the equality gate against the breaks,
// with the drafts forced to plain decode's tokens every fifth one corrupted, so
// every round both accepts and rejects; each break must part from plain decode.
func TestSpecLookupGateDiscriminates(t *testing.T) {
	n := specTokens()
	for _, name := range lookupModels {
		t.Run(strings.TrimSuffix(name, ".gguf"), func(t *testing.T) {
			m := openLookupModel(t, name, noTune, WithJITOptions(nn.WithGEMMExact(true)))
			defer m.Close()
			type fault struct {
				name  string
				fault specFault
				opts  []SpecOption
			}
			faults := []fault{{"none", specFaultNone, nil}, {"accept-unchecked", faultAcceptUnchecked, nil},
				{"keep-rejected-kv", faultKeepRejectedKV, nil}}
			if m.Cfg.Hybrid() {
				faults = append(faults,
					fault{"keep-rejected-rec-rows", faultKeepRejectedRec, []SpecOption{WithSpecRollback(SpecRollbackRows)}},
					fault{"keep-rejected-rec-replay", faultKeepRejectedRec, []SpecOption{WithSpecRollback(SpecRollbackReplay)}})
			}
			for _, f := range faults {
				caught := false
				for _, text := range lookupPrompts {
					prompt := m.Vocab.Encode(text, true)
					n := faultTokens(m, prompt, n)
					want := plainGreedy(t, m, prompt, n, nil)
					got, st := lookupGreedy(t, m, prompt, n, nil, f.fault, want,
						append([]SpecOption{WithSpecDraft(3)}, f.opts...)...)
					i := firstDiff(got, want)
					if f.fault == specFaultNone {
						if i >= 0 {
							t.Fatalf("the forced drafts with no break part from plain decode at token %d", i)
						}
						if st.Accepted == 0 || st.Accepted == st.Drafted {
							t.Fatalf("forced drafts: %d drafted, %d accepted -- the control did not both accept "+
								"and reject", st.Drafted, st.Accepted)
						}
						continue
					}
					if i >= 0 {
						t.Logf("%s: %q diverges at token %d (%d drafted, %d accepted) -- caught", f.name,
							text[:min(len(text), 24)], i, st.Drafted, st.Accepted)
						caught = true
						break
					}
				}
				if f.fault != specFaultNone && !caught {
					t.Errorf("%s: every prompt still matched plain greedy -- the gate cannot see this break", f.name)
				}
			}
		})
	}
}

// TestSpecLookupOnDevice is the contract with the trunk on a device, judged as
// TestSpecGreedyOnDevice judges a flip: inside the band it is a tie.
func TestSpecLookupOnDevice(t *testing.T) {
	n := specTokens()
	specs := testdevices.Specs("cuda:0", "vulkan:0", "metal")
	for _, name := range lookupModels {
		t.Run(strings.TrimSuffix(name, ".gguf"), func(t *testing.T) {
			m := openLookupModel(t, name, noTune, WithKVF16(false))
			defer m.Close()
			ran := 0
			for _, spec := range specs {
				t.Run(spec, func(t *testing.T) {
					g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
						tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					ran++
					var accepted int64
					heads := 0
					for _, text := range lookupPrompts[:2] {
						prompt := m.Vocab.Encode(text, true)
						n := lookupTokens(m, prompt, n)
						want := plainGreedy(t, m, prompt, n, g)
						st := m.NewState(len(prompt) + n + 8)
						if err := st.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						if st.GPULayers() != m.Cfg.NLayer {
							t.Fatalf("%d of %d blocks on %s: this would gate the host", st.GPULayers(), m.Cfg.NLayer, spec)
						}
						st.Close()
						// A random-weight fixture repeats nothing: its drafts are forced
						// (plain decode's tokens, every fifth wrong), so the device's
						// rollback both keeps and rejects rows.
						got, ss := lookupGreedy(t, m, prompt, n, g, specFaultNone, oracleFor(name, want))
						accepted += ss.Accepted
						heads += ss.DeviceHeads
						if i := firstDiff(got, want); i >= 0 {
							gap, pert := devFlip(t, m, g, prompt, want[:i], want[i], got[i])
							if gap > pert {
								t.Fatalf("%q: token %d is %d, plain greedy on %s says %d, %.4f apart against a "+
									"%.4f reach of the band -- not a tie", text, i, got[i], spec, want[i], gap, pert)
							}
							t.Logf("%q: a tie at token %d (%.4f apart, band %.4f); equal before it",
								text[:min(len(text), 24)], i, gap, pert)
						}
						t.Logf("%s %-24q: %d rounds, %d drafted, %d accepted, %.2f tokens a pass, rollback %v",
							spec, text[:min(len(text), 24)], ss.Rounds, ss.Drafted, ss.Accepted, ss.TokensPerRound(),
							ss.Rollback)
					}
					if accepted == 0 {
						t.Fatalf("nothing accepted on %s: the speculation never ran", spec)
					}
					// A hyper-connected model's verifications project on the
					// device after the host's collapse (rowsHead).
					if m.Cfg.streamHead() && heads == 0 {
						t.Fatalf("no verification's head ran on %s: the host projected every one", spec)
					}
				})
			}
			if ran == 0 {
				t.Skip("no device on this host -- this gate proved nothing")
			}
		})
	}
}

// TestLookupSearch pins the search: the longest n first, the latest
// occurrence, the continuation cut at k and at the history's end, and no
// allocation once the destination has room.
func TestLookupSearch(t *testing.T) {
	h := []int32{1, 2, 3, 9, 1, 2, 3, 7, 8, 5, 2, 3}
	cases := []struct {
		lo, hi, k int
		want      []int32
	}{
		{2, 4, 3, []int32{7, 8, 5}}, // "2 3" latest at 5
		{1, 1, 4, []int32{7, 8, 5, 2}},
		{3, 4, 3, nil}, // "5 2 3" never occurred
		{2, 2, 9, []int32{7, 8, 5, 2, 3}},
	}
	for _, c := range cases {
		if got := appendLookup(nil, h, c.lo, c.hi, c.k); !slices.Equal(got, c.want) {
			t.Errorf("n %d..%d k %d: %v, want %v", c.lo, c.hi, c.k, got, c.want)
		}
	}
	if got := appendLookup(nil, []int32{4, 4}, 1, 2, 3); !slices.Equal(got, []int32{4}) {
		t.Errorf("a run: %v", got)
	}
	dst := make([]int32, 0, 8)
	if a := testing.AllocsPerRun(100, func() { dst = appendLookup(dst[:0], h, 2, 4, 5) }); a != 0 {
		t.Errorf("the search allocates %.0f times a call", a)
	}
}
