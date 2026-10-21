//go:build amd64 || arm64

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSpecGreedyOnDevice is the contract on a device: the trunk and the
// prediction block on the card, plain greedy decode on the same card as the
// reference, every rollback mode a device runs.
//
// A verification is a chunk of rows and decode is one row, and on a device
// those are different kernels (the batched matmul against the decode matvec),
// so the logits agree to the f32 reduction band rather than to the bit. A
// disagreement is therefore judged where it happens, as logitJudge judges a
// batch: the device's decode logits at that position against the same
// position computed as a chunk's row, and a flip whose margin is inside
// their disagreement is the model undecided -- the comparison stops there.
// A flip wider than the perturbation is a defect. Every backend the host
// has runs (JITLLM_TEST_DEVICES narrows it).
func TestSpecGreedyOnDevice(t *testing.T) {
	n := specTokens()
	specs := testdevices.Specs("cuda:0", "vulkan:0", "metal") // metal takes no selector
	for name, path := range specModels(t) {
		t.Run(name, func(t *testing.T) {
			m := openSpecModel(t, path, noTune, WithKVF16(false))
			defer m.Close()
			ran := 0
			for _, spec := range specs {
				t.Run(spec, func(t *testing.T) {
					g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
						tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					ran++
					arms := []struct {
						name string
						opts []SpecOption
					}{{"k3", []SpecOption{WithSpecDraft(3)}}, {"adaptive", nil},
						{"schedule", []SpecOption{withSpecSchedule(specSchedule)}}}
					var drafted, accepted int64
					for _, text := range specPrompts {
						prompt := m.Vocab.Encode(text, true)
						want := plainGreedy(t, m, prompt, n, g)
						for _, arm := range arms {
							got, st, onDev := specGreedyDev(t, m, prompt, n, g, oracleFor(name, want), arm.opts...)
							if !onDev {
								t.Fatalf("%s: the trunk and the prediction block are not both on %s: "+
									"this would gate the host", arm.name, spec)
							}
							drafted += st.Drafted
							accepted += st.Accepted
							if i := firstDiff(got, want); i >= 0 {
								gap, pert := devFlip(t, m, g, prompt, want[:i], want[i], got[i])
								if gap > pert {
									t.Fatalf("%s, %q: token %d is %d, plain greedy on %s says %d, %.4f apart "+
										"against a %.4f reach of the band -- not a tie", arm.name, text, i, got[i], spec,
										want[i], gap, pert)
								}
								t.Logf("%s, %q: a tie at token %d (%.4f apart, the band reaching %.4f); equal before it",
									arm.name, text[:min(len(text), 24)], i, gap, pert)
							}
							t.Logf("%s %s %-8s %q: %d rounds, %.2f tokens a pass, rollback %v (%d replayed rows, %d replay passes)",
								name, spec, arm.name, text[:min(len(text), 24)], st.Rounds, st.TokensPerRound(),
								st.Rollback, st.Replayed, st.Commits)
						}
					}
					if drafted == 0 || accepted == 0 {
						t.Fatalf("%d drafted and %d accepted on %s: the speculation never ran", drafted, accepted, spec)
					}
					checkLinearRowsForm(t, m, g)
					// The breaks, on the device: each must part from plain
					// decode somewhere past a tie.
					faults := []struct {
						name  string
						fault specFault
					}{{"accept-unchecked", faultAcceptUnchecked}, {"keep-rejected-kv", faultKeepRejectedKV}}
					if m.Cfg.Hybrid() {
						faults = append(faults, struct {
							name  string
							fault specFault
						}{"keep-rejected-rec", faultKeepRejectedRec})
					}
					for _, f := range faults {
						caught := false
						for _, text := range specPrompts {
							prompt := m.Vocab.Encode(text, true)
							n := faultTokens(m, prompt, n)
							want := plainGreedy(t, m, prompt, n, g)
							got := specFaultDev(t, m, prompt, n, g, f.fault, oracleFor(name, want))
							i := firstDiff(got, want)
							if i < 0 {
								continue
							}
							if gap, pert := devFlip(t, m, g, prompt, want[:i], want[i], got[i]); gap > pert {
								t.Logf("%s on %s: %q parts at token %d, %.4f apart against a %.4f reach of the band "+
									"-- caught", f.name, spec, text[:min(len(text), 24)], i, gap, pert)
								caught = true
								break
							}
						}
						if !caught {
							t.Errorf("%s on %s: every prompt matched plain decode to a tie -- the gate cannot see it",
								f.name, spec)
						}
					}
				})
			}
			if ran == 0 {
				t.Skip("no device on this host -- this gate proved nothing")
			}
		})
	}
}

// specGreedyDev is specGreedy with the trunk and the draft on g, and whether
// every block of both and the trunk's head landed there.
func specGreedyDev(t *testing.T, m *Model, prompt []int32, n int, g *tier.GPU, oracle []int32,
	opts ...SpecOption) ([]int32, SpecStats, bool) {
	t.Helper()
	st := m.NewState(len(prompt) + n + 8)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	sp, err := st.Speculate(opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Close()
	sp.oracle, sp.corrupt = oracle, oracleCorrupt
	d := sp.Draft()
	onDev := st.GPULayers() == m.Cfg.NLayer && st.HeadOnDevice() && d.GPULayers() == 1 && d.HeadOnDevice()
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
	return out[:n], sp.Stats(), onDev
}

// specFaultDev is specGreedyDev with one of the breaks in force, three drafts
// a round.
func specFaultDev(t *testing.T, m *Model, prompt []int32, n int, g *tier.GPU, fault specFault, oracle []int32) []int32 {
	t.Helper()
	st := m.NewState(len(prompt) + 5*n + 8) // as specGreedy's, for the breaks
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	sp, err := st.Speculate(WithSpecDraft(3))
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Close()
	sp.fault, sp.oracle, sp.corrupt = fault, oracle, oracleCorrupt
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
	return out[:n]
}

// devFlip measures a divergence at the position after prompt+prefix on g:
// the margin between a and b in the plain decode's logits there, and the
// widest margin the arithmetic band can reverse -- twice the largest
// disagreement between those logits and the same position computed by any
// other path that legitimately computes it, since a band of p moves one logit
// up by p and the other down by p. A flip inside that reach is the model
// undecided, not a defect. The paths:
//
//   - the last row of a prompt chunk;
//   - a history written as speculation writes it, every generated position
//     through steps of 2 to specMaxRows rows (stepRows), which on a device are
//     ragged steps whose matvecs and recurrence differ from decode's. The
//     history matters, not only the last step: a flip can occur where the
//     chunk, a last four-row step and decode agree exactly at that position;
//   - the host, decoding the same prefix: the device's own band against it. A
//     margin-only judge would call a flip inside that band a defect.
func devFlip(t *testing.T, m *Model, g *tier.GPU, prompt, prefix []int32, a, b int32) (gap, perturb float64) {
	t.Helper()
	all := append(append([]int32(nil), prompt...), prefix...)
	decode := func(dev nn.Device) []float32 {
		st := m.NewState(len(all) + 2)
		defer st.Close()
		if dev != nil {
			if err := st.SetDevice(dev); err != nil {
				t.Fatal(err)
			}
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range prefix {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), lg...)
	}
	ld := decode(g)
	gap = math.Abs(float64(ld[a] - ld[b]))
	perturb = maxAbsDiff(ld, decode(nil))
	ch := m.NewState(len(all) + 2)
	defer ch.Close()
	if err := ch.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	lb, err := ch.Prefill(all)
	if err != nil {
		t.Fatal(err)
	}
	perturb = max(perturb, maxAbsDiff(ld, lb))
	c := m.Cfg
	for step := 2; step <= specMaxRows && step <= len(prefix); step++ {
		rs := m.NewState(len(all) + 2)
		if err := rs.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if _, err := rs.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		var lr []float32
		for i := 0; i < len(prefix); {
			j := i + step
			if i == 0 && len(prefix)%step != 0 {
				j = len(prefix) % step
			}
			k := j - i
			out := rowsOut{logits: make([]float32, k*c.NVocab), hidden: make([]float32, k*c.NEmbd)}
			func() {
				defer m.enterPager()()
				rs.kv.note(rs.seqPos(0), prefix[i:j]...)
				if err := rs.stepRows(embedSrc(rs, prefix[i:j]), prefix[i:j], &out); err != nil {
					t.Fatal(err)
				}
			}()
			lr, i = out.logits[(k-1)*c.NVocab:], j
		}
		rs.Close()
		perturb = max(perturb, maxAbsDiff(ld, lr))
	}
	return gap, 2 * perturb
}

// specMaxRows is the most rows a verification of the device gate's arms
// runs: three drafts and next, behind the replay of a rejected round's
// accepted rows (at most three).
const specMaxRows = 7
