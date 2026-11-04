//go:build jitllmfault

package model

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestStepAcrossSessionsHybridGateDiscriminates runs the hybrid step gate under
// each violation of the recurrent pools' wiring and demands its comparison
// lands outside the band it allows (RULE 10): a gate that passes two sessions
// sharing one summary, or a session reading a half its last step did not
// write, proves nothing about the configuration it was written for.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsHybridGateDiscriminates
func TestStepAcrossSessionsHybridGateDiscriminates(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		for _, c := range []struct {
			fault  string
			shared bool
		}{{"slot", false}, {"parity", false}, {"slot", true}, {"parity", true}} {
			fault := c.fault
			t.Run(fmt.Sprintf("%s/%s/shared=%v", dev, fault, c.shared), func(t *testing.T) {
				var f gpu.Faults
				f.SetRecFault(fault)
				g := stepTier(t, m, dev, c.shared, gpu.WithFaults(&f))
				defer g.Close()
				r := stepGate(t, m, g, true)
				t.Logf("%s: worst logit NMSE %.3e (row %d), %d linear block(s) across sessions",
					fault, r.worst, r.at, r.linear)
				if r.worst < mlaDeviceNMSE {
					t.Fatalf("the gate passed under %q at NMSE %.3e: it cannot tell this violation "+
						"from the wiring it guards", fault, r.worst)
				}
			})
		}
	}
}

// TestStepAcrossSessionsMixtureGateDiscriminates runs the mixture step gate
// with every expert down bank of a joint step's form reading the experts'
// gate output (gpu.Faults.SetMoEFault) and demands it lands outside the band: a
// gate green with the grouped path's wiring broken proves nothing about it.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsMixtureGateDiscriminates
func TestStepAcrossSessionsMixtureGateDiscriminates(t *testing.T) {
	for _, name := range stepMixtures() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openStepMixture(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					var f gpu.Faults
					f.SetMoEFault(true)
					h0, u0, o0 := f.MoEFaultSeen()
					g := stepTier(t, m, dev, false, gpu.WithFaults(&f))
					defer g.Close()
					r := stepGate(t, m, g, false)
					mixtureStepChecks(t, m, r)
					h, u, o := f.MoEFaultSeen()
					t.Logf("worst logit NMSE %.3e (row %d); the fault rewired %d down bank(s), found no source for %d, left %d one-row",
						r.worst, r.at, h-h0, u-u0, o-o0)
					if h == h0 {
						t.Fatalf("the fault rewired no down bank (%d unsourced, %d in one-row forms): the violation never ran", u-u0, o-o0)
					}
					// Against a finite reference, a joint step the fault turned to
					// NaN has parted from it as far as one can (DeepSeek-V4's
					// down bank over the raw gate overflows on CUDA).
					if !r.refFinite {
						t.Fatalf("the sessions alone read a non-finite logit: the reference is degenerate")
					}
					if !math.IsNaN(r.worst) && r.worst < mixtureStepNMSE {
						t.Fatalf("the gate reads NMSE %.3e with every joint step's down bank reading the experts' gate: it "+
							"cannot see the grouped path's wiring", r.worst)
					}
				})
			}
		})
	}
}

// TestStepAcrossSessionsLatentGateDiscriminates runs the latent step gate with
// every row of a step across sessions reading and writing the current
// session's pages ("session"), and demands it fail: rows reading another
// row's latent history -- finite and fluent -- must not pass for their own.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsLatentGateDiscriminates
func TestStepAcrossSessionsLatentGateDiscriminates(t *testing.T) {
	for _, name := range latentModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openLatent(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					var f gpu.Faults
					f.SetPagedFault("session")
					g := stepTier(t, m, dev, false, gpu.WithFaults(&f))
					defer g.Close()
					r := latentStepGate(t, m, g)
					t.Logf("session: worst logit NMSE %.3e (row %d)", r.worst, r.at)
					if r.worst < mlaDeviceNMSE {
						t.Fatalf("the gate passed rows reading the current session's latent pages at NMSE %.3e",
							r.worst)
					}
				})
			}
		})
	}
}

// TestRecurrentPoolGateDiscriminates runs TestRecurrentPoolKeepsASessionAcrossOthers
// with every pool rebuilt empty and demands it sees the session lose its
// state: a gate green with the carry removed would be comparing two sessions
// that never had one to lose.
//
//	go test -tags jitllmfault ./engine/model -run TestRecurrentPoolGateDiscriminates
func TestRecurrentPoolGateDiscriminates(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is", true)
	for _, dev := range stepDevices() {
		for _, shared := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/shared=%v", dev, shared), func(t *testing.T) {
				var f gpu.Faults
				f.SetRecFault("resize")
				r := recPoolCase(t, m, dev, shared, ids, 6, gpu.WithFaults(&f))
				if r.p < 0 {
					t.Fatal("the gate passed with every rebuilt pool empty: it cannot see a lost state")
				}
				t.Logf("resize: token %d logit %d is %g, alone %g", r.p, r.i, r.got, r.want)
			})
		}
	}
}

// TestStepRunsChainGateDiscriminates runs the hybrid's prompt-chunk gates with
// every row of a ragged step its own run -- a prompt chunk's rows each reading
// its sequence's recurrent state from before the step -- and demands both see
// it: the step across sessions (TestStepRunsAdmitsAPromptBesideDecoding) and
// the Batcher (TestSchedulerMixesPromptsIntoDecodeSteps).
//
//	go test -tags jitllmfault ./engine/model -run TestStepRunsChainGateDiscriminates
func TestStepRunsChainGateDiscriminates(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev+"/sessions", func(t *testing.T) {
			var f gpu.Faults
			g := stepTier(t, m, dev, false, gpu.WithFaults(&f))
			defer g.Close()
			plan := runsPlan{early: []string{"The capital of France is", "Water boils at a temperature of"},
				late: "The clerk counted barrels of salt on the upper floor, and", chunk: 3, steps: 4}
			want := plan.alone(t, m, g)
			f.SetRecFault("chain")
			got, _ := plan.together(t, m, g)
			w, at := worstStep(got, want)
			t.Logf("chain: worst logit NMSE %.3e (row %d)", w, at)
			if w < mlaDeviceNMSE {
				t.Fatalf("the gate passed with a chunk's rows not chaining, at NMSE %.3e", w)
			}
		})
		t.Run(dev+"/scheduler", func(t *testing.T) {
			ids := [][]int32{m.Vocab.Encode("The capital of France is", true),
				m.Vocab.Encode("Once upon a time, in a small village by the sea, there lived", true),
				m.Vocab.Encode("1, 2, 3,", true)}
			c := newMixedCase(t, m, ids, []int{8, 8, 8}, dev)
			var f gpu.Faults
			f.SetRecFault("chain")
			c.extra = []gpu.Option{gpu.WithFaults(&f)}
			got, _, _, _ := c.run(nil)
			w, i, j := c.worst(got)
			t.Logf("chain: row %d parts at token %d, %.4f apart on the host", i, j, w)
			if i < 0 || w <= c.tie {
				t.Fatalf("the Batcher gate passed with a chunk's rows not chaining (widest first difference %.4f)", w)
			}
		})
	}
}

// TestStagedPagedPrefillGateDiscriminates launches the staged prefill's softmax
// 64 wide whatever its lanes, a width the kernel does not declare, and demands
// every backend refuse it by the kernel's name and the prompt fail saying so.
// The three disagreed about which width runs (Metal ran the launch's and
// prefilled every staged prompt wrong; CUDA and Vulkan happened to be right),
// so the backend refuses any launch whose width is not the declared one
// rather than leave the answer to the runtime.
//
//	go test -tags jitllmfault ./engine/model -run TestStagedPagedPrefillGateDiscriminates
func TestStagedPagedPrefillGateDiscriminates(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and", true)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			var f gpu.Faults
			f.SetPagedFault("softwidth")
			g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff), gpu.WithFaults(&f))
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			plan := runsPlan{}
			chunk := plan.placed(t, m, g, len(ids)+1)
			defer chunk.Close()
			_, err = chunk.Prefill(ids)
			t.Logf("softwidth: %v", err)
			if err == nil || !strings.Contains(err.Error(), "pagedprefillsoftmax") ||
				!strings.Contains(err.Error(), "launched 64 wide") {
				t.Fatalf("the staged softmax launched 64 wide was not refused by name: %v", err)
			}
		})
	}
}

// TestStepAcrossSessionsSoftcapGateDiscriminates runs the final-softcap step
// gate with the per-row head skipping the cap ("nocap") and demands it fail:
// the cap the step applies must be what the gate compares.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsSoftcapGateDiscriminates
func TestStepAcrossSessionsSoftcapGateDiscriminates(t *testing.T) {
	m := openSoftcapStep(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			bound := denseStepBand * orderControl(t, m, dev)
			var f gpu.Faults
			f.SetRowsFault("nocap")
			g := stepTier(t, m, dev, false, gpu.WithFaults(&f))
			defer g.Close()
			r := stepGate(t, m, g, false)
			t.Logf("nocap: worst logit NMSE %.3e (row %d), %.0fx the bound %.3e", r.worst, r.at, r.worst/bound, bound)
			if r.worst < bound {
				t.Fatalf("the gate passed with the per-row head uncapped, at NMSE %.3e", r.worst)
			}
		})
	}
}

// TestStepAcrossSessionsLlama4GateDiscriminates runs the step gate on the
// Llama 4 and Cohere2 fixtures with each of Llama 4's attention and mixture
// features taken out of the ragged step alone -- the unrotated layers rotated
// ("nope"), the attention temperature skipped ("notemp"), the chunk or window
// ignored ("nowin"), the expert's weight on its output ("wout") -- and demands
// each fail: the features the step runs must be what the gate compares.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsLlama4GateDiscriminates
func TestStepAcrossSessionsLlama4GateDiscriminates(t *testing.T) {
	for _, c := range []struct {
		name string
		rows []string
	}{
		{"synth-llama4.gguf", []string{"nope", "notemp", "nowin"}},
		{"synth-cohere2.gguf", []string{"nope", "nowin"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := Open(stepArchPath(t, c.name), WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range stepDevices() {
				run := func(fault string, set func(*gpu.Faults, string)) {
					t.Run(dev+"/"+fault, func(t *testing.T) {
						var f gpu.Faults
						set(&f, fault)
						g := stepTier(t, m, dev, false, gpu.WithFaults(&f))
						defer g.Close()
						r := stepGate(t, m, g, false)
						t.Logf("%s: worst logit NMSE %.3e (row %d)", fault, r.worst, r.at)
						if r.worst < stepBound(m, c.name, r, 0) {
							t.Fatalf("the gate passed with %q in the step, at NMSE %.3e", fault, r.worst)
						}
					})
				}
				for _, f := range c.rows {
					run(f, (*gpu.Faults).SetRowsFault)
				}
			}
		})
	}
}

// TestStepAcrossSessionsDenseGateDiscriminates runs the dense step gate with
// every row of a step across sessions reading and writing the current
// session's pages ("session") and demands it fail by a wide margin: a row
// attending another session's history is fluent, and the dense bound is
// denseStepBand times an order-only control (orderControl), so the
// violation's reading against it is the margin the gate has.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsDenseGateDiscriminates
func TestStepAcrossSessionsDenseGateDiscriminates(t *testing.T) {
	m := openStepDense(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			bound := denseStepBand * orderControl(t, m, dev)
			var f gpu.Faults
			f.SetPagedFault("session")
			g := stepTier(t, m, dev, false, gpu.WithFaults(&f))
			defer g.Close()
			r := stepGate(t, m, g, false)
			t.Logf("session: worst logit NMSE %.3e (row %d), %.0fx the bound %.3e", r.worst, r.at, r.worst/bound, bound)
			if r.worst < bound {
				t.Fatalf("the gate passed rows reading the current session's pages at NMSE %.3e", r.worst)
			}
		})
	}
}

// TestComposedPromptChainGateDiscriminates runs the hybrid's prompt through
// the composer (ComposedPrompt), its linear blocks scanning the chunk's rows
// as one run, with every row of the chunk made its own run -- each reading
// the state from before the chunk -- and demands the prompt's logits part
// from the same prompt chained: the serving gate's prompt arm
// (TestComposedBlocksServe) can see a chunk that does not
// chain. Both arms must have served every block of the prompt.
//
//	go test -tags jitllmfault ./engine/model -run TestComposedPromptChainGateDiscriminates
func TestComposedPromptChainGateDiscriminates(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and the river rose", true)
	toks := []int32{ids[1], ids[2]}
	parts := gpu.ComposedBlock | gpu.ComposedHead | gpu.ComposedPrompt
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			want, st, _ := composedDecode(t, m, dev, ids, toks, parts, true, false)
			var f gpu.Faults
			f.SetRecFault("chain")
			got, fst, _ := composedDecode(t, m, dev, ids, toks, parts, true, false, gpu.WithFaults(&f))
			for _, s := range []gpu.Stats{st, fst} {
				if n, d := s.ServedPart(gpu.ComposedPrompt); n < m.Cfg.NLayer || d != 0 {
					t.Fatalf("the composer served %d block steps of the prompt and declined %d, of %d blocks", n, d, m.Cfg.NLayer)
				}
			}
			w := logitNMSE(want[0], got[0])
			t.Logf("chain: the prompt's logits %.3e apart (NMSE)", w)
			if w < mlaDeviceNMSE {
				t.Fatalf("the gate passed with a composed chunk's rows not chaining, at NMSE %.3e", w)
			}
		})
	}
}
