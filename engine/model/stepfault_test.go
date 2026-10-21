//go:build jitllmfault

package model

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
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
				g := stepTier(t, m, dev, c.shared)
				defer g.Close()
				g.SetFault("rec", fault)
				defer g.SetFault("rec", "")
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
// with every bank's down reading the gate's output and demands it fail: the
// grouped expert path the step takes must be what the gate compares, on the
// dp4a form and on sm_70's tensor cores alike. The violation follows the
// bank's format -- "downsrc" for a quantized bank, "floatsrc" for a float one
// (the synth fixtures) -- since each is the other's no-op.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsMixtureGateDiscriminates
func TestStepAcrossSessionsMixtureGateDiscriminates(t *testing.T) {
	for _, name := range stepMixtures() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openStepMixture(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g := stepTier(t, m, dev, false)
					defer g.Close()
					fault := downFault(t, m)
					g.SetFault("moe", fault)
					defer g.SetFault("moe", "")
					r := stepGate(t, m, g, false)
					mixtureStepChecks(t, m, r)
					t.Logf("%s: worst logit NMSE %.3e (row %d)", fault, r.worst, r.at)
					if math.IsNaN(r.worst) || r.worst < mixtureStepNMSE {
						t.Fatalf("the gate reads NMSE %.3e with every down reading the gate's output: it "+
							"cannot see the grouped path's wiring", r.worst)
					}
				})
			}
		})
	}
}

// downFault is the moeFault that puts a wrong source under m's down banks:
// "floatsrc" when every mixture block's down bank is float, "downsrc" when
// every one is quantized. A model with both fails, naming it: one fault would
// leave half its blocks unbroken and the other half's gate unproven.
func downFault(t *testing.T, m *Model) string {
	t.Helper()
	nf, nq := 0, 0
	for li := range m.layers {
		l := &m.layers[li]
		if !m.Cfg.MoEAt(li) {
			continue
		}
		_, isFloat := kernels.FloatOf(l.down.typ)
		_, isQuant := kernels.QuantOf(l.down.typ)
		switch {
		case isFloat:
			nf++
		case isQuant:
			nq++
		default:
			t.Fatalf("block %d's down bank is %v, which no device kernel reads", li, l.down.typ)
		}
	}
	switch {
	case nf > 0 && nq > 0:
		t.Fatalf("%d float and %d quantized down banks: one violation cannot break both", nf, nq)
	case nf > 0:
		return "floatsrc"
	case nq > 0:
		return "downsrc"
	}
	t.Fatal("no mixture block: the gate has no bank to break")
	return ""
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
					g := stepTier(t, m, dev, false)
					defer g.Close()
					g.SetFault("paged", "session")
					defer g.SetFault("paged", "")
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
				r := recPoolCase(t, m, dev, shared, ids, 6, tier.WithFault("rec", "resize"))
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
// the Scheduler (TestSchedulerMixesPromptsIntoDecodeSteps).
//
//	go test -tags jitllmfault ./engine/model -run TestStepRunsChainGateDiscriminates
func TestStepRunsChainGateDiscriminates(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev+"/sessions", func(t *testing.T) {
			g := stepTier(t, m, dev, false)
			defer g.Close()
			plan := runsPlan{early: []string{"The capital of France is", "Water boils at a temperature of"},
				late: "The clerk counted barrels of salt on the upper floor, and", chunk: 3, steps: 4}
			want := plan.alone(t, m, g)
			g.SetFault("rec", "chain")
			defer g.SetFault("rec", "")
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
			c.extra = []tier.Option{tier.WithFault("rec", "chain")}
			got, _, _, _ := c.run(nil)
			w, i, j := c.worst(got)
			t.Logf("chain: row %d parts at token %d, %.4f apart on the host", i, j, w)
			if i < 0 || w <= c.tie {
				t.Fatalf("the Scheduler gate passed with a chunk's rows not chaining (widest first difference %.4f)", w)
			}
		})
	}
}

// TestStepRunsOneRowHeadGateDiscriminates runs the one-row head's gate with
// the head launched through decode's own kernel, which reads the wanted row's
// sums where the next row's scales are, and demands it fails.
//
//	go test -tags jitllmfault ./engine/model -run TestStepRunsOneRowHeadGateDiscriminates
func TestStepRunsOneRowHeadGateDiscriminates(t *testing.T) {
	m := openStepDense(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			w, at, ones := oneRowHeadGap(t, m, dev, tier.WithFault("rows", "headone"))
			t.Logf("headone: %d one-row heads, worst logit NMSE %.3e (row %d)", ones, w, at)
			if w < oneRowHeadNMSE {
				t.Fatalf("the gate passed with the one-row head misreading its row, at NMSE %.3e", w)
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
			g, err := openTier(t, tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			g.SetFault("paged", "softwidth")
			defer g.SetFault("paged", "")
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
			g := stepTier(t, m, dev, false)
			defer g.Close()
			g.SetFault("rows", "nocap")
			defer g.SetFault("rows", "")
			r := stepGate(t, m, g, false)
			t.Logf("nocap: worst logit NMSE %.3e (row %d), %.0fx the bound %.3e", r.worst, r.at, r.worst/bound, bound)
			if r.worst < bound {
				t.Fatalf("the gate passed with the per-row head uncapped, at NMSE %.3e", r.worst)
			}
		})
	}
}

// TestStepAcrossSessionsLocalRopeGateDiscriminates runs the step gate on
// gemma-3-1b with the step's local layers handed the global rotary table
// ("swatab"), where the device uploads the host's tables
// (Config.RopeTableHost) so the table reaches the kernels, and demands it
// fail: a local layer rotated at the wrong base must not pass for its own.
//
//	go test -tags jitllmfault ./engine/model -run TestStepAcrossSessionsLocalRopeGateDiscriminates
func TestStepAcrossSessionsLocalRopeGateDiscriminates(t *testing.T) {
	m, err := Open(stepArchPath(t, "gemma-3-1b-it-Q4_K_M.gguf"), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, dev := range stepDevices() {
		for _, fault := range []string{"", "swatab"} {
			t.Run(fmt.Sprintf("%s/%q", dev, fault), func(t *testing.T) {
				bound := denseStepBand * orderControl(t, m, dev)
				g, err := openTier(t, append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
					tier.WithConfig(func(c *tier.Config) { c.RopeTableHost = true })}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				g.SetFault("rows", fault)
				defer g.SetFault("rows", "")
				r := stepGate(t, m, g, false)
				t.Logf("%q, host tables: worst logit NMSE %.3e (row %d), %.2gx the bound %.3e", fault, r.worst, r.at,
					r.worst/bound, bound)
				switch {
				case fault == "" && !(r.worst < bound):
					t.Fatalf("the clean arm with the host's tables reads NMSE %.3e", r.worst)
				case fault != "" && r.worst < bound:
					t.Fatalf("the gate passed with the local layers on the global table, at NMSE %.3e", r.worst)
				}
			})
		}
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
		moe  []string
	}{
		{"synth-llama4.gguf", []string{"nope", "notemp", "nowin"}, []string{"wout"}},
		{"synth-cohere2.gguf", []string{"nope", "nowin"}, nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := Open(stepArchPath(t, c.name), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range stepDevices() {
				run := func(fault, family string) {
					t.Run(dev+"/"+fault, func(t *testing.T) {
						g := stepTier(t, m, dev, false)
						defer g.Close()
						g.SetFault(family, fault)
						r := stepGate(t, m, g, false)
						t.Logf("%s: worst logit NMSE %.3e (row %d)", fault, r.worst, r.at)
						if r.worst < stepBound(m, c.name, r, 0) {
							t.Fatalf("the gate passed with %q in the step, at NMSE %.3e", fault, r.worst)
						}
					})
				}
				for _, f := range c.rows {
					run(f, "rows")
				}
				for _, f := range c.moe {
					run(f, "moe")
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
			g := stepTier(t, m, dev, false)
			defer g.Close()
			g.SetFault("paged", "session")
			defer g.SetFault("paged", "")
			r := stepGate(t, m, g, false)
			t.Logf("session: worst logit NMSE %.3e (row %d), %.0fx the bound %.3e", r.worst, r.at, r.worst/bound, bound)
			if r.worst < bound {
				t.Fatalf("the gate passed rows reading the current session's pages at NMSE %.3e", r.worst)
			}
		})
	}
}
