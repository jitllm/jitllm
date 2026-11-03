package model

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// hostStepNMSE bounds a host step across sessions against each session
// alone: the composed CPU's forms over a step's rows against its forms over
// one, which fold a block in another order (RULE 8: the bar across forms is
// the band, not bit-identity). Measured on the Xeon at 2.4e-3 on
// Llama-3.2-1B, 7.5e-4 on Qwen3.5-0.8B, 9.6e-4 on gemma-3-1b and 7.0e-3 on
// Qwen3-MoE, whose routing turns a near tie the other way; the engine's
// earlier integer-folding GEMM read 8.5e-3 on Llama-3.2-1B. A violation --
// a row reading another session's history, or its own at the wrong
// positions -- must read above it.
const hostStepNMSE = 1e-2

// floatFormsNMSE bounds the composed CPU's float forms (withCPUFloat) over a
// prompt's or a step's rows against their forms over one row: the same
// arithmetic folded in another order, with no int8 activations to round
// apart. Measured on box 41 decoding two tokens and prefilling three
// against decoding five: 1.9e-7 on Qwen2-VL, 1.3e-7 on Qwen3-MoE, 4.5e-9
// on TinyLlama. A row at a wrong position or reading another's history
// reads orders of magnitude above it.
const floatFormsNMSE = 1e-5

// TestStepRunsOnTheCPUSharesOnePass is the gate on StepRuns' host arm
// (stepHost): two sessions decode while a third session's prompt rides beside
// them in chunks of three tokens, then all three decode together, every
// Instance on the host. Teacher-forced against each session alone (Prefill and
// Forward): every decoding row's logits, the late session's at the end of its
// prompt, and its decoding after.
//
// It selects what it gates -- every row ran in a step across sessions
// (Model.HostStepRows) -- and runs two violations: the late prompt's chunks
// laid out at mirrored positions, and every row given the first session's
// history (stepShare). Each must part from the sessions alone by more than
// the bound.
//
// A dense model, a hybrid (recurrent state per row), a mixture and a sliding
// window model.
func TestStepRunsOnTheCPUSharesOnePass(t *testing.T) {
	for _, c := range []struct{ name, file string }{
		{"dense", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
		{"hybrid", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"},
		{"moe", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"},
		{"swa", "gemma-3-1b-it-Q4_K_M.gguf"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := testmodels.Path(c.file)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			m, err := Open(jlmOf(t, p), noTune(), WithKVF16(false), WithJITOptions(nn.WithGEMMExact(true)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if c.name == "hybrid" && !m.Cfg.Hybrid() {
				t.Fatalf("%s is not a hybrid", c.file)
			}
			if c.name == "moe" && !m.Cfg.MoE() {
				t.Fatalf("%s is not a mixture", c.file)
			}
			plan := runsPlan{early: []string{"The capital of France is", "Water boils at a temperature of"},
				late:  "The clerk counted barrels of salt on the upper floor, and",
				chunk: 3, steps: 4}
			want := plan.alone(t, m, nil)
			r0 := m.HostStepRows()
			got, rows := plan.together(t, m, nil)
			if n := m.HostStepRows() - r0; n != int64(rows) {
				t.Fatalf("%d rows ran in host steps across sessions, want %d: StepRuns went one instance after another", n, rows)
			}
			w, at := worstStep(got, want)
			t.Logf("%d rows, %d logit rows compared: worst NMSE %.3e (row %d) against each session alone",
				rows, len(want), w, at)
			if !(w <= hostStepNMSE) {
				t.Fatalf("a host step across sessions parts from each session alone: NMSE %.3e at row %d (bound %.0e)", w, at, hostStepNMSE)
			}

			stepPos = func(base, j, k int) int { return base + k - 1 - j }
			bad, _ := plan.together(t, m, nil)
			stepPos = nil
			if bw, bat := worstStep(bad, want); bw < hostStepNMSE {
				t.Fatalf("chunks at mirrored positions still matched (NMSE %.3e): this gate cannot see positions", bw)
			} else {
				t.Logf("violation: mirrored chunks read NMSE %.3e at row %d", bw, bat)
			}

			stepShare = true
			bad, _ = plan.together(t, m, nil)
			stepShare = false
			if bw, bat := worstStep(bad, want); bw < hostStepNMSE {
				t.Fatalf("rows sharing one session's history still matched (NMSE %.3e): this gate cannot see a row's state", bw)
			} else {
				t.Logf("violation: rows sharing the first session's history read NMSE %.3e at row %d", bw, bat)
			}
		})
	}
}

// hostStepAllocs is TestDecodeDoesNotAllocate's arm for StepRuns' host step
// across sessions: three host instances step together, warm, at a fixed shape --
// one token each, or with chunk the third taking two prompt rows a step with
// no logits -- and the window must make no engine heap allocation. It checks
// the window really ran as host steps (Model.HostStepRows), and skips naming
// why on an architecture the host step refuses.
func hostStepAllocs(t *testing.T, m *Model, chunk bool) {
	prompts := []string{"The capital of France is", "Water boils at a temperature of",
		"The clerk counted barrels of salt on the upper floor, and"}
	warm, n, k := 128, 64, 1
	if chunk {
		// The chunked instance moves two positions a step: a windowed layer's
		// ring of pages fills only past its window, and before that each
		// page is a commit, growth paid once and not per step. 128 warm
		// steps left eight such commits in the window on the windowed
		// synths; 192 reach the ring.
		warm, n, k = 192, 32, 2
	}
	var runs []Run
	toks := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids := m.Vocab.Encode(pr, true)
		st := m.NewInstance(len(ids) + k*(warm+n) + 8)
		defer st.Close()
		if err := st.HostRefusal(); err != nil {
			t.Skipf("the host step refuses this model (%v): StepRuns runs its instances one after another, "+
				"which the host decode arm measured", err)
		}
		if _, err := st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		w, logits := 1, true
		if chunk && i == 2 {
			w, logits = k, false
		}
		toks[i] = make([]int32, w)
		runs = append(runs, Run{Instance: st, Tokens: toks[i], Logits: logits})
	}
	step := func(at int) {
		for i, tk := range toks {
			for j := range tk {
				tk[j] = int32(3 + (at+i+j)%64)
			}
		}
		if _, err := StepRuns(runs); err != nil {
			t.Fatal(err)
		}
	}
	for s := range warm {
		step(s)
	}
	if err := prefaultExperts(m); err != nil {
		t.Fatal(err)
	}
	rows := 0
	for _, r := range runs {
		rows += len(r.Tokens)
	}
	h0, r0 := m.HostStepRows(), m.container.Reads()
	w := countAllocs(func() {
		for s := range n {
			step(warm + s)
		}
	})
	if got := m.HostStepRows() - h0; got != int64(n*rows) {
		t.Fatalf("%d row(s) ran as host steps across sessions, want %d: the arm did not run the step it names",
			got, n*rows)
	}
	allocVerdict(t, fmt.Sprintf("host step, %d run(s) of %d row(s) a step", len(runs), rows), w, n,
		m.container.Reads()-r0, 0)
}
