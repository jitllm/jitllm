//go:build amd64 || arm64

package model

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The speculation gates. Greedy decoding through a Speculator must be greedy
// decoding: token for token what plain decode emits, over hundreds of tokens
// and several prompts, with the drafts actually drafted and actually accepted
// (the acceptance counter is the selection check: a speculator that never
// drafts passes an equality gate trivially).
//
// Exact mode holds the batched verification to the decode path bit for bit:
// with the GEMM's float epilogue every batched matmul sums in the matvec's
// order (TestBatchMatchesForward), so a verified row's logits are decode's and
// any token difference is a defect. The shipped GEMM may reassociate, and is
// judged as the batch gates judge it: a disagreement must sit on a margin the
// perturbation could flip.

// specModels are the models with a prediction block: the real hybrid
// (Qwen3.5-0.8B, linear blocks with a recurrent state to roll back), a
// hybrid fixture of the same architecture and one of its mixture
// (qwen35moe, whose prediction block routes experts; F32, the same weights
// TestMTPMatchesReference holds to transformers), and a dense-attention one
// (DeepSeek-V3's MLA, whose rollback is positions alone). JITLLM_SPEC_MODEL
// names more, comma-separated, under $JITLLM_MODELS; JITLLM_SPEC_ONLY=1 runs
// those alone.
func specModels(t *testing.T) map[string]string {
	ms := map[string]string{
		"qwen35-hybrid":  "qwen35/Qwen3.5-0.8B-MTP-Q8_0.gguf",
		"synth-hybrid":   "synth-qwen35-hybrid-mtp.gguf",
		"synth-moe":      "synth-qwen35moe-hybrid-mtp.gguf",
		"synth-deepseek": "synth-deepseek-mtp",
		"synth-q4e":      "synth-qwen4exp-mtp.gguf",
	}
	if v := os.Getenv("JITLLM_SPEC_MODEL"); v != "" {
		if os.Getenv("JITLLM_SPEC_ONLY") == "1" {
			ms = map[string]string{}
		}
		for _, p := range strings.Split(v, ",") {
			ms[strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))] = p
		}
	}
	return ms
}

func openSpecModel(t *testing.T, name string, opts ...Option) *Model {
	t.Helper()
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory; scripts/mtpgold.py builds "+
			"the fixtures and the real model's GGUF) -- this gate proved nothing", err)
	}
	switch {
	case strings.HasSuffix(p, ".gguf"):
		p = jlmOf(t, p)
	case !strings.HasSuffix(p, ".jlm"):
		p = hfContainer(t, name)
	}
	m, err := Open(p, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if m.Cfg.NMTP == 0 {
		m.Close()
		t.Fatalf("%s carries no prediction block: the gate would test nothing it exists for", name)
	}
	return m
}

var specPrompts = []string{
	"The capital of France is",
	"def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
	"Here is a list of the planets in order from the sun: Mercury, Venus,",
	"Once upon a time, in a small village by the sea, there lived",
}

// plainGreedy is n tokens of greedy decode after prompt.
func plainGreedy(t *testing.T, m *Model, prompt []int32, n int, dev nn.Device) []int32 {
	t.Helper()
	st := m.NewState(len(prompt) + n + 8)
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
	out := make([]int32, 0, n)
	for len(out) < n {
		y := Greedy(lg)
		out = append(out, y)
		if lg, err = st.Forward(y); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// specGreedy is n tokens through a Speculator.
func specGreedy(t *testing.T, m *Model, prompt []int32, n int, dev nn.Device, fault specFault,
	oracle []int32, opts ...SpecOption) ([]int32, SpecStats) {
	t.Helper()
	// A break that keeps rejected rows moves the trunk on by every drafted
	// row while emitting one: room for a round's worth of rows a token.
	st := m.NewState(len(prompt) + 5*n + 8)
	defer st.Close()
	if dev != nil {
		if err := st.SetDevice(dev); err != nil {
			t.Fatal(err)
		}
	}
	sp, err := st.Speculate(opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Close()
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

// oracleFor is the drafts a gate forces on a model whose prediction block
// cannot be expected to agree with its trunk: the synthetic fixtures, whose
// weights are random. A real model drafts for itself (nil).
func oracleFor(name string, want []int32) []int32 {
	if strings.HasPrefix(name, "synth") {
		return want
	}
	return nil
}

// withSpecSchedule fixes the draft count round by round (specOpts.sched).
func withSpecSchedule(k []int) SpecOption { return func(o *specOpts) { o.sched = k } }

// specSchedule is a gate's round-by-round draft counts: runs of no draft long
// enough to owe the prediction block several rows, each followed by a round
// that drafts and so must catch it up first.
var specSchedule = []int{0, 0, 0, 2, 0, 3, 1, 0, 0}

// oracleCorrupt is every how-many-th forced draft is wrong: often enough
// that most rounds of three drafts reject somewhere, rarely enough that most
// accept some.
const oracleCorrupt = 5

// firstDiff is the first index where a and b differ, or -1.
func firstDiff(a, b []int32) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

// faultTokens is how many tokens a break's run decodes: n, or fewer where a
// break that keeps rejected rows -- up to five positions a token (specGreedy)
// -- would run past the model's context. The break must part from plain
// decode by divergence; running the cache out is a different failure, and
// one the gate does not exist to see.
func faultTokens(m *Model, prompt []int32, n int) int {
	if c := m.Cfg.NCtx; c > 0 {
		n = min(n, (c-len(prompt)-8)/5)
	}
	return n
}

// specTokens is how many tokens a gate decodes per prompt: a few hundred, so
// a rollback bug has rounds enough to surface, unless JITLLM_SPEC_TOKENS asks
// for another count.
func specTokens() int {
	if v := os.Getenv("JITLLM_SPEC_TOKENS"); v != "" {
		var n int
		if _, err := fmt.Sscan(v, &n); err == nil && n > 0 {
			return n
		}
	}
	return 256
}

// TestSpecGreedyMatchesDecode is the contract on the host: exact arithmetic,
// every rollback mode, a fixed draft count and the adaptive one.
func TestSpecGreedyMatchesDecode(t *testing.T) {
	n := specTokens()
	for name, path := range specModels(t) {
		t.Run(name, func(t *testing.T) {
			m := openSpecModel(t, path, noTune, WithJITOptions(nn.WithGEMMExact(true)))
			defer m.Close()
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			arms := []struct {
				name string
				opts []SpecOption
			}{
				{"k3", []SpecOption{WithSpecDraft(3)}},
				{"adaptive", nil},
				{"schedule", []SpecOption{withSpecSchedule(specSchedule)}},
			}
			if m.Cfg.Hybrid() {
				arms = append(arms,
					struct {
						name string
						opts []SpecOption
					}{"k3-replay", []SpecOption{WithSpecDraft(3), WithSpecRollback(SpecRollbackReplay)}})
			}
			var drafted, accepted int64
			for _, text := range specPrompts {
				prompt := m.Vocab.Encode(text, true)
				want := plainGreedy(t, m, prompt, n, nil)
				for _, arm := range arms {
					got, st := specGreedy(t, m, prompt, n, nil, specFaultNone, oracleFor(name, want), arm.opts...)
					if i := firstDiff(got, want); i >= 0 {
						t.Fatalf("%s, prompt %q: token %d is %d, plain greedy says %d (%d drafted, %d accepted, "+
							"rollback %v)", arm.name, text, i, got[i], want[i], st.Drafted, st.Accepted, st.Rollback)
					}
					drafted += st.Drafted
					accepted += st.Accepted
					t.Logf("%s %-10s %q: %d rounds, %d drafted, %d accepted, %.2f tokens a pass, rollback %v "+
						"(%d restored, %d replayed rows)", name, arm.name, text[:min(len(text), 24)], st.Rounds,
						st.Drafted, st.Accepted, st.TokensPerRound(), st.Rollback, st.Restored, st.Replayed)
				}
			}
			// The selection check: an equality over rounds that never drafted,
			// or never kept a draft, proves nothing about verification.
			if drafted == 0 || accepted == 0 {
				t.Fatalf("%d drafted and %d accepted: the speculation never ran", drafted, accepted)
			}
		})
	}
}

// TestSpecGateDiscriminates runs the equality gate against the three breaks it
// exists to catch, and demands each one break equality. A gate that has never
// fired is not a gate (RULE 10).
func TestSpecGateDiscriminates(t *testing.T) {
	n := specTokens()
	for name, path := range specModels(t) {
		t.Run(name, func(t *testing.T) {
			m := openSpecModel(t, path, noTune, WithJITOptions(nn.WithGEMMExact(true)))
			defer m.Close()
			faults := []struct {
				name  string
				fault specFault
				opts  []SpecOption
			}{
				{"accept-unchecked", faultAcceptUnchecked, nil},
				{"keep-rejected-kv", faultKeepRejectedKV, nil},
			}
			if m.Cfg.Hybrid() {
				faults = append(faults,
					struct {
						name  string
						fault specFault
						opts  []SpecOption
					}{"keep-rejected-rec-rows", faultKeepRejectedRec, []SpecOption{WithSpecRollback(SpecRollbackRows)}},
					struct {
						name  string
						fault specFault
						opts  []SpecOption
					}{"keep-rejected-rec-replay", faultKeepRejectedRec, []SpecOption{WithSpecRollback(SpecRollbackReplay)}})
			}
			for _, f := range faults {
				caught := false
				for _, text := range specPrompts {
					prompt := m.Vocab.Encode(text, true)
					n := faultTokens(m, prompt, n)
					want := plainGreedy(t, m, prompt, n, nil)
					got, st := specGreedy(t, m, prompt, n, nil, f.fault, oracleFor(name, want),
						append([]SpecOption{WithSpecDraft(3)}, f.opts...)...)
					if i := firstDiff(got, want); i >= 0 {
						t.Logf("%s: %q diverges at token %d (%d drafted, %d accepted) -- caught", f.name,
							text[:min(len(text), 24)], i, st.Drafted, st.Accepted)
						caught = true
						break
					}
				}
				if !caught {
					t.Errorf("%s: every prompt still matched plain greedy -- the gate cannot see this break", f.name)
				}
			}
		})
	}
}
