//go:build jitllmbench && amd64 && linux

package model

import (
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestQuantActDose prices the generated activation quantizer against the Go
// loop it replaces, alternated in one process on the decode path that ships.
// cpu.QuantActStats asserts which path each arm took. Run the A/A control
// first (both arms the same option).
func TestQuantActDose(t *testing.T) {
	if testing.Short() {
		t.Skip("opens a model")
	}
	path := benchModel(t)
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	prompt := envInt("JITLLM_QA_PROMPT", 16)
	decode := envInt("JITLLM_QA_DECODE", 24)
	rounds := envInt("JITLLM_QA_ROUNDS", 15)
	// Setting both to 1 is the A/A control: both arms then run the Go loop.
	// Setting both to 0 is the other control, both on the kernel.
	goA, goB := envInt("JITLLM_QA_A", 0), envInt("JITLLM_QA_B", 1)

	ids := make([]int32, prompt)
	for i := range ids {
		ids[i] = int32(1 + i%64)
	}

	mk := func(goLoop bool) *State {
		m.jit = []nn.Option{nn.WithTune(nn.TuneOff), nn.WithQuietTuner(true),
			nn.WithProfile(true), nn.WithQuantActGo(goLoop)}
		st := m.NewState(prompt + decode + 8)
		// Warm: kernels are mapped and pack widths settle on the first pass.
		st.Reset()
		logits, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			if logits, err = st.Forward(Greedy(logits)); err != nil {
				t.Fatal(err)
			}
		}
		return st
	}

	a, b := mk(goA != 0), mk(goB != 0)
	defer a.Close()
	defer b.Close()

	// One round: a fixed prompt and decode steps, from position zero every
	// time so every round attends over the same history.
	step := func(st *State) func(int) {
		return func(n int) {
			for i := 0; i < n; i++ {
				st.Reset()
				logits, err := st.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				for j := 0; j < decode; j++ {
					if logits, err = st.Forward(Greedy(logits)); err != nil {
						t.Fatal(err)
					}
				}
			}
		}
	}

	// The selection check, around one round of each arm, before the timed
	// comparison: the counters are process-wide.
	census := func(st *State) (wide, narrow, goLoop int64) {
		cpu.ResetQuantActStats()
		step(st)(1)
		return cpu.QuantActStats()
	}
	wa, na, ga := census(a)
	wb, nb, gb := census(b)
	t.Logf("arm A (go=%d): wide %d, narrow %d, go %d", goA, wa, na, ga)
	t.Logf("arm B (go=%d): wide %d, narrow %d, go %d", goB, wb, nb, gb)
	kern := func(w, n int64) bool { return w+n > 0 }
	if goA == 0 && !kern(wa, na) {
		t.Fatal("arm A was built with the kernels and ran none -- the dose would be " +
			"the Go loop against itself")
	}
	if goB != 0 && (kern(wb, nb) || gb == 0) {
		t.Fatal("arm B was built for the Go loop and did not take it")
	}
	if goA == goB {
		t.Logf("both arms identical -- this is the A/A self-control")
	}

	// The statistic is the quantize phase (nn.ProfileNanos), the only thing
	// that differs; the whole-token wall is too noisy to resolve it and is
	// reported but not the claim.
	phase := func(st *State) (quant, wall time.Duration) {
		nn.ResetProfile()
		t0 := time.Now()
		step(st)(1)
		wall = time.Since(t0)
		q, _, _ := nn.ProfileNanos()
		return time.Duration(q), wall
	}
	var ratios []float64
	var qa, qb, wa2, wb2 time.Duration
	for i := 0; i < rounds; i++ {
		// ABBA so a drifting box cancels.
		var pa, pb, la, lb time.Duration
		if i%2 == 0 {
			pa, la = phase(a)
			pb, lb = phase(b)
		} else {
			pb, lb = phase(b)
			pa, la = phase(a)
		}
		qa, qb, wa2, wb2 = qa+pa, qb+pb, wa2+la, wb2+lb
		if pa > 0 {
			ratios = append(ratios, float64(pb)/float64(pa))
		}
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	q1, q3 := ratios[len(ratios)/4], ratios[(3*len(ratios))/4]
	t.Logf("quantize phase: %s %.2f ms/round, %s %.2f ms/round",
		armName(goA), float64(qa.Milliseconds())/float64(rounds),
		armName(goB), float64(qb.Milliseconds())/float64(rounds))
	t.Logf("B/A per-round ratio: median %.4f  IQR/median %.3f  n=%d  (>1 means A is faster)",
		med, (q3-q1)/med, len(ratios))
	tok := float64(rounds * (prompt + decode))
	t.Logf("whole token, NOT the claim: %s %.2f tok/s   %s %.2f tok/s",
		armName(goA), tok/wa2.Seconds(), armName(goB), tok/wb2.Seconds())
}

func armName(goLoop int) string {
	if goLoop != 0 {
		return "quantize/go"
	}
	return "quantize/kernel"
}
