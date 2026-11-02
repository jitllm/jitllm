//go:build jitllmbench && jitllmtest && amd64 && linux

package model

import (
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestRopeTableDose prices the generated rotary table against the float64 Go
// loop it replaced, alternated in one process. It reports the op, where the
// effect is large, and the whole token, where it is expected to be sub-noise,
// each with its dispersion. nn.RopeTableCalls/RopeGoCalls assert which path
// each arm took. Run the A/A control first (both arms the same option).
func TestRopeTableDose(t *testing.T) {
	if testing.Short() {
		t.Skip("opens a model")
	}
	rounds := envInt("JITLLM_RT_ROUNDS", 15)
	goA, goB := envInt("JITLLM_RT_A", 0), envInt("JITLLM_RT_B", 1)

	// ---- half one: the OP, which is the thing that changed ----
	//
	// Llama-3.2-1B's rotary geometry. A table is NRot floats and a decode token
	// builds one (two on an architecture with a second, local rope), so this is
	// the whole of what the kernel replaced.
	r := nn.Rope{NRot: 64, Base: 500000}
	cs := make([]float32, r.NRot)
	const iters = 20000
	// The dose is a JIT setting, so the op arm needs its own JIT.
	opJIT := nn.NewJIT(64, 64, nil, nn.WithTune(nn.TuneOff), nn.WithQuietTuner(true))
	if opJIT == nil {
		t.Skip("no generated tier")
	}
	defer opJIT.Close()
	tableArm := func(goLoop bool) time.Duration {
		opJIT.SetRopeGo(goLoop)
		defer opJIT.SetRopeGo(false)
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			opJIT.RopeTable(r, cs, i&0xFFFF)
		}
		return time.Since(t0)
	}
	// Warm both, then alternate.
	tableArm(false)
	tableArm(true)
	var opRatios []float64
	var opA, opB time.Duration
	for i := 0; i < rounds; i++ {
		var da, db time.Duration
		if i%2 == 0 {
			da, db = tableArm(goA != 0), tableArm(goB != 0)
		} else {
			db, da = tableArm(goB != 0), tableArm(goA != 0)
		}
		opA, opB = opA+da, opB+db
		opRatios = append(opRatios, float64(db)/float64(da))
	}
	sort.Float64s(opRatios)
	om := opRatios[len(opRatios)/2]
	oq1, oq3 := opRatios[len(opRatios)/4], opRatios[(3*len(opRatios))/4]
	t.Logf("the op: arm A %.1f ns/table, arm B %.1f ns/table",
		float64(opA.Nanoseconds())/float64(rounds*iters),
		float64(opB.Nanoseconds())/float64(rounds*iters))
	t.Logf("the op, B/A per-round ratio: median %.4f  IQR/median %.3f  n=%d  (>1 means A is faster)",
		om, (oq3-oq1)/om, len(opRatios))

	// ---- half two: the TOKEN ----
	path := benchModel(t)
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	prompt := envInt("JITLLM_RT_PROMPT", 16)
	decode := envInt("JITLLM_RT_DECODE", 24)
	ids := make([]int32, prompt)
	for i := range ids {
		ids[i] = int32(1 + i%64)
	}
	// One instance with the flag toggled per round, so the two arms share one
	// JIT, one KV cache and one warm-up.
	m.jit = []nn.Option{nn.WithTune(nn.TuneOff), nn.WithQuietTuner(true)}
	st := m.NewInstance(prompt + decode + 8)
	defer st.Close()

	step := func(goLoop bool) {
		st.jit.SetRopeGo(goLoop)
		defer st.jit.SetRopeGo(false)
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
	// Warm: codegen, pack widths and first-touch faults are not per-token
	// costs.
	step(false)
	step(true)

	// The selection check, around one round of each arm, before the timed
	// comparison: the counters are process-wide.
	census := func(goLoop bool) (kern, goCount int64) {
		k0, g0 := nn.RopeTableCalls(), nn.RopeGoCalls()
		step(goLoop)
		return nn.RopeTableCalls() - k0, nn.RopeGoCalls() - g0
	}
	ka, ga := census(goA != 0)
	kb, gb := census(goB != 0)
	t.Logf("arm A (go=%d): %d kernel table(s), %d Go table(s)", goA, ka, ga)
	t.Logf("arm B (go=%d): %d kernel table(s), %d Go table(s)", goB, kb, gb)
	if goA == 0 && (ka == 0 || ga != 0) {
		t.Fatal("arm A was built with the kernel and did not take it")
	}
	if goB != 0 && (gb == 0 || kb != 0) {
		t.Fatal("arm B was built for the Go loop and did not take it")
	}
	if goA == goB {
		t.Logf("both arms identical -- this is the A/A self-control")
	}

	var ratios []float64
	var wa, wb time.Duration
	for i := 0; i < rounds; i++ {
		var la, lb time.Duration
		run := func(goLoop bool) time.Duration {
			t0 := time.Now()
			step(goLoop)
			return time.Since(t0)
		}
		if i%2 == 0 {
			la, lb = run(goA != 0), run(goB != 0)
		} else {
			lb, la = run(goB != 0), run(goA != 0)
		}
		wa, wb = wa+la, wb+lb
		ratios = append(ratios, float64(lb)/float64(la))
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	q1, q3 := ratios[len(ratios)/4], ratios[(3*len(ratios))/4]
	tok := float64(rounds * (prompt + decode))
	t.Logf("the token: arm A %.2f tok/s, arm B %.2f tok/s", tok/wa.Seconds(), tok/wb.Seconds())
	t.Logf("the token, B/A per-round ratio: median %.4f  IQR/median %.3f  n=%d",
		med, (q3-q1)/med, len(ratios))
}
