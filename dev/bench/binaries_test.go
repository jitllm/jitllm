package bench

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

var decodeRate = regexp.MustCompile(`decode [0-9]* tok in [^(]*\(([0-9.]*) tok/s\)`)

// TestBinariesAB is a decode row between two jitllm builds, each a process
// of `jitllm run` a round: ABBA, a soak first, the self-control (A against
// A) before the row, and the median of per-round rate ratios with its IQR,
// the whole comparison twice. It is the shape scripts/vs-llamacpp.sh gives a
// row, as a binary a host that holds no source runs.
//
//	JITLLM_AB_A, JITLLM_AB_B          the two jitllm binaries (required)
//	JITLLM_AB_MODEL                   the container (required)
//	JITLLM_AB_A_ARGS, JITLLM_AB_B_ARGS each arm's run flags (default -devices gpu)
//	JITLLM_AB_A_ENV, JITLLM_AB_B_ENV  each arm's NAME=VALUE environment
//	JITLLM_AB_N                       tokens a run decodes (default 256)
//	JITLLM_AB_ROUNDS                  ABBA rounds (default 10)
//
// A ratio above 1 is B faster.
func TestBinariesAB(t *testing.T) {
	a, b, modelPath := os.Getenv("JITLLM_AB_A"), os.Getenv("JITLLM_AB_B"), os.Getenv("JITLLM_AB_MODEL")
	if a == "" || b == "" || modelPath == "" {
		t.Skip("set JITLLM_AB_A, JITLLM_AB_B and JITLLM_AB_MODEL")
	}
	n := envInt("JITLLM_AB_N", 256)
	rounds := envInt("JITLLM_AB_ROUNDS", 10)
	arm := func(bin, args, env string) func() float64 {
		if args == "" {
			args = "-devices gpu"
		}
		return func() float64 {
			cmd := exec.Command(bin, append(append(append([]string{"run"}, strings.Fields(args)...), "-n", strconv.Itoa(n), modelPath),
				"Once upon a time in a distant land there")...)
			cmd.Env = append(os.Environ(), strings.Fields(env)...)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("%s: %v\n%s", bin, err, out)
			}
			m := decodeRate.FindSubmatch(out)
			if m == nil {
				t.Fatalf("%s: no decode rate in its output:\n%s", bin, out)
			}
			r, _ := strconv.ParseFloat(string(m[1]), 64)
			return r
		}
	}
	runA := arm(a, os.Getenv("JITLLM_AB_A_ARGS"), os.Getenv("JITLLM_AB_A_ENV"))
	runB := arm(b, os.Getenv("JITLLM_AB_B_ARGS"), os.Getenv("JITLLM_AB_B_ENV"))
	for range 2 {
		runA()
		runB()
	}
	row := func(name string, x, y func() float64) (med, spread float64) {
		var ratios, ra, rb []float64
		for range rounds {
			a1 := x()
			b1 := y()
			b2 := y()
			a2 := x()
			ratios = append(ratios, (b1+b2)/(a1+a2))
			ra, rb = append(ra, a1, a2), append(rb, b1, b2)
		}
		sort.Float64s(ratios)
		sort.Float64s(ra)
		sort.Float64s(rb)
		med, spread = medianF(ratios), iqr(ratios)
		t.Logf("%s: %.4fx (IQR/median %.1f%%, %d ABBA rounds); A %.1f tok/s, B %.1f tok/s", name, med, 100*spread/med,
			rounds, medianF(ra), medianF(rb))
		return med, spread
	}
	med, spread := row("A/A self-control", runA, runA)
	if spread/med > 0.10 || med < 0.97 || med > 1.03 {
		t.Fatalf("the self-control reads %.4fx at IQR/median %.1f%%: this host cannot quote a row now", med, 100*spread/med)
	}
	for pass := 1; pass <= 2; pass++ {
		if m, s := row(fmt.Sprintf("B against A, pass %d", pass), runA, runB); s/m > 0.10 {
			t.Errorf("pass %d: IQR/median %.1f%%: do not quote it", pass, 100*s/m)
		}
	}
}
