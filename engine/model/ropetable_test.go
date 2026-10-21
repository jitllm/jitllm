//go:build jitllmtest

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestRopeTableAgreesWithF64 prices the GENERATED rotary table against the
// float64 Go loop it replaced, teacher-forced, on every model available. It
// is behind the jitllmtest tag because the float64 arm (engine/nn/ropego.go) exists
// only in that build:
//
//	go test -tags jitllmtest ./engine/model -run TestRopeTableAgreesWithF64
//
// It is teacher-forced so a near-tie falling the other way is not mistaken for
// the tables differing.
func TestRopeTableAgreesWithF64(t *testing.T) {
	if testing.Short() {
		t.Skip("opens every model")
	}
	var models []string
	for _, p := range languageModels(t) {
		// No toy models: their argmax moves on any perturbation.
		if n := nameOf(p); strings.Contains(n, "stories") || strings.Contains(n, "260K") {
			continue
		}
		// Under 2 GiB: this opens every one of them twice.
		if fi, err := os.Stat(p); err == nil && fi.Size() <= 2<<30 {
			models = append(models, p)
		}
	}
	if len(models) == 0 {
		t.Fatal("no model under 2 GiB -- this gate proved nothing")
	}
	ids := []int32{1, 450, 7483, 310, 3444, 338, 3681, 29889, 450, 2107, 310}

	// The arm is selected through the JIT's options: nn.NewJIT sets the dose
	// from its Config, so it is per model and nothing leaks between models.
	run := func(t *testing.T, path string, goArm bool) [][]float32 {
		t.Helper()
		m, err := Open(path, WithJITOptions(nn.WithRopeGo(goArm)))
		if err != nil {
			t.Skip(err)
		}
		defer m.Close()
		s := m.NewState(len(ids) + 4)
		defer s.Close()
		out := make([][]float32, 0, len(ids))
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}

	// Each arm's selection is checked by a counter, not by the tables
	// differing: the kernel reproduces the float64 table exactly at many
	// positions.
	{
		r := nn.Rope{Base: 500000, NRot: 64}
		// The dose is the JIT's, so the selection check needs one to ask.
		f := nn.NewJIT(64, 64, nil, nn.WithTune(nn.TuneOff), nn.WithQuietTuner(true))
		if f == nil {
			t.Skip("no generated tier")
		}
		kBefore, gBefore := nn.RopeTableCalls(), nn.RopeGoCalls()
		f.SetRopeGo(true)
		f.RopeTable(r, make([]float32, 64), 4096)
		f.SetRopeGo(false)
		f.RopeTable(r, make([]float32, 64), 4096)
		f.Close()
		if got := nn.RopeGoCalls() - gBefore; got != 1 {
			t.Fatalf("the float64 arm ran %d times for one enabled call and one disabled one -- "+
				"the knob reaches nothing", got)
		}
		if got := nn.RopeTableCalls() - kBefore; got != 1 {
			t.Fatalf("the kernel ran %d times for one enabled call and one disabled one -- "+
				"the knob reaches nothing", got)
		}
	}

	// A one-ulp table is the floor for a generated table (see
	// jit_old/cpu/ropetab.go), so the question is whether an argmax moved further
	// than this model's own logit movement w can push it: a perturbation of w
	// cannot separate a pair more than w apart. tieMargin is the floor of w.
	ran := 0
	for _, src := range models {
		t.Run(nameOf(src), func(t *testing.T) {
			path := jlmOf(t, src)
			a, b := run(t, path, true), run(t, path, false)
			var worst float64
			for p := range a {
				for i := range a[p] {
					if d := math.Abs(float64(a[p][i] - b[p][i])); d > worst {
						worst = d
					}
				}
			}
			// worst itself is bounded, or a broken table (huge movement and
			// wide flips) would pass. Ten logits is well above the
			// worst-conditioned model and below a broken table (see
			// engine/nn/ropetabviolation_test.go).
			const ceiling = 10.0
			if worst > ceiling {
				t.Errorf("max|dlogit| %.3e is past the %.0f a one-ulp table can carry -- "+
					"this is not a near-tie moving", worst, ceiling)
			}
			allow := worst
			if allow < tieMargin {
				allow = tieMargin
			}
			flips, ties := 0, 0
			for p := range a {
				// The argmax and its margin: a flip inside allow is a near-tie.
				bi, best, second := 0, float32(math.Inf(-1)), float32(math.Inf(-1))
				for i, v := range a[p] {
					if v > best {
						bi, best, second = i, v, best
					} else if v > second {
						second = v
					}
				}
				bj, bv := 0, float32(math.Inf(-1))
				for i, v := range b[p] {
					if v > bv {
						bj, bv = i, v
					}
				}
				if bi != bj {
					flips++
					if float64(best-second) <= allow {
						ties++
					} else {
						t.Errorf("pos %d: the argmax moved from %d to %d at a margin of "+
							"%.3f -- wider than the %.3f this model's own logits moved, "+
							"so the generated table did more than reorder a tie here",
							p, bi, bj, best-second, allow)
					}
				}
			}
			t.Logf("max|dlogit| %.3e over %d positions; %d argmax flips, %d of them inside "+
				"the %.3f that movement can carry", worst, len(a), flips, ties, allow)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model ran -- this gate proved nothing")
	}
}
