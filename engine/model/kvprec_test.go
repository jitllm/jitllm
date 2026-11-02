//go:build jitllmbench

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// JITLLM_* variables select its parameters.

package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestKVPrecisionCost measures what an f32 KV cache buys over an f16 one. A
// cached K already carries quantization error of the same order as binary16's
// mantissa, but a softmax can amplify, so the output effect is measured: one
// fixed token stream teacher-forced through both at several depths, reporting
// max|dlogit| and argmax agreement. A flip on a margin narrower than the
// perturbation is the model being undecided, not a defect.
func TestKVPrecisionCost(t *testing.T) {
	// Slow (two teacher-forced runs to depth per row). It asserts only the A/A
	// control and otherwise logs a table to read.
	pick := os.Getenv("JITLLM_KVPREC_MODEL")
	paths := modelFiles()
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil || fi.Size() > 2<<30 {
			continue
		}
		name := filepath.Base(p)
		if pick != "" && name != pick {
			continue
		}
		m, err := Open(jlmOf(t, p))
		if err != nil {
			continue
		}
		c := m.Cfg
		for _, depth := range []int{0, 512} {
			// The model's trained context bounds the sweep.
			if depth+40 > c.NCtx {
				continue
			}
			run := func(f16 bool) []float32 {
				m.SetKVF16(f16)
				defer m.SetKVF16(false)
				s := m.NewInstance(depth + 40)
				defer s.Close()
				// A fixed stream, teacher-forced: both arms see identical
				// inputs, so the only difference is the cache's precision.
				var out []float32
				for i := 0; i < depth+16; i++ {
					id := int32((i*2654435761)%(c.NVocab-1)) + 1
					if out, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				return append([]float32(nil), out...)
			}
			// The A/A control first: two f32 runs must be bit-identical, or
			// the numbers below measure the harness.
			if depth == 0 {
				c1, c2 := run(false), run(false)
				var cd float64
				for i := range c1 {
					cd = math.Max(cd, math.Abs(float64(c1[i]-c2[i])))
				}
				if cd != 0 {
					t.Errorf("%s: the f32 arm disagrees with ITSELF by %.3e", name, cd)
				}
			}
			a, b := run(false), run(true)
			_ = b
			var d float64
			for i := range a {
				d = math.Max(d, math.Abs(float64(a[i]-b[i])))
			}
			// argmax and its margin on each side
			am, bm := 0, 0
			for i := range a {
				if a[i] > a[am] {
					am = i
				}
				if b[i] > b[bm] {
					bm = i
				}
			}
			var a2, b2 float64 = math.Inf(-1), math.Inf(-1)
			for i := range a {
				if i != am && float64(a[i]) > a2 {
					a2 = float64(a[i])
				}
				if i != bm && float64(b[i]) > b2 {
					b2 = float64(b[i])
				}
			}
			flag := "same id"
			if am != bm {
				flag = "ID FLIP"
			}
			t.Logf("%-40s depth %4d  max|dlogit| %.4e  margin f32 %.4f f16 %.4f  %s",
				name, depth, d, float64(a[am])-a2, float64(b[bm])-b2, flag)
		}
		m.Close()
	}
}
