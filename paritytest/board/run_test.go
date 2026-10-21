//go:build jitllmtest

package board

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestComposedSamplingRuns runs the host's composed sampling kernel on
// rows of every tail length and against ties, held to the greedy rule: the
// largest logit, the first of equal ones.
func TestComposedSamplingRuns(t *testing.T) {
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return host })
	if plan == nil {
		t.Fatal(err)
	}
	var code *jit.Code
	for _, pp := range plan.Parts {
		if pp.Part.Kind == jit.PartSample && len(pp.Codes) == 1 {
			code = &pp.Codes[0]
		}
	}
	if code == nil || code.CPU == nil {
		t.Fatal("no host sampling kernel in the plan")
	}
	c, err := cpu.Map(code.CPU)
	if err != nil {
		t.Fatal(err)
	}
	extent := 0
	for _, a := range code.Args {
		if a.Length {
			extent = a.Extent
		}
	}
	if extent == 0 {
		t.Fatal("the sampling kernel takes no run-time length")
	}
	rng := rand.New(rand.NewSource(1))
	ran := 0
	for _, n := range []int{1, 3, 7, 8, 9, 15, 16, 17, 64, 100, 1000, extent} {
		for trial := range 4 {
			logits := make([]float32, extent)
			for i := range n {
				logits[i] = float32(rng.NormFloat64())
			}
			switch trial {
			case 1:
				i, j := rng.Intn(n), rng.Intn(n)
				logits[j] = 9
				logits[i] = 9
			case 2:
				for i := range n {
					logits[i] = float32(math.Inf(-1))
				}
			case 3:
				logits[n-1] = 9
			}
			want, wantV := 0, logits[0]
			for i := 1; i < n; i++ {
				if logits[i] > wantV {
					want, wantV = i, logits[i]
				}
			}
			var token [8]int32
			var best [8]float32
			block := make([]uint64, 64)
			for _, a := range code.Args {
				switch {
				case a.Length:
					block[a.Field/8] = uint64(n)
				case a.Value:
					t.Fatalf("argument word %d is a value the test does not compute", a.Field)
				case a.Words != nil:
					block[a.Field/8] = uint64(uintptr(unsafe.Pointer(&a.Words[a.Offset])))
				case a.Operand.Name == "logits":
					block[a.Field/8] = uint64(uintptr(unsafe.Pointer(&logits[a.Offset])))
				case a.Operand.Name == "token":
					block[a.Field/8] = uint64(uintptr(unsafe.Pointer(&token[a.Offset])))
				case a.Operand.Name == "best":
					block[a.Field/8] = uint64(uintptr(unsafe.Pointer(&best[a.Offset])))
				default:
					t.Fatalf("argument word %d binds %q", a.Field, a.Operand.Name)
				}
			}
			c.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
			runtime.KeepAlive(logits)
			runtime.KeepAlive(code)
			if int(token[0]) != want || best[0] != wantV {
				t.Errorf("n %d trial %d: token %d best %v, want %d %v", n, trial, token[0], best[0], want, wantV)
			}
			ran++
		}
	}
	t.Logf("%d rows, the longest %d", ran, extent)
}
