//go:build jitllmtest

package board

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// hostCode is the one kernel the host plan forms for parts of kind k.
func hostCode(t *testing.T, k jit.PartKind) *jit.Code {
	t.Helper()
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return host })
	if plan == nil {
		t.Fatal(err)
	}
	for _, pp := range plan.Parts {
		if pp.Part.Kind == k && len(pp.Codes) == 1 && pp.Codes[0].CPU != nil {
			return &pp.Codes[0]
		}
	}
	t.Fatalf("no host kernel for part kind %d in the plan", k)
	return nil
}

// TestComposedRopeTableRuns runs the host's composed rotary table on the
// planes the engine folds, held to main's host kernel on the same planes:
// the same arithmetic gives the same bits.
func TestComposedRopeTableRuns(t *testing.T) {
	worst, bad := ropeTableAgainstMain(t, hostCode(t, jit.PartTable))
	if bad != "" {
		t.Error(bad)
	}
	t.Logf("worst %d ulp from main's host table", worst)
}

// ropeTableAgainstMain runs a composed host rotary table against main's on
// the same planes: the worst distance in ulp, and the first word past one.
func ropeTableAgainstMain(t *testing.T, code *jit.Code) (int, string) {
	c, err := cpu.Map(code.CPU)
	if err != nil {
		t.Fatal(err)
	}
	npairs := 8
	r := nn.Rope{NRot: 2 * npairs, Base: 10000}
	planes := r.TabPlanes(npairs)
	var konst []uint32
	for _, v := range kernels.RopeTabConsts() {
		konst = append(konst, math.Float32bits(v))
	}
	worst, bad := 0, ""
	for _, pos := range []int{0, 1, 7, 127, 128, 129, 1000, 4097, 16383, 100000, kernels.RopeTabMaxPos - 1} {
		want := make([]float32, 2*npairs)
		r.Table(want, pos)
		got := make([]float32, 2*npairs+8)
		for i := range got {
			got[i] = float32(math.NaN())
		}
		rows := []uint32{1, uint32(pos)}
		block := make([]uint64, 64)
		for _, a := range code.Args {
			var p unsafe.Pointer
			switch {
			case a.Value || a.Moving:
				t.Fatalf("argument word %d is a value the test does not compute", a.Field)
			case a.Words != nil:
				p = unsafe.Pointer(&a.Words[a.Offset])
			case a.Operand.Name == "rope.pos":
				p = unsafe.Pointer(&rows[a.Offset])
			case a.Operand.Name == "rope.planes":
				p = unsafe.Pointer(&planes[a.Offset])
			case a.Operand.Name == "rope.const":
				p = unsafe.Pointer(&konst[a.Offset])
			case a.Operand.Name == "rope.cs":
				p = unsafe.Pointer(&got[a.Offset])
			default:
				t.Fatalf("argument word %d binds %q", a.Field, a.Operand.Name)
			}
			block[a.Field/8] = uint64(uintptr(p))
		}
		c.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
		runtime.KeepAlive(rows)
		runtime.KeepAlive(planes)
		runtime.KeepAlive(konst)
		runtime.KeepAlive(code)
		for i := range want {
			d := int(math.Float32bits(got[i])) - int(math.Float32bits(want[i]))
			if d < 0 {
				d = -d
			}
			if (math.IsNaN(float64(got[i])) || d > 1) && bad == "" {
				bad = fmt.Sprintf("position %d word %d: %v, main's %v", pos, i, got[i], want[i])
			}
			worst = max(worst, d)
		}
		for i := 2 * npairs; i < len(got); i++ {
			if !math.IsNaN(float64(got[i])) && bad == "" {
				bad = fmt.Sprintf("position %d: word %d past the table written", pos, i)
			}
		}
	}
	return worst, bad
}

// TestComposedSamplingRuns runs the host's composed sampling kernel on
// rows of every tail length and against ties, held to the greedy rule: the
// largest logit, the first of equal ones.
func TestComposedSamplingRuns(t *testing.T) {
	code := hostCode(t, jit.PartSample)
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
				case a.Value || a.Moving:
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
