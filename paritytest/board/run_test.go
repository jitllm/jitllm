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
	"github.com/jitllm/jitllm/format/quant"
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
			case a.Word && a.Operand.Name == "rope.pos":
				block[a.Field/8] = uint64(rows[a.Offset])
				continue
			case a.Value || a.Moving || a.Word:
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

// TestComposedRoPERuns runs each host kernel of the block that turns a
// row in place by the rotary table, on head counts the call passes, held to
// the rotation of each pair: x0 c - x1 s and x0 s + x1 c.
func TestComposedRoPERuns(t *testing.T) {
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return host })
	if plan == nil {
		t.Fatal(err)
	}
	ran := 0
	for _, pp := range plan.Parts {
		for ci := range pp.Codes {
			code := &pp.Codes[ci]
			var table, row, heads *jit.CallArg
			for ai := range code.Args {
				a := &code.Args[ai]
				switch {
				case a.Operand.Name == "rope.cs" && !code.Writes(a.Operand):
					table = a
				case a.Length:
					heads = a
				case !a.Value && a.Words == nil && code.Writes(a.Operand):
					row = a
				}
			}
			if table == nil || row == nil || heads == nil || len(code.Args) != 3 {
				continue
			}
			c, err := cpu.Map(code.CPU)
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewSource(2))
			const hd = 16
			for _, n := range []int{1, 2, heads.Extent} {
				x := make([]float32, heads.Extent*hd+8)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				cs := make([]float32, hd)
				for i := range cs {
					cs[i] = float32(rng.NormFloat64())
				}
				want := append([]float32(nil), x...)
				for h := range n {
					for p := range hd / 2 {
						x0, x1 := x[h*hd+2*p], x[h*hd+2*p+1]
						co, si := cs[2*p], cs[2*p+1]
						want[h*hd+2*p] = x0*co - x1*si
						want[h*hd+2*p+1] = x0*si + x1*co
					}
				}
				block := make([]uint64, 64)
				block[row.Field/8] = uint64(uintptr(unsafe.Pointer(&x[row.Offset])))
				block[table.Field/8] = uint64(uintptr(unsafe.Pointer(&cs[table.Offset])))
				block[heads.Field/8] = uint64(n)
				c.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
				runtime.KeepAlive(x)
				runtime.KeepAlive(cs)
				for i := range want {
					if x[i] != want[i] {
						t.Errorf("%s kernel %d, %d heads: word %d %v, want %v", pp.Part, ci, n, i, x[i], want[i])
						break
					}
				}
				ran++
			}
		}
	}
	if ran == 0 {
		t.Fatal("no host rotary kernel in the plan")
	}
	t.Logf("%d rotations", ran)
}

// TestComposedSoftmaxRuns runs the host's fused softmax kernel on one head
// at offset zero, its counts the vector and tail of the row's length, held
// to the softmax in place: each exponential of the row less its maximum
// over their sum.
func TestComposedSoftmaxRuns(t *testing.T) {
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return host })
	if plan == nil {
		t.Fatal(err)
	}
	var code *jit.Code
	for _, pp := range plan.Parts {
		for ci := range pp.Codes {
			if pp.Codes[ci].Name == "reduce+map+reduce+map" && code == nil {
				code = &pp.Codes[ci]
			}
		}
	}
	if code == nil {
		t.Fatal("no fused softmax kernel in the host plan")
	}
	c, err := cpu.Map(code.CPU)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	worst := 0.0
	for _, n := range []int{1, 3, 5, 7, 8, 9, 16, 17, 100} {
		dots := make([]float32, n+8)
		for i := range dots[:n] {
			dots[i] = float32(4 * rng.NormFloat64())
		}
		row := append([]float32(nil), dots...)
		block := make([]uint64, 64)
		for _, a := range code.Args {
			var p unsafe.Pointer
			switch {
			case a.Value && a.Field == 40:
				block[5] = uint64(n >> 3)
				continue
			case a.Value && a.Field == 32:
				block[4] = uint64(n & 7)
				continue
			case a.Value:
				t.Fatalf("argument word %d is a value the test does not compute", a.Field)
			case a.Words != nil:
				p = unsafe.Pointer(&a.Words[0])
			case a.Operand.Name == "attn.dots" && code.Writes(a.Operand):
				p = unsafe.Pointer(&row[0])
			default:
				t.Fatalf("argument word %d binds %q, not the row in place", a.Field, a.Operand.Name)
			}
			block[a.Field/8] = uint64(uintptr(p))
		}
		c.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
		runtime.KeepAlive(row)
		runtime.KeepAlive(code)
		m := math.Inf(-1)
		for _, x := range dots[:n] {
			m = math.Max(m, float64(x))
		}
		sum := 0.0
		for _, x := range dots[:n] {
			sum += math.Exp(float64(x) - m)
		}
		for i := range n {
			w := math.Exp(float64(dots[i])-m) / sum
			d := math.Abs(float64(row[i])-w) / w
			if d > 1e-5 || math.IsNaN(d) {
				t.Errorf("n %d position %d: %v, want %v", n, i, row[i], w)
			}
			worst = max(worst, d)
		}
		for i := n; i < len(row); i++ {
			if row[i] != dots[i] {
				t.Errorf("n %d: position %d past the row written", n, i)
			}
		}
	}
	t.Logf("worst relative %.2g", worst)
}

// TestComposedAttnValuesRun runs the host's weighted-values kernels, one
// key head's query heads over a history of f16 value rows, held to each
// head's sum of its weights times the rows in f32.
func TestComposedAttnValuesRun(t *testing.T) {
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return host })
	if plan == nil {
		t.Fatal(err)
	}
	const dim, stride = 16, 32
	ran := 0
	for _, pp := range plan.Parts {
		for ci := range pp.Codes {
			code := &pp.Codes[ci]
			var count, values *jit.CallArg
			weights, outs := map[int]*jit.CallArg{}, map[int]*jit.CallArg{}
			for ai := range code.Args {
				a := &code.Args[ai]
				switch {
				case a.Value:
					count = a
				case a.Operand.Name == "kv.v":
					values = a
				case a.Operand.Name == "attn.dots":
					weights[a.Operand.Copy] = a
				case a.Operand.Name == "attn" && code.Writes(a.Operand):
					outs[a.Operand.Copy] = a
				}
			}
			if count == nil || values == nil || len(weights) == 0 || len(weights) != len(outs) || len(code.Args) != 2+2*len(weights) {
				continue
			}
			c, err := cpu.Map(code.CPU)
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewSource(3))
			for _, n := range []int{1, 2, 7, 40} {
				v := make([]uint16, n*stride)
				for i := range v {
					v[i] = quant.EncodeHalf(float32(rng.NormFloat64()))
				}
				w := make([][]float32, len(weights))
				o := make([][]float32, len(weights))
				block := make([]uint64, 64)
				block[count.Field/8] = uint64(n)
				block[values.Field/8] = uint64(uintptr(unsafe.Pointer(&v[values.Offset])))
				for h := range w {
					w[h] = make([]float32, n)
					for i := range w[h] {
						w[h][i] = rng.Float32()
					}
					o[h] = make([]float32, dim+8)
					for i := range o[h] {
						o[h][i] = float32(math.NaN())
					}
					block[weights[h].Field/8] = uint64(uintptr(unsafe.Pointer(&w[h][weights[h].Offset])))
					block[outs[h].Field/8] = uint64(uintptr(unsafe.Pointer(&o[h][outs[h].Offset])))
				}
				c.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
				runtime.KeepAlive(v)
				runtime.KeepAlive(w)
				runtime.KeepAlive(o)
				for h := range w {
					for d := range dim {
						var want, mag float64
						for p := range n {
							term := float64(w[h][p]) * quant.DecodeHalf(v[p*stride+d])
							want += term
							mag += math.Abs(term)
						}
						if got := float64(o[h][d]); !(math.Abs(got-want) <= 1e-6*mag) {
							t.Errorf("%s kernel %d, %d positions: head %d dim %d %v, want %v", pp.Part, ci, n, h, d, got, want)
						}
					}
					for d := dim; d < len(o[h]); d++ {
						if !math.IsNaN(float64(o[h][d])) {
							t.Errorf("%s kernel %d: head %d wrote word %d past the head", pp.Part, ci, h, d)
						}
					}
				}
				ran++
			}
		}
	}
	if ran == 0 {
		t.Fatal("no host weighted-values kernel in the plan")
	}
	t.Logf("%d histories", ran)
}
