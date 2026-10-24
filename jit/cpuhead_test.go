package jit

import (
	bin "encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

const (
	headEmbd  = 64
	headVocab = 96
	headEps   = 1e-5
)

// headWeights is a head's stored tensors: the projection's rows, the norm's
// scale and bias, and the projection's bias.
type headWeights struct {
	w                 []byte
	norm, normB, outB []float32
	planes            [3][]uint32
	vecs              map[jlm.Role][]float32
}

func newHeadWeights(t *testing.T, rng *rand.Rand) *headWeights {
	t.Helper()
	h := &headWeights{w: make([]byte, 4*headVocab*headEmbd)}
	for i := 0; i < headVocab*headEmbd; i++ {
		bin.LittleEndian.PutUint32(h.w[4*i:], math.Float32bits(float32(rng.NormFloat64()/8)))
	}
	vec := func(n int, mean, sd float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(mean + sd*rng.NormFloat64())
		}
		return v
	}
	h.norm, h.normB, h.outB = vec(headEmbd, 1, 0.3), vec(headEmbd, 0, 0.5), vec(headVocab, 0, 1)
	qs, d, sc, err := kernels.PackWeights(kernels.Float32, h.w, headVocab, headEmbd)
	if err != nil {
		t.Fatal(err)
	}
	h.planes = [3][]uint32{qs, d, sc}
	h.vecs = map[jlm.Role][]float32{jlm.RoleOutputNorm: h.norm, jlm.RoleOutputNormBias: h.normB, jlm.RoleOutputBias: h.outB}
	return h
}

// headWant is the oracle's head over the rows of x under the head spec s: the
// norm, the projection, the bias, the softcap and the scale, the last three in
// float64.
func headWant(t *testing.T, h *headWeights, s graph.Head, x []float32, rows int) []float32 {
	t.Helper()
	out := make([]float32, rows*headVocab)
	y := make([]float32, headEmbd)
	lg := make([]float32, headVocab)
	for r := 0; r < rows; r++ {
		row := x[r*headEmbd : (r+1)*headEmbd]
		switch {
		case s.LayerNorm && s.NormBias:
			oracle.LayerNorm32(y, row, h.norm, h.normB, s.Eps)
		case s.LayerNorm:
			oracle.LayerNorm32(y, row, h.norm, nil, s.Eps)
		default:
			oracle.RMSNorm32(y, row, h.norm, s.Eps)
		}
		if err := oracle.MatVec(lg, quant.F32, h.w, y, headVocab, headEmbd); err != nil {
			t.Fatal(err)
		}
		for i, v := range lg {
			f := float64(v)
			if s.Bias {
				f += float64(h.outB[i])
			}
			if c := s.Softcap; c != 0 {
				f = c * math.Tanh(f/c)
			}
			if k := s.Scale; k != 0 && k != 1 {
				f /= k
			}
			out[r*headVocab+i] = float32(f)
		}
	}
	return out
}

// runCPUGraph runs codes in order on the CPU over the rows x of graph g's one
// input: the weight's planes, the vectors by role, and a buffer of its own for
// every other value (NaN until a kernel writes it), the scratch and constant
// blocks each where its call words say. It returns g's one output: the
// storage the last kernel writes, which the CPU's in-place pass makes the
// input's or a scratch's where the output goes over a value nothing reads
// after. The input is cloned, for a kernel may run over it.
func runCPUGraph(codes []Code, g *graph.Graph, rows int, x []float32, h *headWeights) ([]float32, error) {
	x = slices.Clone(x)
	vals := map[graph.ValueID][]uint32{}
	nan := math.Float32bits(float32(math.NaN()))
	value := func(v graph.ValueID) []uint32 {
		if w, ok := vals[v]; ok {
			return w
		}
		w := make([]uint32, rows*g.Values[v].Width)
		for i := range w {
			w[i] = nan
		}
		vals[v] = w
		return w
	}
	vals[g.In[0]] = unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x))
	keep := [][]uint32{}
	scratch := map[Operand][]uint32{}
	vecs := map[jlm.Role][]uint32{}
	for r, v := range h.vecs {
		vecs[r] = unsafe.Slice((*uint32)(unsafe.Pointer(&v[0])), len(v))
	}
	for _, c := range codes {
		if c.CPU == nil {
			return nil, fmt.Errorf("%s has no CPU code", c.Name)
		}
		code, err := cpu.Map(c.CPU)
		if err != nil {
			return nil, fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		var args cpu.Args
		base := unsafe.Pointer(&args)
		for _, a := range c.Args {
			if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(args) {
				code.Close()
				return nil, fmt.Errorf("%s: a call word at %d past the argument block", c.Name, a.Field)
			}
			slot := (*uint64)(unsafe.Add(base, a.Field))
			var words []uint32
			o := a.Operand
			switch {
			case a.Length || a.Value:
				*slot = uint64(a.Extent)
				continue
			case a.Words != nil:
				words = append([]uint32(nil), a.Words...)
			case o.Kind == OperandValue:
				words = value(o.Value)
			case o.Kind == OperandVector && vecs[o.Role] != nil:
				words = vecs[o.Role]
			case o.Kind == OperandWeight && o.Region >= 0 && o.Region < 3:
				words = h.planes[o.Region]
			case o.Kind == OperandScratch:
				if scratch[o] == nil {
					size := 1 << 16
					for _, b := range c.Binds {
						if b.Operand == o && b.Bytes > 0 {
							size = int(b.Bytes)/4 + 1
						}
					}
					scratch[o] = make([]uint32, size)
				}
				words = scratch[o]
			default:
				code.Close()
				return nil, fmt.Errorf("%s binds %+v, which the head's call site does not provide", c.Name, o)
			}
			if len(words) == 0 {
				code.Close()
				return nil, fmt.Errorf("%s: %+v is empty", c.Name, o)
			}
			keep = append(keep, words)
			if a.Word {
				*slot = uint64(words[a.Offset])
				continue
			}
			*slot = uint64(uintptr(unsafe.Pointer(&words[0]))) + uint64(4*a.Offset)
		}
		code.Call(&args)
		code.Close()
	}
	runtime.KeepAlive(keep)
	runtime.KeepAlive(x)
	var written []Operand
	for _, b := range codes[len(codes)-1].Binds {
		if b.Writes && b.Operand.Kind != OperandConsts {
			written = append(written, b.Operand)
		}
	}
	if len(written) != 1 {
		return nil, fmt.Errorf("the last kernel writes %d operands (%+v), where the head's output is one", len(written), written)
	}
	var out []uint32
	switch o := written[0]; o.Kind {
	case OperandValue:
		out = value(o.Value)
	case OperandScratch:
		out = scratch[o]
	default:
		return nil, fmt.Errorf("the last kernel writes %+v, which is not where a head's output lives", o)
	}
	n := rows * g.Values[g.Out[0]].Width
	if len(out) < n {
		return nil, fmt.Errorf("the output's storage holds %d words, the head's output is %d", len(out), n)
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&out[0])), n), nil
}

// formHeadOnCPU forms head graph g for the CPU over rows and runs it.
func formHeadOnCPU(t *testing.T, g *graph.Graph, rows int, x []float32, h *headWeights) ([]float32, error) {
	t.Helper()
	formats := map[jlm.Role]jlm.Type{jlm.RoleOutput: jlm.TypeF32}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if w.Rows == 0 {
				formats[w.Role] = jlm.TypeF32
			}
		}
	}
	_, codes, err := Default().WithoutInt8(DeviceCPU).Form(Request{
		Graph:   g,
		Formats: formats,
		Step:    Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true},
	})
	if err != nil {
		return nil, fmt.Errorf("forming: %w", err)
	}
	got, err := runCPUGraph(codes, g, rows, x, h)
	if err != nil {
		return nil, fmt.Errorf("%w; the head forms into:\n%s", err, describeCodes(codes))
	}
	return got, nil
}

// describeCodes is each kernel of codes and the operands its call binds.
func describeCodes(codes []Code) string {
	var b strings.Builder
	for i, c := range codes {
		fmt.Fprintf(&b, "  %d %s:", i, c.Name)
		for _, a := range c.Args {
			switch o := a.Operand; {
			case a.Length || a.Value:
				fmt.Fprintf(&b, " [extent %d]", a.Extent)
			default:
				fmt.Fprintf(&b, " [%v %q v%d +%d moving %v word %v]", o.Kind, o.Name, o.Value, a.Offset, a.Moving, a.Word)
			}
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// TestComposedCPUHead executes the head the front end states on the CPU and
// holds it to the oracle, one arm for each of its ops: the RMSNorm, the
// LayerNorm with and without its bias, the projection's bias, the final
// softcap and the logit scale, then all of them, over several rows run one at
// a time: the head is served at one row. Each op is then taken out of the
// graph and must part from the oracle by far more than any arm does: the
// softcap dropped, the scale inverted, the projection's bias dropped, the
// norm's bias dropped, and an RMSNorm for the LayerNorm.
func TestComposedCPUHead(t *testing.T) {
	const bound, nrows = 1e-9, 4
	rng := rand.New(rand.NewSource(11))
	h := newHeadWeights(t, rng)
	x := make([]float32, nrows*headEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 3)
	}
	row := func(r int) []float32 { return x[r*headEmbd : (r+1)*headEmbd] }
	base := graph.Head{NEmbd: headEmbd, NVocab: headVocab, Eps: headEps, Output: jlm.RoleOutput}
	// The cap bends the logits: half the largest the uncapped head reaches.
	amax := 0.0
	for _, v := range headWant(t, h, graph.Head{NEmbd: headEmbd, NVocab: headVocab, Eps: headEps, LayerNorm: true,
		NormBias: true, Bias: true}, x, nrows) {
		amax = math.Max(amax, math.Abs(float64(v)))
	}
	capAt := amax / 2
	arm := func(f func(s *graph.Head)) graph.Head {
		s := base
		f(&s)
		return s
	}
	arms := []struct {
		name string
		spec graph.Head
	}{
		{"RMSNorm", base},
		{"LayerNorm", arm(func(s *graph.Head) { s.LayerNorm = true })},
		{"LayerNorm with its bias", arm(func(s *graph.Head) { s.LayerNorm, s.NormBias = true, true })},
		{"output bias", arm(func(s *graph.Head) { s.Bias = true })},
		{"softcap", arm(func(s *graph.Head) { s.Softcap = capAt })},
		{"scale", arm(func(s *graph.Head) { s.Scale = 8 })},
		{"all", arm(func(s *graph.Head) {
			s.LayerNorm, s.NormBias, s.Bias, s.Softcap, s.Scale = true, true, true, capAt, 8
		})},
	}
	formed := 0
	for _, a := range arms {
		worst := 0.0
		for r := range nrows {
			got, err := formHeadOnCPU(t, a.spec.Build(), 1, row(r), h)
			if err != nil {
				t.Errorf("%s, row %d: %v", a.name, r, err)
				continue
			}
			e := cpuNMSE(got, headWant(t, h, a.spec, row(r), 1))
			if math.IsNaN(e) || !(e < bound) {
				t.Errorf("%s, row %d: NMSE %.3e against the oracle, bound %.0e", a.name, r, e, bound)
				continue
			}
			worst = math.Max(worst, e)
			formed++
		}
		t.Logf("%s: worst NMSE %.3e against the oracle over %d rows", a.name, worst, nrows)
	}
	all := arms[len(arms)-1].spec
	violations := []struct {
		name string
		spec graph.Head
	}{
		{"softcap dropped", func() graph.Head { s := all; s.Softcap = 0; return s }()},
		{"scale inverted", func() graph.Head { s := all; s.Scale = 1 / all.Scale; return s }()},
		{"output bias dropped", func() graph.Head { s := all; s.Bias = false; return s }()},
		{"norm bias dropped", func() graph.Head { s := all; s.NormBias = false; return s }()},
		{"RMSNorm for LayerNorm", func() graph.Head { s := all; s.LayerNorm, s.NormBias = false, false; return s }()},
	}
	for _, v := range violations {
		least := math.Inf(1)
		for r := range nrows {
			got, err := formHeadOnCPU(t, v.spec.Build(), 1, row(r), h)
			if err != nil {
				t.Errorf("%s, row %d: %v", v.name, r, err)
				least = 0
				continue
			}
			// The row the violation parts from the oracle least: the gate sees
			// it on every row.
			least = math.Min(least, cpuNMSE(got, headWant(t, h, all, row(r), 1)))
		}
		t.Logf("violation %q: NMSE %.3e from the oracle at the row it parts least", v.name, least)
		if math.IsNaN(least) || !(least > 1e-4) {
			t.Errorf("%s: NMSE %.3e from the oracle: the gate cannot see it", v.name, least)
		}
	}
	if want := nrows * len(arms); formed != want {
		t.Fatalf("%d of %d head arms ran", formed, want)
	}
}
