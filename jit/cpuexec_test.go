package jit

import (
	bin "encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestComposedCPUMatVec executes the composer's lone matvec on the CPU: each
// kernel it forms mapped by cpu.Map and called through the trampoline with
// its arguments where its call words say, every packed type at one and four
// tokens, held to the oracle. What does not form is the CPU's work list,
// named; what forms must be right, and with its weight's words moved by one
// it must not be.
func TestComposedCPUMatVec(t *testing.T) {
	const rows, k = 64, 256
	rng := rand.New(rand.NewSource(5))
	right, wrong, unformed := 0, 0, map[string]string{}
	for _, typ := range append([]quant.Type{quant.F32}, quant.PackedTypes...) {
		q, ok := kernels.QuantOf(typ)
		if typ == quant.F32 {
			q, ok = kernels.Float32, true
		}
		jt, jok := jlm.TypeOf(typ)
		if !ok || !jok {
			continue
		}
		w := cpuTestWeights(typ, q, rows, k, rng)
		qs, d, sc, err := kernels.PackWeights(q, w, rows, k)
		if err != nil {
			t.Fatalf("%v: %v", typ, err)
		}
		for _, tokens := range []int{1, 4} {
			name := fmt.Sprintf("%v/%d tokens", typ, tokens)
			x := make([]float32, tokens*k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			want := make([]float32, tokens*rows)
			for tk := range tokens {
				if err := oracle.MatVec(want[tk*rows:(tk+1)*rows], typ, w, x[tk*k:(tk+1)*k], rows, k); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("forming %s", name)
			_, codes, err := Default().Form(Request{
				Graph:   graph.MatVec(jlm.RoleOutput, rows, k),
				Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
				Step:    Step{Rows: tokens, Page: 64, Keys: 1, ActWindow: 32},
				Device:  Device{Kind: DeviceCPU, Slots: 1, Lanes: 8},
			})
			if err != nil {
				unformed[name] = err.Error()
				continue
			}
			planes := [3][]uint32{qs, d, sc}
			got, err := runCPU(codes, x, tokens*rows, planes)
			if err != nil {
				unformed[name] = err.Error()
				continue
			}
			if e := cpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
				wrong++
				t.Errorf("%s: NMSE %.3e against the oracle", name, e)
				continue
			}
			right++
			moved := [3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), d, sc}
			bad, err := runCPU(codes, x, tokens*rows, moved)
			if err != nil {
				t.Errorf("%s: the violation did not run: %v", name, err)
			} else if e := cpuNMSE(bad, want); !(e > 1e-6) {
				t.Errorf("%s: with the weight's words moved by one the NMSE is %.3e: the gate cannot see a wrong kernel", name, e)
			}
		}
	}
	for name, why := range unformed {
		t.Logf("work list: %s: %s", name, why)
	}
	t.Logf("%d formed and right, %d formed and wrong, %d not formed", right, wrong, len(unformed))
	if right == 0 {
		t.Fatal("no matvec formed on the CPU: this gate proved nothing")
	}
}

// runCPU runs codes in order on the CPU: the row x, an output of n floats,
// a weight's planes, the composer's scratch and constant blocks, each where
// its call words say.
func runCPU(codes []Code, x []float32, n int, planes [3][]uint32) ([]float32, error) {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.NaN())
	}
	keep := [][]uint32{}
	scratch := map[Operand][]uint32{}
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
			case o.Kind == OperandValue && o.Value == 0:
				words = unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x))
			case o.Kind == OperandValue && o.Value == 1:
				words = unsafe.Slice((*uint32)(unsafe.Pointer(&out[0])), len(out))
			case o.Kind == OperandWeight && o.Region >= 0 && o.Region < 3:
				words = planes[o.Region]
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
				return nil, fmt.Errorf("%s binds %+v, which a lone matvec does not provide", c.Name, o)
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
	return out, nil
}

func cpuNMSE(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	return num / den
}

// cpuTestWeights is random rows of t with finite scales near one.
func cpuTestWeights(t quant.Type, q kernels.Quant, rows, k int, rng *rand.Rand) []byte {
	if t == quant.F32 {
		b := make([]byte, 4*rows*k)
		for i := 0; i < rows*k; i++ {
			bin.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(rng.NormFloat64())))
		}
		return b
	}
	f16At := map[kernels.Quant][]int{
		kernels.Q4_0: {0}, kernels.Q8_0: {0}, kernels.Q5_0: {0}, kernels.Q5_1: {0, 2},
		kernels.Q4_K: {0, 2}, kernels.Q5_K: {0, 2}, kernels.Q3_K: {108}, kernels.Q6_K: {208}, kernels.TQ2: {64},
	}
	nblk, bb := k/q.Elems(), q.BlockBytes()
	raw := make([]byte, rows*nblk*bb)
	rng.Read(raw)
	for i := 0; i < rows*nblk; i++ {
		switch {
		case q == kernels.MXFP4:
			raw[i*bb] = byte(120 + rng.Intn(10))
		case q == kernels.TQ2F:
			bin.LittleEndian.PutUint32(raw[i*bb+64:], math.Float32bits(0.5+rng.Float32()))
		default:
			for _, off := range f16At[q] {
				bin.LittleEndian.PutUint16(raw[i*bb+off:], uint16(0x2000|rng.Intn(0x0C00)))
			}
		}
	}
	return raw
}
