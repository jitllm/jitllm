package composer

import (
	"crypto/sha256"
	bin "encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/composer/x86"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUMatVec executes the composer's lone matvec on the CPU: each
// kernel it forms mapped by cpu.Map and called through the trampoline with
// its arguments where its call words say, every packed type at one and four
// tokens, held to the oracle. Every shape must form and be right, and with
// its weight's words moved by one it must not be. The int8 arm is held to
// the oracle over the row as the int8 form quantizes it, the float arm to
// the oracle over the row.
func TestComposedCPUMatVec(t *testing.T) {
	const k = 256
	rng := rand.New(rand.NewSource(5))
	right, wrong, apart, shared, sized, unformed := 0, 0, 0, 0, 0, map[string]string{}
	// 64 rows at one worker and six; at six, 48 and 80 rows too, which six
	// workers do not divide; 1024 by 2048 at one worker and one token, a
	// call of rows Streams cuts apart, and at six, a share whose last call
	// holds fewer rows.
	for _, rows := range []int{64, 48, 80, 1024} {
		k := k
		if rows == 1024 {
			k = 2048
		}
		for _, typ := range append([]quant.Type{quant.F32}, quant.PackedTypes...) {
			q, ok := layout.QuantOf(typ)
			if typ == quant.F32 {
				q, ok = layout.Float32, true
			}
			jt, jok := jlm.TypeOf(typ)
			if !ok || !jok {
				continue
			}
			w := cpuTestWeights(typ, q, rows, k, rng)
			qs, d, sc, err := layout.PackWeights(q, w, rows, k)
			if err != nil {
				t.Fatalf("%v: %v", typ, err)
			}
			host := uint64(cpu.CPU().ISA())
			type cpuArm struct {
				name      string
				r         *Registry
				isa       uint64
				inexactDv bool
				win       int
			}
			arms := []cpuArm{{"float", Default().WithoutInt8(DeviceCPU), host, false, 0}, {"int8", withCPUInt8(), host, false, 0},
				{"int8 inexact division", withCPUInt8(), host, true, 0}, {"int8 with no lanes", withCPUInt8NoLanes(), host, false, 0}}
			if vnni := uint64(x86.ISAVNNI); host&vnni != 0 {
				arms = append(arms, cpuArm{"int8 without vnni", withCPUInt8(), host &^ vnni, false, 0})
			}
			if dp := uint64(cpu.ISADotProd); host&dp != 0 {
				arms = append(arms, cpuArm{"int8 without dotprod", withCPUInt8(), host &^ dp, false, 0})
			}
			if l, ok := jlm.Layout(jt); ok && l.Sub*l.SuperSub == 256 {
				arms = append(arms, cpuArm{"int8 window 256", withCPUInt8(), host, false, 256},
					cpuArm{"int8 window 256 without vnni", withCPUInt8(), host &^ uint64(x86.ISAVNNI), false, 256})
			}
			for _, arm := range arms {
				win := max(arm.win, 32)
				for _, at := range []struct{ tokens, slots int }{{1, 1}, {4, 1}, {1, 6}} {
					if rows != 64 && rows != 1024 && at.slots == 1 || rows == 1024 && at.tokens != 1 {
						continue
					}
					tokens := at.tokens
					name := fmt.Sprintf("%v/%d rows/%d tokens/%d workers/%s", typ, rows, tokens, at.slots, arm.name)
					x := make([]float32, tokens*k)
					for i := range x {
						x[i] = float32(rng.NormFloat64())
					}
					ref := x
					if arm.r.Profiles[DeviceCPU].Int8 != nil && typ != quant.F32 {
						ref = int8Row(x, win)
					}
					want := make([]float32, tokens*rows)
					for tk := range tokens {
						if err := oracle.MatVec(want[tk*rows:(tk+1)*rows], typ, w, ref[tk*k:(tk+1)*k], rows, k); err != nil {
							t.Fatal(err)
						}
					}
					t.Logf("forming %s", name)
					req := Request{
						Graph:   graph.MatVec(jlm.RoleOutput, rows, k),
						Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
						Step:    Step{Rows: tokens, Page: 64, Keys: 1, ActWindow: win},
						Device:  Device{Kind: DeviceCPU, Slots: at.slots, Lanes: 8, ISA: arm.isa, ExactDiv: !arm.inexactDv},
					}
					_, codes, err := arm.r.Form(req)
					if err != nil {
						unformed[name] = err.Error()
						continue
					}
					if rows == 1024 {
						if ks, err := arm.r.formKernels(req); err == nil {
							if rowsApart(ks) {
								apart++
								t.Logf("%s: rows apart", name)
							}
							if raggedShare(ks) {
								shared++
							}
						}
					}
					if at.slots > 1 && !dealtOverWorkers(codes) {
						t.Errorf("%s: no kernel's rows are dealt over the %d workers", name, at.slots)
						for _, c := range codes {
							t.Logf("  %s calls %v passes %v", c.Name, c.Calls, c.Passes)
						}
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
					for _, c := range codes {
						if c.Share == 0 {
							continue
						}
						for _, wk := range []int{1, 5, 7, 13} {
							if sp := spaceOf(&c, wk); sp.n > wk {
								t.Errorf("%s: %s runs %d calls on %d workers: in waves", name, c.Name, sp.n, wk)
							}
							got, err := runCPUWorkers(codes, x, tokens*rows, planes, nil, nil, wk)
							if err != nil {
								t.Errorf("%s at %d workers: %v", name, wk, err)
							} else if e := cpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
								t.Errorf("%s at %d workers: NMSE %.3e against the oracle", name, wk, e)
							}
							sized++
						}
					}
					moved := [3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), d, sc}
					bad, err := runCPU(codes, x, tokens*rows, moved)
					if err != nil {
						t.Errorf("%s: the violation did not run: %v", name, err)
					} else if e := cpuNMSE(bad, want); !(e > 1e-6) {
						t.Errorf("%s: with the weight's words moved by one the NMSE is %.3e: the gate cannot see a wrong kernel", name, e)
					}
				}
			}
		}
	}
	for name, why := range unformed {
		t.Errorf("%s does not form on the CPU: %s", name, why)
	}
	t.Logf("%d formed and right, %d formed and wrong, %d not formed, %d with their rows apart", right, wrong, len(unformed), apart)
	if right == 0 {
		t.Fatal("no matvec formed on the CPU: this gate proved nothing")
	}
	if apart == 0 {
		t.Fatal("no matvec formed its rows apart (Streams.Apart): the gate never ran that form")
	}
	if shared == 0 || sized == 0 {
		t.Fatal("no matvec dealt a worker's share a call (Workers): the gate never ran that form")
	}
}

// raggedShare reports whether a kernel's calls are workers' shares.
func raggedShare(ks []*Kernel) bool {
	for _, k := range ks {
		if k.Share > 0 {
			return true
		}
	}
	return false
}

// rowsApart reports whether a kernel runs several loops over a call's rows
// in one loop.
func rowsApart(ks []*Kernel) bool {
	for _, k := range ks {
		n := map[[2]int]int{}
		for _, l := range k.Loops {
			if l.Parent >= 0 && l.Trip > 1 {
				n[[2]int{l.Parent, l.Trip}]++
			}
		}
		for _, c := range n {
			if c > 1 {
				return true
			}
		}
	}
	return false
}

// movingArg is an address word a call moves along its coordinates, at
// its first call's address.
type movingArg struct {
	a    CallArg
	at   uint64
	elem int
}

// dealtOverWorkers reports whether a kernel of codes runs as several
// calls along a space Workers cut.
func dealtOverWorkers(codes []Code) bool {
	for _, c := range codes {
		if slices.Contains(c.Passes, "workers") && len(c.Calls) > 0 && c.Calls[0] > 1 {
			return true
		}
	}
	return false
}

// sseKernelDir names the directory mapComposed writes each SSE-declared
// kernel into, for the VEX-leak gate to read back.
const sseKernelDir = "JITLLM_TEST_SSE_KERNELS"

// mapComposed is cpu.Map, keeping each SSE-declared kernel for the VEX-leak
// gate where it asks for them.
func mapComposed(code []byte) (*cpu.Code, error) {
	if need, ok := x86.KernelISA(code); ok && need&x86.ISAAVX2 == 0 {
		if dir := os.Getenv(sseKernelDir); dir != "" {
			sum := sha256.Sum256(code)
			if err := os.WriteFile(filepath.Join(dir, hex.EncodeToString(sum[:])), code, 0o600); err != nil {
				return nil, err
			}
		}
	}
	return cpu.Map(code)
}

// runCPU runs codes in order on the CPU: the row x, an output of n floats,
// a weight's planes, the composer's scratch and constant blocks, each where
// its call words say.
func runCPU(codes []Code, x []float32, n int, planes [3][]uint32) ([]float32, error) {
	return runCPUVectors(codes, x, n, planes, nil)
}

// runCPUVectors is runCPU with the vector weights vecs binds, by role.
func runCPUVectors(codes []Code, x []float32, n int, planes [3][]uint32, vecs map[jlm.Role][]float32) ([]float32, error) {
	return runCPUBound(codes, x, n, planes, vecs, nil)
}

// runCPUBound is runCPUVectors with bind asked first for each operand: the
// words it returns, or nil for runCPUVectors' own.
func runCPUBound(codes []Code, x []float32, n int, planes [3][]uint32, vecs map[jlm.Role][]float32, bind func(Operand) []uint32) ([]float32, error) {
	return runCPUWorkers(codes, x, n, planes, vecs, bind, harnessWorkers)
}

// runCPUWorkers is runCPUBound with a Share code's calls sized for workers.
func runCPUWorkers(codes []Code, x []float32, n int, planes [3][]uint32, vecs map[jlm.Role][]float32, bind func(Operand) []uint32, workers int) ([]float32, error) {
	const guard = 16
	out := make([]float32, n+guard)
	for i := range out {
		out[i] = float32(math.NaN())
	}
	keep := [][]uint32{}
	scratch := map[Operand][]uint32{}
	for _, c := range codes {
		if c.CPU == nil {
			return nil, fmt.Errorf("%s has no CPU code", c.Name)
		}
		code, err := mapComposed(c.CPU)
		if err != nil {
			return nil, fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		var args cpu.Args
		base := unsafe.Pointer(&args)
		var coords []CallArg
		var moving []movingArg
		for _, a := range c.Args {
			if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(args) {
				code.Close()
				return nil, fmt.Errorf("%s: a call word at %d past the argument block", c.Name, a.Field)
			}
			slot := (*uint64)(unsafe.Add(base, a.Field))
			var words []uint32
			o := a.Operand
			switch {
			case a.Coord > 0:
				coords = append(coords, a)
				continue
			case a.Length || a.Value:
				*slot = argValue(a, spaceOf(&c, workers))
				continue
			case a.Words != nil:
				words = append([]uint32(nil), a.Words...)
			case bind != nil && bind(o) != nil:
				words = bind(o)
			case o.Kind == OperandValue && o.Value == 0:
				words = unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x))
			case o.Kind == OperandValue && o.Value == 1:
				words = unsafe.Slice((*uint32)(unsafe.Pointer(&out[0])), len(out))
			case o.Kind == OperandWeight && o.Region >= 0 && o.Region < 3:
				words = planes[o.Region]
			case o.Kind == OperandVector && len(vecs[o.Role]) > 0:
				v := vecs[o.Role]
				words = unsafe.Slice((*uint32)(unsafe.Pointer(&v[0])), len(v))
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
			elem := 4
			if a.Elem > 0 {
				elem = a.Elem
			}
			*slot = uint64(uintptr(unsafe.Pointer(&words[0]))) + uint64(elem*a.Offset)
			if len(a.Per) > 0 {
				moving = append(moving, movingArg{a, *slot, elem})
			}
		}
		callAll(code, &args, spaceOf(&c, workers), coords, moving)
		code.Close()
	}
	runtime.KeepAlive(keep)
	for i, v := range out[n:] {
		if math.Float32bits(v) != math.Float32bits(float32(math.NaN())) {
			return nil, fmt.Errorf("the kernels wrote %v past the output's end, at element %d", v, n+i)
		}
	}
	return out[:n], nil
}

// callAll calls code once at each coordinate of the calls' space calls, its
// coordinate words and its moving addresses set for each.
func callAll(code *cpu.Code, args *cpu.Args, sp callSpace, coords []CallArg, moving []movingArg) {
	callRange(code, args, sp, coords, moving, 0, sp.n)
}

// callSpace is a code's calls as a test runs them on its workers: their
// count, and for a Share code the rows a call takes.
type callSpace struct {
	calls                       []int
	n, rows, unit, share, field int
}

// harnessWorkers is the workers runCPU sizes a Share code's calls for:
// fewer than any device a test forms for has.
const harnessWorkers = 3

func spaceOf(c *Code, workers int) callSpace {
	sp := callSpace{calls: c.Calls, n: 1}
	if rows, n := c.ShareRows(workers); n > 0 {
		sp.rows, sp.n, sp.unit, sp.share = rows, n, max(c.ShareUnit, 1), c.Share
		for _, a := range c.Args {
			if a.Share {
				sp.field = a.Field
			}
		}
		return sp
	}
	for _, x := range c.Calls {
		sp.n *= x
	}
	return sp
}

func (sp callSpace) coord(at, ax int) int {
	if sp.rows > 0 {
		return at * sp.rows / sp.unit
	}
	for a := len(sp.calls) - 1; a > ax; a-- {
		at /= sp.calls[a]
	}
	return at % sp.calls[ax]
}

// argValue is the value word a: a Share code's first call's rows, or its
// extent.
func argValue(a CallArg, sp callSpace) uint64 {
	if a.Share {
		return uint64(sp.rows)
	}
	return uint64(a.Extent)
}

// withCPUInt8 is the default registry with the CPU keeping the int8 form,
// whether or not its profile does yet.
func withCPUInt8() *Registry {
	r := Default()
	p := r.Profiles[DeviceCPU]
	p.Int8 = &Int8Form{Window: 32, Word: 4, Sums: 16, UnsignedDot: uint64(x86.ISAVNNI), Slot: 16}
	r.Profiles[DeviceCPU] = p
	return r
}

// withCPUInt8NoLanes is withCPUInt8 with no loop across lanes, a float's
// bits in the general registers (Kernel.Unvector).
func withCPUInt8NoLanes() *Registry {
	r := withCPUInt8()
	p := r.Profiles[DeviceCPU]
	p.Off = map[string]bool{Vectorize{}.Name(): true}
	r.Profiles[DeviceCPU] = p
	return r
}

// int8Row is x as the int8 form leaves it: each window of w scaled by
// d = amax/127 and its inverse 1/d, each correctly rounded, each element
// d times its code, the element times the inverse rounded half away from
// zero.
func int8Row(x []float32, w int) []float32 {
	out := make([]float32, len(x))
	for s := 0; s < len(x); s += w {
		var amax float32
		for _, v := range x[s : s+w] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := amax / 127
		var inv float32
		if d > 0 {
			inv = 1 / d
		}
		for i, v := range x[s : s+w] {
			t := v * inv
			code := float32(int32(t + float32(math.Copysign(0.5, float64(t)))))
			out[s+i] = d * code
		}
	}
	return out
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
func cpuTestWeights(t quant.Type, q layout.Quant, rows, k int, rng *rand.Rand) []byte {
	if t == quant.F32 {
		b := make([]byte, 4*rows*k)
		for i := 0; i < rows*k; i++ {
			bin.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(rng.NormFloat64())))
		}
		return b
	}
	f16At := map[layout.Quant][]int{
		layout.Q4_0: {0}, layout.Q8_0: {0}, layout.Q5_0: {0}, layout.Q5_1: {0, 2},
		layout.Q4_K: {0, 2}, layout.Q5_K: {0, 2}, layout.Q3_K: {108}, layout.Q6_K: {208}, layout.TQ2: {64},
	}
	nblk, bb := k/q.Elems(), q.BlockBytes()
	raw := make([]byte, rows*nblk*bb)
	rng.Read(raw)
	for i := 0; i < rows*nblk; i++ {
		switch {
		case q == layout.MXFP4:
			raw[i*bb] = byte(120 + rng.Intn(10))
		case q == layout.TQ2F:
			bin.LittleEndian.PutUint32(raw[i*bb+64:], math.Float32bits(0.5+rng.Float32()))
		default:
			for _, off := range f16At[q] {
				bin.LittleEndian.PutUint16(raw[i*bb+off:], uint16(0x2000|rng.Intn(0x0C00)))
			}
		}
	}
	return raw
}

// TestComposedCPUHonoursTheDevicesISA forms every packed matvec for a CPU
// stated without BMI2: each forms, with no SHRX or SHLX, which fault there,
// and each is right. Its kernels are mapped with the ISA declared, so an
// instruction past it fails at assembly.
func TestComposedCPUHonoursTheDevicesISA(t *testing.T) {
	const rows, k = 64, 256
	rng := rand.New(rand.NewSource(7))
	isa := uint64(cpu.CPU().ISA()) &^ uint64(x86.ISABMI2)
	r := Default().WithoutInt8(DeviceCPU)
	right := 0
	for _, typ := range append([]quant.Type{quant.F32}, quant.PackedTypes...) {
		q, ok := layout.QuantOf(typ)
		if typ == quant.F32 {
			q, ok = layout.Float32, true
		}
		jt, jok := jlm.TypeOf(typ)
		if !ok || !jok {
			continue
		}
		w := cpuTestWeights(typ, q, rows, k, rng)
		qs, d, sc, err := layout.PackWeights(q, w, rows, k)
		if err != nil {
			t.Fatalf("%v: %v", typ, err)
		}
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, rows)
		if err := oracle.MatVec(want, typ, w, x, rows, k); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{
			Graph:   graph.MatVec(jlm.RoleOutput, rows, k),
			Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
			Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
			Device:  Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: isa, ExactDiv: true},
		})
		if err != nil {
			t.Errorf("%v without BMI2 does not form: %v", typ, err)
			continue
		}
		got, err := runCPU(codes, x, rows, [3][]uint32{qs, d, sc})
		if err != nil {
			t.Errorf("%v: %v", typ, err)
			continue
		}
		if e := cpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
			t.Errorf("%v without BMI2: NMSE %.3e against the oracle", typ, e)
			continue
		}
		right++
	}
	t.Logf("without BMI2: %d formed and right", right)
}
