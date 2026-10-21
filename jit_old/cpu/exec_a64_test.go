//go:build arm64 && darwin

package cpu

import (
	"encoding/binary"
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// Gates that run only on Apple hardware, in the order they must be believed.
//
// If the binary carries the hardened runtime flag without
// com.apple.security.cs.allow-unsigned-executable-memory, the kernel SIGKILLs
// the process on its first instruction fetch from the RX page (exit 137, no Go
// error, no output). So the acceptance criterion is "the binary printed PASS",
// never "the command exited 0".

// TestA64ArgsLayout pins the struct offsets the generated code addresses by.
// The amd64 test asserts the same numbers: Args is shared by both
// architectures, and a field inserted in the middle repoints every load.
func TestA64ArgsLayout(t *testing.T) {
	var a Args
	for _, tc := range []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"Out", unsafe.Offsetof(a.Out), 0},
		{"W", unsafe.Offsetof(a.W), 8},
		{"A", unsafe.Offsetof(a.A), 16},
		{"AScale", unsafe.Offsetof(a.AScale), 24},
		{"Rows", unsafe.Offsetof(a.Rows), 32},
		{"K", unsafe.Offsetof(a.K), 40},
		{"RowStr", unsafe.Offsetof(a.RowStr), 48},
		{"Scr", unsafe.Offsetof(a.Scr), 56},
		{"AHalf", unsafe.Offsetof(a.AHalf), 64},
		// The packed layout's spans, loaded from exactly these offsets by
		// every packed emitter on both architectures.
		{"PD", unsafe.Offsetof(a.PD), 144},
		{"PSC", unsafe.Offsetof(a.PSC), 152},
		{"DStr", unsafe.Offsetof(a.DStr), 160},
	} {
		if tc.got != tc.want {
			t.Errorf("Args.%s is at offset %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

// TestA64ExecuteScalar is the smallest end-to-end proof: RX mapping survives
// the security policy, instruction fetch from an anonymous page, *Args arrives
// in x0, a load, a store, and the return into Go through the trampoline.
func TestA64ExecuteScalar(t *testing.T) {
	var a A64
	a.LDRx(X1, X0, 32) // args.Rows
	a.LDRx(X2, X0, 56) // args.Scr
	a.STRx(X1, X2, 0)
	a.RET()

	code, err := Map(a.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer code.Close()

	scr := make([]byte, 8)
	args := Args{Rows: 0x1234_5678_9abc, Scr: &scr[0]}
	code.Call(&args)

	if got := binary.LittleEndian.Uint64(scr); got != 0x1234_5678_9abc {
		t.Fatalf("kernel wrote %#x, want %#x", got, 0x1234_5678_9abc)
	}
}

// TestA64ExecuteSDOT checks SDOT's arithmetic against a scalar reference. The
// point is the signedness: SDOT is signed x signed (VPDPBUSD is unsigned x
// signed), which is why matvec_a64.go has no bias correction, so both operands
// are negative in places.
func TestA64ExecuteSDOT(t *testing.T) {
	var a A64
	a.LDRx(X1, X0, 16) // args.A
	a.LDRx(X2, X0, 8)  // args.W
	a.LDRx(X3, X0, 56) // args.Scr
	a.MOVIzero(V0)
	a.LDRq(V1, X1, 0)
	a.LDRq(V2, X2, 0)
	a.SDOT(V0, V1, V2)
	a.STRq(V0, X3, 0)
	a.RET()

	code, err := Map(a.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer code.Close()

	act := make([]int8, 16)
	w := make([]byte, 16)
	for i := range act {
		act[i] = int8(i*37%251 - 125) // spans the negative half
		w[i] = byte(int8(i*13%127 - 63))
	}
	scr := make([]byte, 16)
	args := Args{A: &act[0], W: &w[0], Scr: &scr[0]}
	code.Call(&args)

	for lane := 0; lane < 4; lane++ {
		var want int32
		for i := 0; i < 4; i++ {
			k := lane*4 + i
			want += int32(act[k]) * int32(int8(w[k])) // BOTH signed
		}
		got := int32(binary.LittleEndian.Uint32(scr[lane*4:]))
		if got != want {
			t.Errorf("lane %d: kernel got %d, reference %d", lane, got, want)
		}
	}
}

// TestA64ExecuteLoop exercises a backward CBNZ, which every kernel inner loop
// ends in, and the label fixup that produced its displacement.
func TestA64ExecuteLoop(t *testing.T) {
	var a A64
	a.LDRx(X1, X0, 16) // A
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X3, X0, 56) // Scr
	a.LDRx(X4, X0, 40) // K = number of 16-byte blocks
	a.MOVIzero(V0)
	top := a.Label()
	a.Bind(top)
	a.LDRq(V1, X1, 0)
	a.LDRq(V2, X2, 0)
	a.SDOT(V0, V1, V2)
	a.ADDimm(X1, X1, 16)
	a.ADDimm(X2, X2, 16)
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, top)
	a.STRq(V0, X3, 0)
	a.RET()

	code, err := Map(a.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer code.Close()

	const blocks = 64
	act := make([]int8, blocks*16)
	w := make([]byte, blocks*16)
	for i := range act {
		act[i] = int8(i%127 - 63)
		w[i] = byte(int8(i%97 - 48))
	}
	scr := make([]byte, 16)
	args := Args{A: &act[0], W: &w[0], K: blocks, Scr: &scr[0]}
	code.Call(&args)

	var want [4]int32
	for b := 0; b < blocks; b++ {
		for lane := 0; lane < 4; lane++ {
			for i := 0; i < 4; i++ {
				k := b*16 + lane*4 + i
				want[lane] += int32(act[k]) * int32(int8(w[k]))
			}
		}
	}
	for lane := 0; lane < 4; lane++ {
		got := int32(binary.LittleEndian.Uint32(scr[lane*4:]))
		if got != want[lane] {
			t.Errorf("lane %d: kernel got %d, reference %d", lane, got, want[lane])
		}
	}
}

// TestA64GoroutineRegisterIntact: x28 holds g on arm64, and a kernel that
// clobbered it would break the runtime later, elsewhere. Allocating and
// scheduling right after kernel calls makes the damage local.
func TestA64GoroutineRegisterIntact(t *testing.T) {
	for _, wt := range a64KernelTypes {
		code, err := EmitA64(Spec{W: wt, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatal(err)
		}
		kern := mustMap(t, code)
		const k, rows = 256, 4
		nb := BlocksPerRow(wt, k)
		packed := make([]byte, rows*nb*int(wt.BlockBytes()))
		q := make([]int8, k)
		pairs := make([]float32, 2*(k/Q8Block))
		for i := range pairs {
			pairs[i] = 1
		}
		out := make([]float32, rows)
		konst := KernelConst(wt)
		scratch := make([]float32, 64)
		args := Args{Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
			Rows: rows, K: int64(nb), RowStr: int64(RowBytes(wt, k)),
			Scr: &konst[0], Scratch: (*byte)(unsafe.Pointer(&scratch[0]))}
		done := make(chan int, 8)
		for i := 0; i < 8; i++ {
			go func() {
				kern.Call(&args)
				s := 0
				for j := 0; j < 4096; j++ {
					s += len(make([]byte, 64)) // allocate; a bad g dies here
				}
				done <- s
			}()
		}
		for i := 0; i < 8; i++ {
			<-done
		}
		kern.Close()
	}
}

// TestA64KernelMatchesReference is T1 for arm64, held to the x86 bar: NMSE <
// 1e-10 against a float64 evaluation of the same int8 activations and
// per-block scales the kernel uses (unquantized activations would need a
// ~1e-2 tolerance). Weight values come from quant.Dequant, which is verified
// against libggml.
//
// SDOT is signed x signed, so Q4_0, Q8_0, Q3_K and Q6_K never read
// pairs[2b+1]; a sub-test proves the value is ignored (BiasC's /8 is sized
// for eight AVX2 lanes). Q4_K and Q5_K do read it: w = d*sc*q - dmin*m needs
// sum(a) per 32-element block, which QuantizeQ8 stores as -sum(q)*biasC/8 with
// BiasC 1, so the kernel recovers it as -8*pairs[2b+1] instead of recomputing
// it per row.
func TestA64KernelMatchesReference(t *testing.T) {
	for _, wt := range a64KernelTypes {
		t.Run(wt.String(), func(t *testing.T) { a64KernelVsReference(t, wt, BiasC(wt)) })
	}
	t.Run("bias-is-ignored", func(t *testing.T) {
		for _, wt := range a64KernelTypes {
			if a64NeedsActivationSum(wt) {
				continue // reads pairs[2b+1] for the min term, by design
			}
			a64KernelVsReference(t, wt, 0)
		}
	})
}

// a64NeedsActivationSum reports whether the arm64 kernel reads pairs[2b+1].
func a64NeedsActivationSum(t quant.Type) bool {
	return t == quant.Q4_K || t == quant.Q5_K
}

func a64KernelVsReference(t *testing.T, wt quant.Type, biasC float64) {
	t.Helper()
	code, err := EmitA64(Spec{W: wt, Rows: 1, Accs: 1, Cols: 1})
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	shapes := []struct{ rows, k int }{{1, 32}, {4, 64}, {8, 256}, {32, 2048}, {3, 96}}
	if wt.BlockElems() == 256 {
		shapes = []struct{ rows, k int }{{1, 256}, {4, 512}, {8, 1024}, {32, 2048}, {3, 768}}
	}
	konst := KernelConst(wt)
	scratch := make([]float32, 64)
	for _, shape := range shapes {
		rng := rand.New(rand.NewSource(int64(shape.rows*1000 + shape.k)))
		packed, exact := buildWeights(rng, wt, shape.rows, shape.k)

		// Four input distributions. Uniform alone is not enough: real
		// activations are heavy-tailed, and the interesting failures are at the
		// extremes of the quantization range.
		for _, dist := range []struct {
			name string
			gen  func(i int) float64
		}{
			{"uniform", func(int) float64 { return rng.Float64()*2 - 1 }},
			{"student-t", func(int) float64 { return rng.NormFloat64() / (0.1 + math.Abs(rng.NormFloat64())) }},
			{"one-huge-per-block", func(i int) float64 {
				if i%32 == 7 {
					return 1000
				}
				return rng.Float64() * 0.01
			}},
			{"saturating", func(i int) float64 {
				if i%2 == 0 {
					return 1
				}
				return -1
			}},
		} {
			x := make([]float32, shape.k)
			for i := range x {
				x[i] = float32(dist.gen(i))
			}
			q := make([]int8, shape.k)
			pairs := make([]float32, 2*(shape.k/Q8Block))
			if err := oracle.QuantizeQ8(q, pairs, x, biasC); err != nil {
				t.Fatal(err)
			}
			out := make([]float32, shape.rows)
			args := Args{
				Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
				Rows: int64(shape.rows), K: int64(BlocksPerRow(wt, shape.k)),
				RowStr: int64(RowBytes(wt, shape.k)),
				Scr:    &konst[0], Scratch: (*byte)(unsafe.Pointer(&scratch[0])),
			}
			kern.Call(&args)

			// Reference: the same arithmetic, in float64.
			var sse, sy2 float64
			for r := 0; r < shape.rows; r++ {
				var want float64
				for b := 0; b < shape.k/32; b++ {
					dx := float64(pairs[2*b])
					for i := 0; i < 32; i++ {
						want += exact[r][b*32+i] * float64(q[b*32+i]) * dx
					}
				}
				d := float64(out[r]) - want
				sse += d * d
				sy2 += want * want
			}
			nmse := 0.0
			if sy2 > 0 {
				nmse = sse / sy2
			}
			if nmse > 1e-10 || math.IsNaN(nmse) {
				t.Errorf("%s rows=%d k=%d %s biasC=%g: NMSE %.3e exceeds 1e-10",
					wt, shape.rows, shape.k, dist.name, biasC, nmse)
			}
		}
	}
}

// TestA64MapRejects: an empty program must not become an executable page.
func TestA64MapRejects(t *testing.T) {
	if _, err := Map(nil); err == nil {
		t.Error("Map accepted empty code")
	}
	var r A64
	r.RET()
	c, err := Map(r.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Error("double Close should be a no-op")
	}
	defer func() {
		if recover() == nil {
			t.Error("Call on closed code should panic, not segfault")
		}
	}()
	c.Call(&Args{})
}
