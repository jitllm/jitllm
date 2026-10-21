//go:build arm64

package cpu

import (
	"encoding/hex"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// NEON gates for the kernels quantact_a64.go and argmax_a64.go generate. They
// call the emitters directly rather than through Native(), so a gap in the
// emitter table cannot read as a pass.
//
// The bar is bit equality against the Go reference (QuantizeQ8Window,
// QuantizeHalfSums, model.Greedy's scan): a last-bit scale difference moves a
// quantized weight at the boundary, and a different tie-break picks another
// token.
//
// The shared helpers live in quantact_test.go, built on both architectures
// so both tiers are compared against one oracle.

// quantActKernelsA64 builds both shapes straight from the arm64 emitters.
func quantActKernelsA64(t *testing.T, q quant.Type) (QuantActKernels, func()) {
	t.Helper()
	half := NeedsHalfSums(q)
	kern := QuantActKernels{Konst: QuantActConsts(q)}
	wide, err := MapNamed(emitQuantActA64(half), "quantact_a64_gate")
	if err != nil {
		t.Fatalf("%v: mapping the wide NEON kernel: %v", q, err)
	}
	narrow, err := MapNamed(emitQuantActNarrowA64(half), "quantact_narrow_a64_gate")
	if err != nil {
		wide.Close()
		t.Fatalf("%v: mapping the narrow NEON kernel: %v", q, err)
	}
	kern.Wide, kern.Narrow = wide, narrow
	return kern, func() { wide.Close(); narrow.Close() }
}

// TestQuantActA64MatchesTheGoLoopExactly is the bit-equality gate, over the
// same (format, shape) sweep the amd64 one uses so the two tiers are held to
// one standard.
func TestQuantActA64MatchesTheGoLoopExactly(t *testing.T) {
	wide, narrow := 0, 0
	for _, q := range quant.PackedTypes {
		kern, done := quantActKernelsA64(t, q)
		for _, sh := range []struct{ k, window int }{
			// Whole groups of eight blocks, a group plus a remainder
			// (2048/32 = 64 blocks split raggedly), a k with fewer than eight
			// blocks in total, and both wide windows.
			{64, 32}, {256, 32}, {2048, 32}, {512, 256}, {2048, 256}, {2304, 256},
		} {
			// Count what ran, so a declined shape cannot pass by comparing
			// the Go loop with itself.
			if !kern.Applies(sh.window) {
				t.Fatalf("%v k=%d window=%d: both NEON kernels are built and the driver "+
					"still declined -- this shape would compare the Go loop with itself",
					q, sh.k, sh.window)
			}
			x := quantActInput(sh.k, int64(sh.k+sh.window))
			wq, wp, wh := quantActGo(q, x, sh.k, sh.window)
			ResetQuantActStats()
			gq, gp, gh := quantActRun(t, kern, q, x, sh.k, sh.window, quantActSplits(sh.k/Q8Block))
			if _, _, goLoop := QuantActStats(); goLoop != 0 {
				t.Fatalf("%v k=%d window=%d: %d range(s) fell back to the Go loop, so this "+
					"comparison is partly the oracle against itself", q, sh.k, sh.window, goLoop)
			}
			for i := range wq {
				if gq[i] != wq[i] {
					t.Fatalf("%v k=%d window=%d: int8 %d differs: kernel %d, Go %d",
						q, sh.k, sh.window, i, gq[i], wq[i])
				}
			}
			for i := range wp {
				if math.Float32bits(gp[i]) != math.Float32bits(wp[i]) {
					t.Fatalf("%v k=%d window=%d: pair %d differs: kernel %v (%#08x), Go %v (%#08x)",
						q, sh.k, sh.window, i, gp[i], math.Float32bits(gp[i]),
						wp[i], math.Float32bits(wp[i]))
				}
			}
			for i := range wh {
				if math.Float32bits(gh[i]) != math.Float32bits(wh[i]) {
					t.Fatalf("%v k=%d window=%d: half %d differs: kernel %v (%#08x), Go %v (%#08x)",
						q, sh.k, sh.window, i, gh[i], math.Float32bits(gh[i]),
						wh[i], math.Float32bits(wh[i]))
				}
			}
			if sh.window > Q8Block {
				wide++
			} else {
				narrow++
			}
		}
		done()
	}
	if wide == 0 || narrow == 0 {
		t.Fatalf("ran %d wide and %d narrow shapes -- both kernels must be exercised or "+
			"this gate covers only half the window range", wide, narrow)
	}
	t.Logf("%d wide and %d narrow (format, shape) pairs are bit-identical to the Go loop on NEON",
		wide, narrow)
}

// quantActZeroInput is quantActInput with three whole ranges zeroed, each
// putting an all-zero amax window in front of a different kernel:
//
//	block 0          the wide kernel at a one-block window -- the driver hands
//	                 it the ragged splits the narrow group of eight cannot fill
//	the second window  the wide kernel's own window path
//	blocks 8..15     a whole narrow group, whose eight amaxes are one vector
//
// quantActInput's zero blocks sit inside non-zero 256-element windows, so the
// divide-by-zero path needs this. An all-zero window is routine: MatMul
// zero-fills the padding rows of a ragged tile. (FCVTZS turns NaN into 0 on
// arm64, so the zero mask is parity with amd64 rather than load-bearing here.)
func quantActZeroInput(k, window int, seed int64) []float32 {
	x := quantActInput(k, seed)
	zero := func(lo, hi int) {
		for i := lo; i < min(hi, k); i++ {
			x[i] = 0
		}
	}
	zero(0, Q8Block)
	zero(window, 2*window)
	zero(8*Q8Block, 16*Q8Block)
	return x
}

// TestQuantActA64AllZeroWindowsAgree is the ragged tile's padding, on both
// kernels: d is zero, inv is masked to zero rather than +Inf, every int8 is
// zero and the pair's correction is a negative zero on both sides -- which a
// bit-equality gate checks and a tolerance would not.
func TestQuantActA64AllZeroWindowsAgree(t *testing.T) {
	for _, q := range quant.PackedTypes {
		kern, done := quantActKernelsA64(t, q)
		for _, sh := range []struct{ k, window int }{
			{2048, 32}, {2048, 256}, {2304, 256}, {512, 256},
		} {
			x := quantActZeroInput(sh.k, sh.window, int64(sh.k))
			wq, wp, wh := quantActGo(q, x, sh.k, sh.window)
			ResetQuantActStats()
			gq, gp, gh := quantActRun(t, kern, q, x, sh.k, sh.window, quantActSplits(sh.k/Q8Block))
			wide, narrow, goLoop := QuantActStats()
			if goLoop != 0 {
				t.Fatalf("%v k=%d window=%d: %d range(s) fell back to the Go loop",
					q, sh.k, sh.window, goLoop)
			}
			if wide == 0 {
				t.Fatalf("%v k=%d window=%d: the wide kernel never ran (%d narrow), so its "+
					"zero mask is not what this compared", q, sh.k, sh.window, narrow)
			}
			for i := range wq {
				if gq[i] != wq[i] {
					t.Fatalf("%v k=%d window=%d: int8 %d differs: kernel %d, Go %d",
						q, sh.k, sh.window, i, gq[i], wq[i])
				}
			}
			for i := range wp {
				if math.Float32bits(gp[i]) != math.Float32bits(wp[i]) {
					t.Fatalf("%v k=%d window=%d: pair %d differs: kernel %v (%#08x), Go %v (%#08x)",
						q, sh.k, sh.window, i, gp[i], math.Float32bits(gp[i]),
						wp[i], math.Float32bits(wp[i]))
				}
			}
			for i := range wh {
				if math.Float32bits(gh[i]) != math.Float32bits(wh[i]) {
					t.Fatalf("%v k=%d window=%d: half %d differs: kernel %v (%#08x), Go %v (%#08x)",
						q, sh.k, sh.window, i, gh[i], math.Float32bits(gh[i]),
						wh[i], math.Float32bits(wh[i]))
				}
			}
		}
		done()
	}
}

// TestQuantActA64GateSeesAWrongConstant runs the gate above against a
// violation, because a comparison that has never failed is not a comparison.
// The constant block is where the pair layout differs from the GEMM's, so
// flipping the sign of -biasC/8 is the smallest real defect: every int8 stays
// right and every pair's correction is wrong.
func TestQuantActA64GateSeesAWrongConstant(t *testing.T) {
	q := quant.Q4_K
	kern, done := quantActKernelsA64(t, q)
	defer done()

	// Both shapes: they have different prologues and a violation the wide
	// kernel fails says nothing about the narrow one.
	for _, window := range []int{256, Q8Block} {
		const k = 512
		nb := k / Q8Block
		x := quantActInput(k, 7)
		_, want, _ := quantActGo(q, x, k, window)

		bad := kern
		bad.Konst = QuantActConsts(q)
		bad.Konst[6] = -bad.Konst[6] // the sign the correction carries
		dst, got := make([]int8, k), make([]float32, 2*nb)
		scr := make([]float32, QuantActNarrowScratch)
		bad.Run(q, dst, got, nil, scr, x, k, 0, nb, window)

		diff := 0
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				diff++
			}
		}
		if diff == 0 {
			t.Fatalf("window %d: the NEON kernel agreed with the Go loop through a flipped "+
				"bias constant -- this comparison cannot fail and certifies nothing", window)
		}
		t.Logf("window %d: a flipped -biasC/8 moves %d of %d pair words", window, diff, len(want))
	}
}

// TestQuantActA64GateSeesAWrongScale is the second violation and it aims at the
// half the first cannot reach: the correction constant never touches an int8,
// so a gate that only flips it says nothing about the amax, the divisions or
// the round. Dividing by 128 instead of 127 is a scale that is very nearly
// right -- every quantized value stays inside the clamp and the model would
// still be fluent -- which is exactly the defect a tolerance would pass.
func TestQuantActA64GateSeesAWrongScale(t *testing.T) {
	q := quant.Q4_K
	kern, done := quantActKernelsA64(t, q)
	defer done()

	for _, window := range []int{256, Q8Block} {
		const k = 512
		nb := k / Q8Block
		x := quantActInput(k, 11)
		wantQ, wantP, _ := quantActGo(q, x, k, window)

		bad := kern
		bad.Konst = QuantActConsts(q)
		bad.Konst[0] = 128 // the divisor that makes d, and inv with it
		dst, pairs := make([]int8, k), make([]float32, 2*nb)
		scr := make([]float32, QuantActNarrowScratch)
		bad.Run(q, dst, pairs, nil, scr, x, k, 0, nb, window)

		bytes, scales := 0, 0
		for i := range wantQ {
			if dst[i] != wantQ[i] {
				bytes++
			}
		}
		for i := 0; i < len(wantP); i += 2 {
			if math.Float32bits(pairs[i]) != math.Float32bits(wantP[i]) {
				scales++
			}
		}
		if bytes == 0 || scales == 0 {
			t.Fatalf("window %d: dividing the amax by 128 moved %d int8 and %d scales -- "+
				"the payload is not being compared", window, bytes, scales)
		}
		t.Logf("window %d: an amax divided by 128 moves %d of %d int8 and %d of %d scales",
			window, bytes, len(wantQ), scales, len(wantP)/2)
	}
}

// ---------------------------------------------------------------------------
// argmax

// argmaxA64Run drives the kernel exactly as nn.Argmax32JIT does.
func argmaxA64Run(t *testing.T, c *Code, konst []float32, x []float32) (int32, float32) {
	t.Helper()
	// Any positive length: Args.K is the element count and the kernel handles
	// the ragged tail itself.
	if len(x) == 0 {
		t.Fatalf("n=%d is not a usable length", len(x))
	}
	var idx int32
	var val float32
	args := Args{
		Q32:  &x[0],
		K:    int64(len(x)),
		Out:  &val,
		ASum: &idx,
		Scr:  (*byte)(unsafe.Pointer(&konst[0])),
	}
	c.Call(&args)
	return idx, val
}

// argmaxA64Go is model.Greedy's scan, copied (as in engine/nn/argmax_test.go) so
// the gate compares against what the engine actually ran.
func argmaxA64Go(x []float32) int32 {
	best, bi := float32(math.Inf(-1)), int32(0)
	for i, v := range x {
		if v > best {
			best, bi = v, int32(i)
		}
	}
	return bi
}

func TestArgmaxA64MatchesTheScan(t *testing.T) {
	c, err := MapNamed(emitArgmaxA64(), "argmax_a64_gate")
	if err != nil {
		t.Fatalf("mapping the NEON argmax: %v", err)
	}
	defer c.Close()
	konst := ArgmaxConsts()

	// The real vocabularies plus the shapes around them: the
	// kernel is only worth having if it serves the ones that ship.
	ran := 0
	// The ragged lengths reach the tail: 1..9 cover a run with no vector at
	// all, and the +/-1 pairs cover a tail beside a full sweep.
	for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 17, 255, 256, 257, 32000, 32767,
		32768, 49152, 128255, 128256, 151936, 201088, 256000} {
		r := rand.New(rand.NewSource(int64(n)))
		for _, shape := range []string{"random", "ascending", "descending", "flat", "negative", "allneginf"} {
			x := make([]float32, n)
			for i := range x {
				switch shape {
				case "ascending":
					x[i] = float32(i) * 1e-3
				case "descending":
					x[i] = float32(n-i) * 1e-3
				case "flat":
					x[i] = 1.25 // every element ties; the FIRST must win
				case "negative":
					x[i] = -float32(r.Float64()) * 40
				case "allneginf":
					// The initial vmax is -Inf, so this is the case where no
					// lane ever improves and the fold runs on its own initial
					// index vector. The scan returns 0 and so must the kernel.
					x[i] = float32(math.Inf(-1))
				default:
					x[i] = float32(r.NormFloat64()) * 8
				}
			}
			got, _ := argmaxA64Run(t, c, konst, x)
			if want := argmaxA64Go(x); got != want {
				// Range-check before indexing: a broken fold returns
				// 0x7FFFFFFF and x[got] would panic in the failure message.
				if got < 0 || int(got) >= n {
					t.Fatalf("n=%d %s: kernel returned %d, which is not an index into %d "+
						"elements; the scan says %d", n, shape, got, n, want)
				}
				t.Fatalf("n=%d %s: kernel %d, scan %d (x[%d]=%v, x[%d]=%v)",
					n, shape, got, want, got, x[got], want, x[want])
			}
			ran++
		}
	}
	t.Logf("%d (vocabulary, shape) pairs agree with the scan on NEON", ran)
}

// TestArgmaxA64KeepsTheFirstOfATie is the property greedy decoding rests on,
// tested where a lane-parallel fold is most likely to get it wrong: the two
// equal maxima land in different vector lanes, in the two different halves of
// the eight-element group (a NEON iteration is two vectors, so the fold has a
// seam amd64's does not), in the same lane of different groups, and at the ends.
func TestArgmaxA64KeepsTheFirstOfATie(t *testing.T) {
	c, err := MapNamed(emitArgmaxA64(), "argmax_a64_tie")
	if err != nil {
		t.Fatalf("mapping the NEON argmax: %v", err)
	}
	defer c.Close()
	konst := ArgmaxConsts()

	const n = 32000
	for _, pair := range [][2]int{
		{0, 1}, {0, n - 1}, {5, 13}, {7, 8}, {3, 4}, {n - 9, n - 1}, {1000, 24000},
	} {
		x := make([]float32, n)
		for i := range x {
			x[i] = -1
		}
		x[pair[0]], x[pair[1]] = 9, 9
		got, _ := argmaxA64Run(t, c, konst, x)
		if want := int32(pair[0]); got != want {
			t.Fatalf("maxima at %d and %d: kernel chose %d, and the lower index must win -- "+
				"a greedy chain that breaks ties differently from llama.cpp diverges exactly "+
				"where two logits are equal", pair[0], pair[1], got)
		}
	}
}

// TestArgmaxA64GateSeesAWrongFold runs the two gates above against violations:
// a last-of-tie oracle must disagree with the kernel, and zeroing the
// 0x7FFFFFFF sentinel a non-tying lane contributes to the SMINV fold must
// change the answer (proving the fold decides the index).
func TestArgmaxA64GateSeesAWrongFold(t *testing.T) {
	c, err := MapNamed(emitArgmaxA64(), "argmax_a64_violation")
	if err != nil {
		t.Fatalf("mapping the NEON argmax: %v", err)
	}
	defer c.Close()

	lastOfTie := func(x []float32) int32 {
		best, bi := float32(math.Inf(-1)), int32(0)
		for i, v := range x {
			if v >= best {
				best, bi = v, int32(i)
			}
		}
		return bi
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = -1
	}
	x[3], x[200] = 5, 5
	got, _ := argmaxA64Run(t, c, ArgmaxConsts(), x)
	if got == lastOfTie(x) {
		t.Fatal("the kernel agrees with a last-of-tie scan, so this gate cannot tell the two " +
			"rules apart and certifies nothing about the tie order")
	}
	if got != 3 {
		t.Fatalf("kernel chose %d, want 3", got)
	}

	bad := ArgmaxConsts()
	bad[9] = math.Float32frombits(0) // the sentinel a non-tying lane contributes
	y := make([]float32, 256)
	for i := range y {
		y[i] = float32(i)
	}
	if idx, _ := argmaxA64Run(t, c, bad, y); idx == argmaxA64Go(y) {
		t.Fatalf("a zeroed tie sentinel still produced %d, the right answer -- the SMINV "+
			"fold is not deciding the index and this gate certifies nothing", idx)
	}
}

// TestArgmaxA64ConstantsAreTheSharedBlock checks this kernel and the amd64 one
// read one layout: ArgmaxConsts is built twice (argmax.go, argmax_other.go),
// and drift would silently give wrong initial indices.
func TestArgmaxA64ConstantsAreTheSharedBlock(t *testing.T) {
	c := ArgmaxConsts()
	if len(c) != 11 {
		t.Fatalf("ArgmaxConsts has %d floats, and the kernel reads bytes 0..43", len(c))
	}
	for i := 0; i < ArgmaxLanes; i++ {
		if got := math.Float32bits(c[i]); got != uint32(i) {
			t.Fatalf("index lane %d holds %d", i, got)
		}
	}
	if got := math.Float32bits(c[8]); got != uint32(ArgmaxLanes) {
		t.Fatalf("the per-iteration step is %d and the kernel consumes %d lanes", got, ArgmaxLanes)
	}
	if got := math.Float32bits(c[9]); got != 0x7FFFFFFF {
		t.Fatalf("the tie sentinel is %#08x, want 0x7FFFFFFF -- the fold is a SIGNED min", got)
	}
	if !math.IsInf(float64(c[10]), -1) {
		t.Fatalf("the initial maximum is %v, want -Inf", c[10])
	}
}

// TestQuantActArgmaxA64Disassembles checks the disassembler agrees on which
// instructions the bytes are (a wrong opcode on this ISA decodes to some other
// NEON instruction rather than faulting), and applies the register-discipline
// check: x28 is the goroutine pointer and corrupting it breaks the runtime far
// from the cause.
func TestQuantActArgmaxA64Disassembles(t *testing.T) {
	mc := lookAny("llvm-mc-18", "llvm-mc", "llvm-mc-17", "llvm-mc-19")
	if mc == "" {
		t.Skip("no llvm-mc to disassemble aarch64")
	}
	for _, k := range []struct {
		name string
		code []byte
		want []string
	}{
		{"quantact wide", emitQuantActA64(true),
			[]string{"fabs", "fmaxv", "fdiv", "fcmeq", "bic", "fmul", "fadd",
				"fcvtzs", "sqxtn", "addv", "scvtf", "cbnz", "ret"}},
		{"quantact narrow", emitQuantActNarrowA64(true),
			[]string{"fabs", "fmaxv", "fdiv", "fcmeq", "bic", "str\tq",
				"dup", "sqxtn", "addv", "scvtf", "cbnz", "ret"}},
		{"argmax", emitArgmaxA64(),
			[]string{"fcmgt", "bit", "add\tv", "fmaxv", "fcmeq", "smin", "sminv",
				"cbnz", "ret"}},
	} {
		text := a64Disasm(t, mc, k.code)
		// llvm-mc prints an undecodable word as a warning plus a .byte directive.
		for _, bad := range []string{"warning:", "error:", ".byte", "invalid"} {
			if strings.Contains(text, bad) {
				t.Errorf("%s: the disassembler rejected part of the stream (%q):\n%s",
					k.name, bad, text)
			}
		}
		for _, want := range k.want {
			if !strings.Contains(text, want) {
				t.Errorf("%s: the disassembler never saw %q:\n%s", k.name, want, text)
			}
		}
		for _, reg := range []string{"x18", "x27", "x28", "x29", "x30", "sp", "xzr", "wzr"} {
			if regexp.MustCompile(`\b` + reg + `\b`).MatchString(text) {
				t.Errorf("%s touches %s, which the runtime or the ABI owns:\n%s",
					k.name, reg, text)
			}
		}
		t.Logf("%s: %d instructions, clean round-trip", k.name, len(k.code)/4)
	}
}

// TestA64QuantActArgmaxEncodings pins the four instructions these kernels added
// to the assembler against clang, as a64_test.go does for the rest.
func TestA64QuantActArgmaxEncodings(t *testing.T) {
	as := lookAny("clang-18", "clang", "clang-17", "clang-19")
	od := lookAny("llvm-objdump-18", "llvm-objdump", "llvm-objdump-17", "llvm-objdump-19")
	if as == "" || od == "" {
		t.Skip("no clang + llvm-objdump to cross-assemble aarch64")
	}
	cases := []struct {
		name string
		emit func(*A64)
	}{
		{"fcmgt v9.4s, v7.4s, v0.4s", func(a *A64) { a.FCMGT4s(V9, V7, V0) }},
		{"fcmgt v31.4s, v30.4s, v29.4s", func(a *A64) { a.FCMGT4s(V31, V30, V29) }},
		{"fcmeq v12.4s, v0.4s, v11.4s", func(a *A64) { a.FCMEQ4s(V12, V0, V11) }},
		{"fcmeq v31.4s, v30.4s, v29.4s", func(a *A64) { a.FCMEQ4s(V31, V30, V29) }},
		{"smin v15.4s, v15.4s, v16.4s", func(a *A64) { a.SMIN4s(V15, V15, V16) }},
		{"smin v31.4s, v30.4s, v29.4s", func(a *A64) { a.SMIN4s(V31, V30, V29) }},
		{"sminv s15, v15.4s", func(a *A64) { a.SMINV(V15, V15) }},
		{"sminv s31, v30.4s", func(a *A64) { a.SMINV(V31, V30) }},
	}
	var src strings.Builder
	src.WriteString(".arch armv8.2-a+dotprod+fp16\n.text\n")
	for _, tc := range cases {
		src.WriteString(tc.name)
		src.WriteByte('\n')
	}
	dir := t.TempDir()
	sp, op := filepath.Join(dir, "a.s"), filepath.Join(dir, "a.o")
	if err := os.WriteFile(sp, []byte(src.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(as, "--target=aarch64-linux-gnu", "-c", sp, "-o", op).CombinedOutput(); err != nil {
		t.Skipf("%s cannot cross-assemble aarch64: %v\n%s", as, err, out)
	}
	out, err := exec.Command(od, "-d", op).CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", od, err, out)
	}
	// "       4: f9401807     \tldr\tx7, [x0, #0x30]"
	re := regexp.MustCompile(`(?m)^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s`)
	ms := re.FindAllStringSubmatch(string(out), -1)
	if len(ms) != len(cases) {
		t.Fatalf("assembled %d instructions, table has %d\n%s", len(ms), len(cases), out)
	}
	for i, m := range ms {
		var a A64
		cases[i].emit(&a)
		// objdump prints the 32-bit word; the file holds it little-endian.
		w := m[1]
		want := w[6:8] + w[4:6] + w[2:4] + w[0:2]
		if got := hex.EncodeToString(a.Bytes()); got != want {
			t.Errorf("%s: this assembler emits %s, clang says %s", cases[i].name, got, want)
		}
	}
	t.Logf("%d added encodings regenerated from %s and matched", len(ms), as)
}
