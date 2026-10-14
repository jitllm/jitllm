//go:build arm64

package cpu

import (
	"encoding/binary"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The kernels without FEAT_DotProd (sdotemu.go). Two gates beside the package's
// own: every SDOT-emitting family emits none once the probe says the chip lacks
// it, and the widened decode kernels give the SDOT kernels' bits, against the
// oracle, at ragged k and row counts with a guard after the output. The whole
// suite runs widened too, with JITLLM_FORCE_NO_DOTPROD=1 or under
// `qemu-aarch64-static -cpu cortex-a72` (sdotemu_env_test.go).

// withDot sets, for the rest of t, whether the emitters widen SDOT: absent
// forces the probe off, and present forces SDOT's encoding, which only a chip
// with the feature may then run (chipHasDot).
func withDot(t *testing.T, absent bool) {
	t.Helper()
	oldN, oldE := forceNoDotProd, forceDotEncoding
	t.Cleanup(func() { forceNoDotProd, forceDotEncoding = oldN, oldE })
	forceNoDotProd, forceDotEncoding = absent, !absent
	if absent != a64EmulateDot() {
		t.Fatalf("forced FEAT_DotProd absent=%v and the emitters disagree", absent)
	}
}

// chipHasDot is the probe with no override: whether SDOT may be executed.
func chipHasDot() bool {
	old := forceNoDotProd
	defer func() { forceNoDotProd = old }()
	forceNoDotProd = false
	return hasDotProd()
}

// sdotWords counts SDOT and SDOTelem encodings (any register, any Q) in code.
func sdotWords(code []byte) int {
	n := 0
	for i := 0; i+4 <= len(code); i += 4 {
		w := binary.LittleEndian.Uint32(code[i:])
		if w&0xBFE0FC00 == 0x0E809400 || w&0xBFC0F400 == 0x0F80E000 {
			n++
		}
	}
	return n
}

// dotFamilies emits every kernel family that uses SDOT, at shapes the engine
// asks for, keyed by a name.
func dotFamilies(t *testing.T) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	add := func(name string, code []byte, err error) {
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		out[name] = code
	}
	for _, w := range a64KernelTypes {
		for _, win := range []int16{0, 256} {
			code, err := EmitA64(Spec{W: w, Rows: 1, Accs: 1, Cols: 1, ActWin: win})
			add("rowmajor/"+w.String()+"/win"+itoa(int(win)), code, err)
		}
		mr, nr := 2, 1
		code, err := EmitGEMMWindow(w, mr, nr, 8, 32)
		if w == quant.Q5_0 {
			continue // no row-major GEMM for Q5_0
		}
		add("gemm/"+w.String(), code, err)
	}
	for _, w := range quant.PackedTypes {
		code, err := EmitA64PackedMatVecPF(w, PackedRows, 0)
		add("packed/"+w.String(), code, err)
		code, err = EmitA64PackedMatVecFused(w, 16, 0)
		add("fused/"+w.String(), code, err)
		code, err = EmitA64PackedMatVecFusedWin(w, 16, 0, 256)
		add("fusedwin/"+w.String(), code, err)
		q, _ := kernels.QuantOf(w)
		k := PackedOuterElems(w) * 2
		group := packedFusedGroupK(q)
		for tok := 1; tok <= MaxTiledTokensA64(w); tok++ {
			code, err = EmitA64PackedMatMulTiled(w, k, group*2, tok)
			add("tiled/"+w.String()+"/tok"+itoa(tok), code, err)
		}
		for _, win := range []int{32, 256} {
			code, err = EmitA64PackedMatMulStationary(w, k, 16, win)
			if err != nil {
				continue // a format with no stationary form is declined by name
			}
			add("stationary/"+w.String()+"/win"+itoa(win), code, err)
		}
	}
	return out
}

// TestNoDotProdKernelsEmitNoSDOT: with the feature present every family uses
// SDOT (so the scan can see it), and with it absent none does and the widening
// counter moved. A family that forgot DotScratch fails here.
func TestNoDotProdKernelsEmitNoSDOT(t *testing.T) {
	withDot(t, false)
	with := dotFamilies(t)
	for name, code := range with {
		if sdotWords(code) == 0 {
			t.Errorf("%s: no SDOT with the feature present -- the scan sees nothing", name)
		}
	}
	withDot(t, true)
	before := DotEmulated()
	without := dotFamilies(t)
	if len(without) != len(with) {
		t.Fatalf("%d families without the feature, %d with", len(without), len(with))
	}
	for name, code := range without {
		if n := sdotWords(code); n != 0 {
			t.Errorf("%s: %d SDOTs on a chip without FEAT_DotProd", name, n)
		}
	}
	if DotEmulated() == before {
		t.Fatal("no SDOT was widened -- the forced path emitted nothing it claims to")
	}
	t.Logf("%d kernel families, %d SDOTs widened", len(without), DotEmulated()-before)
}

// TestNoDotProdMatchesDotBitForBit runs the container decode kernels (the tiled
// matvec, the fused one, and the fused one's integer fold) for every packed format with and without
// FEAT_DotProd: the widened sequence computes SDOT's int32 lanes exactly, so
// the outputs are the same bits, and both are held to the oracle. k sweeps one
// to three outer steps and the rows one to three tiles, with a guard after the
// output that neither kernel may write.
func TestNoDotProdMatchesDotBitForBit(t *testing.T) {
	// On a chip without the feature the SDOT arm cannot run; the widened arm is
	// still held to the oracle and the guard.
	both := chipHasDot()
	if !both {
		t.Log("no FEAT_DotProd on this chip: the widened arm alone, against the oracle")
	}
	ran := 0
	for _, f := range packedFormats {
		// The tiled kernel, the fused one, and the fused one at a window that
		// takes the integer fold, where the format has it.
		for arm := 0; arm < 3; arm++ {
			if arm == 2 && !StationaryIntAccA64(f.g, 256) {
				continue
			}
			// A k the packer takes: whole source blocks as well as whole steps.
			step := max(PackedOuterElems(f.g), int(f.g.BlockElems()))
			rows := PackedRows
			if arm > 0 {
				rows = 16
			}
			for ks := 1; ks <= 3; ks++ {
				for tiles := 1; tiles <= 3; tiles++ {
					k, nrows := ks*step, tiles*rows
					name := []string{"", "fused/", "fusedwin/"}[arm] + f.name + "/k" + itoa(k) + "/rows" + itoa(nrows)
					t.Run(name, func(t *testing.T) {
						emit := func(absent bool) []byte {
							withDot(t, absent)
							var code []byte
							var err error
							switch arm {
							case 2:
								code, err = EmitA64PackedMatVecFusedWin(f.g, rows, 0, 256)
							case 1:
								code, err = EmitA64PackedMatVecFused(f.g, rows, 0)
							default:
								code, err = EmitA64PackedMatVecPF(f.g, rows, 0)
							}
							if err != nil {
								t.Fatal(err)
							}
							return code
						}
						dot, wide := emit(false), emit(true)
						if sdotWords(wide) != 0 || sdotWords(dot) == 0 {
							t.Fatalf("the arms are not what they claim: %d SDOTs widened, %d plain",
								sdotWords(wide), sdotWords(dot))
						}
						gotW := runPackedGuarded(t, wide, f.g, f.k, rows, nrows, k)
						got := gotW
						if both {
							got = runPackedGuarded(t, dot, f.g, f.k, rows, nrows, k)
						}
						for r := range got.out {
							if math.Float32bits(got.out[r]) != math.Float32bits(gotW.out[r]) {
								t.Fatalf("row %d: SDOT %v, widened %v -- not the same bits", r, got.out[r], gotW.out[r])
							}
						}
						if arm == 2 {
							// The integer fold reads the stationary GEMM's
							// padded activation streams (GEMMPad), which this
							// harness does not lay out, so the oracle is not its
							// reference here; the bits above still hold the
							// widening to SDOT on the same bytes, and nn's gates
							// hold the fold to the oracle.
							ran++
							return
						}
						var num, den float64
						for r := range gotW.out {
							dv := float64(gotW.out[r]) - float64(gotW.ref[r])
							num += dv * dv
							den += float64(gotW.ref[r]) * float64(gotW.ref[r])
						}
						if den == 0 || math.IsNaN(num) || math.IsNaN(den) {
							t.Fatalf("degenerate oracle (num=%v den=%v) -- this gate proved nothing", num, den)
						}
						// The bound is the int8 activation's error, which a k of one
						// block (32) does not average down as k=512 does: Q5_1 at
						// k=32 reads 1.06e-3 on both arms. The bit comparison
						// above is the kernel gate; this one holds the pair to
						// the arithmetic.
						if nmse := num / den; nmse > 4e-3 {
							t.Fatalf("widened against the oracle: NMSE %.3e", nmse)
						}
						ran++
					})
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no case ran")
	}
}

type packedRun struct{ out, ref []float32 }

// runPackedGuarded runs a container decode kernel over nrows rows of k with a
// NaN-poisoned output and a guard after it, and returns the output and the
// oracle's. The output accumulates (out[r] +=), so it starts at zero.
func runPackedGuarded(t *testing.T, code []byte, g quant.Type, kq kernels.Quant, rows, nrows, k int) packedRun {
	t.Helper()
	kern := mustMap(t, code)
	defer kern.Close()
	rng := rand.New(rand.NewSource(int64(nrows*131 + k)))
	nb := uint64(nrows*k) / g.BlockElems()
	src := make([]byte, nb*g.BlockBytes())
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	if !quant.PlantScales(g, src, 0) {
		t.Fatalf("%s: no scale planter", g)
	}
	qs, d, sc, err := kernels.PackWeights(kq, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q8 := make([]int8, k)
	pairs := make([]float32, 2*(k/Q8Block))
	if err := oracle.QuantizeQ8(q8, pairs, x, BiasC(g)); err != nil {
		t.Fatal(err)
	}
	half := make([]float32, k/16)
	oracle.QuantizeHalfSums(half, q8, BiasC(g))
	scr := make([]byte, PackedScratchBytes)
	if err := PackedScratch(scr); err != nil {
		t.Fatal(err)
	}
	const guard = 16
	buf := make([]float32, nrows+guard)
	for i := nrows; i < len(buf); i++ {
		buf[i] = float32(math.NaN())
	}
	qb, db, scb := u32b(qs), u32b(d), u32b(sc)
	args := Args{
		Out: &buf[0], W: &qb[0], A: &q8[0], AScale: &pairs[0],
		Rows: int64(nrows / rows), RowStr: int64(nrows * 4),
		K: int64(k / PackedOuterElems(g)), Scr: &scr[0], PD: &db[0], AHalf: &half[0],
		DStr: int64(DSuperBytes(g, nrows)),
	}
	if len(scb) > 0 {
		args.PSC = &scb[0]
	}
	kern.Call(&args)
	for i := nrows; i < len(buf); i++ {
		if !math.IsNaN(float64(buf[i])) {
			t.Fatalf("guard %d after the output was written: %v", i-nrows, buf[i])
		}
	}
	ref := make([]float32, nrows)
	if err := oracle.MatVecPacked(ref, kq, qb, db, scb, x, nrows, k); err != nil {
		t.Fatal(err)
	}
	return packedRun{out: buf[:nrows], ref: ref}
}

// TestNoDotProdChainCapIsTheOverflowBound runs a chain of DotChain links on the
// worst case -- every weight byte -wmax, every activation -127, so every
// product is +wmax*127 and nothing cancels -- and holds its four lanes to the
// int32 sum at DotChainCap(wmax) links, then shows the gate can fire: one link
// past the cap wraps the int16 pair and the lanes are wrong. A cap that is too
// generous fails the first half; a gate that could not see an overflow fails
// the second.
func TestNoDotProdChainCapIsTheOverflowBound(t *testing.T) {
	withDot(t, true)
	run := func(wmax, links int) (got, want [4]int32) {
		var a A64
		a.DotScratch(V3, V4)
		a.LDRx(X1, X0, 0)  // Out, four int32
		a.LDRx(X2, X0, 8)  // W
		a.LDRx(X3, X0, 16) // A
		a.MOVIzero(V0)
		for i := 0; i < links; i++ {
			a.LDRq(V1, X2, int32(16*i))
			a.LDRq(V2, X3, int32(16*i))
			a.DotChain(V0, V1, V2, i == links-1)
		}
		a.STRq(V0, X1, 0)
		a.RET()
		kern := mustMap(t, a.Bytes())
		defer kern.Close()
		w := make([]int8, 16*links)
		x := make([]int8, 16*links)
		for i := range w {
			w[i], x[i] = int8(-wmax), -127
			want[i%16/4] += int32(-wmax) * -127
		}
		out := make([]int32, 4+4)
		for i := 4; i < len(out); i++ {
			out[i] = 0x7fc00000 // a guard the kernel must not write
		}
		args := Args{
			Out: (*float32)(unsafe.Pointer(&out[0])),
			W:   (*byte)(unsafe.Pointer(&w[0])),
			A:   &x[0],
		}
		kern.Call(&args)
		runtime.KeepAlive(w)
		runtime.KeepAlive(x)
		for i := 4; i < len(out); i++ {
			if out[i] != 0x7fc00000 {
				t.Fatalf("wmax %d: the kernel wrote past its four lanes", wmax)
			}
		}
		copy(got[:], out[:4])
		return got, want
	}
	for _, wmax := range []int{7, 12, 15, 31, 63, 127, 128} {
		n := DotChainCap(wmax)
		if got, want := run(wmax, n); got != want {
			t.Errorf("wmax %d at the cap (%d links): %v, want %v", wmax, n, got, want)
		}
		if got, want := run(wmax, n+1); got == want {
			t.Errorf("wmax %d one link past the cap (%d): %v is still exact -- the cap is not the bound", wmax, n+1, got)
		}
		t.Logf("wmax %3d: cap %2d links exact, %2d wraps", wmax, n, n+1)
	}
}

// TestNoDotProdChainsTheDecodeDots counts, for every packed format's decode
// kernels, the instructions the widened kernel spends beyond the SDOT kernel,
// per SDOT the SDOT kernel issues. The counts are deterministic (the payload
// loop is unrolled), so the bound is a property of the emitted code, not a
// timing: a nibble or merged-plane format whose two dots a word share one
// ADDP and one SADALP (DotChain, broadcasts hoisted) costs about two more a
// dot, an 8-bit one three, and only the secondary plane's two-pass loop keeps
// the per-SDOT widening (DUP, SMULL, SMULL2, ADDP, SADALP: four more).
func TestNoDotProdChainsTheDecodeDots(t *testing.T) {
	type arm struct {
		name string
		emit func(quant.Type) ([]byte, error)
	}
	arms := []arm{
		{"tiled", func(w quant.Type) ([]byte, error) { return EmitA64PackedMatVecPF(w, PackedRows, 0) }},
		{"fused", func(w quant.Type) ([]byte, error) { return EmitA64PackedMatVecFused(w, 16, 0) }},
		{"fusedwin", func(w quant.Type) ([]byte, error) { return EmitA64PackedMatVecFusedWin(w, 16, 0, 256) }},
	}
	for _, w := range quant.PackedTypes {
		q, _ := kernels.QuantOf(w)
		sub, bits, _, _ := kernels.Layout(q)
		hi := kernels.HiPlane(q)
		bound := 4.0
		switch {
		case bits == 4 && (hi == 0 || sub*hi/32 == 1):
			bound = 2.5
		case bits == 8:
			bound = 3.5
		}
		for _, r := range arms {
			withDot(t, false)
			dot, err := r.emit(w)
			if err != nil {
				t.Fatal(err)
			}
			withDot(t, true)
			wide, err := r.emit(w)
			if err != nil {
				t.Fatal(err)
			}
			n := sdotWords(dot)
			extra := float64(len(wide)-len(dot)) / 4 / float64(n)
			t.Logf("%-8s %-8s %6d words with SDOT, %6d widened: %5d SDOTs, %+.2f instructions each",
				w, r.name, len(dot)/4, len(wide)/4, n, extra)
			if extra > bound {
				t.Errorf("%s %s: %.2f more instructions a dot, want at most %.1f", w, r.name, extra, bound)
			}
		}
	}
}
