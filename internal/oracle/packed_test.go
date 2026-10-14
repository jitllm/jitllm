package oracle_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The host reader is gated against the GGUF dequantizer, not the device
// kernel: agreeing with quant.Dequant32 (the source layout) on every format
// proves the pack is a pure permutation, where two readers of the same layout
// agreeing would prove nothing. The formats come from quant.PackedTypes so a
// new one cannot be missed.
var packedRefFormats = func() (fs []struct {
	name string
	g    quant.Type
	k    kernels.Quant
}) {
	for _, g := range quant.PackedTypes {
		k, ok := kernels.QuantOf(g)
		if !ok {
			panic(g.String() + " is in quant.PackedTypes and has no device layout")
		}
		fs = append(fs, struct {
			name string
			g    quant.Type
			k    kernels.Quant
		}{g.String(), g, k})
	}
	return fs
}()

// plant gives every block finite scales; random bytes are not a valid block.
func plant(t *testing.T, g quant.Type, src []byte, seed int) {
	t.Helper()
	if !quant.PlantScales(g, src, seed) {
		t.Fatalf("%s: no scale planter -- add it rather than testing noise", g)
	}
}

func u32b(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func TestPackedRefMatchesDequant(t *testing.T) {
	// Row counts that are not multiples of eight on purpose: a row-stride bug
	// is invisible at a count the tile divides. 300 also crosses PackWeights'
	// 256-row fan-out, exercising both packers.
	for _, f := range packedRefFormats {
		for _, nrows := range []int{9, 300} {
			t.Run(f.name, func(t *testing.T) {
				const k = 512
				rng := rand.New(rand.NewSource(1))
				nb := uint64(nrows*k) / f.g.BlockElems()
				bb := f.g.BlockBytes()
				src := make([]byte, nb*bb)
				for i := range src {
					src[i] = byte(rng.Intn(256))
				}
				// Random bytes are not a valid block: a random f16 scale can be
				// Inf or NaN, which poisons both arms. Only the scale fields
				// are constrained; every payload bit stays random.
				plant(t, f.g, src, nrows)
				qs, d, sc, err := kernels.PackWeights(f.k, src, nrows, k)
				if err != nil {
					t.Fatalf("pack: %v", err)
				}
				x := make([]float32, k)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				got := make([]float32, nrows)
				if err := oracle.MatVecPacked(got, f.k,
					u32b(qs), u32b(d), u32b(sc),
					x, nrows, k); err != nil {
					t.Fatalf("MatVecPacked: %v", err)
				}

				w := make([]float32, nrows*k)
				if err := quant.Dequant32(f.g, src, w); err != nil {
					t.Fatalf("dequant: %v", err)
				}
				var num, den float64
				for r := 0; r < nrows; r++ {
					var want float64
					for j := 0; j < k; j++ {
						want += float64(w[r*k+j]) * float64(x[j])
					}
					dv := float64(got[r]) - want
					num += dv * dv
					den += want * want
				}
				nmse := num / den
				if !(nmse < 1e-12) {
					t.Fatalf("%s nrows=%d: NMSE %.3e against the GGUF dequantizer -- "+
						"the device layout and the source layout disagree", f.name, nrows, nmse)
				}
				if math.IsNaN(nmse) || den == 0 {
					t.Fatalf("%s: degenerate oracle (den=%v) -- this gate proved nothing", f.name, den)
				}
				t.Logf("%s nrows=%d: NMSE %.2e", f.name, nrows, nmse)
			})
		}
	}
}

// The row reader is gated the same way: a wrong embedding row is fluent wrong
// text. Every row is checked, since a bug in the `si*words*nrows + r` stride is
// invisible on row 0.
func TestRowPackedMatchesDequant(t *testing.T) {
	for _, f := range packedRefFormats {
		for _, nrows := range []int{9, 300} {
			t.Run(f.name, func(t *testing.T) {
				const k = 512
				rng := rand.New(rand.NewSource(2))
				nb := uint64(nrows*k) / f.g.BlockElems()
				bb := f.g.BlockBytes()
				src := make([]byte, nb*bb)
				for i := range src {
					src[i] = byte(rng.Intn(256))
				}
				plant(t, f.g, src, nrows)
				qs, d, sc, err := kernels.PackWeights(f.k, src, nrows, k)
				if err != nil {
					t.Fatalf("pack: %v", err)
				}
				w := make([]float32, nrows*k)
				if err := quant.Dequant32(f.g, src, w); err != nil {
					t.Fatalf("dequant: %v", err)
				}
				got := make([]float32, k)
				var num, den float64
				for r := 0; r < nrows; r++ {
					for i := range got {
						got[i] = float32(math.NaN()) // RULE 13: poison, do not trust the allocator
					}
					if err := oracle.RowPacked(got, f.k, u32b(qs), u32b(d), u32b(sc), r, nrows, k); err != nil {
						t.Fatalf("RowPacked row %d: %v", r, err)
					}
					for j := 0; j < k; j++ {
						want := float64(w[r*k+j])
						dv := float64(got[j]) - want
						num += dv * dv
						den += want * want
					}
				}
				nmse := num / den
				if math.IsNaN(nmse) || den == 0 {
					t.Fatalf("%s: degenerate oracle (den=%v) -- this gate proved nothing", f.name, den)
				}
				if !(nmse < 1e-12) {
					t.Fatalf("%s nrows=%d: NMSE %.3e -- RowPacked and the GGUF dequantizer disagree",
						f.name, nrows, nmse)
				}
				// A row out of range is an error, not a silent read of somebody
				// else's weights: the token id comes from a sampler.
				if err := oracle.RowPacked(got, f.k, u32b(qs), u32b(d), u32b(sc), nrows, nrows, k); err == nil {
					t.Fatalf("%s: row %d of %d was accepted", f.name, nrows, nrows)
				}
				t.Logf("%s nrows=%d: %d rows, NMSE %.2e", f.name, nrows, nrows, nmse)
			})
		}
	}
}

// The payload is asserted at its native width: correctness cannot see a
// widened payload, and at the memory wall the byte count is the run time.
func TestPackedPayloadIsNativeWidth(t *testing.T) {
	// The native width of each format, which is a fact about the format and
	// so is written here; the sweep is quant.PackedTypes, and a format missing
	// from this map fails rather than going unchecked.
	native := map[quant.Type]int{
		quant.Q4_0: 4, quant.Q5_0: 5, quant.Q5_1: 5, quant.Q8_0: 8, quant.Q4_K: 4,
		quant.Q3_K:    4, // 3 bits would need a 2-bit primary plane
		quant.Q5_K:    5, // 4-bit primary + 1-bit secondary, like Q6_K's 4+2
		quant.Q6_K:    6,
		quant.MXFP4:   4,
		quant.TQ2_0:   2, // the 2-bit plane: four fields a byte (kernels.Fields)
		quant.TQ2_F32: 2,
	}
	for _, f := range packedRefFormats {
		bits, ok := native[f.g]
		if !ok {
			t.Errorf("%s: no native width recorded here", f.name)
			continue
		}
		const nrows, k = 8, 512
		nq, _, _, err := kernels.PackedWords(f.k, nrows, k)
		if err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		got := nq * 32 / (nrows * k) // payload bits per weight
		if got != bits {
			t.Errorf("%s: payload is %d bits per weight, want %d", f.name, got, bits)
		}
		// And against the source, so the expansion is a number rather than a
		// claim: the GGUF block plus what the packer adds for scales.
		src := nrows * k / int(f.g.BlockElems()) * int(f.g.BlockBytes())
		_, nd, nsc, _ := kernels.PackedWords(f.k, nrows, k)
		tot := (nq + nd + nsc) * 4
		t.Logf("%-5s %d bits/weight, %d bytes against the GGUF's %d (%+.1f%%)",
			f.name, got, tot, src, 100*(float64(tot)/float64(src)-1))
	}
}

// TestMXFP4ScaleTheF32CannotHoldIsRefused: the d plane stores MXFP4's E8M0
// exponent natively, so the only exponents left to refuse are the ones an f32
// cannot express as a normal scale -- unless every code in the block is a zero,
// where 0 is exact at any scale. e = 103 and e = 144, outside an f16's exact
// powers of two, are ordinary scales.
func TestMXFP4ScaleTheF32CannotHoldIsRefused(t *testing.T) {
	const nrows, k = 2, 32
	blk := func(e, code byte) []byte {
		b := make([]byte, 17)
		b[0] = e
		for i := 1; i < 17; i++ {
			b[i] = code
		}
		return b
	}
	for _, c := range []struct {
		e, code byte
		ok      bool
	}{
		{2, 0x11, true}, {254, 0x11, true}, // 2^-126 and 2^126: the f32 ends
		{103, 0x11, true}, {144, 0x11, true}, // refused by the f16 plane, fine now
		{1, 0x11, false},                     // 2^-127: e8m0Store would make it 0.0
		{0, 0x11, false}, {255, 0x11, false}, // 2^-128 and E8M0's NaN
		{0, 0x88, true}, {255, 0x80, true}, // +-0 codes: exact at any scale
		{0, 0x10, false},
	} {
		src := append(blk(127, 0x21), blk(c.e, c.code)...)
		_, _, _, err := kernels.PackWeights(kernels.MXFP4, src, nrows, k)
		if (err == nil) != c.ok {
			t.Errorf("e=%d codes %#02x: err=%v, want ok=%v", c.e, c.code, err, c.ok)
		}
	}
}
