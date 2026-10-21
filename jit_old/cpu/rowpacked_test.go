//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestPackedRowMatchesTheLayout dequantizes every row of an odd-row-count
// tensor in every packed format and compares it with oldkernels.RowPacked. The
// narrow formats pair rows in their d plane (even row low half, odd row high
// half), and thirteen rows add an unpaired last row. The scales vary per block
// (quant.PlantScales), since a constant plane hides reading the wrong one.
func TestPackedRowMatchesTheLayout(t *testing.T) {
	ran := 0
	for _, gt := range quant.PackedTypes {
		q, ok := kernels.QuantOf(gt)
		if !ok {
			t.Fatalf("%s has no device layout", gt)
		}
		c := onHost(t)(hostTable().PackedRow(q))
		k := 512
		if k%int(gt.BlockElems()) != 0 {
			k = int(gt.BlockElems()) * 2
		}
		const nrows = 13
		src := make([]byte, uint64(nrows*k)/gt.BlockElems()*gt.BlockBytes())
		rand.New(rand.NewSource(int64(gt))).Read(src)
		if !quant.PlantScales(gt, src, 0) {
			t.Fatalf("%s: no scale planter", gt)
		}
		qs32, d32, sc32, err := kernels.PackWeights(q, src, nrows, k)
		if err != nil {
			t.Fatal(err)
		}
		b := func(v []uint32) []byte {
			if len(v) == 0 {
				return nil
			}
			return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
		}
		qs, d, sc := b(qs32), b(d32), b(sc32)
		konst := cpu.PackedRowConsts(q)
		scratch := make([]byte, cpu.PackedRowScratch)
		perSuper, _ := kernels.ScaleLayout(q)
		sub, _, _, _ := kernels.Layout(q)
		for r := 0; r < nrows; r++ {
			want := make([]float32, k)
			if err := oracle.RowPacked(want, q, qs, d, sc, r, nrows, k); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, k+8)
			for i := k; i < len(got); i++ {
				got[i] = -3
			}
			di, slot := kernels.DIndex(q, 0, nrows, r)
			pd := di*4 + slot*cpu.DRowBytes(gt)
			args := cpu.Args{
				Out: &got[0], W: &qs[r*4], RowStr: int64(nrows * 4),
				PD: &d[pd], DStr: int64(cpu.DSuperBytes(gt, nrows)),
				K:   int64(k / (sub * perSuper)),
				Scr: &konst[0], Scratch: &scratch[0],
			}
			if len(sc) > 0 {
				args.PSC = &sc[r*4]
			}
			c.Call(&args)
			for i := 0; i < k; i++ {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-6*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%s row %d element %d: %v, want %v", gt, r, i, got[i], want[i])
				}
			}
			for i := k; i < len(got); i++ {
				if got[i] != -3 {
					t.Fatalf("%s row %d: wrote element %d past k", gt, r, i)
				}
			}
			ran++
		}
		c.Close()
	}
	t.Logf("%d rows over %d formats, each equal to the layout's own reading", ran, len(quant.PackedTypes))
}

// TestWidenEveryLength converts F16 and BF16 at lengths that exercise the
// vector loop, the single-element loop and both, with a guard after dst.
func TestWidenEveryLength(t *testing.T) {
	for _, bf := range []bool{false, true} {
		c := onHost(t)(hostTable().Widen(bf))
		lanes := cpu.ElemLanes
		for _, n := range []int{1, 3, lanes, lanes + 1, 3*lanes + 2, 40, 1023} {
			src := make([]uint16, n)
			want := make([]float32, n)
			rng := rand.New(rand.NewSource(int64(n)))
			for i := range src {
				v := float32(rng.NormFloat64() * 10)
				if bf {
					src[i] = uint16(math.Float32bits(v) >> 16)
					want[i] = math.Float32frombits(uint32(src[i]) << 16)
				} else {
					src[i] = quant.EncodeHalf(v)
					want[i] = float32(quant.DecodeHalf(src[i]))
				}
			}
			got := make([]float32, n+4)
			got[n] = -9
			c.Call(&cpu.Args{Out: &got[0], W: (*byte)(unsafe.Pointer(&src[0])),
				K: int64(n / lanes), Rows: int64(n % lanes)})
			for i := 0; i < n; i++ {
				if got[i] != want[i] {
					t.Fatalf("bf16=%v n=%d element %d: %v, want %v", bf, n, i, got[i], want[i])
				}
			}
			if got[n] != -9 {
				t.Fatalf("bf16=%v n=%d: wrote past the end", bf, n)
			}
		}
		c.Close()
	}
}
