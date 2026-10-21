package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGemmVoltaGroupedMatchesPerExpert is the tensor-core half of a batched
// mixture on sm_70: NTok sorted columns in token blocks, each block reading the
// ONE expert pSel names for it. Two oracles, because they catch different
// faults:
//
//   - the DENSE GemmVolta run on that expert's own sheet, same tile, same
//     columns -- the same arithmetic in the same order, so the bar is BIT
//     equality. A kernel that reads the wrong expert, the wrong plane base or
//     the wrong token block fails here.
//   - a float64 reference over the GGUF dequantization, which the dense kernel
//     cannot be for a format whose decode is new (MXFP4's e2m1 on the f16 path):
//     a decode both kernels share would pass the first check wrong.
//
// A block past `used` is never launched and must stay poisoned, which pins the
// grid order (row block fastest) the tier relies on to launch only the blocks
// it filled. The second tile of each format carries an expert BIAS bank
// ([expert][row], gpt-oss's shape) added in the epilogue, held to the dense
// kernel with that expert's bias row. The activations are built the way the
// tier builds a gate's: ActF16TGather from source rows through a permutation
// with padding columns, held BYTE-equal to GatherRows followed by ActF16T.
func TestGemmVoltaGroupedMatchesPerExpert(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []mainkernels.Quant{mainkernels.Q4_K, mainkernels.MXFP4, mainkernels.Q6_K, mainkernels.Q8_0, mainkernels.Q5_0,
			mainkernels.Q5_K, mainkernels.Q4_0, mainkernels.Q5_1} {
			sub, _, _, _ := mainkernels.Layout(q)
			kb := max(32/sub, 1)
			for _, c := range []struct {
				rows, k, experts int
				tl               mainkernels.VoltaTile
				sel              []uint32 // one expert per token block
				used             int      // blocks launched; the rest must stay poisoned
				bias             bool
			}{
				{128, 512, 4, mainkernels.VoltaTile{MT: 4, NT: 1, WM: 1, WN: 4, KB: kb}, []uint32{2, 0, 0, 3, 1}, 5, false},
				{192, 256, 4, mainkernels.VoltaTile{MT: 2, NT: 1, WM: 1, WN: 4, KB: kb}, []uint32{3, 3, 1, 0}, 3, true},
				{256, 512, 3, mainkernels.VoltaTile{MT: 4, NT: 2, WM: 1, WN: 4, KB: kb}, []uint32{1, 2, 0}, 3, false},
				// gpt-oss's shape: 96-row blocks, whose 96 (or 192) staging
				// items do not nest in 128 threads.
				{192, 512, 4, mainkernels.VoltaTile{MT: 3, NT: 2, WM: 1, WN: 4, KB: kb}, []uint32{0, 3, 3, 1}, 4, true},
			} {
				s := mainkernels.MatVecShape{T: q, K: c.k, Rows: c.rows, NTok: c.tl.Toks() * len(c.sel), Experts: c.experts,
					Bias: c.bias}
				kk, err := mainkernels.GemmVolta(s, c.tl)
				if err != nil && c.tl.MT == 3 && kb > 1 {
					// 192 staging items do not nest in 128 threads; the tier
					// takes the next tile (voltaGroupTiles).
					continue
				}
				if err != nil {
					t.Fatalf("%v %+v: %v", q, c.tl, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					if d.API() == "ptx" {
						t.Fatalf("%s %v %+v: %v", d.API(), q, c.tl, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				groupedGemmCase(t, d, kern, s, voltaGrouped(c.tl), c.sel, c.used)
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

// groupedGemm is what groupedGemmCase needs of a grouped staged GEMM: its
// token block, its workgroup width, how many workgroups a launch of `used`
// token blocks takes, and the DENSE form of the same kernel (the bit-equality
// oracle) with its grid.
type groupedGemm struct {
	tile        string
	toks, width int
	groups      func(s mainkernels.MatVecShape, used int) int
	dense       func(s mainkernels.MatVecShape) (*ir.Kernel, error)
	denseGroups func(s mainkernels.MatVecShape) int
}

func voltaGrouped(tl mainkernels.VoltaTile) groupedGemm {
	return groupedGemm{fmt.Sprintf("%+v", tl), tl.Toks(), tl.Threads(),
		func(s mainkernels.MatVecShape, used int) int { return mainkernels.GemmVoltaGrouped(s, tl, used) },
		func(s mainkernels.MatVecShape) (*ir.Kernel, error) { return mainkernels.GemmVolta(s, tl) },
		func(s mainkernels.MatVecShape) int { return mainkernels.GemmVoltaGroups(s, tl) }}
}

func tileGrouped(tl mainkernels.TileGemm) groupedGemm {
	return groupedGemm{fmt.Sprintf("tile %+v", tl), tl.Toks(), tl.Threads(),
		func(s mainkernels.MatVecShape, used int) int { return mainkernels.GemmTileGrouped(s, tl, used) },
		func(s mainkernels.MatVecShape) (*ir.Kernel, error) { return mainkernels.GemmTile(s, tl) },
		func(s mainkernels.MatVecShape) int { return mainkernels.GemmTileGroups(s, tl) }}
}

func groupedGemmCase(t *testing.T, d backend.Device, kern backend.Kernel, s mainkernels.MatVecShape,
	gg groupedGemm, sel []uint32, used int) {
	t.Helper()
	name := fmt.Sprintf("%s %v %dx%d e%d %s used %d/%d bias %v", d.API(), s.T, s.Rows, s.K, s.Experts, gg.tile, used,
		len(sel), s.Bias)
	rng := rand.New(rand.NewSource(int64(s.Rows*31 + s.K + int(s.T))))
	g := newGPU(t, d)
	defer g.free()
	gt := map[mainkernels.Quant]quant.Type{mainkernels.MXFP4: quant.MXFP4}[s.T]
	if gt == 0 {
		gt = ggufOf[s.T]
	}
	type sheet struct {
		qs, dw, scw []uint32
		wf          []float32
	}
	sheets := make([]sheet, s.Experts)
	var qs, dw, scw []uint32
	for e := range sheets {
		raw := randWeights(s.T, s.Rows, s.K, rng)
		q1, d1, s1, err := mainkernels.PackWeights(s.T, raw, s.Rows, s.K)
		if err != nil {
			t.Fatal(err)
		}
		if len(s1) == 0 {
			s1 = []uint32{0}
		}
		wf := make([]float32, s.Rows*s.K)
		if err := quant.Dequant32(gt, raw, wf); err != nil {
			t.Fatal(err)
		}
		sheets[e] = sheet{q1, d1, s1, wf}
		// A sheet's scale plane is PackedWords' length, not padded: the base
		// the kernel adds is e*nsc, and a one-word stand-in for an empty plane
		// is only ever read at index 0 of a format with no plane.
		_, _, nsc, err := mainkernels.PackedWords(s.T, s.Rows, s.K)
		if err != nil {
			t.Fatal(err)
		}
		qs, dw = append(qs, q1...), append(dw, d1...)
		if nsc > 0 {
			scw = append(scw, s1...)
		}
	}
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	BN := gg.toks
	// The columns' activations: nsrc source rows, each column reading
	// perm[col], and every fifth column a padding column (perm = nsrc) that
	// must read zeros.
	nsrc := s.NTok/2 + 3
	srcRows := make([]float32, nsrc*s.K)
	for i := range srcRows {
		srcRows[i] = float32(rng.NormFloat64())
	}
	perm := make([]uint32, s.NTok)
	x := make([]float32, s.NTok*s.K)
	for col := range perm {
		perm[col] = uint32(rng.Intn(nsrc))
		if col%5 == 4 {
			perm[col] = uint32(nsrc)
			continue
		}
		copy(x[col*s.K:(col+1)*s.K], srcRows[int(perm[col])*s.K:])
	}
	n := s.Rows * s.NTok
	sub, _, _, _ := mainkernels.Layout(s.T)
	at := s.NTok * s.K / sub
	ak, err := mainkernels.ActF16T(s.T, s.NTok, s.K)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := d.Compile(ak)
	if err != nil {
		t.Fatal(err)
	}
	defer ac.Close()
	gk, err := mainkernels.ActF16TGather(s.T, s.NTok, s.K, nsrc)
	if err != nil {
		t.Fatal(err)
	}
	gc, err := d.Compile(gk)
	if err != nil {
		t.Fatal(err)
	}
	defer gc.Close()
	bB := g.up(make([]byte, s.NTok*s.K*2))
	bRef := g.up(make([]byte, s.NTok*s.K*2))
	if err := ac.Launch((at+127)/128, 128, g.up(f32bytes(x)), bRef); err != nil {
		t.Fatal(err)
	}
	if err := gc.Launch((at+127)/128, 128, g.up(f32bytes(srcRows)), g.up(u32bytes(perm)), bB); err != nil {
		t.Fatal(err)
	}
	rawG, rawR := make([]byte, s.NTok*s.K*2), make([]byte, s.NTok*s.K*2)
	if err := bB.Read(rawG); err != nil {
		t.Fatal(err)
	}
	if err := bRef.Read(rawR); err != nil {
		t.Fatal(err)
	}
	for i := range rawG {
		if rawG[i] != rawR[i] {
			t.Fatalf("%s: ActF16TGather byte %d (column %d, perm %d) is %#x, GatherRows+ActF16T %#x",
				name, i, i/(s.K*2), perm[i/(s.K*2)], rawG[i], rawR[i])
		}
	}
	// The bias bank, one row of s.Rows per expert.
	biasBank := make([]float32, s.Experts*s.Rows)
	for i := range biasBank {
		biasBank[i] = float32(rng.NormFloat64())
	}
	read := func(b backend.Buf) []float32 {
		raw := make([]byte, n*4)
		if err := b.Read(raw); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return v
	}
	got := g.up(f32bytes(nanFill(n)))
	args := []backend.Buf{g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw)), bB, got}
	if s.Bias {
		args = append(args, g.up(f32bytes(biasBank)))
	}
	if err := kern.Launch(gg.groups(s, used), gg.width,
		append(args, g.up(u32bytes(sel)))...); err != nil {
		t.Fatal(err)
	}
	gv := read(got)
	// The dense kernel on each expert's own sheet, over every column.
	dense := make([][]float32, s.Experts)
	ds := s
	ds.Experts = 0
	dk0, err := gg.dense(ds)
	if err != nil {
		t.Fatal(err)
	}
	dk, err := d.Compile(dk0)
	if err != nil {
		t.Fatal(err)
	}
	defer dk.Close()
	for e, sh := range sheets {
		out := g.up(f32bytes(nanFill(n)))
		dargs := []backend.Buf{g.up(u32bytes(sh.qs)), g.up(u32bytes(sh.dw)), g.up(u32bytes(sh.scw)), bB, out}
		if s.Bias {
			dargs = append(dargs, g.up(f32bytes(biasBank[e*s.Rows:(e+1)*s.Rows])))
		}
		if err := dk.Launch(gg.denseGroups(ds), gg.width, dargs...); err != nil {
			t.Fatal(err)
		}
		dense[e] = read(out)
	}
	var se, sy float64
	for tk := 0; tk < s.NTok; tk++ {
		blk := tk / BN
		for r := 0; r < s.Rows; r++ {
			i := tk*s.Rows + r
			v := gv[i]
			if blk >= used {
				if !math.IsNaN(float64(v)) {
					t.Fatalf("%s: column %d row %d is in block %d, past the %d launched, and was written (%v)",
						name, tk, r, blk, used, v)
				}
				continue
			}
			e := sel[blk]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: column %d row %d (expert %d) is %v -- never written", name, tk, r, e, v)
			}
			if w := dense[e][i]; v != w {
				t.Fatalf("%s: column %d row %d (block %d, expert %d): grouped %v, dense on the expert's sheet %v",
					name, tk, r, blk, e, v, w)
			}
			var ref float64
			if s.Bias {
				ref = float64(biasBank[int(e)*s.Rows+r])
			}
			wf := sheets[e].wf
			for k := 0; k < s.K; k++ {
				ref += float64(wf[r*s.K+k]) * float64(x[tk*s.K+k])
			}
			se += (float64(v) - ref) * (float64(v) - ref)
			sy += ref * ref
		}
	}
	nmse := se / sy
	t.Logf("%s: NMSE %.3e against the float64 reference, bit-equal to the dense kernel per expert", name, nmse)
	if !(nmse <= 1e-5) {
		t.Fatalf("%s: NMSE %.3e against the float64 reference", name, nmse)
	}
}
