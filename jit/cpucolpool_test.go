package jit

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// colPoolOracle is main's compressor pool (engine/model/ds4.go ds4Compress,
// nn.ColPool32JIT) of the window ending at position p: for each column the
// softmax over the window's rows of their gates weighting their values; with
// overlap the Span before's first halves (where they exist), then the
// window's second halves. shift moves the window, swap reads the halves the
// other way round, flat ignores the gate.
func colPoolOracle(rows [][]float32, p, span, D, off, gate int, overlap bool, shift int, swap, flat bool) []float64 {
	type slot struct{ t, half int }
	var slots []slot
	p0 := p + 1 - span + shift
	h0, h1 := 0, 0
	if overlap {
		h0, h1 = 0, D
		if swap {
			h0, h1 = D, 0
		}
		if p0-span >= 0 {
			for j := range span {
				slots = append(slots, slot{p0 - span + j, h0})
			}
		}
	}
	for j := range span {
		slots = append(slots, slot{p0 + j, h1})
	}
	out := make([]float64, D)
	for d := range D {
		mx := math.Inf(-1)
		for _, s := range slots {
			if !flat {
				mx = math.Max(mx, float64(rows[s.t][gate+s.half+d]))
			}
		}
		var sum, acc float64
		for _, s := range slots {
			w := 1.0
			if !flat {
				w = math.Exp(float64(rows[s.t][gate+s.half+d]) - mx)
			}
			sum += w
			acc += w * float64(rows[s.t][off+s.half+d])
		}
		out[d] = acc / sum
	}
	return out
}

// TestComposedCPUColPool executes the composer's gated column pool
// (graph.OpColPool, DeepSeek V4's compressor) on the CPU over a paged
// binary16 history whose values are its keys, one row at a time, and its
// entry appended to the entry history (graph.EntHistory, f32) at the slot
// the row's descriptor names, held to main's pool: windows of four of one
// half (HCA's) and of two halves (CSA's), at a row closing its window and,
// for CSA, the first window, which has no window before it. A window one
// position early, the halves read the other way round and the gate ignored
// must each move it.
func TestComposedCPUColPool(t *testing.T) {
	for _, tc := range []struct {
		name    string
		p       int
		overlap bool
	}{{"hca", 11, false}, {"csa", 11, true}, {"csa-first", 3, true}, {"hca-open", 9, false}} {
		t.Run(tc.name, func(t *testing.T) { composedCPUColPool(t, tc.p, tc.overlap) })
	}
}

func composedCPUColPool(t *testing.T, pos int, overlap bool) {
	const page, span, D = 32, 4, 8
	halves := 1
	if overlap {
		halves = 2
	}
	W := 16 + 2*halves*D
	off, gate := 16, 16+halves*D
	b := graph.New()
	k := b.In("k", W)
	h := graph.StateRef{Name: "kv", Version: b.KVAppend(k, k, 0)}
	y := b.ColPool(h, graph.Attrs{HeadDim: D, Span: span, Offset: off, GateAt: gate, Width: W, Overlap: overlap})
	b.EntAppend(y, 0)
	b.Out(y)
	g := b.G
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := Default().Form(Request{Graph: g, History: History{ValuesInKeys: true},
		Step:   Step{Rows: 1, Page: page, Keys: 4 * page, ActWindow: 32},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	n := pos + 1
	rows := make([][]float32, n)
	kpool := make([]uint32, 4*page*W/2)
	for p := range n {
		rows[p] = make([]float32, W)
		for d := range W {
			rows[p][d] = float32(quant.DecodeHalf(quant.EncodeHalf(float32(rng.NormFloat64()))))
			if p == pos {
				continue
			}
			x, i := uint32(quant.EncodeHalf(rows[p][d])), p*W+d
			if i%2 == 0 {
				kpool[i/2] = kpool[i/2]&0xFFFF0000 | x
			} else {
				kpool[i/2] = kpool[i/2]&0xFFFF | x<<16
			}
		}
	}
	const slot = 5
	ent := make([]float32, 4*page*D)
	for i := range ent {
		ent[i] = float32(math.NaN())
	}
	got := make([]float32, D)
	bufs := map[string][]uint32{"k": f32Words(append([]float32(nil), rows[pos]...)), "rows": {0, 0, uint32(n), uint32(pos)},
		"pages": {0, 1, 2, 3}, "q.off": {0}, "kv.k": kpool, "ent.k": f32Words(ent), "ent.rows": {0, 0, slot + 1, slot},
		"ent.pages": {0, 1, 2, 3}, g.Values[y].Name: f32Words(got)}
	if err := runCPUCalls(codes, g, bufs); err != nil {
		t.Fatal(err)
	}
	nmse := func(want []float64, got []float32) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		return num / den
	}
	want := colPoolOracle(rows, pos, span, D, off, gate, overlap, 0, false, false)
	e, ee := nmse(want, got), nmse(want, ent[slot*D:(slot+1)*D])
	t.Logf("NMSE %.3e against main's pool, the entry %.3e", e, ee)
	if !(e < 1e-10) || !(ee < 1e-10) {
		t.Fatalf("the composed pool: NMSE %.3e, its entry %.3e", e, ee)
	}
	for i, v := range ent {
		if i/D != slot && !math.IsNaN(float64(v)) {
			t.Fatalf("entry %d written: only the row's slot %d is", i/D, slot)
		}
	}
	for _, f := range []struct {
		name        string
		shift       int
		swap, flat  bool
		needOverlap bool
	}{{"the window one position early", -1, false, false, false}, {"the halves the other way round", 0, true, false, true},
		{"the gate ignored", 0, false, true, false}} {
		if f.needOverlap && !overlap || f.shift < 0 && pos+1-span+f.shift < 0 {
			continue
		}
		e := nmse(colPoolOracle(rows, pos, span, D, off, gate, overlap, f.shift, f.swap, f.flat), got)
		t.Logf("violation %q: NMSE %.3e", f.name, e)
		if !(e > 1e-6) {
			t.Errorf("against main's pool with %s the NMSE is %.3e: the gate cannot see it", f.name, e)
		}
	}
}
