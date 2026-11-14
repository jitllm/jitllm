package composer

import (
	le "encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestComposedCPUAttendQ8 executes the composer's attention over q8 pools
// (Pool.Row) on the CPU: a paged history of q8 head rows as oracle.QuantizeQ8
// writes them, the row's own key and value appended through the composed
// quantizer, held to the definition over every row widened (code times its
// block's scale). Head widths below a block and between blocks pad the last
// block with zeros. Two pages swapped, and a row read from a history held at
// f32, must fail it.
func TestComposedCPUAttendQ8(t *testing.T) {
	for _, sh := range []struct{ heads, kvh, dim int }{{6, 6, 48}, {32, 8, 64}, {16, 8, 128}, {8, 2, 64}, {15, 5, 64},
		{4, 2, 2}, {3, 1, 6}, {4, 2, 40}} {
		t.Run(fmt.Sprintf("%dq%dkv%dd", sh.heads, sh.kvh, sh.dim), func(t *testing.T) {
			composedCPUAttendQ8(t, hostArm(), sh.heads, sh.kvh, sh.dim, 0)
		})
	}
}

// composedCPUAttendQ8 is the gate over pools of q8 rows, or of f's blocks
// where it is set.
func composedCPUAttendQ8(t *testing.T, cpuOn cpuArm, heads, kvh, dim int, f graph.BlockFmt) {
	const page, pos = 32, 70
	r := Default().WithoutHalves(DeviceCPU)
	g := attendGraph(heads, kvh, dim)
	pool := Pool{Row: dim, Block: f}
	_, codes, err := r.Form(Request{Graph: g,
		Step:    Step{Rows: 1, Page: page, Keys: 4 * page, ActWindow: 32},
		History: History{Keys: pool, Values: pool},
		Device:  Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	w := kvh * dim
	rw := graph.Q8RowWords(dim)
	if f != 0 {
		rw = graph.BlockRowWords(f, dim)
	}
	nb := (dim + 31) / 32
	// quantized is a head row as the history holds it, and as its d*q.
	quantized := func(row []float32) ([]uint32, []float64) {
		pad := make([]float32, nb*32)
		copy(pad, row)
		if f != 0 {
			b, d := make([]byte, nb*f.Bytes()), make([]float32, len(pad))
			if err := oracle.QuantizeBlocks(b, pad, f); err != nil {
				t.Fatal(err)
			}
			if err := oracle.DequantBlocks(d, b, f); err != nil {
				t.Fatal(err)
			}
			words := oracle.BlockRow(b, f, dim)
			deq := make([]float64, len(row))
			for i := range row {
				deq[i] = float64(d[i])
			}
			return words, deq
		}
		q, pairs := make([]int8, len(pad)), make([]float32, 2*nb)
		if err := oracle.QuantizeQ8(q, pairs, pad, 128); err != nil {
			t.Fatal(err)
		}
		b := make([]byte, 4*rw)
		for i, c := range q {
			b[i] = byte(c)
		}
		for i, p := range pairs {
			le.LittleEndian.PutUint32(b[len(q)+4*i:], math.Float32bits(p))
		}
		words := make([]uint32, rw)
		for i := range words {
			words[i] = le.LittleEndian.Uint32(b[4*i:])
		}
		deq := make([]float64, len(row))
		for i := range row {
			deq[i] = float64(pairs[2*(i/32)] * float32(q[i]))
		}
		return words, deq
	}
	kpool, vpool := make([]uint32, 4*page*kvh*rw), make([]uint32, 4*page*kvh*rw)
	ks, vs := make([][]float64, pos+1), make([][]float64, pos+1)
	put := func(pool []uint32, p int, row []float32) []float64 {
		var deq []float64
		for h := range kvh {
			words, d := quantized(row[h*dim : (h+1)*dim])
			copy(pool[((p/page*page+p%page)*kvh+h)*rw:], words)
			deq = append(deq, d...)
		}
		return deq
	}
	randRow := func() []float32 {
		x := make([]float32, w)
		for d := range x {
			x[d] = float32(rng.NormFloat64())
		}
		return x
	}
	for p := range pos {
		ks[p], vs[p] = put(kpool, p, randRow()), put(vpool, p, randRow())
	}
	q, k, v := make([]float32, heads*dim), randRow(), randRow()
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	scratch := make([]uint32, len(kpool))
	ks[pos], vs[pos] = put(scratch, pos, k), put(scratch, pos, v)
	want := make([]float64, heads*dim)
	for h := range heads {
		kv := h / (heads / kvh)
		sc := make([]float64, pos+1)
		mx := math.Inf(-1)
		for p := range pos + 1 {
			var s float64
			for d := range dim {
				s += float64(q[h*dim+d]) * ks[p][kv*dim+d]
			}
			sc[p] = s * 0.125
			mx = math.Max(mx, sc[p])
		}
		var sum float64
		for p := range sc {
			sc[p] = math.Exp(sc[p] - mx)
			sum += sc[p]
		}
		for d := range dim {
			var o float64
			for p := range sc {
				o += sc[p] / sum * vs[p][kv*dim+d]
			}
			want[h*dim+d] = o
		}
	}
	run := func(pages []uint32, kp, vp []uint32) []float32 {
		out := make([]float32, heads*dim)
		bufs := map[string][]uint32{"q": f32Words(q), "k": f32Words(k), "v": f32Words(v),
			"rows": {0, 0, pos + 1, pos}, "pages": pages, "q.off": {0},
			"kv.k": append([]uint32(nil), kp...), "kv.v": append([]uint32(nil), vp...)}
		bufs[g.Values[g.Out[0]].Name] = f32Words(out)
		if err := runCPUCalls(cpuOn, codes, g, bufs); err != nil {
			t.Fatal(err)
		}
		return out
	}
	errOf := func(got []float32) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		return num / den
	}
	e := errOf(run([]uint32{0, 1, 2, 3}, kpool, vpool))
	t.Logf("NMSE %.3e against the definition over the widened rows", e)
	if math.IsNaN(e) || e > 1e-10 {
		t.Fatalf("the composed CPU attention over %v pools: NMSE %.3e", pool, e)
	}
	if e := errOf(run([]uint32{0, 2, 1, 3}, kpool, vpool)); !(e > 1e-6) {
		t.Errorf("with two pages swapped the NMSE is %.3e: the gate cannot see a wrong page", e)
	}
	// The keys' pool holding f32 rows, read as q8 rows: a read that ignored
	// the layout would not see it.
	f32k := make([]uint32, len(kpool))
	for i := range f32k {
		f32k[i] = math.Float32bits(float32(rng.NormFloat64()))
	}
	if e := errOf(run([]uint32{0, 1, 2, 3}, f32k, vpool)); !(e > 1e-6) {
		t.Errorf("with the keys' pool not q8 the NMSE is %.3e: the gate cannot see the layout", e)
	}
}
