package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// crossGraph is a cross-attention: rows of queries attending, both ways,
// a history this graph does not write -- version 0 of "kv", the keys and
// values another part's rows appended (a resampler's learned queries over
// a picture's patches).
func crossGraph(heads, kvh, dim int) *graph.Graph {
	b := graph.New()
	b.Out(b.CrossAttend(b.In("q", heads*dim), graph.StateRef{Name: "kv"}, graph.Attrs{NHead: heads, NKVHead: kvh, HeadDim: dim,
		Scale: 1 / math.Sqrt(float64(dim))}))
	return b.G
}

// crossCase is a cross-attention gate's case: Q query rows over N keys.
type crossCase struct{ heads, kvh, dim, Q, N int }

func crossCases() []crossCase {
	var out []crossCase
	for _, sh := range [][3]int{{4, 4, 128}, {6, 2, 48}, {3, 3, 72}} {
		for _, qn := range [][2]int{{1, 1}, {7, 5}, {64, 33}, {9, 100}, {64, 300}} {
			out = append(out, crossCase{sh[0], sh[1], sh[2], qn[0], qn[1]})
		}
	}
	return out
}

// crossData is a case's queries, its keys and values by position, the
// binary16 pools holding them (a position's row at p*w, two halves a word)
// and the definition: each query row's softmax over every key of the
// binary16 history, times the values.
func crossData(c crossCase, page, pages int, seed int64) (q []float32, kp, vp []uint32, want []float64) {
	rng := rand.New(rand.NewSource(seed))
	w, qw := c.kvh*c.dim, c.heads*c.dim
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	q, ks, vs := rnd(c.Q*qw), rnd(c.N*w), rnd(c.N*w)
	pool := func(x []float32) []uint32 {
		out := make([]uint32, pages*page*w/2)
		for i, v := range x {
			h := uint32(quant.EncodeHalf(v))
			if i%2 == 0 {
				out[i/2] = out[i/2]&0xFFFF0000 | h
			} else {
				out[i/2] = out[i/2]&0xFFFF | h<<16
			}
		}
		return out
	}
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	scale := 1 / math.Sqrt(float64(c.dim))
	want = make([]float64, c.Q*qw)
	for r := range c.Q {
		for h := range c.heads {
			kv := h / (c.heads / c.kvh)
			sc := make([]float64, c.N)
			mx := math.Inf(-1)
			for p := range c.N {
				var s float64
				for d := range c.dim {
					s += float64(q[r*qw+h*c.dim+d]) * round(ks[p*w+kv*c.dim+d])
				}
				sc[p] = s * float64(float32(scale))
				mx = math.Max(mx, sc[p])
			}
			var sum float64
			for p := range sc {
				sc[p] = math.Exp(sc[p] - mx)
				sum += sc[p]
			}
			for d := range c.dim {
				var o float64
				for p := range sc {
					o += sc[p] / sum * round(vs[p*w+kv*c.dim+d])
				}
				want[r*qw+h*c.dim+d] = o
			}
		}
	}
	return q, pool(ks), pool(vs), want
}

// crossNMSE is got's squared error from want over want's energy.
func crossNMSE(got []float32, want []float64) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - want[i]
		num += d * d
		den += want[i] * want[i]
	}
	return num / den
}

// TestComposedCPUCrossAttend executes the composer's cross-attention on the
// CPU: query rows attending, both ways, a history the graph does not write
// (crossGraph), every row's span the history's N keys, at head widths of
// 128, 72 and 48, grouped and not, query and key counts ragged and apart
// (one query over one key through 64 queries over 300 keys), held to the
// definition. A span cut by its last key or starting one key late must
// fail it.
func TestComposedCPUCrossAttend(t *testing.T) {
	const page = 32
	ran := 0
	for _, c := range crossCases() {
		t.Run(fmt.Sprintf("%dq%dkv%dd/Q%d/N%d", c.heads, c.kvh, c.dim, c.Q, c.N), func(t *testing.T) {
			pages := (c.N + page - 1) / page
			g := crossGraph(c.heads, c.kvh, c.dim)
			_, codes, err := Default().Form(Request{Graph: g,
				Step:   Step{Rows: c.Q, Page: page, Keys: pages * page, ActWindow: 32},
				Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
			if err != nil {
				t.Fatal(err)
			}
			q, kp, vp, want := crossData(c, page, pages, int64(c.Q*1000+c.N))
			qw := c.heads * c.dim
			pg := make([]uint32, pages)
			for i := range pg {
				pg[i] = uint32(i)
			}
			run := func(lo, hi uint32) []float32 {
				out := make([]float32, c.Q*qw)
				rows := make([]uint32, c.Q*4)
				for r := range c.Q {
					copy(rows[r*4:], []uint32{0, lo, hi, hi - 1})
				}
				bufs := map[string][]uint32{"q": f32Words(q), "rows": rows, "rows.win": rows, "pages": pg,
					"q.off": make([]uint32, c.Q*4), "rows.n": {uint32(c.Q)}, "kv.k": kp, "kv.v": vp}
				bufs[g.Values[g.Out[0]].Name] = f32Words(out)
				if err := runCPURows(codes, g, bufs, c.Q); err != nil {
					t.Fatal(err)
				}
				for _, b := range codes[len(codes)-1].Binds {
					if o := b.Operand; b.Writes && o.Kind == OperandScratch {
						copy(out, wordsF32(bufs["own:"+o.Name]))
					}
				}
				return out
			}
			e := crossNMSE(run(0, uint32(c.N)), want)
			t.Logf("%d queries over %d keys: NMSE %.3e against the definition", c.Q, c.N, e)
			if math.IsNaN(e) || !(e < 1e-8) {
				t.Fatalf("NMSE %.3e against the definition", e)
			}
			ran++
			if c.N < 2 {
				return
			}
			if e := crossNMSE(run(0, uint32(c.N-1)), want); !(e > 1e-6) {
				t.Errorf("with the span cut by its last key the NMSE is %.3e: the gate cannot see it", e)
			}
			if e := crossNMSE(run(1, uint32(c.N)), want); !(e > 1e-6) {
				t.Errorf("with the span starting one key late the NMSE is %.3e: the gate cannot see it", e)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no cross-attention held to the definition: the gate proved nothing")
	}
}
