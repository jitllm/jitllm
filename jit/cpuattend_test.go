package jit

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

func attendGraph(heads, kv, dim int) *graph.Graph {
	b := graph.New()
	n := b.KVAppend(b.In("k", kv*dim), b.In("v", kv*dim), 0)
	b.Out(b.Attend(b.In("q", heads*dim), n, graph.Attrs{NHead: heads, NKVHead: kv, HeadDim: dim, Scale: 0.125}))
	return b.G
}

// runCPUCalls runs codes in order on the CPU, each over its calls' space,
// every operand's words where bufs names it by kind and name (a value by
// its graph name), the composer's own buffers made here.
func runCPUCalls(codes []Code, g *graph.Graph, bufs map[string][]uint32) error {
	own := map[Operand][]uint32{}
	var keep [][]uint32
	for _, c := range codes {
		if c.CPU == nil {
			return fmt.Errorf("%s has no CPU code", c.Name)
		}
		code, err := cpu.Map(c.CPU)
		if err != nil {
			return err
		}
		n := 1
		for _, e := range c.Calls {
			n *= e
		}
		for at := range n {
			var args cpu.Args
			base := unsafe.Pointer(&args)
			for _, a := range c.Args {
				slot := (*uint64)(unsafe.Add(base, a.Field))
				o := a.Operand
				switch {
				case a.Coord > 0:
					v := at
					for ax := len(c.Calls) - 1; ax > a.Coord-1; ax-- {
						v /= c.Calls[ax]
					}
					*slot = uint64(v % c.Calls[a.Coord-1])
					continue
				case a.Length:
					*slot = uint64(a.Extent)
					continue
				case a.Value || a.Moving:
					code.Close()
					return fmt.Errorf("%s: a word the caller computes", c.Name)
				}
				o.Copy = 0
				var words []uint32
				switch {
				case a.Words != nil:
					if own[o] == nil {
						own[o] = append([]uint32(nil), a.Words...)
					}
					words = own[o]
				case o.Kind == OperandValue && bufs[g.Values[o.Value].Name] != nil:
					words = bufs[g.Values[o.Value].Name]
				case o.Kind == OperandTable && bufs[o.Name] != nil:
					words = bufs[o.Name]
				case o.Kind == OperandValue || o.Kind == OperandScratch:
					if own[o] == nil {
						own[o] = make([]uint32, ownWords(codes, o))
					}
					words = own[o]
				default:
					code.Close()
					return fmt.Errorf("%s binds %+v, which the gate does not provide", c.Name, o)
				}
				keep = append(keep, words)
				if a.Word {
					*slot = uint64(words[a.Offset])
					continue
				}
				*slot = uint64(uintptr(unsafe.Pointer(&words[0]))) + uint64(4*a.Offset)
			}
			code.Call(&args)
		}
		code.Close()
	}
	runtime.KeepAlive(keep)
	return nil
}

// ownWords is the words operand o's storage needs over every kernel that
// binds it, a pair's copies with it, each at its offset.
func ownWords(codes []Code, o Operand) int {
	n := 0
	for _, c := range codes {
		reach := map[int]int{}
		for _, a := range c.Args {
			if x := a.Operand; x.Copy > 0 {
				x.Copy = 0
				if x == o {
					reach[a.Operand.Copy] = a.Offset
				}
			}
		}
		for _, b := range c.Binds {
			x := b.Operand
			cp := x.Copy
			x.Copy = 0
			if x == o {
				n = max(n, int(b.Bytes)/4+reach[cp]+1)
			}
		}
	}
	return n
}

func f32Words(x []float32) []uint32 { return unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x)) }

// TestComposedCPUAttend executes the composer's decode attention on the CPU:
// a row's key and value appended to a paged binary16 history, its query
// attending every position, held to the definition over the history as
// binary16 holds it. Two violations must fail it: the full second page
// swapped with the partly written third (two full pages swapped only reorder
// the keys, which attention cannot see), and the first attended row moved by
// one.
func TestComposedCPUAttend(t *testing.T) {
	// The served models' attention shapes: stories15M's, Llama-3.2-1B's and
	// Qwen3-0.6B's, and a pairing of two query heads to a kv head.
	for _, sh := range []struct{ heads, kvh, dim int }{{6, 6, 48}, {32, 8, 64}, {16, 8, 128}, {8, 2, 64}} {
		t.Run(fmt.Sprintf("%dq%dkv%dd", sh.heads, sh.kvh, sh.dim), func(t *testing.T) {
			composedCPUAttend(t, sh.heads, sh.kvh, sh.dim)
		})
	}
}

func composedCPUAttend(t *testing.T, heads, kvh, dim int) {
	const page, pos = 32, 70
	r := Default()
	g := attendGraph(heads, kvh, dim)
	// The CPU's profile keeps the history binary16 itself (Halves): the
	// request states the pools' layout, a position's row of elements.
	_, codes, err := r.Form(Request{Graph: g,
		Step:   Step{Rows: 1, Page: page, Keys: 4 * page, ActWindow: 32},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	w := kvh * dim
	hist := func() ([]uint32, [][]float32, [][]float32) {
		pool := make([]uint32, 4*page*w/2)
		ks, vs := make([][]float32, pos+1), make([][]float32, pos+1)
		for p := range pos {
			ks[p], vs[p] = make([]float32, w), make([]float32, w)
			for d := range w {
				ks[p][d] = float32(rng.NormFloat64())
				vs[p][d] = float32(rng.NormFloat64())
			}
		}
		return pool, ks, vs
	}
	kpool, ks, vs := hist()
	vpool := make([]uint32, len(kpool))
	put := func(pool []uint32, p int, row []float32) {
		for d := range row {
			h := uint32(quant.EncodeHalf(row[d]))
			i := (p/page*page+p%page)*w + d
			if d%2 == 0 {
				pool[i/2] = pool[i/2]&0xFFFF0000 | h
			} else {
				pool[i/2] = pool[i/2]&0xFFFF | h<<16
			}
		}
	}
	for p := range pos {
		put(kpool, p, ks[p])
		put(vpool, p, vs[p])
	}
	q, k, v := make([]float32, heads*dim), make([]float32, w), make([]float32, w)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	for i := range k {
		k[i], v[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	ks[pos], vs[pos] = k, v
	want := make([]float64, heads*dim)
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	for h := range heads {
		kv := h / (heads / kvh)
		sc := make([]float64, pos+1)
		mx := math.Inf(-1)
		for p := range pos + 1 {
			var s float64
			for d := range dim {
				s += float64(q[h*dim+d]) * round(ks[p][kv*dim+d])
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
				o += sc[p] / sum * round(vs[p][kv*dim+d])
			}
			want[h*dim+d] = o
		}
	}
	run := func(pages []uint32, start uint32) []float32 {
		out := make([]float32, heads*dim)
		bufs := map[string][]uint32{"q": f32Words(q), "k": f32Words(k), "v": f32Words(v),
			"rows": {0, start, pos + 1, pos}, "pages": pages, "q.off": {0},
			"kv.k": append([]uint32(nil), kpool...), "kv.v": append([]uint32(nil), vpool...)}
		bufs[g.Values[g.Out[0]].Name] = f32Words(out)
		if err := runCPUCalls(codes, g, bufs); err != nil {
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
	e := errOf(run([]uint32{0, 1, 2, 3}, 0))
	t.Logf("NMSE %.3e against the definition", e)
	if math.IsNaN(e) || e > 1e-10 {
		t.Fatalf("the composed CPU attention: NMSE %.3e", e)
	}
	if e := errOf(run([]uint32{0, 2, 1, 3}, 0)); !(e > 1e-6) {
		t.Errorf("with two pages swapped the NMSE is %.3e: the gate cannot see a wrong page", e)
	}
	if e := errOf(run([]uint32{0, 1, 2, 3}, 1)); !(e > 1e-6) {
		t.Errorf("with the first row moved the NMSE is %.3e: the gate cannot see a wrong row", e)
	}
}
