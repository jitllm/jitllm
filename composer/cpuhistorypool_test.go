package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUHistoryPool executes OpHistoryPool on the CPU over a
// chunk of rows appended to a paged binary16 history, each row's window of
// Span positions (and for a two-segment pool the window before it) pooled
// per channel by the softmax of gate plus bias, held to the definition row
// by row, closing rows and not, a window before the history's first no
// term. A zeroed bias and a write position moved by one must fail it.
func TestComposedCPUHistoryPool(t *testing.T) {
	for _, sh := range []struct {
		name  string
		p, w  int
		segs  [][4]uint64
		width int
	}{
		{"one", 8, 6, [][4]uint64{{0, 0, 6, 0}}, 12},
		{"one-ragged", 4, 5, [][4]uint64{{0, 0, 5, 0}}, 10},
		{"overlap", 4, 6, [][4]uint64{{1, 0, 12, 0}, {0, 6, 18, 6}}, 24},
		{"overlap-ragged", 2, 5, [][4]uint64{{1, 0, 10, 0}, {0, 5, 15, 5}}, 20},
	} {
		t.Run(sh.name, func(t *testing.T) { composedCPUHistoryPool(t, sh.p, sh.w, sh.segs, sh.width) })
	}
}

func composedCPUHistoryPool(t *testing.T, P, W int, segs [][4]uint64, RW int) {
	const page, p0, R, capRows = 8, 13, 7, 8
	var words []uint64
	bw := 0
	for _, s := range segs {
		words = append(words, s[:]...)
		bw = max(bw, int(s[3])+W)
	}
	b := graph.New()
	n := b.KVAppend(b.In("k", RW), b.In("v", RW), 0)
	b.Out(b.HistoryPool(graph.StateRef{State: graph.StateKV, Version: n}, graph.Attrs{Span: P, Width: RW, Words: words}, jlm.RoleCompAPE, W))
	g := b.G
	_, codes, err := Default().Form(Request{Graph: g,
		Step:   Step{Rows: capRows, Page: page, Keys: 4 * page, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(hostArm().tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	rnd := func(n int, m float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * m)
		}
		return x
	}
	rows := make([][]float32, p0+R)
	for p := range rows {
		rows[p] = rnd(RW, 2)
	}
	bias := rnd(P*bw, 1)
	k := make([]float32, capRows*RW)
	for r := range R {
		copy(k[r*RW:], rows[p0+r])
	}
	pool := func() []uint32 {
		out := make([]uint32, 4*page*RW/2)
		for p := range p0 {
			for d, x := range rows[p] {
				h, i := uint32(quant.EncodeHalf(x)), p*RW+d
				if d%2 == 0 {
					out[i/2] = out[i/2]&0xFFFF0000 | h
				} else {
					out[i/2] = out[i/2]&0xFFFF | h<<16
				}
			}
		}
		return out
	}
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	want := func(bias []float32) []float64 {
		out := make([]float64, R*W)
		for r := range R {
			t := p0 + r
			ws := t / P * P
			for ch := range W {
				var sc, val []float64
				for _, s := range segs {
					base := ws - int(s[0])*P
					if base < 0 {
						continue
					}
					for j := range P {
						p := base + j
						if p > t {
							continue
						}
						sc = append(sc, round(rows[p][int(s[2])+ch])+float64(bias[j*bw+int(s[3])+ch]))
						val = append(val, round(rows[p][int(s[1])+ch]))
					}
				}
				mx := math.Inf(-1)
				for _, x := range sc {
					mx = math.Max(mx, x)
				}
				var num, den float64
				for i, x := range sc {
					e := math.Exp(x - mx)
					num += e * val[i]
					den += e
				}
				out[r*W+ch] = num / den
			}
		}
		return out
	}
	run := func(bias []float32, shift uint32) []float32 {
		out := make([]float32, capRows*W)
		desc := make([]uint32, capRows*4)
		for r := range R {
			p := uint32(p0+r) + shift
			copy(desc[r*4:], []uint32{0, 0, p + 1, p})
		}
		bufs := map[string][]uint32{"k": f32Words(k), "v": f32Words(k), "rows": desc, "pages": {0, 1, 2, 3}, "rows.n": {R},
			"kv.k": pool(), "kv.v": pool(), jlm.RoleCompAPE.String(): f32Words(bias), "q.off": make([]uint32, capRows*4)}
		bufs[g.Values[g.Out[0]].Name] = f32Words(out)
		if err := runCPURows(hostArm(), codes, g, bufs, R); err != nil {
			t.Fatal(err)
		}
		for _, b := range codes[len(codes)-1].Binds {
			if o := b.Operand; b.Writes && o.Kind == OperandScratch {
				copy(out, wordsF32(bufs["own:"+o.Name]))
			}
		}
		return out
	}
	nmse := func(got []float32, want []float64) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		return num / den
	}
	ref := want(bias)
	e := nmse(run(bias, 0), ref)
	t.Logf("%d rows, windows of %d, %d segments: NMSE %.3e against the definition", R, P, len(segs), e)
	if math.IsNaN(e) || e > 1e-9 {
		t.Fatalf("the composed pool: NMSE %.3e against the definition", e)
	}
	if e := nmse(run(make([]float32, len(bias)), 0), ref); !(e > 1e-6) {
		t.Errorf("with the bias zeroed the NMSE is %.3e: the gate cannot see the bias", e)
	}
	if e := nmse(run(bias, 1), ref); !(e > 1e-6) {
		t.Errorf("with the write position moved the NMSE is %.3e: the gate cannot see the window", e)
	}
}

// TestComposedCPUPeriodicAppend executes an append of one entry each Span
// positions (OpKVAppend's Span) on the CPU: a row closing a window writes
// its entry at its page's slot (position%page)/Span, every other row writes
// zeros to the page's last slot, and no other word of the pool changes.
func TestComposedCPUPeriodicAppend(t *testing.T) {
	for _, P := range []int{1, 2, 3, 4, 16} {
		t.Run(fmt.Sprint(P), func(t *testing.T) { composedCPUPeriodicAppend(t, P) })
	}
}

func composedCPUPeriodicAppend(t *testing.T, P int) {
	const page, p0, R, capRows, w = 8, 5, 7, 8, 6
	b := graph.New()
	e := b.In("e", w)
	b.KVAppendEvery(1, e, e, 0, P)
	g := b.G
	// The CPU's profile keeps the history binary16 itself (Halves): the
	// request states the pool's layout, a position's row of elements.
	ent := History{ValuesInKeys: true, Period: P}
	_, codes, err := Default().Form(Request{Graph: g, More: []History{ent},
		Step:   Step{Rows: capRows, Page: page, Keys: 4 * page, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(hostArm().tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	slots := ent.slots(page)
	rng := rand.New(rand.NewSource(5))
	in := make([]float32, capRows*w)
	for i := range in {
		in[i] = float32(rng.NormFloat64())
	}
	sentinel := uint32(0x7fc0beef)
	poolWords := 4 * slots * w / 2
	pool := make([]uint32, poolWords)
	for i := range pool {
		pool[i] = sentinel
	}
	desc := make([]uint32, capRows*4)
	for r := range R {
		p := uint32(p0 + r)
		copy(desc[r*4:], []uint32{0, 0, p + 1, p})
	}
	bufs := map[string][]uint32{"e": f32Words(in), "rows": desc, "rows.n": {R}, "q.off": make([]uint32, capRows*4)}
	for _, c := range codes {
		for _, bd := range c.Binds {
			if o := bd.Operand; o.Kind == OperandTable && o.Table == TableKeys {
				bufs[o.Name] = pool
			}
			if o := bd.Operand; o.Kind == OperandTable && o.Table == TablePages {
				bufs[o.Name] = []uint32{0, 1, 2, 3}
			}
		}
	}
	if err := runCPURows(hostArm(), codes, g, bufs, R); err != nil {
		t.Fatal(err)
	}
	want := make([]uint32, poolWords)
	for i := range want {
		want[i] = sentinel
	}
	half := func(x float32) uint32 { return uint32(quant.EncodeHalf(x)) }
	for r := range R {
		p := p0 + r
		closes := (p+1)%P == 0
		slot := slots - 1
		if closes {
			slot = p % page / P
		}
		for d := 0; d < w; d += 2 {
			v := half(in[r*w+d]) | half(in[r*w+d+1])<<16
			if !closes {
				v = 0
			}
			want[((p/page)*slots+slot)*w/2+d/2] = v
		}
	}
	for i := range want {
		if pool[i] != want[i] {
			t.Fatalf("word %d (page %d, slot %d): %#x, want %#x", i, i/(slots*w/2), i/(w/2)%slots, pool[i], want[i])
		}
	}
}
