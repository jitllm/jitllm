package composer

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestComposedCPUAttendTwo executes the composer's attention over a window
// of the history and the entries of an entry history (OpAttend of two
// reads, DeepSeek V4's compressed blocks) on the CPU, one decode row: H
// query heads over one key head that is its value, the window's keys from a
// paged binary16 history and the entries from an f32 one, under one softmax
// with the heads' sinks, held to main's procedure (engine/model/ds4.go
// ds4AttnRow): every visible entry (HCA's), or the entries a selection keeps
// (CSA's: the indexer's ReLU'd, weighted heads over the key that follows
// the entry's, the best TopK). The window alone, the sinks dropped and, for
// CSA, every entry read must each move it.
func TestComposedCPUAttendTwo(t *testing.T) {
	for _, csa := range []bool{false, true} {
		name := "hca"
		if csa {
			name = "csa"
		}
		t.Run(name, func(t *testing.T) { composedCPUAttendTwo(t, csa) })
	}
}

func composedCPUAttendTwo(t *testing.T, csa bool) {
	const page, H, D, window, pos, nvis, Hi, Di, topK = 32, 4, 16, 8, 40, 10, 2, 8, 4
	W := D + 8
	EW := D
	if csa {
		EW = D + Di
	}
	b := graph.New()
	k := b.In("k", W)
	h := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, k, 0)}
	e := graph.StateRef{State: graph.StateEnt}
	q := b.In("q", H*D)
	a := graph.Attrs{NHead: H, NKVHead: 1, HeadDim: D, Scale: 0.25, Window: window, Width: W, EntWidth: EW}
	var sel *graph.ValueID
	if csa {
		iq, iw := b.In("iq", Hi*Di), b.In("iw", Hi)
		s := b.Select(iq, e, graph.Attrs{NHead: Hi, NKVHead: 1, HeadDim: Di, Offset: D, Width: EW, Span: 1, TopK: topK, Fn: graph.MapMax}, iw)
		sel = &s
	}
	y := b.AttendTwo(q, h, e, sel, a, jlm.RoleAttnSinks)
	b.Out(y)
	g := b.G
	if err := graph.Verify(g, graph.Facts{Inventory: map[jlm.Role]graph.Shape{jlm.RoleAttnSinks: {K: H}}}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := Default().Form(Request{Graph: g, History: History{ValuesInKeys: true},
		Step:   Step{Rows: 1, Page: page, Keys: 4 * page, ActWindow: 32},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(17))
	vec := func(m int) []float32 {
		x := make([]float32, m)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	n := pos + 1
	rows := make([][]float32, n)
	kpool := make([]uint32, 4*page*W/2)
	for p := range n {
		rows[p] = vec(W)
		for d := range W {
			rows[p][d] = float32(quant.DecodeHalf(quant.EncodeHalf(rows[p][d])))
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
	ents := make([][]float32, nvis)
	epool := make([]float32, 4*page*EW)
	for i := range ents {
		ents[i] = vec(EW)
		copy(epool[i*EW:], ents[i])
	}
	qv, iqv, iwv, sinks := vec(H*D), vec(Hi*Di), vec(Hi), vec(H)
	for i := range iwv {
		iwv[i] = float32(math.Abs(float64(iwv[i])))
	}
	// kept is the entries main reads: every visible one, or the indexer's best.
	kept := make([]bool, nvis)
	for i := range kept {
		kept[i] = !csa
	}
	if csa {
		sc := make([]float64, nvis)
		for i := range sc {
			for hh := range Hi {
				var d float64
				for j := range Di {
					d += float64(iqv[hh*Di+j]) * float64(ents[i][D+j])
				}
				sc[i] += math.Max(d, 0) * float64(iwv[hh])
			}
		}
		ord := make([]int, nvis)
		for i := range ord {
			ord[i] = i
		}
		sort.SliceStable(ord, func(x, y int) bool { return sc[ord[x]] > sc[ord[y]] })
		for _, i := range ord[:topK] {
			kept[i] = true
		}
	}
	want := func(entries []bool, sinked bool) []float64 {
		out := make([]float64, H*D)
		for hh := range H {
			var keys [][]float32
			for p := pos - window + 1; p <= pos; p++ {
				keys = append(keys, rows[p][:D])
			}
			for i, ok := range entries {
				if ok {
					keys = append(keys, ents[i][:D])
				}
			}
			s := make([]float64, len(keys))
			mx := math.Inf(-1)
			for i, kk := range keys {
				for d := range D {
					s[i] += float64(qv[hh*D+d]) * float64(kk[d])
				}
				s[i] *= 0.25
				mx = math.Max(mx, s[i])
			}
			var sum float64
			for i := range s {
				s[i] = math.Exp(s[i] - mx)
				sum += s[i]
			}
			if sinked {
				sum += math.Exp(float64(sinks[hh]) - mx)
			}
			for d := range D {
				var o float64
				for i, kk := range keys {
					o += s[i] / sum * float64(kk[d])
				}
				out[hh*D+d] = o
			}
		}
		return out
	}
	got := make([]float32, H*D)
	bufs := map[string][]uint32{"q": f32Words(qv), "k": f32Words(append([]float32(nil), rows[pos]...)),
		"rows": {0, 0, uint32(n), uint32(pos)}, "rows.win": {0, uint32(pos - window + 1), uint32(n), uint32(pos)},
		"pages": {0, 1, 2, 3}, "q.off": {0}, "kv.k": kpool, "ent.k": f32Words(epool), "ent.rows": {0, 0, nvis, nvis},
		"ent.pages": {0, 1, 2, 3}, jlm.RoleAttnSinks.String(): f32Words(sinks), g.Values[y].Name: f32Words(got)}
	if csa {
		bufs["iq"], bufs["iw"] = f32Words(iqv), f32Words(iwv)
		bufs[g.Values[*sel].Name] = make([]uint32, 1)
	}
	if err := runCPUCalls(codes, g, bufs); err != nil {
		t.Fatal(err)
	}
	nmse := func(want []float64) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		return num / den
	}
	ee := nmse(want(kept, true))
	t.Logf("NMSE %.3e against main's attention", ee)
	if math.IsNaN(ee) || ee > 1e-10 {
		t.Fatalf("the composed attention over two histories: NMSE %.3e", ee)
	}
	if csa {
		if c := math.Float32frombits(bufs[g.Values[*sel].Name][0]); c != topK {
			t.Errorf("the selection keeps %v entries, main %d", c, topK)
		}
	}
	vs := []struct {
		name string
		e    float64
	}{{"the window alone", nmse(want(make([]bool, nvis), true))}, {"the sinks dropped", nmse(want(kept, false))}}
	if csa {
		all := make([]bool, nvis)
		for i := range all {
			all[i] = true
		}
		vs = append(vs, struct {
			name string
			e    float64
		}{"every entry read", nmse(want(all, true))})
	}
	for _, v := range vs {
		t.Logf("violation %q: NMSE %.3e", v.name, v.e)
		if !(v.e > 1e-6) {
			t.Errorf("against main's attention with %s the NMSE is %.3e: the gate cannot see it", v.name, v.e)
		}
	}
}
