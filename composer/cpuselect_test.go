package composer

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// selectCase is one selection's shape: the attention's heads over a history
// row of width W (keys at its start), the selection's index heads of Di at
// Offset, in G groups, blocks of B, K kept, L forced, a ReLU and weights
// where Relu, the values the keys' own pool where Latent, a block scored by
// its first position where Firsts.
type selectCase struct {
	name                 string
	H, KV, Dk, W, Hi, G  int
	Di, Off, B, K, L     int
	Relu, Latent, Firsts bool
}

// selectOracle is the selection the case states over n positions' keys: for
// each group, whether each position is kept (main's procedure: the forced
// blocks first, then blocks in the order of their best position -- of their
// first where Firsts -- until K).
func selectOracle(sc selectCase, keys [][]float32, iq, iw []float32, n int, noLocal bool) [][]bool {
	J := sc.Hi / sc.G
	nb := (n + sc.B - 1) / sc.B
	own := (n - 1) / sc.B
	keep := make([][]bool, sc.G)
	for g := range sc.G {
		keep[g] = make([]bool, n)
		if nb <= sc.K {
			for t := range n {
				keep[g][t] = true
			}
			continue
		}
		s := make([]float64, n)
		for t := range n {
			for j := range J {
				h := g*J + j
				var d float64
				for e := range sc.Di {
					d += float64(iq[h*sc.Di+e]) * float64(keys[t][sc.Off+e])
				}
				if sc.Relu {
					d = math.Max(d, 0)
					d *= float64(iw[h])
				}
				s[t] += d
			}
		}
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return s[order[a]] > s[order[b]] })
		kept := make([]bool, nb)
		have := 0
		for l := 0; l < sc.L && !noLocal; l++ {
			if b := max(own-l, 0); !kept[b] {
				kept[b] = true
				have++
			}
		}
		for _, t := range order {
			if have >= sc.K {
				break
			}
			if sc.Firsts && t%sc.B != 0 {
				continue
			}
			if b := t / sc.B; !kept[b] {
				kept[b] = true
				have++
			}
		}
		for t := range n {
			keep[g][t] = kept[t/sc.B]
		}
	}
	return keep
}

// TestComposedCPUSelect executes the composer's key selection and the
// attention over what it keeps on the CPU, one decode row over a paged f32
// history: a lightning indexer's shape (one selection of ReLU'd, weighted
// heads per position, its key the tail of a latent row that is its own
// value) and a block selection's (a selection per kv group over blocks of
// eight positions, two forced in, the key one more head of the key row,
// the values a pool of their own; and the same where its blocks number no
// more than it keeps, so it keeps all), each held to the definition with
// every group's count. A selection that keeps less must drop positions
// (its output parts from the dense attention's), and the definition
// without the forced blocks must fail it. The indexer's key sits past the
// pool row's start in a binary16 history: a constant offset into a buffer
// of halves counted in floats reads another position's key.
func TestComposedCPUSelect(t *testing.T) {
	for _, sc := range []selectCase{
		{name: "indexer", H: 4, KV: 1, Dk: 16, W: 24, Hi: 4, G: 1, Di: 8, Off: 16, B: 1, K: 9, L: 0, Relu: true, Latent: true},
		{name: "blocks", H: 4, KV: 2, Dk: 8, W: 24, Hi: 2, G: 2, Di: 8, Off: 16, B: 8, K: 4, L: 2},
		{name: "blocksJ", H: 4, KV: 1, Dk: 16, W: 24, Hi: 4, G: 2, Di: 8, Off: 16, B: 8, K: 4, L: 2, Latent: true},
		{name: "dense", H: 4, KV: 2, Dk: 8, W: 24, Hi: 2, G: 2, Di: 8, Off: 16, B: 8, K: 9, L: 2},
		{name: "blocksFirst", H: 4, KV: 2, Dk: 8, W: 24, Hi: 2, G: 2, Di: 8, Off: 16, B: 8, K: 4, L: 2, Firsts: true},
	} {
		t.Run(sc.name, func(t *testing.T) { composedCPUSelect(t, hostArm(), sc) })
	}
}

func composedCPUSelect(t *testing.T, cpuOn cpuArm, sc selectCase) {
	const page, pos = 32, 70
	n := pos + 1
	b := graph.New()
	k := b.In("k", sc.W)
	v := k
	if !sc.Latent {
		v = b.In("v", sc.W)
	}
	h := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, v, 0)}
	sa := graph.Attrs{NHead: sc.Hi, NKVHead: sc.G, HeadDim: sc.Di, Offset: sc.Off, Width: sc.W, Span: sc.B, TopK: sc.K, Local: sc.L,
		Firsts: sc.Firsts}
	iq := b.In("iq", sc.Hi*sc.Di)
	var w []graph.ValueID
	if sc.Relu {
		sa.Fn = graph.MapMax
		w = append(w, b.In("iw", sc.Hi))
	}
	sel := b.Select(iq, h, sa, w...)
	q := b.In("q", sc.H*sc.Dk)
	b.Out(b.AttendSelected(q, h, sel, graph.Attrs{NHead: sc.H, NKVHead: sc.KV, HeadDim: sc.Dk, Scale: 0.125, Width: sc.W}))
	g := b.G
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	hist := History{ValuesInKeys: true}
	if !sc.Latent {
		hist = History{Keys: Pool{ByDim: true}}
	}
	_, codes, err := Default().Form(Request{Graph: g, History: hist,
		Step:   Step{Rows: 1, Page: page, Keys: 4 * page, ActWindow: 32},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	vec := func(m int) []float32 {
		x := make([]float32, m)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	keys, vals := make([][]float32, n), make([][]float32, n)
	for p := range n {
		keys[p], vals[p] = vec(sc.W), vec(sc.W)
		if sc.Latent {
			vals[p] = keys[p]
		}
	}
	// The CPU keeps the history binary16 (Halves): an element of a pool is a
	// half at its element index, the keys and values the definition reads
	// those halves.
	kpool, vpool := make([]uint32, 4*page*sc.W/2), make([]uint32, 4*page*sc.W/2)
	put := func(pool []uint32, i int, x float32) {
		h := uint32(quant.EncodeHalf(x))
		if i%2 == 0 {
			pool[i/2] = pool[i/2]&0xFFFF0000 | h
		} else {
			pool[i/2] = pool[i/2]&0xFFFF | h<<16
		}
	}
	appended := [2][]float32{append([]float32(nil), keys[pos]...), append([]float32(nil), vals[pos]...)}
	for p := range n {
		for d := range sc.W {
			keys[p][d] = float32(quant.DecodeHalf(quant.EncodeHalf(keys[p][d])))
			if !sc.Latent {
				vals[p][d] = float32(quant.DecodeHalf(quant.EncodeHalf(vals[p][d])))
			}
			if p == pos {
				continue
			}
			if sc.Latent {
				put(kpool, p*sc.W+d, keys[p][d])
			} else {
				put(kpool, (p/page*sc.W+d)*page+p%page, keys[p][d])
				put(vpool, p*sc.W+d, vals[p][d])
			}
		}
	}
	qv, iqv, iwv := vec(sc.H*sc.Dk), vec(sc.Hi*sc.Di), vec(sc.Hi)
	for i := range iwv {
		iwv[i] = float32(math.Abs(float64(iwv[i])))
	}
	want := func(keep [][]bool) []float64 {
		out := make([]float64, sc.H*sc.Dk)
		gqa := sc.H / sc.KV
		for hh := range sc.H {
			kv, grp := hh/gqa, hh/(sc.H/sc.G)
			mx := math.Inf(-1)
			s := make([]float64, n)
			for p := range n {
				if keep != nil && !keep[grp][p] {
					s[p] = math.Inf(-1)
					continue
				}
				for d := range sc.Dk {
					s[p] += float64(qv[hh*sc.Dk+d]) * float64(keys[p][kv*sc.Dk+d])
				}
				s[p] *= 0.125
				mx = math.Max(mx, s[p])
			}
			var sum float64
			for p := range s {
				s[p] = math.Exp(s[p] - mx)
				sum += s[p]
			}
			for d := range sc.Dk {
				var o float64
				for p := range s {
					o += s[p] / sum * float64(vals[p][kv*sc.Dk+d])
				}
				out[hh*sc.Dk+d] = o
			}
		}
		return out
	}
	bufs := map[string][]uint32{"q": f32Words(qv), "k": f32Words(appended[0]), "iq": f32Words(iqv),
		"rows": {0, 0, uint32(n), pos}, "pages": {0, 1, 2, 3}, "q.off": {0}, "kv.k": kpool}
	if !sc.Latent {
		bufs["v"], bufs["kv.v"] = f32Words(appended[1]), vpool
	}
	if sc.Relu {
		bufs["iw"] = f32Words(iwv)
	}
	count := make([]uint32, sc.G)
	bufs[g.Values[sel].Name] = count
	got := make([]float32, sc.H*sc.Dk)
	bufs[g.Values[g.Out[0]].Name] = f32Words(got)
	if err := runCPUCalls(cpuOn, codes, g, bufs); err != nil {
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
	keep := selectOracle(sc, keys, iqv, iwv, n, false)
	kept := 0
	for _, row := range keep {
		for _, k := range row {
			if k {
				kept++
			}
		}
	}
	for grp, row := range keep {
		n := 0
		for _, k := range row {
			if k {
				n++
			}
		}
		if c := math.Float32frombits(count[grp]); c != float32(n) {
			t.Errorf("group %d keeps %v positions, the definition %d", grp, c, n)
		}
	}
	e := nmse(want(keep))
	t.Logf("%d of %d positions kept over %d groups: NMSE %.3e against the definition", kept, sc.G*n, sc.G, e)
	if math.IsNaN(e) || e > 1e-10 {
		t.Fatalf("the composed selection: NMSE %.3e", e)
	}
	if kept == sc.G*n {
		return
	}
	if e := nmse(want(nil)); !(e > 1e-6) {
		t.Errorf("against the dense attention the NMSE is %.3e: the selection dropped nothing", e)
	}
	if sc.L > 0 {
		if e := nmse(want(selectOracle(sc, keys, iqv, iwv, n, true))); !(e > 1e-6) {
			t.Errorf("against the definition without its forced blocks the NMSE is %.3e: the gate cannot see them", e)
		}
	}
	if sc.B > 1 {
		other := sc
		other.Firsts = !sc.Firsts
		if e := nmse(want(selectOracle(other, keys, iqv, iwv, n, false))); !(e > 1e-6) {
			t.Errorf("against the definition scoring a block by its %s the NMSE is %.3e: the gate cannot tell the two",
				map[bool]string{true: "first position", false: "best"}[other.Firsts], e)
		}
	}
}
