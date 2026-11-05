package composer

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// pooledCase is a pooled selection's shape (QSA's): the attention's H heads
// over KV kv heads of Dk in a history row of W, the selection's Hi index
// heads of Di at Off in one group, blocks of B, K complete blocks kept, the
// pooled key normed and turned by a rotary of NRot (NEOX where Neox).
type pooledCase struct {
	name                 string
	H, KV, Dk, W, Hi, Di int
	Off, B, K, NRot      int
	Neox                 bool
	pos                  int
}

// pooledFault is a departure from the definition a gate must see.
type pooledFault int

const (
	pooledExact     pooledFault = iota
	pooledFirstKey              // a block's pool its first key alone
	pooledTailRanks             // the incomplete block ranked, not kept
	pooledTurnLate              // a block turned at its last position
)

// pooledOracle is the selection main's QSA makes (engine/model/qsa.go
// qsaMask) over n positions: each complete block's keys averaged, RMS-normed
// by w, turned at the block's first position; scored by the sum over heads
// of the ReLU of each query head's dot with it; the K best complete blocks
// and the incomplete tail kept, every position where the complete blocks
// number at most K.

func pooledOracle(pc pooledCase, keys [][]float32, iq, w []float32, n int, fault pooledFault) []bool {
	keep := make([]bool, n)
	nb := n / pc.B
	if nb <= pc.K {
		for t := range keep {
			keep[t] = true
		}
		return keep
	}
	const eps = 1e-6
	score := make([]float64, nb)
	key := make([]float64, pc.Di)
	for b := range nb {
		clear(key)
		cnt := pc.B
		if fault == pooledFirstKey {
			cnt = 1
		}
		for t := b * pc.B; t < b*pc.B+cnt; t++ {
			for e := range pc.Di {
				key[e] += float64(keys[t][pc.Off+e])
			}
		}
		var ss float64
		for e := range key {
			key[e] /= float64(cnt)
			ss += key[e] * key[e]
		}
		inv := 1 / math.Sqrt(ss/float64(pc.Di)+eps)
		for e := range key {
			key[e] *= inv * float64(w[e])
		}
		p := float64(b * pc.B)
		if fault == pooledTurnLate {
			p += float64(pc.B - 1)
		}
		half := pc.NRot / 2
		for i := range half {
			th := p * math.Pow(10000, -2*float64(i)/float64(pc.NRot))
			a, c := 2*i, 2*i+1
			if pc.Neox {
				a, c = i, i+half
			}
			x0, x1 := key[a], key[c]
			key[a], key[c] = x0*math.Cos(th)-x1*math.Sin(th), x0*math.Sin(th)+x1*math.Cos(th)
		}
		for h := range pc.Hi {
			var d float64
			for e := range pc.Di {
				d += float64(iq[h*pc.Di+e]) * key[e]
			}
			score[b] += math.Max(d, 0)
		}
	}
	order := make([]int, nb)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return score[order[a]] > score[order[b]] })
	for _, b := range order[:pc.K] {
		for t := b * pc.B; t < (b+1)*pc.B; t++ {
			keep[t] = true
		}
	}
	for t := nb * pc.B; t < n && fault != pooledTailRanks; t++ {
		keep[t] = true
	}
	return keep
}

// TestComposedCPUSelectPooled executes the composer's pooled selection
// (graph.Attrs.Pooled, QSA's) and the attention over what it keeps on the
// CPU, one decode row over a paged binary16 history: the indexer's key one
// more head of the key row, blocks of four, held to main's procedure with
// the group's count. A NEOX and a pairwise rotary, a row whose tail is
// incomplete and one whose blocks are all complete, and one whose complete
// blocks number no more than it keeps. Each departure -- a block pooled
// from its first key, the incomplete tail ranked rather than kept, a block
// turned at its last position -- must move the output.
func TestComposedCPUSelectPooled(t *testing.T) {
	for _, pc := range []pooledCase{
		{name: "neox", H: 4, KV: 2, Dk: 8, W: 24, Hi: 4, Di: 8, Off: 16, B: 4, K: 4, NRot: 8, Neox: true, pos: 70},
		{name: "pairs", H: 4, KV: 2, Dk: 8, W: 24, Hi: 4, Di: 8, Off: 16, B: 4, K: 4, NRot: 4, pos: 70},
		{name: "whole", H: 4, KV: 2, Dk: 8, W: 24, Hi: 4, Di: 8, Off: 16, B: 4, K: 4, NRot: 8, Neox: true, pos: 75},
		{name: "dense", H: 4, KV: 2, Dk: 8, W: 24, Hi: 4, Di: 8, Off: 16, B: 4, K: 18, NRot: 8, Neox: true, pos: 70},
	} {
		t.Run(pc.name, func(t *testing.T) { composedCPUSelectPooled(t, hostArm(), pc) })
	}
}

func composedCPUSelectPooled(t *testing.T, cpuOn cpuArm, pc pooledCase) {
	const page = 32
	n := pc.pos + 1
	b := graph.New()
	k, v := b.In("k", pc.W), b.In("v", pc.W)
	h := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, v, 0)}
	layout := graph.RopeNorm
	if pc.Neox {
		layout = graph.RopeNeox
	}
	sa := graph.Attrs{NHead: pc.Hi, NKVHead: 1, HeadDim: pc.Di, Offset: pc.Off, Width: pc.W, Span: pc.B, TopK: pc.K,
		Fn: graph.MapMax, Eps: 1e-6, NRot: pc.NRot, Base: 10000, Layout: layout}
	iq := b.In("iq", pc.Hi*pc.Di)
	sel := b.SelectPooled(iq, h, sa, jlm.RoleIdxKNorm)
	q := b.In("q", pc.H*pc.Dk)
	b.Out(b.AttendSelected(q, h, sel, graph.Attrs{NHead: pc.H, NKVHead: pc.KV, HeadDim: pc.Dk, Scale: 0.125, Width: pc.W}))
	g := b.G
	if err := graph.Verify(g, graph.Facts{Inventory: map[jlm.Role]graph.Shape{jlm.RoleIdxKNorm: {K: pc.Di}}}); err != nil {
		t.Fatal(err)
	}
	_, codes, err := Default().Form(Request{Graph: g, History: History{Keys: Pool{ByDim: true}},
		Step:   Step{Rows: 1, Page: page, Keys: 4 * page, ActWindow: 32},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	vec := func(m int) []float32 {
		x := make([]float32, m)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	keys, vals := make([][]float32, n), make([][]float32, n)
	for p := range n {
		keys[p], vals[p] = vec(pc.W), vec(pc.W)
	}
	kpool, vpool := make([]uint32, 4*page*pc.W/2), make([]uint32, 4*page*pc.W/2)
	put := func(pool []uint32, i int, x float32) {
		h := uint32(quant.EncodeHalf(x))
		if i%2 == 0 {
			pool[i/2] = pool[i/2]&0xFFFF0000 | h
		} else {
			pool[i/2] = pool[i/2]&0xFFFF | h<<16
		}
	}
	appended := [2][]float32{append([]float32(nil), keys[pc.pos]...), append([]float32(nil), vals[pc.pos]...)}
	for p := range n {
		for d := range pc.W {
			keys[p][d] = float32(quant.DecodeHalf(quant.EncodeHalf(keys[p][d])))
			vals[p][d] = float32(quant.DecodeHalf(quant.EncodeHalf(vals[p][d])))
			if p == pc.pos {
				continue
			}
			put(kpool, (p/page*pc.W+d)*page+p%page, keys[p][d])
			put(vpool, p*pc.W+d, vals[p][d])
		}
	}
	qv, iqv, wv := vec(pc.H*pc.Dk), vec(pc.Hi*pc.Di), vec(pc.Di)
	for i := range wv {
		wv[i] = 0.5 + float32(math.Abs(float64(wv[i])))
	}
	want := func(keep []bool) []float64 {
		out := make([]float64, pc.H*pc.Dk)
		gqa := pc.H / pc.KV
		for hh := range pc.H {
			kv := hh / gqa
			mx := math.Inf(-1)
			s := make([]float64, n)
			for p := range n {
				if keep != nil && !keep[p] {
					s[p] = math.Inf(-1)
					continue
				}
				for d := range pc.Dk {
					s[p] += float64(qv[hh*pc.Dk+d]) * float64(keys[p][kv*pc.Dk+d])
				}
				s[p] *= 0.125
				mx = math.Max(mx, s[p])
			}
			var sum float64
			for p := range s {
				s[p] = math.Exp(s[p] - mx)
				sum += s[p]
			}
			for d := range pc.Dk {
				var o float64
				for p := range s {
					o += s[p] / sum * float64(vals[p][kv*pc.Dk+d])
				}
				out[hh*pc.Dk+d] = o
			}
		}
		return out
	}
	planes := nn.Rope{NRot: pc.NRot, Base: 10000, Neox: pc.Neox}.TabPlanes(pc.NRot / 2)
	bufs := map[string][]uint32{"q": f32Words(qv), "k": f32Words(appended[0]), "v": f32Words(appended[1]), "iq": f32Words(iqv),
		"rows": {0, 0, uint32(n), uint32(pc.pos)}, "pages": {0, 1, 2, 3}, "q.off": {0}, "kv.k": kpool, "kv.v": vpool,
		jlm.RoleIdxKNorm.String(): f32Words(wv), "rope.planes": f32Words(planes)}
	count := make([]uint32, 1)
	bufs[g.Values[sel].Name] = count
	got := make([]float32, pc.H*pc.Dk)
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
	keep := pooledOracle(pc, keys, iqv, wv, n, pooledExact)
	kept := 0
	for _, k := range keep {
		if k {
			kept++
		}
	}
	if c := math.Float32frombits(count[0]); c != float32(kept) {
		t.Errorf("the selection keeps %v positions, the definition %d", c, kept)
	}
	e := nmse(want(keep))
	t.Logf("%d of %d positions kept: NMSE %.3e against the definition", kept, n, e)
	if math.IsNaN(e) || e > 1e-10 {
		t.Fatalf("the composed pooled selection: NMSE %.3e", e)
	}
	if kept == n {
		return
	}
	if e := nmse(want(nil)); !(e > 1e-6) {
		t.Errorf("against the dense attention the NMSE is %.3e: the selection dropped nothing", e)
	}
	for _, f := range []struct {
		fault pooledFault
		name  string
	}{{pooledFirstKey, "a block pooled from its first key"}, {pooledTailRanks, "the incomplete tail ranked"},
		{pooledTurnLate, "a block turned at its last position"}} {
		if f.fault == pooledTailRanks && n%pc.B == 0 {
			continue
		}
		e := nmse(want(pooledOracle(pc, keys, iqv, wv, n, f.fault)))
		t.Logf("violation %q: NMSE %.3e", f.name, e)
		if !(e > 1e-6) {
			t.Errorf("against the definition with %s the NMSE is %.3e: the gate cannot see it", f.name, e)
		}
	}
}
