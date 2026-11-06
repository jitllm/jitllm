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

// TestComposedCPUAttendBoth executes an attention over two histories on the
// CPU (OpAttend with two Reads): a chunk of rows appending to a paged
// binary16 history of rows, values their keys, each row attending its
// window of them and the entries of a second history, one each Span
// positions, those its position+1 over Span closes, all in one softmax with
// a per-head sink counted once. Held to the explicit concatenation row by
// row -- a row too early to see an entry among them -- and the answer
// without the entries and the one without the rows must each fail it.
func TestComposedCPUAttendBoth(t *testing.T) {
	for _, sh := range []struct {
		heads, dim, win, period int
		sinks                   bool
	}{{4, 8, 0, 4, true}, {4, 8, 0, 4, false}, {2, 6, 5, 3, true}, {3, 4, 6, 8, true}} {
		t.Run(fmt.Sprintf("%dq%dd/win%d/every%d/sinks%v", sh.heads, sh.dim, sh.win, sh.period, sh.sinks), func(t *testing.T) {
			composedCPUAttendBoth(t, sh.heads, sh.dim, sh.win, sh.period, sh.sinks)
		})
	}
}

func composedCPUAttendBoth(t *testing.T, H, D, win, P int, sinks bool) {
	const page, p0, R, capRows = 8, 2, 9, 16
	b := graph.New()
	k := b.In("k", D)
	n := b.KVAppend(k, k, 0)
	a := graph.Attrs{NHead: H, NKVHead: 1, HeadDim: D, Scale: 0.3, Window: win}
	var roles []jlm.Role
	if sinks {
		roles = append(roles, jlm.RoleAttnSinks)
	}
	b.Out(b.AttendBoth(b.In("q", H*D), graph.StateRef{State: graph.StateKV, Version: n},
		graph.StateRef{State: graph.StateKV, Hist: 1}, P, D, a, roles...))
	g := b.G
	ents := History{ValuesInKeys: true, Period: P}
	_, codes, err := Default().Form(Request{Graph: g, History: History{ValuesInKeys: true}, More: []History{ents},
		Step:   Step{Rows: capRows, Page: page, Keys: 4 * page, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(hostArm().tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(13))
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	last := p0 + R
	rows := make([][]float32, last)
	for p := range rows {
		rows[p] = rnd(D)
	}
	nEnt := last / P
	entries := make([][]float32, nEnt)
	for j := range entries {
		entries[j] = rnd(D)
	}
	q := rnd(capRows * H * D)
	sink := rnd(H)
	kin := make([]float32, capRows*D)
	for r := range R {
		copy(kin[r*D:], rows[p0+r])
	}
	put := func(pool []uint32, at int, row []float32) {
		for d, x := range row {
			h, i := uint32(quant.EncodeHalf(x)), at*D+d
			if d%2 == 0 {
				pool[i/2] = pool[i/2]&0xFFFF0000 | h
			} else {
				pool[i/2] = pool[i/2]&0xFFFF | h<<16
			}
		}
	}
	slots := ents.slots(page)
	rowPool, entPool := make([]uint32, 4*page*D/2), make([]uint32, 4*slots*D/2)
	for p := range p0 {
		put(rowPool, p, rows[p])
	}
	for j := range entries {
		tp := (j+1)*P - 1
		put(entPool, tp/page*slots+tp%page/P, entries[j])
	}
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	// want is the definition, with the rows and the entries each kept or not.
	want := func(useRows, useEnts bool) []float64 {
		out := make([]float64, R*H*D)
		for r := range R {
			t := p0 + r
			var keys [][]float32
			if useRows {
				first := 0
				if win > 0 {
					first = max(0, t+1-win)
				}
				keys = append(keys, rows[first:t+1]...)
			}
			if useEnts {
				keys = append(keys, entries[:(t+1)/P]...)
			}
			for h := range H {
				sc := make([]float64, len(keys))
				mx := math.Inf(-1)
				for i, key := range keys {
					var s float64
					for d := range D {
						s += float64(q[(r*H+h)*D+d]) * round(key[d])
					}
					sc[i] = s * 0.3
					mx = math.Max(mx, sc[i])
				}
				if sinks {
					mx = math.Max(mx, float64(sink[h]))
				}
				var sum float64
				for i := range sc {
					sc[i] = math.Exp(sc[i] - mx)
					sum += sc[i]
				}
				if sinks {
					sum += math.Exp(float64(sink[h]) - mx)
				}
				for d := range D {
					var o float64
					for i, key := range keys {
						o += sc[i] / sum * round(key[d])
					}
					out[(r*H+h)*D+d] = o
				}
			}
		}
		return out
	}
	out := make([]float32, capRows*H*D)
	desc, wdesc := make([]uint32, capRows*4), make([]uint32, capRows*4)
	for r := range R {
		p := uint32(p0 + r)
		copy(desc[r*4:], []uint32{0, 0, p + 1, p})
		first := uint32(0)
		if win > 0 {
			first = uint32(max(0, int(p)+1-win))
		}
		copy(wdesc[r*4:], []uint32{0, first, p + 1, p})
	}
	bufs := map[string][]uint32{"k": f32Words(kin), "q": f32Words(q), "rows.n": {R}, "q.off": make([]uint32, capRows*4)}
	if sinks {
		bufs[jlm.RoleAttnSinks.String()] = f32Words(sink)
	}
	for _, c := range codes {
		for _, bd := range c.Binds {
			o := bd.Operand
			if o.Kind != OperandTable {
				continue
			}
			switch {
			case o.Table == TableRows:
				bufs[o.Name] = desc
			case o.Table == TableWindows:
				bufs[o.Name] = wdesc
			case o.Table == TablePages:
				bufs[o.Name] = []uint32{0, 1, 2, 3}
			case o.Table == TableKeys && o.Instance == 0:
				bufs[o.Name] = rowPool
			case o.Table == TableKeys && o.Instance == 1:
				bufs[o.Name] = entPool
			}
		}
	}
	bufs[g.Values[g.Out[0]].Name] = f32Words(out)
	if err := runCPURows(hostArm(), codes, g, bufs, R); err != nil {
		t.Fatal(err)
	}
	for _, bd := range codes[len(codes)-1].Binds {
		if o := bd.Operand; bd.Writes && o.Kind == OperandScratch {
			copy(out, wordsF32(bufs["own:"+o.Name]))
		}
	}
	nmse := func(w []float64) float64 {
		var num, den float64
		for i := range w {
			d := float64(out[i]) - w[i]
			num += d * d
			den += w[i] * w[i]
		}
		return num / den
	}
	e := nmse(want(true, true))
	t.Logf("%d rows over a window of %d and an entry each %d: NMSE %.3e against the concatenation", R, win, P, e)
	if math.IsNaN(e) || e > 1e-9 {
		t.Fatalf("the composed attention over two histories: NMSE %.3e against the concatenation", e)
	}
	if e := nmse(want(true, false)); !(e > 1e-6) {
		t.Errorf("without the entries the NMSE is %.3e: the gate cannot see them", e)
	}
	if e := nmse(want(false, true)); !(e > 1e-6) {
		t.Errorf("without the rows the NMSE is %.3e: the gate cannot see them", e)
	}
}
