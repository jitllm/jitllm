package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/nn"
)

// TestComposedGPUSelectPooled is TestComposedCPUSelectPooled on every GPU:
// the composer's pooled selection (graph.Attrs.Pooled, DeepSeek V4's and
// QSA's) and the attention over what it keeps, over the paged history a
// card holds (keys float32 by dimension, values binary16 by position), at a
// decode row and at a prompt's rows, each row its own selection over the
// positions up to its own, held to the definition; and each departure -- a
// block pooled from its first key, the incomplete tail ranked, a block
// turned at its last position -- must move the output.
func TestComposedGPUSelectPooled(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	type pcase struct {
		name                      string
		h, kv, dk, w, hi, di, off int
		b, k, nrot                int
		neox                      bool
		pos, rows                 int
	}
	for _, d := range devs {
		for _, pc := range []pcase{
			{"neox", 4, 2, 8, 24, 4, 8, 16, 4, 4, 8, true, 70, 1},
			{"pairs", 4, 2, 8, 24, 4, 8, 16, 4, 4, 4, false, 70, 1},
			{"whole", 4, 2, 8, 24, 4, 8, 16, 4, 4, 8, true, 75, 1},
			{"dense", 4, 2, 8, 24, 4, 8, 16, 4, 18, 8, true, 70, 1},
			{"prompt", 4, 2, 8, 24, 4, 8, 16, 4, 4, 8, true, 75, 9},
			{"prompt-from-start", 4, 2, 8, 24, 4, 8, 16, 4, 4, 8, true, 40, 41},
		} {
			t.Run(fmt.Sprintf("%s/%s", d.API(), pc.name), func(t *testing.T) {
				gpuSelectPooled(t, d, pc.h, pc.kv, pc.dk, pc.w, pc.hi, pc.di, pc.off, pc.b, pc.k, pc.nrot, pc.neox, pc.pos, pc.rows)
			})
		}
	}
}

func gpuSelectPooled(t *testing.T, dev backend.Device, h, kvh, dk, w, hi, di, off, blk, topk, nrot int, neox bool, pos, R int) {
	const page = 32
	n, p0 := pos+1, pos+1-R
	b := graph.New()
	k, v := b.In("k", w), b.In("v", w)
	hist := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, v, 0)}
	layout := graph.RopeNorm
	if neox {
		layout = graph.RopeNeox
	}
	sa := graph.Attrs{NHead: hi, NKVHead: 1, HeadDim: di, Offset: off, Width: w, Span: blk, TopK: topk,
		Fn: graph.MapMax, Eps: 1e-6, NRot: nrot, Base: 10000, Layout: layout}
	iq := b.In("iq", hi*di)
	sel := b.SelectPooled(iq, hist, sa, jlm.RoleIdxKNorm)
	q := b.In("q", h*dk)
	b.Out(b.AttendSelected(q, hist, sel, graph.Attrs{NHead: h, NKVHead: kvh, HeadDim: dk, Scale: 0.125, Width: w}))
	g := b.G
	if err := graph.Verify(g, graph.Facts{Inventory: map[jlm.Role]graph.Shape{jlm.RoleIdxKNorm: {K: di}}}); err != nil {
		t.Fatal(err)
	}
	keysCap := (n + page - 1) / page * page
	_, codes, err := composer.Default().Form(composer.Request{Graph: g,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}},
		Step:    composer.Step{Rows: R, Page: page, Keys: keysCap, Seqs: 1}, Device: gpuDevice(dev)})
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
		keys[p], vals[p] = vec(w), vec(w)
	}
	var kin, vin []float32
	for p := p0; p < n; p++ {
		kin, vin = append(kin, keys[p]...), append(vin, vals[p]...)
	}
	npg := keysCap / page
	kpool, vpool := poisoned(npg*page*w), poisoned(npg*page*w/2)
	for p := range n {
		for e := range w {
			vals[p][e] = float32(quant.DecodeHalf(quant.EncodeHalf(vals[p][e])))
			if p >= p0 {
				continue
			}
			kpool[(p/page*w+e)*page+p%page] = math.Float32bits(keys[p][e])
			hv, i := uint32(quant.EncodeHalf(vals[p][e])), p*w+e
			if i%2 == 0 {
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			} else {
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	qv, iqv, wv := vec(R*h*dk), vec(R*hi*di), vec(di)
	for i := range wv {
		wv[i] = 0.5 + float32(math.Abs(float64(wv[i])))
	}
	// want is row r's definition over the positions up to its own, those
	// keep holds where keep is set.
	want := func(r int, keep []bool) []float64 {
		m := p0 + r + 1
		out := make([]float64, h*dk)
		gqa := h / kvh
		for hh := range h {
			kv := hh / gqa
			mx := math.Inf(-1)
			s := make([]float64, m)
			for p := range m {
				if keep != nil && !keep[p] {
					s[p] = math.Inf(-1)
					continue
				}
				for e := range dk {
					s[p] += float64(qv[(r*h+hh)*dk+e]) * float64(keys[p][kv*dk+e])
				}
				s[p] *= 0.125
				mx = math.Max(mx, s[p])
			}
			var sum float64
			for p := range s {
				s[p] = math.Exp(s[p] - mx)
				sum += s[p]
			}
			for e := range dk {
				var o float64
				for p := range s {
					o += s[p] / sum * float64(vals[p][kv*dk+e])
				}
				out[hh*dk+e] = o
			}
		}
		return out
	}
	keepOf := func(r, fault int) []bool {
		return composer.PooledKeep(blk, topk, nrot, neox, hi, di, off, keys, iqv[r*hi*di:(r+1)*hi*di], wv, p0+r+1, fault)
	}
	planes := nn.Rope{NRot: nrot, Base: 10000, Neox: neox}.TabPlanes(nrot / 2)
	rows := make([]uint32, 0, 4*R)
	for r := range R {
		rows = append(rows, 0, 0, uint32(p0+r+1), uint32(p0+r))
	}
	pages := make([]uint32, npg)
	for i := range pages {
		pages[i] = uint32(i)
	}
	vs := map[graph.ValueID][]uint32{g.In[0]: f32w(kin), g.In[1]: f32w(vin), g.In[2]: f32w(iqv), g.In[3]: f32w(qv),
		g.Out[0]: poisoned(R * h * dk)}
	tables := map[string][]uint32{"rows": rows, "rows.win": rows, "rows.n": {uint32(R)}, "pages": pages, "q.off": {0},
		"kv.k": kpool, "kv.v": vpool, "rope.planes": f32w(planes)}
	vecs := map[jlm.Role][]uint32{jlm.RoleIdxKNorm: f32w(wv)}
	out, err := launchAttend(dev, codes, vs, tables, vecs, g.Out[0])
	if err != nil {
		t.Fatal(err)
	}
	got := wf32(out)
	nmse := func(r int, want []float64) float64 {
		var num, den float64
		for i := range want {
			e := float64(got[r*h*dk+i]) - want[i]
			num += e * e
			den += want[i] * want[i]
		}
		return num / den
	}
	worst, kept, dropped := 0.0, 0, false
	for r := range R {
		keep := keepOf(r, 0)
		e := nmse(r, want(r, keep))
		if math.IsNaN(e) || e > 1e-6 {
			t.Errorf("row %d at position %d: the composed pooled selection on the card is %.3e NMSE from the definition", r, p0+r, e)
		}
		worst = max(worst, e)
		c := 0
		for _, k := range keep {
			if k {
				c++
			}
		}
		kept += c
		dropped = dropped || c < p0+r+1
	}
	t.Logf("%d rows, %d positions kept in all: worst NMSE %.3e against the definition", R, kept, worst)
	if !dropped {
		return
	}
	for fault, name := range []string{1: "a block pooled from its first key", 2: "the incomplete tail ranked",
		3: "a block turned at its last position"} {
		if fault == 0 {
			continue
		}
		seen, applies := 0.0, false
		for r := range R {
			if fault == 2 && (p0+r+1)%blk == 0 {
				continue
			}
			applies = true
			seen = max(seen, nmse(r, want(r, keepOf(r, fault))))
		}
		if !applies {
			continue
		}
		t.Logf("violation %q: NMSE %.3e at worst", name, seen)
		if !(seen > 1e-4) {
			t.Errorf("against the definition with %s the NMSE is %.3e: the gate cannot see it", name, seen)
		}
	}
}
