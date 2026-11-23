package composer

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// attendBiasCase is an attention a bias gate forms: a score bias of extent
// distances a head (graph.Attrs.Extent; 0 none), its sinks (0 none, 1 a
// head's weight, 2 a row's input, graph.Attrs.RowSinks) and its window (0
// none).
type attendBiasCase struct {
	extent, sinks, window int
}

// TestComposedCPUAttendBias executes the composer's attention over a chunk
// of rows on the CPU with a score bias by backward distance and sinks taken
// from a row's input, each beside the per-head sink weight, causal and
// sliding: each row's key and value appended to the paged binary16 history
// at positions past the extent, each row's scores gaining its head's bias at
// the key's distance below the extent and nothing past it, held row by row to
// the definition, at ragged head counts and widths, fused and unfused. The
// violations must fail: the bias zeroed, a row's sinks every row's first's
// or a head's sinks zeroed, and the form must be no definition by the
// distance's sign turned or a distance past the extent clamped to its last.
func TestComposedCPUAttendBias(t *testing.T) { gateComposedCPUAttendBias(t, hostArm()) }
func gateComposedCPUAttendBias(t *testing.T, cpuOn cpuArm) {
	unfused := Default()
	p := unfused.Profiles[DeviceCPU]
	p.Off = map[string]bool{"fuse": true}
	unfused.Profiles[DeviceCPU] = p
	cases := []attendBiasCase{{5, 0, 0}, {19, 2, 0}, {0, 2, 0}, {0, 1, 0}, {7, 1, 12}, {19, 2, 12}, {1, 0, 0}}
	for _, sh := range []struct{ heads, kvh, dim int }{{3, 1, 6}, {6, 2, 8}, {5, 5, 4}, {8, 2, 64}} {
		for _, c := range cases {
			for _, arm := range []struct {
				name string
				reg  *Registry
			}{{"fused", Default()}, {"unfused", unfused}} {
				t.Run(fmt.Sprintf("%dq%dkv%dd/extent%d/sinks%d/window%d/%s", sh.heads, sh.kvh, sh.dim, c.extent, c.sinks, c.window, arm.name), func(t *testing.T) {
					composedCPUAttendBias(t, cpuOn, arm.reg, sh.heads, sh.kvh, sh.dim, c)
				})
			}
		}
	}
}

func composedCPUAttendBias(t *testing.T, cpuOn cpuArm, reg *Registry, heads, kvh, dim int, c attendBiasCase) {
	const page, p0, R, capRows, scale = 32, 40, 9, 16, 0.125
	E := c.extent
	b := graph.New()
	n := b.KVAppend(b.In("k", kvh*dim), b.In("v", kvh*dim), 0)
	q := b.In("q", heads*dim)
	var extra []graph.ValueID
	if E > 0 {
		extra = append(extra, b.In("bias", heads*E))
	}
	if c.sinks == 2 {
		extra = append(extra, b.In("sinks", heads))
	}
	var roles []jlm.Role
	inv := map[jlm.Role]graph.Shape{}
	if c.sinks == 1 {
		roles = []jlm.Role{jlm.RoleAttnSinks}
		inv[jlm.RoleAttnSinks] = graph.Shape{K: heads}
	}
	b.Out(b.AttendWith(q, graph.StateRef{State: graph.StateKV, Version: n}, graph.Attrs{NHead: heads, NKVHead: kvh, HeadDim: dim,
		Scale: scale, Window: c.window, Extent: E, RowSinks: c.sinks == 2}, extra, roles...))
	g := b.G
	if err := graph.Verify(g, graph.Facts{Inventory: inv}); err != nil {
		t.Fatal(err)
	}
	at := g.Ops[len(g.Ops)-1]
	if _, bias, sinks, ok := graph.AttendInputs(at); !ok || (bias >= 0) != (E > 0) || (sinks >= 0) != (c.sinks == 2) {
		t.Fatalf("the attention reads bias %d and sinks %d: the gate did not state its case", bias, sinks)
	}
	_, codes, err := reg.Form(Request{Graph: g,
		Step:   Step{Rows: capRows, Page: page, Keys: 4 * page, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	reads := func(name string) bool {
		for _, cd := range codes {
			for _, bd := range cd.Binds {
				if bd.Read && bd.Operand.Kind == OperandValue && g.Values[bd.Operand.Value].Name == name {
					return true
				}
			}
		}
		return false
	}
	if E > 0 && !reads("bias") || c.sinks == 2 && !reads("sinks") {
		t.Fatalf("no kernel reads the bias (%v) or the row's sinks (%v) it states", reads("bias"), reads("sinks"))
	}
	rng := rand.New(rand.NewSource(int64(17 + heads + E + c.sinks + c.window)))
	w, qw := kvh*dim, heads*dim
	rnd := func(n int, m float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * m)
		}
		return x
	}
	ks, vs := make([][]float32, p0+R), make([][]float32, p0+R)
	for p := range p0 + R {
		ks[p], vs[p] = rnd(w, 1), rnd(w, 1)
	}
	qv := rnd(capRows*qw, 1)
	k, v := make([]float32, capRows*w), make([]float32, capRows*w)
	for r := range R {
		copy(k[r*w:], ks[p0+r])
		copy(v[r*w:], vs[p0+r])
	}
	bias := rnd(capRows*heads*max(1, E), 1)
	sinkv := rnd(heads, 1)
	if c.sinks == 2 {
		sinkv = rnd(capRows*heads, 1)
	}
	sinkOf := func(sv []float32, r, h int) float64 {
		if c.sinks == 2 {
			return float64(sv[r*heads+h])
		}
		return float64(sv[h])
	}
	pool := func(rows [][]float32) []uint32 {
		out := make([]uint32, 4*page*w/2)
		for p := range p0 {
			for d, x := range rows[p] {
				h, i := uint32(quant.EncodeHalf(x)), p*w+d
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
	first := func(r int) int {
		if c.window > 0 {
			return max(0, p0+r+1-c.window)
		}
		return 0
	}
	// biasAt is the bias row r's key at p gains by the definition's distance
	// (mode 0), or by a violation's: the distance's sign turned (1), a
	// distance past the extent clamped to its last (2).
	biasAt := func(mode, r, h, p int) float64 {
		d := p0 + r - p
		switch mode {
		case 1:
			d = -d
		case 2:
			d = min(d, E-1)
		}
		if E == 0 || d < 0 || d >= E {
			return 0
		}
		return float64(bias[(r*heads+h)*E+d])
	}
	def := func(mode int) []float64 {
		want := make([]float64, R*qw)
		for r := range R {
			for h := range heads {
				kv := h / (heads / kvh)
				var sc []float64
				for p := first(r); p <= p0+r; p++ {
					var s float64
					for d := range dim {
						s += float64(qv[r*qw+h*dim+d]) * round(ks[p][kv*dim+d])
					}
					sc = append(sc, s*scale+biasAt(mode, r, h, p))
				}
				nk := len(sc)
				if c.sinks > 0 {
					sc = append(sc, sinkOf(sinkv, r, h))
				}
				oracle.Softmax(sc)
				for d := range dim {
					var o float64
					for i := range nk {
						o += sc[i] * round(vs[first(r)+i][kv*dim+d])
					}
					want[r*qw+h*dim+d] = o
				}
			}
		}
		return want
	}
	run := func(sv []float32) []float32 {
		out := make([]float32, capRows*qw)
		rows, win := make([]uint32, capRows*4), make([]uint32, capRows*4)
		for r := range R {
			p := uint32(p0 + r)
			copy(rows[r*4:], []uint32{0, 0, p + 1, p})
			copy(win[r*4:], []uint32{0, uint32(first(r)), p + 1, p})
		}
		bufs := map[string][]uint32{"q": f32Words(qv), "k": f32Words(k), "v": f32Words(v), "bias": f32Words(bias),
			"rows": rows, "rows.win": win, "pages": {0, 1, 2, 3}, "q.off": make([]uint32, capRows*4), "rows.n": {R},
			"kv.k": pool(ks), "kv.v": pool(vs)}
		if c.sinks == 1 {
			bufs[jlm.RoleAttnSinks.String()] = f32Words(sv)
		} else {
			bufs["sinks"] = f32Words(sv)
		}
		bufs[g.Values[g.Out[0]].Name] = f32Words(out)
		if err := runCPURows(cpuOn, codes, g, bufs, R); err != nil {
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
		for i := range R * qw {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		return num / den
	}
	want := def(0)
	got := run(sinkv)
	worst := 0.0
	for r := range R {
		var num, den float64
		for i := r * qw; i < (r+1)*qw; i++ {
			d := float64(got[i]) - want[i]
			num += d * d
			den += want[i] * want[i]
		}
		e := num / den
		if math.IsNaN(e) || e > 1e-8 {
			t.Errorf("row %d at position %d: NMSE %.3e against the definition", r, p0+r, e)
		}
		worst = max(worst, e)
	}
	t.Logf("%d rows: worst NMSE %.3e against the definition", R, worst)
	if E > 0 {
		kept := slices.Clone(bias)
		clear(bias)
		e := nmse(run(sinkv), want)
		copy(bias, kept)
		if !(e > 1e-6) {
			t.Errorf("with the bias zeroed the NMSE is %.3e: the gate cannot see the bias", e)
		}
		modes := []int{2}
		if E > 1 {
			modes = []int{1, 2}
		}
		for _, m := range modes {
			if m == 2 && c.window > 0 && c.window <= E {
				// The window holds no key at a distance past the extent.
				continue
			}
			if e := nmse(got, def(m)); !(e > 1e-6) {
				t.Errorf("the form is NMSE %.3e from the bias by violation %d's distance: the gate cannot tell them apart", e, m)
			}
		}
	}
	switch c.sinks {
	case 1:
		if e := nmse(run(make([]float32, heads)), want); !(e > 1e-6) {
			t.Errorf("with the sinks zeroed the NMSE is %.3e: the gate cannot see the sinks", e)
		}
	case 2:
		firsts := slices.Clone(sinkv)
		for i := range firsts {
			firsts[i] = sinkv[i%heads]
		}
		if e := nmse(run(firsts), want); !(e > 1e-6) {
			t.Errorf("with every row's sinks the first's the NMSE is %.3e: the gate cannot see a row's sinks", e)
		}
	}
}
