package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
)

// GridAttendCase is a grid's attention over its keys and values downsampled
// at stride S (1: the grid's own rows), Heads heads of KD reading one key
// and value head.
type GridAttendCase struct {
	GridRowsCase
	Heads, KD int
}

func (g GridAttendCase) String() string {
	return fmt.Sprintf("%v/%d heads of %d", g.GridRowsCase, g.Heads, g.KD)
}

// GridAttendCases is MobileNet-V5's attention at ragged grids.
func GridAttendCases() []GridAttendCase {
	return []GridAttendCase{{GridRowsCase{8, 8, 16, 3, 2, 0}, 4, 8}, {GridRowsCase{7, 9, 5, 3, 2, 0}, 3, 4},
		{GridRowsCase{5, 5, 17, 5, 2, 0}, 2, 9}, {GridRowsCase{9, 7, 8, 3, 1, 0}, 2, 8}, {GridRowsCase{1, 7, 5, 3, 2, 0}, 1, 4}}
}

// GridAttendGraph is a MobileNet-V5 attention block as the front end states
// it: the grid normed, its queries, its keys and values each from its own
// depthwise convolution at the case's stride and norm, appended at their
// grid's rows, every query attending every key, projected and added back.
func GridAttendGraph(g GridAttendCase) (*graph.Graph, error) {
	b := graph.New()
	x := b.In("x", g.C)
	xn := b.Norm(x, jlm.RoleVAttnNorm, 1e-6)
	q := b.MatVec(xn, jlm.RoleVAttnQ, g.Heads*g.KD)
	side := func(r, down, norm jlm.Role) graph.ValueID {
		in := xn
		if g.S > 1 {
			d := b.Emit(graph.OpConv, []graph.ValueID{xn}, []graph.Result{{Name: down.String(), Width: g.C, Rows: g.Out()}},
				[]graph.Binding{{Role: down, K: g.C, Rows: g.K * g.K}}, graph.Attrs{Taps: g.K, GridW: g.W, Stride: g.S}, nil, nil)[0]
			in = b.Norm(d, norm, 1e-6)
		}
		return b.MatVec(in, r, g.KD)
	}
	k := side(jlm.RoleVAttnK, jlm.RoleVMNKDown, jlm.RoleVMNKDownNorm)
	v := side(jlm.RoleVAttnV, jlm.RoleVMNVDown, jlm.RoleVMNVDownNorm)
	a := b.Attend(q, b.KVAppend(k, v, 0), graph.Attrs{NHead: g.Heads, NKVHead: 1, HeadDim: g.KD, Scale: 1 / math.Sqrt(float64(g.KD)), Bidir: true})
	b.Out(b.Add(b.MatVec(a, jlm.RoleVAttnOut, g.C), x))
	inv := map[jlm.Role]graph.Shape{jlm.RoleVAttnNorm: {K: g.C}, jlm.RoleVAttnQ: {K: g.C, Rows: g.Heads * g.KD},
		jlm.RoleVAttnK: {K: g.C, Rows: g.KD}, jlm.RoleVAttnV: {K: g.C, Rows: g.KD}, jlm.RoleVAttnOut: {K: g.Heads * g.KD, Rows: g.C}}
	if g.S > 1 {
		for _, r := range []jlm.Role{jlm.RoleVMNKDown, jlm.RoleVMNVDown} {
			inv[r] = graph.Shape{K: g.C, Rows: g.K * g.K}
		}
		inv[jlm.RoleVMNKDownNorm], inv[jlm.RoleVMNVDownNorm] = graph.Shape{K: g.C}, graph.Shape{K: g.C}
	}
	return b.G, graph.Verify(b.G, graph.Facts{Inventory: inv})
}

// GridAttendData is a case's weights by role, its input, and the oracle's
// result.
func GridAttendData(g GridAttendCase, rng *rand.Rand) (map[jlm.Role][]float32, []float32, []float32) {
	n, m, qd := g.H*g.W, g.Out(), g.Heads*g.KD
	roles := map[jlm.Role][]float32{
		jlm.RoleVAttnNorm: RandRow(rng, g.C, 1), jlm.RoleVAttnQ: RandRow(rng, qd*g.C, 1/math.Sqrt(float64(g.C))),
		jlm.RoleVAttnK: RandRow(rng, g.KD*g.C, 1/math.Sqrt(float64(g.C))), jlm.RoleVAttnV: RandRow(rng, g.KD*g.C, 1/math.Sqrt(float64(g.C))),
		jlm.RoleVAttnOut: RandRow(rng, g.C*qd, 1/math.Sqrt(float64(qd))),
		jlm.RoleVMNKDown: RandRow(rng, g.K*g.K*g.C, 0.5), jlm.RoleVMNVDown: RandRow(rng, g.K*g.K*g.C, 0.5),
		jlm.RoleVMNKDownNorm: RandRow(rng, g.C, 1), jlm.RoleVMNVDownNorm: RandRow(rng, g.C, 1),
	}
	if g.S == 1 {
		m = n
	}
	x := RandRow(rng, n*g.C, 1)
	xn := make([]float32, n*g.C)
	for r := range n {
		oracle.RMSNorm32(xn[r*g.C:(r+1)*g.C], x[r*g.C:(r+1)*g.C], roles[jlm.RoleVAttnNorm], 1e-6)
	}
	side := func(r, down, norm jlm.Role) []float32 {
		in := xn
		if g.S > 1 {
			in = make([]float32, m*g.C)
			oracle.DWConv32(in, xn, g.H, g.W, g.C, g.K, g.S, roles[down])
			for i := range m {
				oracle.RMSNorm32(in[i*g.C:(i+1)*g.C], in[i*g.C:(i+1)*g.C], roles[norm], 1e-6)
			}
		}
		out := make([]float32, m*g.KD)
		for i := range m {
			oracle.MatVec(out[i*g.KD:(i+1)*g.KD], quant.F32, F32Bytes(roles[r]), in[i*g.C:(i+1)*g.C], g.KD, g.C)
		}
		return out
	}
	k := side(jlm.RoleVAttnK, jlm.RoleVMNKDown, jlm.RoleVMNKDownNorm)
	v := side(jlm.RoleVAttnV, jlm.RoleVMNVDown, jlm.RoleVMNVDownNorm)
	want := make([]float32, n*g.C)
	q, att, p := make([]float32, qd), make([]float32, qd), make([]float64, m)
	o := make([]float32, g.C)
	for r := range n {
		oracle.MatVec(q, quant.F32, F32Bytes(roles[jlm.RoleVAttnQ]), xn[r*g.C:(r+1)*g.C], qd, g.C)
		for h := range g.Heads {
			mx := math.Inf(-1)
			for j := range m {
				s := 0.0
				for d := range g.KD {
					s += float64(q[h*g.KD+d]) * float64(k[j*g.KD+d])
				}
				p[j] = s / math.Sqrt(float64(g.KD))
				mx = max(mx, p[j])
			}
			sum := 0.0
			for j := range m {
				p[j] = math.Exp(p[j] - mx)
				sum += p[j]
			}
			for d := range g.KD {
				a := 0.0
				for j := range m {
					a += p[j] * float64(v[j*g.KD+d])
				}
				att[h*g.KD+d] = float32(a / sum)
			}
		}
		oracle.MatVec(o, quant.F32, F32Bytes(roles[jlm.RoleVAttnOut]), att, g.C, qd)
		for i := range o {
			want[r*g.C+i] = o[i] + x[r*g.C+i]
		}
	}
	return roles, x, want
}

// GridAttendTables is a case's history tables: one page of the key grid's
// rows m, every row attending [0, m) and writing at its own row, the pools
// poisoned to keyRows rows past the page and as many pages listed.
func GridAttendTables(g GridAttendCase, page, keyRows int) map[string][]uint32 {
	n := g.H * g.W
	rows := make([]uint32, n*4)
	m := g.Out()
	if g.S == 1 {
		m = n
	}
	for r := range n {
		copy(rows[r*4:], []uint32{0, 0, uint32(m), uint32(r)})
	}
	pages := (keyRows + page - 1) / page
	pg := make([]uint32, pages)
	for i := range pg {
		pg[i] = uint32(i)
	}
	nan := math.Float32bits(float32(math.NaN()))
	pool := func() []uint32 {
		w := make([]uint32, pages*page*g.KD)
		for i := range w {
			w[i] = nan
		}
		return w
	}
	return map[string][]uint32{"rows": rows, "pages": pg, "rows.n": {uint32(n)}, "q.off": make([]uint32, n*4),
		"kv.k": pool(), "kv.v": pool()}
}

// GridAttendFormats is the case's formats: every weight F32.
func GridAttendFormats() map[jlm.Role]jlm.Type {
	f := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleVAttnNorm, jlm.RoleVAttnQ, jlm.RoleVAttnK, jlm.RoleVAttnV, jlm.RoleVAttnOut,
		jlm.RoleVMNKDown, jlm.RoleVMNVDown, jlm.RoleVMNKDownNorm, jlm.RoleVMNVDownNorm} {
		f[r] = jlm.TypeF32
	}
	return f
}

// TestComposedCPUGridAttend executes MobileNet-V5's attention on the CPU
// device -- keys and values of a stride-2 grid appended at that grid's rows,
// every query row of the full grid attending them -- held to the oracle,
// guards past every value and the pools poisoned past the key grid. The
// append run at the query grid's rows must fail: it writes past the key
// grid's rows.
func TestComposedCPUGridAttend(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	right, seen := 0, 0
	for _, g := range GridAttendCases() {
		gr, err := GridAttendGraph(g)
		if err != nil {
			t.Fatalf("%v: %v", g, err)
		}
		roles, x, want := GridAttendData(g, rng)
		n, m := g.H*g.W, g.Out()
		if g.S == 1 {
			m = n
		}
		step := Step{Rows: n, Page: m, Keys: m, ActWindow: 32}
		run := func(r *Registry) ([]float32, map[string][]uint32, error) {
			_, codes, err := r.Form(Request{Graph: gr, Formats: GridAttendFormats(), Step: step, Device: cpuDevice()})
			if err != nil {
				return nil, nil, err
			}
			tabs := GridAttendTables(g, m, n+m)
			vals := map[graph.ValueID][]uint32{gr.In[0]: f32Words(x)}
			byRole := map[jlm.Role][]uint32{}
			for r, v := range roles {
				byRole[r] = f32Words(v)
			}
			err = runCPUGridValues(codes, gr, vals, [3][]uint32{}, tabs, gridValueRows(gr, n), byRole)
			if err != nil {
				return nil, tabs, err
			}
			return wordsF32(vals[gr.Out[0]]), tabs, nil
		}
		got, tabs, err := run(Default().WithoutInt8(DeviceCPU))
		if err != nil {
			t.Errorf("%v: %v", g, err)
			continue
		}
		if e := rowsNMSE(got[:n*g.C], want); math.IsNaN(e) || !(e < 1e-7) {
			t.Errorf("%v: NMSE %.3e against the oracle", g, e)
			continue
		}
		if w := PoolWrittenPast(tabs, m*g.KD); w >= 0 {
			t.Errorf("%v: the key pool written at word %d, past the key grid's %d rows", g, w, m)
			continue
		}
		right++
		if g.S == 1 {
			continue
		}
		_, tabs, err = run(Misrowed(Default().WithoutInt8(DeviceCPU), graph.OpKVAppend, n))
		switch {
		case err != nil:
			t.Logf("%v: the append at the query grid's rows: %v", g, err)
			seen++
		case PoolWrittenPast(tabs, m*g.KD) >= 0:
			seen++
		}
	}
	t.Logf("%d of %d grids' attention held to the oracle; the append at the query grid's rows seen at %d", right, len(GridAttendCases()), seen)
	if right != len(GridAttendCases()) {
		t.Fatalf("%d of %d grids' attention held to the oracle", right, len(GridAttendCases()))
	}
	if seen == 0 {
		t.Fatal("the append at the query grid's rows is seen at no grid: the gate cannot see it")
	}
}

// PoolWrittenPast is the first word of the key or value pool past words
// that is not the poison, -1 for none.
func PoolWrittenPast(tabs map[string][]uint32, words int) int {
	nan := math.Float32bits(float32(math.NaN()))
	for _, name := range []string{"kv.k", "kv.v"} {
		for i, w := range tabs[name][words:] {
			if w != nan {
				return words + i
			}
		}
	}
	return -1
}

// gridValueRows is graph g's value rows at a step of step rows.
func gridValueRows(g *graph.Graph, step int) func(graph.ValueID) int {
	return func(v graph.ValueID) int {
		if r := g.Values[v].Rows; r > 0 {
			return r
		}
		return step
	}
}
