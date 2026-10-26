package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// recRuns is a step's rows as runs: each run's rows of the step, in the
// order its sequence steps them.
type recRuns [][]int

// desc is the run descriptor of n rows (kernels.RunDescWords' layout):
// run j reads slot in+j and writes slot out+j, a step of S runs repeating
// the last in the entries past it.
func (rs recRuns) desc(n, in, out int) []uint32 {
	S := len(rs)
	w := make([]uint32, 1+7*n)
	w[0] = uint32(n)
	at := 0
	for j := range n {
		s, start := min(j, S-1), at
		if j >= S {
			start = at - len(rs[s])
		}
		w[1+2*j], w[2+2*j] = uint32(in+s), uint32(out+s)
		w[1+2*n+2*j], w[2+2*n+2*j] = uint32(start), uint32(len(rs[s]))
		if j < S {
			for k, r := range rs[j] {
				w[1+4*n+at+k] = uint32(r)
				w[1+5*n+r] = uint32(at + k)
				w[1+6*n+r] = uint32(j)
			}
			at += len(rs[j])
		}
	}
	return w
}

// recRun is a form's launch over its bindings: graph values by id, the
// step's tables by name, weights by role, its own scratch poisoned.
type recRun struct {
	d      backend.Device
	codes  []jit.Code
	kerns  []backend.Kernel
	vals   map[graph.ValueID][]uint32
	tables map[string][]uint32
	wts    map[jlm.Role][]uint32
	bufs   map[string]backend.Buf
}

func (r *recRun) close() {
	for _, k := range r.kerns {
		k.Close()
	}
	for _, b := range r.bufs {
		b.Free()
	}
}

func (r *recRun) buf(key string, w []uint32) (backend.Buf, error) {
	if b, ok := r.bufs[key]; ok {
		return b, nil
	}
	b, err := r.d.Alloc(max(4*len(w), 4))
	if err != nil {
		return nil, err
	}
	p := make([]byte, 4*len(w))
	for i, v := range w {
		binary.LittleEndian.PutUint32(p[4*i:], v)
	}
	if len(p) > 0 {
		if err := b.Write(p); err != nil {
			return nil, err
		}
	}
	r.bufs[key] = b
	return b, nil
}

// launch runs every code at its full grid and reads every value and table
// back.
func (r *recRun) launch() error {
	for ci, c := range r.codes {
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			var key string
			var w []uint32
			switch {
			case o.Kind == jit.OperandValue:
				key, w = fmt.Sprintf("v%d", o.Value), r.vals[o.Value]
			case o.Kind == jit.OperandTable && r.tables[o.Name] != nil:
				key, w = "t"+o.Name, r.tables[o.Name]
			case o.Kind == jit.OperandWeight:
				key, w = fmt.Sprintf("w%d", o.Role), r.wts[o.Role]
			case bd.Words != nil:
				key, w = fmt.Sprintf("c%d.%d", ci, i), bd.Words
			default:
				key, w = fmt.Sprintf("s%d.%s", o.Kind, o.Name), poisoned(int(max(bd.Bytes, 4)/4))
			}
			if w == nil {
				return fmt.Errorf("%s binds %+v, which the gate does not hold", c.Name, o)
			}
			b, err := r.buf(key, w)
			if err != nil {
				return err
			}
			args[i] = b
		}
		if err := r.kerns[ci].Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
			return fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	read := func(key string, w []uint32) error {
		b, ok := r.bufs[key]
		if !ok {
			return nil
		}
		p := make([]byte, 4*len(w))
		if err := b.Read(p); err != nil {
			return err
		}
		for i := range w {
			w[i] = binary.LittleEndian.Uint32(p[4*i:])
		}
		return nil
	}
	for id, w := range r.vals {
		if err := read(fmt.Sprintf("v%d", id), w); err != nil {
			return err
		}
	}
	for n, w := range r.tables {
		if err := read("t"+n, w); err != nil {
			return err
		}
	}
	return nil
}

func formRec(d backend.Device, dev jit.Device, g *graph.Graph, rows int, formats map[jlm.Role]jlm.Type) (*recRun, error) {
	_, codes, err := jit.Default().Form(jit.Request{Graph: g, Formats: formats, Step: jit.Step{Rows: rows, Page: 64, Keys: 1}, Device: dev})
	if err != nil {
		return nil, err
	}
	r := &recRun{d: d, codes: codes, vals: map[graph.ValueID][]uint32{}, tables: map[string][]uint32{},
		wts: map[jlm.Role][]uint32{}, bufs: map[string]backend.Buf{}}
	for _, c := range codes {
		if c.GPU == nil {
			r.close()
			return nil, fmt.Errorf("%s has no GPU code", c.Name)
		}
		k, err := d.Compile(c.GPU)
		if err != nil {
			r.close()
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		r.kerns = append(r.kerns, k)
	}
	return r, nil
}

func f32w(x []float32) []uint32 {
	w := make([]uint32, len(x))
	for i, v := range x {
		w[i] = math.Float32bits(v)
	}
	return w
}

func wf32(w []uint32) []float32 {
	x := make([]float32, len(w))
	for i, v := range w {
		x[i] = math.Float32frombits(v)
	}
	return x
}

// recSteps are the steps a recurrent gate runs at 8 rows: a prompt chunk's
// rows as one run, fewer live rows than the form holds, and ragged steps of
// several sequences, their rows interleaved.
var recSteps = []struct {
	name string
	n    int
	runs recRuns
}{
	{"chunk of 8", 8, recRuns{{0, 1, 2, 3, 4, 5, 6, 7}}},
	{"chunk of 5", 5, recRuns{{0, 1, 2, 3, 4}}},
	{"chunk of 1", 1, recRuns{{0}}},
	{"ragged 1+3+1", 5, recRuns{{2}, {0, 3, 4}, {1}}},
	{"ragged decode of 6", 6, recRuns{{0}, {1}, {2}, {3}, {4}, {5}}},
	{"ragged 4+4", 8, recRuns{{0, 2, 4, 6}, {1, 3, 5, 7}}},
}

// TestComposedGPURecurrentRuns forms OpDeltaRule and OpStateConv at a step
// of 8 rows on every GPU and runs them over a prompt chunk's rows (one run)
// and ragged steps (a run a sequence, rows interleaved, as the run
// descriptor lists them), padding rows NaN: each run's rows chain through
// its state from its slot read, held to the oracle at every row's output
// and at the state its slot written holds after; every other slot stays
// untouched. With each row made its own run -- the state not carried from
// row to row -- the forms must part.
func TestComposedGPURecurrentRuns(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		// And as a device that guarantees no subgroup: each sum through the
		// workgroup's memory, read by every thread before the next row
		// writes it.
		for _, lanes := range []int{0, 1} {
			name := d.API()
			dev := gpuDevice(d)
			if lanes > 0 {
				name, dev.Lanes = name+"/no subgroup", lanes
			}
			t.Run(name, func(t *testing.T) { recurrentRuns(t, d, dev) })
		}
	}
}

// recurrentRuns is TestComposedGPURecurrentRuns on device d as dev states it.
func recurrentRuns(t *testing.T, d backend.Device, dev jit.Device) {
	const R = 8
	rng := rand.New(rand.NewSource(23))
	ran, parted := 0, 0
	for _, s := range []deltaShape{
		{h: 4, kh: 2, dk: 16, dv: 8},
		{h: 4, kh: 2, dk: 7, dv: 5, tiled: true},
		{h: 8, kh: 2, dk: 40, dv: 4, chans: true},
		{h: 2, kh: 1, dk: 128, dv: 3, chans: true},
		{h: 6, kh: 3, dk: 33, dv: 2, tiled: true},
	} {
		g := deltaGraphOf(s)
		fr, err := formRec(d, dev, g, R, map[jlm.Role]jlm.Type{})
		if err != nil {
			t.Fatalf("%+v: %v", s, err)
		}
		for _, st := range recSteps {
			for _, chained := range []bool{true, false} {
				worst, err := runDelta(fr, g, s, st.n, st.runs, chained, rng)
				if err != nil {
					fr.close()
					t.Fatalf("%+v %s: %v", s, st.name, err)
				}
				longest := 0
				for _, run := range st.runs {
					longest = max(longest, len(run))
				}
				switch {
				case !chained && longest > 1 && worst > 1e-3:
					parted++
				case chained && worst > 2e-5:
					fr.close()
					t.Fatalf("%+v %s: the composed delta rule is %g from the oracle", s, st.name, worst)
				}
				if chained {
					ran++
				}
			}
		}
		fr.close()
	}
	for _, sh := range [][2]int{{2, 3}, {4, 7}, {4, 33}, {3, 64}, {5, 9}} {
		T, C := sh[0], sh[1]
		b := graph.New()
		x := b.In("x", C)
		b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: C}},
			[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: T, Rows: C}}, graph.Attrs{Taps: T},
			[]graph.StateRef{{Name: "rec.conv"}}, []graph.StateRef{{Name: "rec.conv", Version: 1}})[0])
		fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32})
		if err != nil {
			t.Fatalf("%d taps over %d: %v", T, C, err)
		}
		for _, st := range recSteps {
			for _, chained := range []bool{true, false} {
				worst, err := runConv(fr, b.G, T, C, st.n, st.runs, chained, rng)
				if err != nil {
					fr.close()
					t.Fatalf("%d taps over %d %s: %v", T, C, st.name, err)
				}
				longest := 0
				for _, run := range st.runs {
					longest = max(longest, len(run))
				}
				switch {
				case !chained && longest > 1 && worst > 1e-3:
					parted++
				case chained && worst > 1e-5:
					fr.close()
					t.Fatalf("%d taps over %d %s: the composed convolution is %g from the oracle", T, C, st.name, worst)
				}
				if chained {
					ran++
				}
			}
		}
		fr.close()
	}
	if parted == 0 {
		t.Fatal("a state not carried from row to row never parts from the oracle: the gate cannot see it")
	}
	t.Logf("%d steps held to the oracle; unchained runs parted %d times", ran, parted)
}

// deltaShape is a delta rule's geometry: value heads, key heads, the key's
// and the value's head widths, the pairing and the decay's width.
type deltaShape struct {
	h, kh, dk, dv int
	tiled, chans  bool
}

func deltaGraphOf(s deltaShape) *graph.Graph {
	b := graph.New()
	nd := s.h
	if s.chans {
		nd = s.h * s.dk
	}
	q, k := b.In("q", s.kh*s.dk), b.In("k", s.kh*s.dk)
	v, d, beta := b.In("v", s.h*s.dv), b.In("decay", nd), b.In("beta", s.h)
	b.Out(b.Emit(graph.OpDeltaRule, []graph.ValueID{q, k, v, d, beta}, []graph.Result{{Name: "o", Width: s.h * s.dv}}, nil,
		graph.Attrs{NHead: s.h, NKVHead: s.kh, HeadDim: s.dk, Tiled: s.tiled},
		[]graph.StateRef{{Name: "rec.state"}}, []graph.StateRef{{Name: "rec.state", Version: 1}})[0])
	return b.G
}

// slots is a pool of runs+2 slots of n floats, NaN but where set.
func slotsNaN(runs, n int) []uint32 {
	w := make([]uint32, (runs+2)*n)
	for i := range w {
		w[i] = math.Float32bits(float32(math.NaN()))
	}
	return w
}

// runDelta runs one step of fr over the runs (each row its own run unless
// chained) and returns the worst distance from the chained oracle.
func runDelta(fr *recRun, g *graph.Graph, s deltaShape, n int, runs recRuns, chained bool, rng *rand.Rand) (float64, error) {
	const R = 8
	N := s.h * s.dv * s.dk
	nd := s.h
	if s.chans {
		nd = s.h * s.dk
	}
	rnd := func(per int, lo, hi float64) []float32 {
		x := make([]float32, R*per)
		for i := range x {
			x[i] = float32(lo + (hi-lo)*rng.Float64())
			if i >= n*per {
				x[i] = float32(math.NaN())
			}
		}
		return x
	}
	q, k, v := rnd(s.kh*s.dk, -1, 1), rnd(s.kh*s.dk, -1, 1), rnd(s.h*s.dv, -1, 1)
	dc, beta := rnd(nd, 0.5, 0.99), rnd(s.h, 0.1, 0.9)
	in, out := slotsNaN(len(runs), N), slotsNaN(len(runs), N)
	ref := make([][]float64, len(runs))
	for j := range runs {
		ref[j] = make([]float64, N)
		for i := range N {
			x := float32(rng.NormFloat64() * 0.3)
			in[(1+j)*N+i], ref[j][i] = math.Float32bits(x), float64(x)
		}
	}
	steps := runs
	if !chained {
		steps = nil
		for _, run := range runs {
			for _, r := range run {
				steps = append(steps, []int{r})
			}
		}
	}
	tables := map[string][]uint32{"rec.state": in, "rec.state.out": out, "rec.rows": steps.desc(n, 1, 1)}
	if !chained {
		// Each row its own run, each reading its sequence's state.
		w := tables["rec.rows"]
		at := 0
		for j, run := range runs {
			for range run {
				w[1+2*at], w[2+2*at] = uint32(1+j), uint32(1+j)
				at++
			}
		}
		for e := at; e < n; e++ {
			w[1+2*e], w[2+2*e] = w[1+2*(at-1)], w[2+2*(at-1)]
		}
	}
	fr.vals = map[graph.ValueID][]uint32{g.In[0]: f32w(q), g.In[1]: f32w(k), g.In[2]: f32w(v), g.In[3]: f32w(dc), g.In[4]: f32w(beta),
		g.Out[0]: poisoned(R * s.h * s.dv)}
	fr.tables = tables
	clear(fr.bufs)
	if err := fr.launch(); err != nil {
		return 0, err
	}
	got := wf32(fr.vals[g.Out[0]])
	worst := 0.0
	for j, run := range runs {
		for _, r := range run {
			at := func(x []float32, per int) []float32 { return x[r*per : (r+1)*per] }
			want := deltaOracleOf(s, ref[j], at(q, s.kh*s.dk), at(k, s.kh*s.dk), at(v, s.h*s.dv), at(dc, nd), at(beta, s.h))
			for i, w := range want {
				worst = max(worst, math.Abs(float64(got[r*s.h*s.dv+i])-w))
			}
		}
		if chained {
			for i := range N {
				worst = max(worst, math.Abs(float64(math.Float32frombits(out[(1+j)*N+i]))-ref[j][i]))
			}
		}
	}
	if math.IsNaN(worst) {
		worst = math.Inf(1)
	}
	for i, w := range out {
		if (i < N || i >= (1+len(runs))*N) && !math.IsNaN(float64(math.Float32frombits(w))) {
			return 0, fmt.Errorf("the form wrote %v at %d of the pool, outside the step's slots", math.Float32frombits(w), i)
		}
	}
	return worst, nil
}

// deltaOracleOf is one row of the delta rule over every value head of the
// state st (float64, [h][dv][dk]) in the oracle's arithmetic.
func deltaOracleOf(s deltaShape, st []float64, q, k, v, d, beta []float32) []float64 {
	f := func(x []float32) []float64 {
		out := make([]float64, len(x))
		for i, e := range x {
			out[i] = float64(e)
		}
		return out
	}
	o := make([]float64, s.h*s.dv)
	for h := range s.h {
		kh := h / (s.h / s.kh)
		if s.tiled {
			kh = h % s.kh
		}
		n := max(s.dv, s.dk)
		sq := make([]float64, n*n)
		for j := range s.dv {
			copy(sq[j*n:j*n+s.dk], st[(h*s.dv+j)*s.dk:][:s.dk])
		}
		pad := func(x []float64) []float64 { return append(append([]float64(nil), x...), make([]float64, n-len(x))...) }
		ko, qo := pad(f(k[kh*s.dk:][:s.dk])), pad(f(q[kh*s.dk:][:s.dk]))
		vo, oo := pad(f(v[h*s.dv:][:s.dv])), make([]float64, n)
		if s.chans {
			oracle.GatedDeltaChan(oo, sq, ko, qo, vo, pad(f(d[h*s.dk:][:s.dk])), float64(beta[h]))
		} else {
			oracle.GatedDelta(oo, sq, ko, qo, vo, float64(d[h]), float64(beta[h]))
		}
		copy(o[h*s.dv:], oo[:s.dv])
		for j := range s.dv {
			copy(st[(h*s.dv+j)*s.dk:][:s.dk], sq[j*n:j*n+s.dk])
		}
	}
	return o
}

// runConv is runDelta's step for the state convolution of T taps over C
// channels.
func runConv(fr *recRun, g *graph.Graph, T, C, n int, runs recRuns, chained bool, rng *rand.Rand) (float64, error) {
	const R = 8
	W := (T - 1) * C
	taps, planes := make([]float32, C*T), make([]float64, C*T)
	for c := range C {
		for k := range T {
			v := float32(rng.NormFloat64())
			taps[c*T+k], planes[k*C+c] = v, float64(v)
		}
	}
	x := make([]float32, R*C)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
		if i >= n*C {
			x[i] = float32(math.NaN())
		}
	}
	in, out := slotsNaN(len(runs), W), slotsNaN(len(runs), W)
	ref := make([][]float64, len(runs))
	for j := range runs {
		ref[j] = make([]float64, W)
		for i := range W {
			v := float32(rng.NormFloat64())
			in[(1+j)*W+i], ref[j][i] = math.Float32bits(v), float64(v)
		}
	}
	var steps recRuns
	if chained {
		steps = runs
	} else {
		for _, run := range runs {
			for _, r := range run {
				steps = append(steps, []int{r})
			}
		}
	}
	desc := steps.desc(n, 1, 1)
	if !chained {
		at := 0
		for j, run := range runs {
			for range run {
				desc[1+2*at], desc[2+2*at] = uint32(1+j), uint32(1+j)
				at++
			}
		}
		for e := at; e < n; e++ {
			desc[1+2*e], desc[2+2*e] = desc[1+2*(at-1)], desc[2+2*(at-1)]
		}
	}
	fr.vals = map[graph.ValueID][]uint32{g.In[0]: f32w(x), g.Out[0]: poisoned(R * C)}
	fr.tables = map[string][]uint32{"rec.conv": in, "rec.conv.out": out, "rec.rows": desc}
	fr.wts = map[jlm.Role][]uint32{jlm.RoleSSMConv1d: f32w(taps)}
	clear(fr.bufs)
	if err := fr.launch(); err != nil {
		return 0, err
	}
	got := wf32(fr.vals[g.Out[0]])
	worst := 0.0
	for j, run := range runs {
		for _, r := range run {
			want := make([]float64, C)
			x64 := make([]float64, C)
			for i := range C {
				x64[i] = float64(x[r*C+i])
			}
			oracle.Conv1d(want, ref[j], planes, x64, T, C)
			for i, w := range want {
				worst = max(worst, math.Abs(float64(got[r*C+i])-w))
			}
		}
		if chained {
			for i := range W {
				worst = max(worst, math.Abs(float64(math.Float32frombits(out[(1+j)*W+i]))-ref[j][i]))
			}
		}
	}
	if math.IsNaN(worst) {
		worst = math.Inf(1)
	}
	for i, w := range out {
		if (i < W || i >= (1+len(runs))*W) && !math.IsNaN(float64(math.Float32frombits(w))) {
			return 0, fmt.Errorf("the form wrote %v at %d of the pool, outside the step's slots", math.Float32frombits(w), i)
		}
	}
	return worst, nil
}
