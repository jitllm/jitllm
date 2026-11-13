package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
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
	codes  []composer.Code
	kerns  []backend.Kernel
	vals   map[graph.ValueID][]uint32
	tables map[string][]uint32
	wts    map[jlm.Role][]uint32
	bufs   map[string]backend.Buf
	// share binds a table to another's buffer, as the shared pool binds
	// rec.state.out to rec.state.
	share map[string]string
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
			case o.Kind == composer.OperandValue:
				key, w = fmt.Sprintf("v%d", o.Value), r.vals[o.Value]
			case o.Kind == composer.OperandTable && r.share[o.Name] != "":
				key, w = "t"+r.share[o.Name], r.tables[r.share[o.Name]]
			case o.Kind == composer.OperandTable && r.tables[o.Name] != nil:
				key, w = "t"+o.Name, r.tables[o.Name]
			case o.Kind == composer.OperandWeight:
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

func formRec(d backend.Device, dev composer.Device, g *graph.Graph, rows int, formats map[jlm.Role]jlm.Type) (*recRun, error) {
	return formRecWith(composer.Default(), d, dev, g, rows, formats)
}

// formRecWith is formRec through registry reg.
func formRecWith(reg *composer.Registry, d backend.Device, dev composer.Device, g *graph.Graph, rows int, formats map[jlm.Role]jlm.Type) (*recRun, error) {
	_, codes, err := reg.Form(composer.Request{Graph: g, Formats: formats, Step: composer.Step{Rows: rows, Page: 64, Keys: 1}, Device: dev})
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

// recSteps() are the steps a recurrent gate runs at 8 rows: a prompt chunk's
// rows as one run, fewer live rows than the form holds, and ragged steps of
// several sequences, their rows interleaved.
func recSteps() []struct {
	name string
	n    int
	runs recRuns
} {
	return []struct {
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
}

// TestComposedGPURecurrentRuns forms OpDeltaRule and OpStateConv at a step
// of 8 rows on every GPU and runs them over a prompt chunk's rows (one run)
// and ragged steps (a run a sequence, rows interleaved, as the run
// descriptor lists them), padding rows NaN: each run's rows chain through
// its state from its slot read, held to the oracle at every row's output
// and at the state its slot written holds after; every other slot stays
// untouched. With each row made its own run -- the state not carried from
// row to row -- the forms must part. Each is formed keeping its rows too
// (graph.Attrs.Keeps): every entry's slot of the keep table holds the
// oracle's state after that entry's row, the entries past the live rows
// untouched.
func TestComposedGPURecurrentRuns(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
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
func recurrentRuns(t *testing.T, d backend.Device, dev composer.Device) {
	const R = 8
	rng := rand.New(rand.NewSource(23))
	ran, parted, kept := 0, 0, 0
	// The keeping forms draw from a stream of their own, so the plain forms'
	// inputs are what they were.
	krng := rand.New(rand.NewSource(29))
	pick := func(keeps bool) *rand.Rand {
		if keeps {
			return krng
		}
		return rng
	}
	for _, s := range []deltaShape{
		{h: 4, kh: 2, dk: 16, dv: 8},
		{h: 4, kh: 2, dk: 7, dv: 5, tiled: true},
		{h: 8, kh: 2, dk: 40, dv: 4, chans: true},
		{h: 2, kh: 1, dk: 128, dv: 3, chans: true},
		{h: 6, kh: 3, dk: 33, dv: 2, tiled: true},
		{h: 4, kh: 2, dk: 16, dv: 8, additive: true},
		{h: 40, kh: 1, dk: 16, dv: 1, chans: true, additive: true},
		{h: 1536, kh: 1, dk: 16, dv: 1, chans: true, additive: true},
		{h: 5120, kh: 1, dk: 16, dv: 1, chans: true, additive: true},
		{h: 32, kh: 8, dk: 128, dv: 8, additive: true},
	} {
		for _, keeps := range []bool{false, true} {
			s.keeps = keeps
			g := deltaGraphOf(s)
			fr, err := formRec(d, dev, g, R, map[jlm.Role]jlm.Type{})
			if err != nil {
				t.Fatalf("%+v: %v", s, err)
			}
			for _, st := range recSteps() {
				for _, chained := range []bool{true, false} {
					worst, err := runDelta(fr, g, s, st.n, st.runs, chained, pick(keeps))
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
					// A row's output sums dk products: its f32 error grows with
					// dk, held relative to its size (relErr).
					case chained && worst > 2e-5*max(1, float64(s.dk)/32):
						fr.close()
						t.Fatalf("%+v %s: the composed delta rule is %g from the oracle", s, st.name, worst)
					}
					if chained {
						ran++
						if keeps {
							kept++
						}
					}
				}
			}
			fr.close()
		}
	}
	for _, sh := range [][2]int{{2, 3}, {4, 7}, {4, 33}, {3, 64}, {5, 9}, {4, 1536}, {4, 6144}} {
		for _, keeps := range []bool{false, true} {
			T, C := sh[0], sh[1]
			b := graph.New()
			x := b.In("x", C)
			b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: C}},
				[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: T, Rows: C}}, graph.Attrs{Taps: T, Keeps: keeps},
				[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0])
			fr, err := formRec(d, dev, b.G, R, map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32})
			if err != nil {
				t.Fatalf("%d taps over %d: %v", T, C, err)
			}
			for _, st := range recSteps() {
				for _, chained := range []bool{true, false} {
					worst, err := runConv(fr, b.G, T, C, st.n, st.runs, chained, keeps, pick(keeps))
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
						if keeps {
							kept++
						}
					}
				}
			}
			fr.close()
		}
	}
	if parted == 0 {
		t.Fatal("a state not carried from row to row never parts from the oracle: the gate cannot see it")
	}
	if kept == 0 {
		t.Fatal("no step kept its rows' states: the keep arm never ran")
	}
	t.Logf("%d steps held to the oracle, %d of them keeping their rows; unchained runs parted %d times", ran, kept, parted)
}

// deltaShape is a delta rule's geometry: value heads, key heads, the key's
// and the value's head widths, the pairing and the decay's width.
type deltaShape struct {
	h, kh, dk, dv          int
	tiled, chans, additive bool
	// keeps forms the rule keeping its rows (graph.Attrs.Keeps).
	keeps bool
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
		graph.Attrs{NHead: s.h, NKVHead: s.kh, HeadDim: s.dk, Tiled: s.tiled, Additive: s.additive, Keeps: s.keeps},
		[]graph.StateRef{{State: graph.StateRecState}}, []graph.StateRef{{State: graph.StateRecState, Version: 1}})[0])
	return b.G
}

// relErr is got's distance from want, relative to want's size past 1: an
// f32 sum's error grows with the sum.
func relErr(got, want float64) float64 { return math.Abs(got-want) / max(1, math.Abs(want)) }

// keptState holds a keep table of R slots of n floats to the oracle's states
// after each entry's row (want, entry by entry) and the slots past them
// untouched, and returns the worst distance (relErr).
func keptState(keep []uint32, want [][]float64, n int) (float64, error) {
	worst := 0.0
	for e, st := range want {
		for i, w := range st {
			worst = max(worst, relErr(float64(math.Float32frombits(keep[e*n+i])), w))
		}
	}
	for i := len(want) * n; i < len(keep); i++ {
		if v := math.Float32frombits(keep[i]); !math.IsNaN(float64(v)) {
			return 0, fmt.Errorf("the form kept %v at word %d, past the step's %d entries", v, i, len(want))
		}
	}
	if math.IsNaN(worst) {
		worst = math.Inf(1)
	}
	return worst, nil
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
	if s.keeps {
		tables["rec.state.keep"] = slotsNaN(R-2, N)
	}
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
	var after [][]float64
	for j, run := range runs {
		for _, r := range run {
			at := func(x []float32, per int) []float32 { return x[r*per : (r+1)*per] }
			want := deltaOracleOf(s, ref[j], at(q, s.kh*s.dk), at(k, s.kh*s.dk), at(v, s.h*s.dv), at(dc, nd), at(beta, s.h))
			for i, w := range want {
				worst = max(worst, relErr(float64(got[r*s.h*s.dv+i]), w))
			}
			after = append(after, slices.Clone(ref[j]))
		}
		if chained {
			for i := range N {
				worst = max(worst, relErr(float64(math.Float32frombits(out[(1+j)*N+i])), ref[j][i]))
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
	if s.keeps && chained {
		k, err := keptState(tables["rec.state.keep"], after, N)
		if err != nil {
			return 0, err
		}
		worst = max(worst, k)
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
		switch {
		case s.additive && s.chans:
			// Each row's decay exp(dt a) with dt beta: a is the decay's log
			// over beta.
			dt, a := make([]float64, n), make([]float64, n*n)
			for j := range n {
				dt[j] = float64(beta[h])
				for i := range s.dk {
					a[j*n+i] = math.Log(float64(d[h*s.dk+i])) / float64(beta[h])
				}
			}
			oracle.SelScan(oo, sq, ko, qo, vo, dt, a, make([]float64, n))
		case s.additive:
			oracle.GatedSSD(oo, sq, ko, qo, vo, float64(d[h]), float64(beta[h]), 0)
		case s.chans:
			oracle.GatedDeltaChan(oo, sq, ko, qo, vo, pad(f(d[h*s.dk:][:s.dk])), float64(beta[h]))
		default:
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
func runConv(fr *recRun, g *graph.Graph, T, C, n int, runs recRuns, chained, keeps bool, rng *rand.Rand) (float64, error) {
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
	if keeps {
		fr.tables["rec.conv.keep"] = slotsNaN(R-2, W)
	}
	fr.wts = map[jlm.Role][]uint32{jlm.RoleSSMConv1d: f32w(taps)}
	clear(fr.bufs)
	if err := fr.launch(); err != nil {
		return 0, err
	}
	got := wf32(fr.vals[g.Out[0]])
	worst := 0.0
	var after [][]float64
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
			after = append(after, slices.Clone(ref[j]))
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
	if keeps && chained {
		k, err := keptState(fr.tables["rec.conv.keep"], after, W)
		if err != nil {
			return 0, err
		}
		worst = max(worst, k)
	}
	return worst, nil
}

// TestComposedGPUScanWidths forms the selective scan's float matvecs at a
// real checkpoint's widths on every GPU device present -- B and C's 16 rows
// off a channel row of 1536 and 5120, dt's bottleneck of 48 and 160 back to
// the channels -- and holds each to the oracle's compensated dot.
func TestComposedGPUScanWidths(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	rng := rand.New(rand.NewSource(5))
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, sh := range [][2]int{{16, 1536}, {48, 1536}, {1536, 48}, {16, 5120}, {160, 5120}, {5120, 160}, {16, 128}} {
				rows, k := sh[0], sh[1]
				b := graph.New()
				x := b.In("x", k)
				b.Out(b.MatVec(x, jlm.RoleSSMXB, rows))
				fr, err := formRec(d, gpuDevice(d), b.G, 1, map[jlm.Role]jlm.Type{jlm.RoleSSMXB: jlm.TypeF32})
				if err != nil {
					t.Fatalf("%dx%d: %v", rows, k, err)
				}
				xs, ws := make([]float32, k), make([]float32, rows*k)
				for i := range xs {
					xs[i] = float32(rng.NormFloat64())
				}
				for i := range ws {
					ws[i] = float32(rng.NormFloat64() * 0.05)
				}
				fr.vals = map[graph.ValueID][]uint32{b.G.In[0]: f32w(xs), b.G.Out[0]: poisoned(rows)}
				fr.wts = map[jlm.Role][]uint32{jlm.RoleSSMXB: f32w(ws)}
				if err := fr.launch(); err != nil {
					fr.close()
					t.Fatalf("%dx%d: %v", rows, k, err)
				}
				got := wf32(fr.vals[b.G.Out[0]])
				fr.close()
				worst := 0.0
				for r := range rows {
					want := oracle.Dot32(ws[r*k:(r+1)*k], xs)
					worst = max(worst, math.Abs(float64(got[r]-want)))
				}
				t.Logf("%d rows of %d: worst %.3g, %d kernels", rows, k, worst, len(fr.codes))
				if !(worst < 1e-3) {
					t.Errorf("%d rows of %d: the composed matvec is %.3g from the oracle", rows, k, worst)
				}
			}
		})
	}
}
