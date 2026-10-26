package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// deltaShape is a delta rule's geometry: value heads, key heads, the key's
// and the value's head widths, the pairing and the decay's width.
type deltaShape struct {
	h, kh, dk, dv int
	tiled, chans  bool
}

// deltaGraph is OpDeltaRule alone over q, k, v, decay and beta.
func deltaGraph(s deltaShape) *graph.Graph {
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

// deltaOracle is one row of the delta rule over every value head of the
// state st (float64, [h][dv][dk]) in the oracle's arithmetic.
func deltaOracle(s deltaShape, st []float64, q, k, v, d, beta []float32) []float64 {
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
		// The oracle's state is square; a head of dv rows by dk is held in
		// one of max(dv, dk) with the rest zero.
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

// TestComposedCPUDeltaRule forms OpDeltaRule on the CPU device over ragged
// head widths, both pairings of value heads with key heads and both decay
// widths, and runs it three rows on, the state carried in its table, held
// to the oracle's delta rule at every row's output and at the state after.
// With the pairing turned, the decay dropped or beta dropped, the form
// must part from the oracle.
func TestComposedCPUDeltaRule(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	shapes := []deltaShape{
		{h: 2, kh: 1, dk: 4, dv: 4},
		{h: 4, kh: 2, dk: 7, dv: 5},
		{h: 4, kh: 2, dk: 7, dv: 5, tiled: true},
		{h: 8, kh: 2, dk: 9, dv: 8, chans: true},
		{h: 4, kh: 2, dk: 16, dv: 16, tiled: true, chans: true},
	}
	type fault uint8
	const (
		none fault = iota
		turned
		noDecay
		noBeta
	)
	parted := map[fault]int{}
	for _, s := range shapes {
		for _, fl := range []fault{none, turned, noDecay, noBeta} {
			if fl == turned && s.kh == 1 {
				continue
			}
			name := fmt.Sprintf("%+v fault %d", s, fl)
			fs := s
			if fl == turned {
				fs.tiled = !s.tiled
			}
			g := deltaGraph(fs)
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice()})
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			N := s.h * s.dv * s.dk
			// A pool of three slots, the sequence's the middle one: the
			// slots either side are another sequence's and must not move.
			pool := make([]uint32, 3*N+16)
			for i := range pool {
				pool[i] = math.Float32bits(float32(math.NaN()))
			}
			state := pool[N : 2*N]
			ref := make([]float64, N)
			for i := range N {
				x := float32(rng.NormFloat64() * 0.3)
				state[i], ref[i] = math.Float32bits(x), float64(x)
			}
			tables := map[string][]uint32{"rec.state": pool, "rec.state.out": pool, "rec.rows": {1, 1, 1}}
			worst := 0.0
			for row := range 3 {
				rnd := func(n int, lo, hi float64) []float32 {
					x := make([]float32, n)
					for i := range x {
						x[i] = float32(lo + (hi-lo)*rng.Float64())
					}
					return x
				}
				nd := s.h
				if s.chans {
					nd = s.h * s.dk
				}
				q, k, v := rnd(s.kh*s.dk, -1, 1), rnd(s.kh*s.dk, -1, 1), rnd(s.h*s.dv, -1, 1)
				d, beta := rnd(nd, 0.5, 0.99), rnd(s.h, 0.1, 0.9)
				words := func(x []float32) []uint32 { return unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x)) }
				vals := map[graph.ValueID][]uint32{g.In[0]: words(q), g.In[1]: words(k), g.In[2]: words(v), g.In[3]: words(d), g.In[4]: words(beta)}
				rd, rb := d, beta
				switch fl {
				case noDecay:
					rd = rnd(nd, 1, 1)
				case noBeta:
					rb = rnd(s.h, 1, 1)
				}
				if fl == noDecay || fl == noBeta {
					vals[g.In[3]], vals[g.In[4]] = words(rd), words(rb)
				}
				if err := runCPUTables(codes, g, vals, [3][]uint32{}, tables); err != nil {
					t.Fatalf("%s row %d: %v", name, row, err)
				}
				want := deltaOracle(s, ref, q, k, v, d, beta)
				got := wordsF32(vals[g.Out[0]])
				for i, w := range want {
					worst = max(worst, math.Abs(float64(got[i])-w))
				}
				for i := range N {
					worst = max(worst, math.Abs(float64(math.Float32frombits(state[i]))-ref[i]))
				}
				for i, w := range pool {
					if (i < N || i >= 2*N) && !math.IsNaN(float64(math.Float32frombits(w))) {
						t.Fatalf("%s row %d: the form wrote %v at %d, outside its slot [%d, %d)", name, row, math.Float32frombits(w), i, N, 2*N)
					}
				}
			}
			if fl != none {
				if worst > 1e-3 {
					parted[fl]++
				}
				continue
			}
			if worst > 2e-5 {
				t.Fatalf("%s: the composed delta rule is %g from the oracle", name, worst)
			}
		}
	}
	for _, fl := range []fault{turned, noDecay, noBeta} {
		if parted[fl] == 0 {
			t.Fatalf("fault %d never parts from the oracle: the gate cannot see it", fl)
		}
	}
	t.Logf("%d shapes held to the oracle over three rows; faults parted %v", len(shapes), parted)
}

// TestComposedCPUStateConv forms OpStateConv on the CPU device over ragged
// channel counts and window lengths and runs it four rows on, the window in
// the middle slot of a pool, read and written where the step's descriptor
// says, held to the oracle's convolution at every row's output and at the
// window after; the slots either side stay untouched. With the window read
// as zeros every row, it must part.
func TestComposedCPUStateConv(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	parted := 0
	for _, sh := range [][2]int{{2, 3}, {3, 1}, {4, 7}, {4, 33}, {3, 64}, {5, 9}} {
		for _, fault := range []bool{false, true} {
			T, C := sh[0], sh[1]
			b := graph.New()
			x := b.In("x", C)
			b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x}, []graph.Result{{Name: "conv", Width: C}},
				[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: T, Rows: C}}, graph.Attrs{Taps: T},
				[]graph.StateRef{{Name: "rec.conv"}}, []graph.StateRef{{Name: "rec.conv", Version: 1}})[0])
			_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32},
				Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice()})
			if err != nil {
				t.Errorf("%d taps over %d: %v", T, C, err)
				continue
			}
			// The file's taps, a channel's contiguous; the oracle's, plane-major.
			W := (T - 1) * C
			taps, planes := make([]float32, C*T), make([]float64, C*T)
			for c := range C {
				for k := range T {
					v := float32(rng.NormFloat64())
					taps[c*T+k], planes[k*C+c] = v, float64(v)
				}
			}
			pool := make([]uint32, 3*W+16)
			for i := range pool {
				pool[i] = math.Float32bits(float32(math.NaN()))
			}
			ref := make([]float64, W)
			for i := range W {
				v := float32(rng.NormFloat64())
				pool[W+i], ref[i] = math.Float32bits(v), float64(v)
			}
			tables := map[string][]uint32{"rec.conv": pool, "rec.conv.out": pool, "rec.rows": {1, 1, 1}}
			wt := unsafe.Slice((*uint32)(unsafe.Pointer(&taps[0])), len(taps))
			worst := 0.0
			for row := range 4 {
				in := make([]float32, C)
				x64 := make([]float64, C)
				for i := range in {
					in[i] = float32(rng.NormFloat64())
					x64[i] = float64(in[i])
				}
				if fault {
					for i := range W {
						pool[W+i] = 0
					}
				}
				vals := map[graph.ValueID][]uint32{b.G.In[0]: unsafe.Slice((*uint32)(unsafe.Pointer(&in[0])), C)}
				if err := runCPUTables(codes, b.G, vals, [3][]uint32{wt}, tables); err != nil {
					t.Fatalf("%d taps over %d row %d: %v", T, C, row, err)
				}
				want := make([]float64, C)
				oracle.Conv1d(want, ref, planes, x64, T, C)
				got := wordsF32(vals[b.G.Out[0]])
				for i, w := range want {
					worst = max(worst, math.Abs(float64(got[i])-w))
				}
				for i := range W {
					worst = max(worst, math.Abs(float64(math.Float32frombits(pool[W+i]))-ref[i]))
				}
				for i, w := range pool {
					if (i < W || i >= 2*W) && !math.IsNaN(float64(math.Float32frombits(w))) {
						t.Fatalf("%d taps over %d row %d: the form wrote %v at %d, outside its slot", T, C, row, math.Float32frombits(w), i)
					}
				}
			}
			switch {
			case fault && worst > 1e-3:
				parted++
			case !fault && worst > 1e-5:
				t.Fatalf("%d taps over %d: the composed convolution is %g from the oracle", T, C, worst)
			}
		}
	}
	if parted == 0 {
		t.Fatal("a window read as zeros never parts from the oracle: the gate cannot see it")
	}
	t.Logf("six shapes held to the oracle over four rows; the zeroed window parted %d", parted)
}
