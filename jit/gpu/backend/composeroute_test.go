package backend_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// routeVariant is one router shape: the menu's (kernels.MoERoute) and the
// composed fragments' (compose.Route) statement of it.
type routeVariant struct {
	name string
	m    kernels.MoERoute
	ties bool // logits on a coarse grid, so keys tie
}

func (v routeVariant) route() compose.Route {
	m := v.m
	rt := compose.Route{Experts: m.NExpert, K: m.K, Bias: m.Bias, Groups: m.NGroup, GroupsUsed: m.NGroupUsed,
		Norm: m.Norm, Scale: m.Scale, Jitter: m.SparseMixer}
	switch {
	case m.Sigmoid:
		rt.Score = compose.ScoreSigmoid
	case m.SqrtSoftplus:
		rt.Score = compose.ScoreSqrtSoftplus
	}
	return rt
}

var routeVariants = []routeVariant{
	{name: "softmax-norm", m: kernels.MoERoute{NExpert: 64, K: 8, Norm: true}},
	{name: "softmax-norm-ties", m: kernels.MoERoute{NExpert: 60, K: 4, Norm: true}, ties: true},
	{name: "softmax-all", m: kernels.MoERoute{NExpert: 64, K: 8}},
	{name: "softmax-scaled", m: kernels.MoERoute{NExpert: 40, K: 6, Norm: true, Scale: 16}},
	{name: "softmax-wide", m: kernels.MoERoute{NExpert: 384, K: 8, Norm: true}, ties: true},
	{name: "softmax-bias", m: kernels.MoERoute{NExpert: 64, K: 6, Norm: true, Bias: true}},
	{name: "sigmoid-norm", m: kernels.MoERoute{NExpert: 32, K: 4, Sigmoid: true, Norm: true}},
	{name: "sigmoid-bias-scaled", m: kernels.MoERoute{NExpert: 48, K: 8, Sigmoid: true, Bias: true, Scale: 2.5}},
	{name: "sigmoid-grouped", m: kernels.MoERoute{NExpert: 64, K: 6, Sigmoid: true, Bias: true, NGroup: 8, NGroupUsed: 3,
		Norm: true, Scale: 2.5}},
	{name: "sigmoid-grouped-ties", m: kernels.MoERoute{NExpert: 64, K: 6, Sigmoid: true, NGroup: 8, NGroupUsed: 4,
		Norm: true}, ties: true},
	{name: "sigmoid-groups-of-one", m: kernels.MoERoute{NExpert: 16, K: 4, Sigmoid: true, NGroup: 16, NGroupUsed: 6,
		Norm: true}},
	{name: "sqrtsoftplus-bias", m: kernels.MoERoute{NExpert: 64, K: 6, SqrtSoftplus: true, Bias: true, RowBias: true,
		Norm: true, Scale: 1.5}},
	{name: "sparsemixer", m: kernels.MoERoute{NExpert: 16, K: 2, SparseMixer: 0.01}},
	{name: "sparsemixer-ties", m: kernels.MoERoute{NExpert: 16, K: 2, SparseMixer: 0.01}, ties: true},
}

// rowGroup is the composed group a row of width w runs under: the tier's
// (rowSchedule's) for a group with no row reduction.
func rowGroup(w int) int {
	g := 32
	for g < w && g < 256 {
		g *= 2
	}
	return g
}

// TestComposedRouteMatchesTheMenu holds the router's selection formed from
// the fold primitive -- GroupMask, TopK and RouteWeights -- to the menu's
// ExpertGroupScore and ExpertGroupMask, ExpertRank and ExpertWeights bit for
// bit on every route the menu forms: softmax renormalised, over every
// expert, scaled and biased; sigmoid with and without groups and bias;
// sqrt-softplus; sparsemixer; with keys that tie. Each stage reads the
// menu's previous outputs, so a parting names its stage. The violation
// arms: the tie rule reversed, a group too many kept, and the weights
// formed without their renormalisation (or sparsemixer's jitter widened),
// each of which must part.
func TestComposedRouteMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, v := range routeVariants {
				t.Run(v.name, func(t *testing.T) { routeMatches(t, d, v) })
			}
		})
	}
}

func routeMatches(t *testing.T, d backend.Device, v routeVariant) {
	g := newGPU(t, d)
	defer g.free()
	m, rt := v.m, v.route()
	if err := rt.Check(); err != nil {
		t.Fatal(err)
	}
	E, K := m.NExpert, m.K
	rng := rand.New(rand.NewSource(int64(E*31 + K)))
	logits := make([]float32, E)
	bias := make([]float32, E)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 2)
		if v.ties {
			logits[i] = float32(math.Round(float64(logits[i])*2) / 2)
		}
		bias[i] = float32(rng.NormFloat64() * 0.05)
	}
	launch := func(k *ir.Kernel, groups, width int, bufs ...backend.Buf) {
		t.Helper()
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if err := c.Launch(groups, width, bufs...); err != nil {
			t.Fatal(err)
		}
	}
	read := func(b backend.Buf, n int) []byte {
		t.Helper()
		raw := make([]byte, n*4)
		if err := b.Read(raw); err != nil {
			t.Fatal(err)
		}
		return raw
	}
	emit := func(r *compose.Row) *ir.Kernel {
		t.Helper()
		k, err := r.Emit(compose.Schedule{Group: rowGroup(r.Width)})
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	pL, pB := g.up(f32bytes(logits)), g.up(f32bytes(bias))
	zeros := func(n int) backend.Buf { return g.up(make([]byte, n*4)) }

	// The selection plane of a grouped route.
	var plane backend.Buf
	if m.NGroup > 1 {
		gs, bm := zeros(m.NGroup), zeros(E)
		sk, err := kernels.ExpertGroupScore(m)
		if err != nil {
			t.Fatal(err)
		}
		mk, err := kernels.ExpertGroupMask(m)
		if err != nil {
			t.Fatal(err)
		}
		if m.Bias {
			launch(sk, (m.NGroup+127)/128, 128, pL, gs, pB)
			launch(mk, (E+127)/128, 128, pL, gs, bm, pB)
		} else {
			launch(sk, (m.NGroup+127)/128, 128, pL, gs)
			launch(mk, (E+127)/128, 128, pL, gs, bm)
		}
		want := read(bm, E)
		mask := func(rt compose.Route) []byte {
			r := &compose.Row{Width: E, Rows: 1}
			l := r.Buf(compose.BufRow)
			b := -1
			if rt.Bias {
				b = r.Buf(compose.BufVec)
			}
			out := r.Buf(compose.BufOut)
			r.GroupMask(rt, l, b, out)
			o := zeros(E)
			if rt.Bias {
				launch(emit(r), 1, rowGroup(E), pL, pB, o)
			} else {
				launch(emit(r), 1, rowGroup(E), pL, o)
			}
			return read(o, E)
		}
		if got := mask(rt); !bytes.Equal(got, want) {
			t.Errorf("the composed group mask parts from ExpertGroupScore and ExpertGroupMask")
		}
		wrong := rt
		wrong.GroupsUsed++
		if got := mask(wrong); bytes.Equal(got, want) {
			t.Errorf("a group too many kept matched the menu's plane")
		}
		plane = bm
	}

	// The selection: ids and scores at their slots (slot K is the bin every
	// expert not kept writes in turn, so only 0..K-1 are compared).
	sel, top := zeros(K+1), zeros(K+1)
	rk, err := kernels.ExpertRank(m)
	if err != nil {
		t.Fatal(err)
	}
	switch {
	case plane != nil:
		launch(rk, (E+127)/128, 128, pL, sel, top, plane)
	case m.Bias:
		launch(rk, (E+127)/128, 128, pL, sel, top, pB)
	default:
		launch(rk, (E+127)/128, 128, pL, sel, top)
	}
	wantSel, wantTop := read(sel, K+1)[:K*4], read(top, K+1)[:K*4]
	pick := func(rt compose.Route, rev bool) ([]byte, []byte) {
		r := &compose.Row{Width: E, Rows: 1}
		l, s, tp := r.Buf(compose.BufRow), r.Buf(compose.BufOutU32), r.Buf(compose.BufOut)
		pl, b := -1, -1
		third := plane
		switch {
		case plane != nil:
			pl = r.Buf(compose.BufRow)
		case rt.Bias:
			b = r.Buf(compose.BufVec)
			third = pB
		}
		if rev {
			topKTiesHigh(r, rt, l, s, tp)
		} else {
			r.TopK(rt, l, s, tp, pl, b)
		}
		gs, gt := zeros(K+1), zeros(K+1)
		if third != nil {
			launch(emit(r), 1, rowGroup(E), pL, gs, gt, third)
		} else {
			launch(emit(r), 1, rowGroup(E), pL, gs, gt)
		}
		return read(gs, K+1)[:K*4], read(gt, K+1)[:K*4]
	}
	gotSel, gotTop := pick(rt, false)
	if !bytes.Equal(gotSel, wantSel) || !bytes.Equal(gotTop, wantTop) {
		t.Errorf("the composed top-k parts from ExpertRank: %v %v against %v %v",
			u32s(gotSel), bytesF32(gotTop), u32s(wantSel), bytesF32(wantTop))
	}
	if v.ties && !m.Bias && !m.Gated() && plane == nil {
		if s, _ := pick(rt, true); bytes.Equal(s, wantSel) {
			t.Errorf("ties broken to the higher index matched ExpertRank's selection")
		}
	}

	// The weights, from the menu's selection.
	wk, err := kernels.ExpertWeights(m)
	if err != nil {
		t.Fatal(err)
	}
	reads := m.SparseMixer != 0 || !m.Sigmoid && !m.SqrtSoftplus && !m.Norm
	w := zeros(K)
	if reads {
		launch(wk, 1, 1, top, w, pL)
	} else {
		launch(wk, 1, 1, top, w)
	}
	wantW := read(w, K)
	weigh := func(rt compose.Route) []byte {
		r := &compose.Row{Width: K, Rows: 1}
		tp, out, l := r.Buf(compose.BufRow), r.Buf(compose.BufOut), r.Buf(compose.BufRow)
		r.RouteWeights(rt, tp, l, out)
		o := zeros(K)
		launch(emit(r), 1, rowGroup(K), top, o, pL)
		return read(o, K)
	}
	if got := weigh(rt); !bytes.Equal(got, wantW) {
		t.Errorf("the composed weights part from ExpertWeights: %v against %v", bytesF32(got), bytesF32(wantW))
	}
	wrong := rt
	if rt.Jitter != 0 {
		wrong.Jitter = 0.5
	} else {
		wrong.Norm = !wrong.Norm
	}
	if got := weigh(wrong); bytes.Equal(got, wantW) {
		t.Errorf("the weights formed the wrong way matched ExpertWeights")
	}
}

// topKTiesHigh is TopK's plain logit rank with the tie rule reversed (an
// equal key at a higher index ranks first): the violation the tie rule is
// held against.
func topKTiesHigh(r *compose.Row, rt compose.Route, l, sel, top int) {
	e := r.Column()
	mine := r.LoadAt(l, e)
	c := r.Loop(rt.Experts)
	f := r.Iter(c)
	rank := r.Fold(c, r.UConst(0))
	kf := r.LoadAt(l, f)
	gt, lt := r.Flag(r.Less(mine, kf)), r.Flag(r.Less(kf, mine))
	eq := r.UBin(compose.OpSub, r.UConst(1), r.UBin(compose.OpAdd, gt, lt))
	higher := r.Flag(r.Less(e, f))
	r.SetFold(rank, r.UBin(compose.OpAdd, rank, r.UBin(compose.OpAdd, gt, r.UBin(compose.OpMul, eq, higher))))
	slot := r.UBin(compose.OpMin, rank, r.UConst(rt.K))
	r.StoreAt(sel, e, slot)
	r.StoreAt(top, mine, slot)
}

// TestComposedRouterDotsMatchTheMenu holds the router's logits (an f32 and
// an f16 router, RouterMatVecOf) and a shared expert's gate logit over rows
// (SharedRouterLogits) formed as RowDot to the menu bit for bit; the
// violation arm is the dot summed under a group of 256, whose tree parts.
// The router's bias (kernels.Add) and the per-expert weight scale
// (ExpertWeightScale) are held too, each with its violation: a subtraction,
// and the scale read at the slot rather than the expert.
func TestComposedRouterDotsMatchTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const E, k, rows, K = 24, 1000, 3, 6
	rng := rand.New(rand.NewSource(7))
	x := randActs(rows*k, rng)
	wf := randActs(E*k, rng)
	wh := make([]byte, E*k*2)
	for i := range wf {
		binary.LittleEndian.PutUint16(wh[2*i:], f16bits(wf[i]))
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			launch := func(k *ir.Kernel, groups, width int, bufs ...backend.Buf) {
				t.Helper()
				c, err := d.Compile(k)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				if err := c.Launch(groups, width, bufs...); err != nil {
					t.Fatal(err)
				}
			}
			read := func(b backend.Buf, n int) []byte {
				t.Helper()
				raw := make([]byte, n*4)
				if err := b.Read(raw); err != nil {
					t.Fatal(err)
				}
				return raw
			}
			pX := g.up(f32bytes(x))
			for _, f16 := range []bool{false, true} {
				pW := g.up(f32bytes(wf))
				wk := compose.BufRow
				if f16 {
					pW, wk = g.up(wh), compose.BufRowF16
				}
				mk, err := kernels.RouterMatVecOf(E, k, f16)
				if err != nil {
					t.Fatal(err)
				}
				out := g.up(make([]byte, E*4))
				launch(mk, kernels.RouterThreads(E)/128, 128, pW, pX, out)
				want := read(out, E)
				dot := func(group int) []byte {
					r := &compose.Row{Width: k, Rows: E}
					w, xb, o := r.Buf(wk), r.Buf(compose.BufVec), r.Buf(compose.BufOut)
					r.RowDot(r.Load(w), r.Load(xb), o)
					ker, err := r.Emit(compose.Schedule{Group: group})
					if err != nil {
						t.Fatal(err)
					}
					res := g.up(make([]byte, E*4))
					launch(ker, E, group, pW, pX, res)
					return read(res, E)
				}
				if got := dot(128); !bytes.Equal(got, want) {
					t.Errorf("f16 %v: the composed router logits part from RouterMatVecOf", f16)
				}
				if got := dot(256); bytes.Equal(got, want) {
					t.Errorf("f16 %v: the dot under another tree matched RouterMatVecOf", f16)
				}
			}
			// The shared expert's gate over rows: one weight vector, a row each.
			sk, err := kernels.SharedRouterLogits(rows, k)
			if err != nil {
				t.Fatal(err)
			}
			pV := g.up(f32bytes(wf[:k]))
			out := g.up(make([]byte, rows*4))
			launch(sk, kernels.SharedRouterThreads(rows)/128, 128, pV, pX, out)
			want := read(out, rows)
			r := &compose.Row{Width: k, Rows: rows}
			w, xb, o := r.Buf(compose.BufVec), r.Buf(compose.BufRow), r.Buf(compose.BufOut)
			r.RowDot(r.Load(w), r.Load(xb), o)
			ker, err := r.Emit(compose.Schedule{Group: 128})
			if err != nil {
				t.Fatal(err)
			}
			res := g.up(make([]byte, rows*4))
			launch(ker, rows, 128, pV, pX, res)
			if got := read(res, rows); !bytes.Equal(got, want) {
				t.Errorf("the composed gate logit parts from SharedRouterLogits")
			}

			// The router's bias, and the per-expert scale.
			lg, bs := randActs(E, rng), randActs(E, rng)
			pL, pB := g.up(f32bytes(lg)), g.up(f32bytes(bs))
			ak, err := kernels.Add(E)
			if err != nil {
				t.Fatal(err)
			}
			sum := g.up(make([]byte, E*4))
			launch(ak, 1, 128, pL, pB, sum)
			wantSum := read(sum, E)
			for _, op := range []compose.Op{compose.OpAdd, compose.OpSub} {
				r := &compose.Row{Width: E, Rows: 1}
				a, b, o := r.Buf(compose.BufRow), r.Buf(compose.BufVec), r.Buf(compose.BufOut)
				r.Store(o, r.Bin(op, r.Load(a), r.Load(b)))
				res := g.up(make([]byte, E*4))
				launch(emitRow(t, r), 1, rowGroup(E), pL, pB, res)
				if got := read(res, E); bytes.Equal(got, wantSum) != (op == compose.OpAdd) {
					t.Errorf("the router bias as %v: equal to kernels.Add %v", op, op != compose.OpAdd)
				}
			}
			wts, scale := randActs(K, rng), randActs(E, rng)
			ids := []uint32{17, 3, 9, 22, 0, 5, 11} // K+1: the bin slot too
			pWt, pSel, pSc := g.up(f32bytes(wts)), g.up(u32bytes(ids)), g.up(f32bytes(scale))
			ek, err := kernels.ExpertWeightScale(1, K)
			if err != nil {
				t.Fatal(err)
			}
			sc := g.up(make([]byte, K*4))
			launch(ek, 1, 128, pWt, pSel, pSc, sc)
			wantSc := read(sc, K)
			for _, atSlot := range []bool{false, true} {
				r := &compose.Row{Width: K, Rows: 1}
				w, s, c, o := r.Buf(compose.BufRow), r.Buf(compose.BufU32), r.Buf(compose.BufVec), r.Buf(compose.BufOut)
				if atSlot {
					r.Store(o, r.Bin(compose.OpMul, r.Load(w), r.LoadAt(c, r.Column())))
				} else {
					r.ScaleWeights(w, s, c, o)
				}
				res := g.up(make([]byte, K*4))
				launch(emitRow(t, r), 1, rowGroup(K), pWt, pSel, pSc, res)
				if got := read(res, K); bytes.Equal(got, wantSc) == atSlot {
					t.Errorf("the weight scale read at the slot %v: equal to ExpertWeightScale %v", atSlot, atSlot)
				}
			}
		})
	}
}

func emitRow(t *testing.T, r *compose.Row) *ir.Kernel {
	t.Helper()
	k, err := r.Emit(compose.Schedule{Group: rowGroup(r.Width)})
	if err != nil {
		t.Fatal(err)
	}
	return k
}
