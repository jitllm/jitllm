package backend_test

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// attnStep is one launch of a paged decode attention plan: its kernel, its
// grid, and its arguments by name.
type attnStep struct {
	k             *ir.Kernel
	groups, width int
	args          []string
}

// attnPlanSteps is a decode plan's launches as the tier runs them
// (pagedDecode): FlashDecodeKV or the staged trio, then the merge where
// there are partials. form takes the composed kernels, else the menu's.
func attnPlanSteps(t *testing.T, p kernels.DecodePlan, lanes int, items, form bool) []attnStep {
	t.Helper()
	s := p.Shape
	must := func(k *ir.Kernel, err error) *ir.Kernel {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	var out []attnStep
	merged := s.Splits > 1
	if p.Path == kernels.PathStaged {
		merged = true
		scoreLanes := 1
		if s.MLA > 0 && lanes == 32 {
			scoreLanes = 32
		}
		sc, sm, ac := &compose.StagedScores{S: s, Lanes: scoreLanes}, &compose.StagedSoftmax{S: s, Lanes: lanes}, &compose.StagedAcc{S: s}
		var ks, kf, ka *ir.Kernel
		if form {
			ks, kf, ka = must(sc.Emit()), must(sm.Emit()), must(ac.Emit())
		} else {
			ks, kf, ka = must(kernels.PagedStagedScores(s, scoreLanes)), must(kernels.PagedStagedSoftmax(s, lanes)), must(kernels.PagedStagedAcc(s))
		}
		out = append(out,
			attnStep{ks, sc.Groups(), sc.Width(), []string{"q", "k", "n", "sc", "tab", "desc"}},
			attnStep{kf, sm.Groups(), sm.Width(), []string{"sc", "n", "wt", "part", "tab", "desc"}},
			attnStep{ka, ac.Groups(), ac.Width(), []string{"wt", "v", "n", "part", "tab", "desc"}})
	} else {
		a := &compose.Attend{S: s}
		k := must(kernels.FlashDecodeKV(s))
		if form {
			k = must(a.Emit())
		}
		dst := "out"
		if merged {
			dst = "part"
		}
		args := []string{"q", "k", "v", "n", dst, "tab", "desc"}
		if s.Sink {
			args = append(args, "sink")
		}
		out = append(out, attnStep{k, a.Groups(), a.Width(), args})
	}
	if merged {
		ms := s
		if ms.MLA > 0 {
			ms.Dim, ms.MLA = ms.MLA, 0
		}
		m := &compose.Merge{S: ms, Wide: !items}
		var k *ir.Kernel
		switch {
		case form:
			k = must(m.Emit())
		case items:
			k = must(kernels.FlashAttentionMerge(ms))
		default:
			k = must(kernels.FlashAttentionMergeWide(ms))
		}
		args := []string{"part", "out"}
		if s.Sink {
			args = append(args, "sink")
		}
		out = append(out, attnStep{k, m.Groups(), m.Width(), args})
	}
	return out
}

// TestComposedAttendMatchesTheMenu holds the formed decode attention --
// compose.Attend, the staged scores, softmax and accumulate, and both merges
// -- to the menu kernels a plan names, bit for bit, on every device: over the
// plans kernels.ChooseDecodePlan picks across key counts and the schedule
// knobs one at a time (splits, binary16 V and K, the K address form, vector
// V, a sink, a softcap, a group, warps, a latent row), over rows whose
// windows start mid-page and mid-tile, a padded row and one long row, in
// shuffled pages. The violation: the formed plan run over every page id one
// past its own (pagedPool.wrong), and the plan formed at a score scale one
// ulp off, must part from the menu's bits.
func TestComposedAttendMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const page = 64
	type plan struct {
		p     kernels.DecodePlan
		rows  string
		items bool
	}
	var plans []plan
	base := []kernels.FlashShape{
		{Heads: 8, KVHeads: 2, Dim: 64, Page: page, Scale: .125},
		{Heads: 4, KVHeads: 2, Dim: 34, Page: page, Scale: .17},
		{Heads: 16, KVHeads: 4, Dim: 128, Page: page, Scale: .0884},
	}
	knobs := []func(s kernels.FlashShape) kernels.FlashShape{
		func(s kernels.FlashShape) kernels.FlashShape { return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Splits = 3; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.F16 = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.F16K = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.F16K, s.KImm = true, true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Sink, s.Splits = true, 2; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Sink = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Softcap = 30; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Group = 1; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Warps = 2; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.KImm = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.VecV = true; return s },
	}
	for _, s := range base {
		for i, f := range knobs {
			sh := f(s)
			rows := []string{"rows", "one-long"}[i%2]
			plans = append(plans, plan{kernels.DecodePlan{Path: kernels.PathFlashKV, Shape: sh}, rows, false},
				plan{kernels.DecodePlan{Path: kernels.PathStaged, Shape: kernels.PagedStagedAt(sh, 6*page, 3)}, rows, i%3 == 0})
		}
		for _, keys := range []int{64, 4096, 65536} {
			plans = append(plans, plan{kernels.ChooseDecodePlan("ptx", s, keys, 2560), "rows", false})
		}
	}
	mla := kernels.FlashShape{Heads: 8, KVHeads: 1, Dim: 96, MLA: 64, Page: page, Scale: .09}
	plans = append(plans, plan{kernels.ChooseDecodePlan("ptx", mla, 512, 0), "rows", false},
		plan{kernels.DecodePlan{Path: kernels.PathStaged, Shape: kernels.PagedStagedAt(mla, 6*page, 2)}, "one-long", true})
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			lanes := 1
			if ok, _ := backend.GuaranteedLanes(d, 32); ok {
				lanes = 32
			}
			ran := 0
			for i, pl := range plans {
				if pl.p.Path == kernels.PathFlashKV && lanes != 32 {
					continue // FlashDecodeKV reduces across a warp
				}
				if pl.p.Shape.VecV && d.API() == "spirv" {
					continue // vector loads lower on PTX and MSL (FlashShape.VecV)
				}
				rows := pagedCases(page)[pl.rows]
				p := pl.p
				p.Shape.Rows = len(rows)
				s := p.Shape
				if p.Path == kernels.PathStaged {
					// The chunk covers every row's keys at the plan's splits.
					p.Shape = kernels.PagedStagedAt(s, 6*page, max(s.Splits, 1))
					s = p.Shape
				}
				name := fmt.Sprintf("%d/%s/%s", i, p.Path, pl.rows)
				pool := newPagedPoolOrder(rand.New(rand.NewSource(int64(5+i))), page, s.KVHeads*s.Dim, s.F16, rows, false, s.MLA > 0)
				if s.F16K {
					pool.withF16K()
				}
				want := runAttnSteps(t, d, s, attnPlanSteps(t, p, lanes, pl.items, false), pool, pool.tab)
				formed := attnPlanSteps(t, p, lanes, pl.items, true)
				got := runAttnSteps(t, d, s, formed, pool, pool.tab)
				if !bytes.Equal(got, want) {
					t.Errorf("%s (%+v): the formed attention parts from the menu's", name, s)
				}
				if bad := runAttnSteps(t, d, s, formed, pool, pool.wrong); bytes.Equal(bad, want) {
					t.Errorf("%s: the formed attention over the wrong pages matched the menu's", name)
				}
				// And a formed plan whose score scale is one ulp off.
				ulp := p
				ulp.Shape.Scale = math.Nextafter32(s.Scale, 1)
				if bad := runAttnSteps(t, d, s, attnPlanSteps(t, ulp, lanes, pl.items, true), pool, pool.tab); bytes.Equal(bad, want) {
					t.Errorf("%s: an attention formed at a scale one ulp off matched the menu's", name)
				}
				ran++
			}
			if ran < 30 {
				t.Fatalf("only %d plans ran: the gate proves nothing", ran)
			}
			t.Logf("%d plans formed bit for bit (lanes %d)", ran, lanes)
		})
	}
}

// runAttnSteps runs a plan's launches over pool through table tab and returns the
// output's bytes, a NaN guard after it checked.
func runAttnSteps(t *testing.T, d backend.Device, s kernels.FlashShape, steps []attnStep, p *pagedPool, tab []uint32) []byte {
	t.Helper()
	g := newGPU(t, d)
	defer g.free()
	rng := rand.New(rand.NewSource(97))
	vd := kernels.StagedValue(s)
	q := make([]float32, s.Rows*s.Heads*s.Dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	sinks := make([]float32, s.Heads)
	for h := range sinks {
		sinks[h] = float32(h%4) - 1.5
	}
	nan := func(n int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = nan32
		}
		return f32bytes(f)
	}
	const guard = 32
	n := s.Rows * s.Heads * vd
	ms := s
	if ms.MLA > 0 {
		ms.Dim, ms.MLA = ms.MLA, 0
	}
	ms.Splits = max(ms.Splits, 1)
	plane := 1
	if s.Chunk > 0 && s.Splits > 0 {
		plane = kernels.PagedStagedPlane(s)
	}
	kb := g.up(p.kBytes())
	vb := kb
	if s.MLA == 0 {
		vb = g.up(p.vBytes())
	}
	bufs := map[string]backend.Buf{
		"q": g.up(f32bytes(q)), "k": kb, "v": vb, "n": g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF})),
		"out": g.up(nan(n + guard)), "tab": g.up(u32bytes(tab)), "desc": g.up(u32bytes(p.desc)),
		"sink": g.up(f32bytes(sinks)), "part": g.up(nan(kernels.FlashPartialFloats(ms))),
		"sc": g.up(nan(plane)), "wt": g.up(nan(plane)),
	}
	for _, st := range steps {
		if !laneOK(t, d, st.k) {
			t.Fatalf("%s needs %d lanes", st.k.Name, st.k.Lanes)
		}
		c, err := d.Compile(st.k)
		if err != nil {
			t.Fatal(err)
		}
		args := make([]backend.Buf, len(st.args))
		for i, a := range st.args {
			args[i] = bufs[a]
		}
		err = c.Launch(st.groups, st.width, args...)
		c.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	raw := make([]byte, 4*(n+guard))
	if err := bufs["out"].Read(raw); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw[4*n:], nan(guard)) {
		t.Fatal("the output's guard was written")
	}
	return raw[:4*n]
}
