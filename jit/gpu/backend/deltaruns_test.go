package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestRunsStepEachSequenceAlone holds the run forms of a linear block's
// recurrence -- GatedDeltaFused with Runs, Conv1dRowsRuns, Conv1dShiftRuns --
// to each sequence run alone through the chunk kernels, Conv1dRows,
// Conv1dShift and GatedDeltaFused over its own rows from its own state. The
// step carries three sequences: one decoding (a run of one row) and two
// prompt chunks (runs of three and two), its rows in an order that is
// neither by sequence nor by position, as a scheduled step lays them out (its
// logit rows lead). Bit for bit on ptx, where the two forms do the same
// arithmetic in the same order; 5e-5 elsewhere, since SPIR-V may contract a
// multiply-add differently in two different kernels.
//
// The failures it guards are the two ways a run goes wrong: sequences sharing
// a history, and a chunk's rows not chaining -- each reading its sequence's
// state as it was before the step, which is a batched decode's semantics and
// fluent on a chunk's first row only. The second is run here as the violation:
// the same rows with every row its own run must miss the chunks alone.
//
// The runs' states sit at scattered slots of a pool larger than the step, the
// delta rule writes each at an out slot different from its in slot, and the
// step has fewer runs than rows, so the descriptor's surplus entries repeat the
// last run: a kernel that indexes by row rather than by the descriptor reads
// another sequence's history, and the slots no run names must come back
// untouched.
func TestRunsStepEachSequenceAlone(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	const taps = 4
	ran := 0
	for _, c := range []oldkernels.DeltaScan{
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, Lanes: ir.SubgroupLanes},
		{VHeads: 2, VDim: 32, KDim: 32, Rep: 1, PerChan: true, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, TiledK: 2, Lanes: ir.SubgroupLanes},
		// The one-thread form a device without the subgroup steps its rows on
		// (the tier's prepRagLinear), at ragged widths: a key width under,
		// between and over the subgroup, one with no power-of-two lane split
		// past four (12) and one with none at all (7), value widths that are
		// not a multiple of anything, and Qwen3.5's own 128.
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, Lanes: 1},
		{VHeads: 2, VDim: 32, KDim: 32, Rep: 1, PerChan: true, Lanes: 1},
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, TiledK: 2, Lanes: 1},
		{VHeads: 4, VDim: 9, KDim: 8, Rep: 2, Lanes: 1},
		{VHeads: 2, VDim: 13, KDim: 12, Rep: 1, PerChan: true, Lanes: 1},
		{VHeads: 2, VDim: 5, KDim: 7, Rep: 2, Lanes: 1},
		{VHeads: 2, VDim: 24, KDim: 48, Rep: 1, Lanes: 1},
		{VHeads: 4, VDim: 128, KDim: 128, Rep: 2, Lanes: 1},
		{VHeads: 2, VDim: 128, KDim: 128, Rep: 1, PerChan: true, Lanes: 1},
	} {
		for _, d := range devs {
			if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok && c.Lanes != 1 {
				continue
			}
			t.Run(fmt.Sprintf("%s/v%dx%dx%d-chan%v-tiled%d-lanes%d", d.API(), c.VHeads, c.VDim, c.KDim,
				c.PerChan, c.TiledK, c.Lanes), func(t *testing.T) {
				runsCase(t, d, c, taps)
			})
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no device")
	}
}

// stepRuns is a step's rows grouped into runs: runs[j] are run j's rows in
// position order.
type stepRuns [][]int

// desc is the run descriptor for a step compiled for n rows (see
// oldkernels.RunDescWords), run j's state read at in[j] and written at out[j].
func (rs stepRuns) desc(n int, in, out []int) []byte {
	w := make([]uint32, oldkernels.RunDescWords(n))
	w[0] = uint32(n)
	at := 0
	for j := 0; j < n; j++ {
		s := min(j, len(rs)-1) // surplus entries repeat the last run
		w[1+2*j], w[2+2*j] = uint32(in[s]), uint32(out[s])
		start := at
		if j >= len(rs) {
			start = at - len(rs[s])
		}
		w[1+2*n+2*j], w[2+2*n+2*j] = uint32(start), uint32(len(rs[s]))
		if j < len(rs) {
			for k, r := range rs[j] {
				w[1+4*n+at+k] = uint32(r)
				w[1+5*n+r] = uint32(at + k)
				w[1+6*n+r] = uint32(j)
			}
			at += len(rs[j])
		}
	}
	var b []byte
	for _, v := range w {
		b = append(b, u32le(v)...)
	}
	return b
}

func runsCase(t *testing.T, d backend.Device, c oldkernels.DeltaScan, taps int) {
	// Rows 0..5: sequence 0 is a chunk at rows 4, 0, 2 (positions in that
	// order), sequence 1 decodes at row 5, sequence 2 is a chunk at rows 1, 3.
	runs := stepRuns{{4, 0, 2}, {5}, {1, 3}}
	const R = 6
	S := len(runs)
	kHeads := c.VHeads / c.Rep
	if c.TiledK > 0 {
		kHeads = c.TiledK
	}
	qk, vw := kHeads*c.KDim, c.VHeads*c.VDim
	chans := 2*qk + vw
	baRep := c.VHeads / kHeads
	nGate := c.VHeads
	if c.PerChan {
		nGate = c.VHeads * c.KDim
	}
	eps := float32(1e-6 / float64(c.KDim))
	sc := float32(1 / math.Sqrt(float64(c.KDim)))
	rng := rand.New(rand.NewSource(int64(R*1000 + c.VHeads)))
	fill := func(n int, f func() float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(f())
		}
		return x
	}
	sLen, cLen := c.VHeads*c.VDim*c.KDim, (taps-1)*chans
	st := fill(S*sLen, func() float64 { return rng.NormFloat64() * 0.3 })
	cst := fill(S*cLen, rng.NormFloat64)
	mixed := fill(R*chans, rng.NormFloat64)
	w := fill(taps*chans, rng.NormFloat64)
	conv := fill(R*chans, rng.NormFloat64)
	ba := fill(R*2*c.VHeads, rng.NormFloat64)
	alpha := fill(R*nGate, rng.NormFloat64)
	bRaw := fill(R*c.VHeads, rng.NormFloat64)
	dt := fill(nGate, rng.NormFloat64)
	aa := fill(nGate, func() float64 { return -0.1 - 2*rng.Float64() })

	g := newGPU(t, d)
	defer g.free()
	buf := func(n int) backend.Buf { return g.up(make([]byte, n*4)) }
	lin := func(k *ir.Kernel, err error, n int, args ...backend.Buf) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("%s does not validate: %v", k.Name, err)
		}
		ck, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		defer ck.Close()
		if err := ck.Launch((n+127)/128, 128, args...); err != nil {
			t.Fatal(err)
		}
	}
	read := func(b backend.Buf, n int) []float32 {
		p := make([]byte, n*4)
		if err := b.Read(p); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = math.Float32frombits(uint32(p[4*i]) | uint32(p[4*i+1])<<8 | uint32(p[4*i+2])<<16 | uint32(p[4*i+3])<<24)
		}
		return v
	}
	// rowsOf gathers rows rs of x, w floats each, in the order given.
	rowsOf := func(x []float32, w int, rs []int) []float32 {
		var o []float32
		for _, r := range rs {
			o = append(o, x[r*w:(r+1)*w]...)
		}
		return o
	}
	fuse := oldkernels.DeltaFuse{Chans: chans, Eps: eps, Scale: sc, BARep: baRep}
	bDt, bA := g.up(f32bytes(dt)), g.up(f32bytes(aa))
	bW := g.up(f32bytes(w))

	// The step, each run's state at its own slots of a pool of P.
	P := S + 4
	in, out := []int{3, 0, 5}, []int{1, 6, 2}
	scatter := func(src []float32, per int, at []int) []float32 {
		p := make([]float32, P*per)
		for i := range p {
			p[i] = float32(rng.NormFloat64())
		}
		for j := 0; j < S; j++ {
			copy(p[at[j]*per:(at[j]+1)*per], src[j*per:(j+1)*per])
		}
		return p
	}
	gather := func(pool []float32, per int, at []int) []float32 {
		var o []float32
		for j := 0; j < S; j++ {
			o = append(o, pool[at[j]*per:(at[j]+1)*per]...)
		}
		return o
	}
	pS, pC := scatter(st, sLen, in), scatter(cst, cLen, in)
	bConv, bBA, bAlpha, bBRaw := g.up(f32bytes(conv)), g.up(f32bytes(ba)), g.up(f32bytes(alpha)), g.up(f32bytes(bRaw))
	bMixed := g.up(f32bytes(mixed))
	cr := c
	cr.Rows, cr.Runs = R, true
	// step runs the run forms over rs and returns the outputs and the states
	// the runs wrote, run by run.
	step := func(rs stepRuns, in, out []int, lanes int) (dOut, dS, cOut, cS []float32) {
		cr := cr
		cr.Lanes = lanes
		bN := g.up(rs.desc(R, in, out))
		outI, sI := buf(R*vw), g.up(f32bytes(poisonPool(P*sLen)))
		k, err := oldkernels.GatedDeltaFused(cr, fuse)
		bS := g.up(f32bytes(pS))
		if c.PerChan {
			lin(k, err, cr.Threads(), bS, bConv, bAlpha, bBRaw, bDt, bA, outI, sI, bN)
		} else {
			lin(k, err, cr.Threads(), bS, bConv, bBA, bDt, bA, outI, sI, bN)
		}
		convI, cstI := buf(R*chans), g.up(f32bytes(poisonPool(P*cLen)))
		bC := g.up(f32bytes(pC))
		k, err = oldkernels.Conv1dRowsRuns(taps, chans, R)
		lin(k, err, R*chans, bC, bMixed, bW, convI, bN)
		k, err = oldkernels.Conv1dShiftRuns(taps, chans, R)
		lin(k, err, R*cLen, bC, bMixed, cstI, bN)
		poolS, poolC := read(sI, P*sLen), read(cstI, P*cLen)
		poisonKept(t, "state pool", f64of(poolS), sLen, out...)
		poisonKept(t, "conv state pool", f64of(poolC), cLen, in...)
		return read(outI, R*vw), gather(poolS, sLen, out), read(convI, R*chans), gather(poolC, cLen, in)
	}
	gotOut, gotS, gotConv, gotCst := step(runs, in, out, c.Lanes)

	// Each sequence alone, its rows in position order through the chunk
	// kernels, from its own state. The outputs land at the rows they came
	// from, so they compare row for row with the step's.
	wantOut, wantConv := make([]float32, R*vw), make([]float32, R*chans)
	var wantS, wantCst []float32
	for j, rs := range runs {
		m := len(rs)
		cj := c
		cj.Rows = m
		chunk := g.up(recDesc(uint32(m), 0, 0))
		o, s := buf(m*vw), buf(sLen)
		k, err := oldkernels.GatedDeltaFused(cj, fuse)
		bs1, bc1 := g.up(f32bytes(st[j*sLen:(j+1)*sLen])), g.up(f32bytes(rowsOf(conv, chans, rs)))
		if c.PerChan {
			lin(k, err, cj.Threads(), bs1, bc1, g.up(f32bytes(rowsOf(alpha, nGate, rs))),
				g.up(f32bytes(rowsOf(bRaw, c.VHeads, rs))), bDt, bA, o, s, chunk)
		} else {
			lin(k, err, cj.Threads(), bs1, bc1, g.up(f32bytes(rowsOf(ba, 2*c.VHeads, rs))), bDt, bA, o, s, chunk)
		}
		ov := read(o, m*vw)
		cv, cs := buf(m*chans), buf(cLen)
		bcs1, bx1 := g.up(f32bytes(cst[j*cLen:(j+1)*cLen])), g.up(f32bytes(rowsOf(mixed, chans, rs)))
		k, err = oldkernels.Conv1dRows(taps, chans, m)
		lin(k, err, m*chans, bcs1, bx1, bW, cv, chunk)
		k, err = oldkernels.Conv1dShift(taps, chans, m)
		lin(k, err, cLen, bcs1, bx1, cs, chunk)
		cvv := read(cv, m*chans)
		for i, r := range rs {
			copy(wantOut[r*vw:(r+1)*vw], ov[i*vw:(i+1)*vw])
			copy(wantConv[r*chans:(r+1)*chans], cvv[i*chans:(i+1)*chans])
		}
		wantS, wantCst = append(wantS, read(s, sLen)...), append(wantCst, read(cs, cLen)...)
	}
	exact := d.API() == "ptx"
	nmse := func(got, want []float32) float64 {
		var sse, sy2 float64
		for i := range want {
			d := float64(got[i]) - float64(want[i])
			sse, sy2 = sse+d*d, sy2+float64(want[i])*float64(want[i])
		}
		return sse / sy2
	}
	for _, cmp := range []struct {
		what      string
		got, want []float32
	}{{"output", gotOut, wantOut}, {"state", gotS, wantS}, {"conv", gotConv, wantConv}, {"conv state", gotCst, wantCst}} {
		if exact {
			for i := range cmp.want {
				if cmp.got[i] != cmp.want[i] {
					t.Fatalf("%s[%d] = %g, alone %g", cmp.what, i, cmp.got[i], cmp.want[i])
				}
			}
		}
		if n := nmse(cmp.got, cmp.want); math.IsNaN(n) || n > 5e-5 {
			t.Fatalf("%s: NMSE %.3e against each sequence alone", cmp.what, n)
		}
	}

	// The violation: every row its own run, a chunk's rows each reading the
	// state from before the step. The decoding row is right either way; the
	// chunks' later rows must not be.
	var flat stepRuns
	var fin, fout []int
	for j, rs := range runs {
		for _, r := range rs {
			flat, fin, fout = append(flat, []int{r}), append(fin, in[j]), append(fout, out[j])
		}
	}
	badOut, _, badConv, _ := step(flat, fin, fout, c.Lanes)
	for _, cmp := range []struct {
		what      string
		got, want []float32
	}{{"output", badOut, wantOut}, {"conv", badConv, wantConv}} {
		n := nmse(cmp.got, cmp.want)
		if !(n > 1e-3) {
			t.Fatalf("%s: rows that do not chain read NMSE %.3e against each sequence alone: "+
				"this gate cannot see a chunk's rows chaining", cmp.what, n)
		}
		t.Logf("violation (no chaining): %s NMSE %.3e", cmp.what, n)
	}

	// The one-thread form against the lane groups on the same step, where the
	// device has the subgroup and the key width splits: the same rule with
	// each dot summed serially rather than by a butterfly, so they part by
	// reassociation alone. The run forms above are held to the chunk forms of
	// their own lane count; this is what holds the two counts to each other,
	// and the chunk's lane-group form is held to the unfused chain and that to
	// the host oracle (TestGatedDeltaFusedMatchesTheUnfusedChain,
	// TestGatedDeltaScanOnEveryDevice). The violation is the unchained step
	// at one lane, which the bound must refuse.
	wide := c
	wide.Lanes = ir.SubgroupLanes
	if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); c.Lanes != 1 || !ok || wide.ScanLanes() == 1 {
		return
	}
	lanesOut, lanesS, _, _ := step(runs, in, out, ir.SubgroupLanes)
	for _, cmp := range []struct {
		what      string
		got, want []float32
	}{{"output", gotOut, lanesOut}, {"state", gotS, lanesS}} {
		n := nmse(cmp.got, cmp.want)
		if math.IsNaN(n) || n > formBand {
			t.Fatalf("%s: the one-thread form reads NMSE %.3e against %d lanes a row, over %.0e",
				cmp.what, n, wide.ScanLanes(), formBand)
		}
		t.Logf("one thread against %d lanes a row: %s NMSE %.3e", wide.ScanLanes(), cmp.what, n)
	}
	n := nmse(badOut, lanesOut)
	if !(n > formBand) {
		t.Fatalf("the unchained one-thread step reads NMSE %.3e against the lane groups: the bound "+
			"cannot see a wrong step", n)
	}
	t.Logf("violation (no chaining) against the lane groups: output NMSE %.3e", n)
}

// formBand is how far the scan's one-thread and lane-group forms may part on
// one step: f32 reassociation of two dots and a norm, a few ulp of each
// output.
const formBand = 1e-10
