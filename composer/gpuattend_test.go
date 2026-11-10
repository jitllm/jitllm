package composer_test

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// attendRowsCase is an attention a gate forms at a prompt chunk's rows: its
// heads, its window (0 none), whether it carries sinks and whether its
// history holds the values as float32 rather than binary16.
type attendRowsCase struct {
	heads, kvh, dim, window int
	sinks, f32v             bool
}

// attendInputs is what a gate's factors hold: the positions before the
// chunk (2100 where 0), the factors' deviation (1 where 0; at 4 most
// exponentials fall below binary16's normal range), whether the row and
// position factors share a large part whose dots cancel, and the chunk's
// rows [run[0], run[1]) that attend to the run's end, as a picture's rows
// do, each window still starting from the row's own position.
type attendInputs struct {
	ctx    int
	spread float64
	cancel bool
	run    [2]int
}

// TestComposedGPUAttendRows forms an appended history and the attention
// over it at a prompt chunk's rows on every GPU, the served models' pool
// layout (keys float32 by dimension, values by position, binary16 or, as a
// float32 layer holds them, float32), and runs
// fewer live rows than the form holds, past a workgroup's chunk of
// positions so a causal row's span takes two partitions, starting inside a
// page and crossing into the next: each row attends causally from the history's
// start, or from its window's where the attention slides, with a sink a head
// where it carries them, held row by row to the definition over the history
// as the pools hold it. The form must run its partitions in order (the
// running pass), or where the device's matrix unit is stated, run on the
// unit (the flash pass), a window a bound of its loop, every
// device whose backend lowers a tile in both arms, which a CUDA device must;
// its scratch is counted at a key bound far past the live
// positions: no binding a kernel reads or writes may hold more than four
// words a row, head and dimension, where a [rows, heads, keys] buffer at the
// bound is sixty-four of them. Two violations must fail: every row's first
// attended position moved by one, and the sinks zeroed.
func TestComposedGPUAttendRows(t *testing.T) {
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
	cases := []struct {
		c  attendRowsCase
		in attendInputs
	}{
		{attendRowsCase{8, 2, 64, 0, false, false}, attendInputs{}},
		{attendRowsCase{8, 2, 64, 0, true, false}, attendInputs{}},
		{attendRowsCase{16, 8, 128, 0, false, false}, attendInputs{}},
		{attendRowsCase{64, 8, 64, 16, true, false}, attendInputs{}},
		{attendRowsCase{64, 8, 64, 0, true, false}, attendInputs{}},
		{attendRowsCase{12, 4, 64, 16, false, false}, attendInputs{}},
		{attendRowsCase{8, 2, 64, 0, false, true}, attendInputs{}},
		{attendRowsCase{16, 8, 128, 0, false, true}, attendInputs{}},
		{attendRowsCase{64, 8, 64, 0, true, true}, attendInputs{}},
		{attendRowsCase{10, 2, 80, 0, false, false}, attendInputs{}},
		{attendRowsCase{12, 4, 96, 0, true, true}, attendInputs{}},
		{attendRowsCase{16, 8, 128, 0, false, false}, attendInputs{cancel: true}},
		{attendRowsCase{16, 8, 128, 0, false, true}, attendInputs{cancel: true}},
		{attendRowsCase{8, 2, 64, 0, true, false}, attendInputs{spread: 4}},
		{attendRowsCase{8, 2, 64, 0, false, true}, attendInputs{spread: 4}},
		{attendRowsCase{8, 2, 128, 0, false, false}, attendInputs{ctx: 8192}},
		{attendRowsCase{8, 2, 128, 0, true, true}, attendInputs{ctx: 8192}},
		{attendRowsCase{12, 4, 64, 16, false, false}, attendInputs{run: [2]int{0, 20}}},
		{attendRowsCase{8, 2, 64, 0, false, false}, attendInputs{run: [2]int{5, 20}}},
	}
	matrices, cuda := 0, false
	for _, d := range devs {
		cuda = cuda || d.API() == "ptx"
		tile, _ := backend.LowersTile(d, backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32})
		for _, matrix := range []bool{false, true} {
			if matrix && !tile {
				continue
			}
			for _, tc := range cases {
				if matrix {
					matrices++
				}
				c, in := tc.c, tc.in
				name := fmt.Sprintf("%s/matrix%v/%dq%dkv%dd/window%d/sinks%v/f32values%v/ctx%d/spread%g/cancel%v/run%d-%d", d.API(), matrix,
					c.heads, c.kvh, c.dim, c.window, c.sinks, c.f32v, in.ctx, in.spread, in.cancel, in.run[0], in.run[1])
				t.Run(name, func(t *testing.T) { gpuAttendRows(t, d, c, in, matrix) })
			}
		}
	}
	if cuda && matrices == 0 {
		t.Error("no device lowers a matrix tile, a CUDA device among them: the gate did not run the matrix unit's form")
	}
}

// gpuAttendRows is one case on d, the device's matrix unit stated where
// matrix is: an attention then forms on it (the flash pass), its blocks of
// positions run in order, and without it the split's running form.
func gpuAttendRows(t *testing.T, d backend.Device, c attendRowsCase, in attendInputs, matrix bool) {
	const page, capRows = 64, 64
	p0, spread, R := cmp.Or(in.ctx, 2100), cmp.Or(in.spread, 1), 37
	if in.ctx > 0 {
		R = 9
	}
	keys := (p0 + R + 64*page - 1) / (64 * page) * 64 * page
	b := graph.New()
	n := b.KVAppend(b.In("k", c.kvh*c.dim), b.In("v", c.kvh*c.dim), 0)
	var sinks []jlm.Role
	if c.sinks {
		sinks = []jlm.Role{jlm.RoleAttnSinks}
	}
	scale := float32(1 / math.Sqrt(float64(c.dim)))
	b.Out(b.Attend(b.In("q", c.heads*c.dim), n, graph.Attrs{NHead: c.heads, NKVHead: c.kvh, HeadDim: c.dim, Scale: float64(scale),
		Window: c.window}, sinks...))
	g := b.G
	dev := gpuDevice(d)
	if matrix {
		dev.Matrix = [3]int{16, 16, 16}
	}
	_, codes, err := composer.Default().Form(composer.Request{Graph: g, Step: composer.Step{Rows: capRows, Page: page, Keys: keys, Seqs: 1},
		Device: dev, History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: !c.f32v}}})
	if err != nil {
		t.Fatal(err)
	}
	var scratch int64
	running, flash := 0, 0
	for _, cd := range codes {
		if slices.Contains(cd.Passes, "running") {
			running++
		}
		if slices.Contains(cd.Passes, "flash") {
			flash++
			scaled := cd.GPU != nil && slices.ContainsFunc(cd.GPU.Ops, func(o ir.Op) bool { return o.Kind == ir.OpTileScale })
			if scaled != dev.ScaleRows {
				t.Errorf("%s scales its sums' rows in place %v on a device that states it %v", cd.Name, scaled, dev.ScaleRows)
			}
		}
		for _, bd := range cd.Binds {
			if bd.Operand.Kind != composer.OperandScratch || !bd.Read && !bd.Writes {
				continue
			}
			scratch = max(scratch, bd.Bytes)
			if bd.Bytes > int64(4*4*capRows*c.heads*c.dim) {
				t.Errorf("%s binds %s of %d bytes: more than four words a row, head and dimension at a key bound of %d",
					cd.Name, bd.Operand.Name, bd.Bytes, keys)
			}
		}
	}
	switch {
	case matrix && (flash != 1 || running != 0):
		t.Errorf("%d kernels run on the matrix unit and %d the partitions in order: the gate did not select the matrix unit's form", flash, running)
	case !matrix && (running != 1 || flash != 0):
		t.Errorf("%d kernels run the partitions in order and %d on the matrix unit: the gate did not select the running form", running, flash)
	}
	t.Logf("%d kernels, the largest scratch the attention reads or writes %d bytes (a [rows, heads, keys] buffer is %d)",
		len(codes), scratch, 4*capRows*c.heads*keys)

	rng := rand.New(rand.NewSource(11))
	w, qw := c.kvh*c.dim, c.heads*c.dim
	rnd := func(n int, sd float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * sd)
		}
		return x
	}
	// Cancelling factors: a position's is 8 at every element and a row's 8
	// and -8 in turn, beside a deviation of 1/8.
	factor := func(n int, row bool) []float32 {
		if !in.cancel {
			return rnd(n, spread)
		}
		x := rnd(n, 0.125)
		for i := range x {
			if row && i%2 == 1 {
				x[i] -= 8
			} else {
				x[i] += 8
			}
		}
		return x
	}
	ks, vs := make([][]float32, p0+R), make([][]float32, p0+R)
	for p := range p0 + R {
		ks[p], vs[p] = factor(w, false), rnd(w, 1)
	}
	q := factor(capRows*qw, true)
	k, v := make([]float32, capRows*w), make([]float32, capRows*w)
	for r := range R {
		copy(k[r*w:], ks[p0+r])
		copy(v[r*w:], vs[p0+r])
	}
	kpool := poisoned(keys / page * page * w)
	vpool := poisoned(keys / page * page * w / 2)
	if c.f32v {
		vpool = poisoned(keys / page * page * w)
	}
	for p := range p0 {
		pg, off := p/page, p%page
		for e := range w {
			kpool[(pg*w+e)*page+off] = math.Float32bits(ks[p][e])
			hv, i := uint32(quant.EncodeHalf(vs[p][e])), (pg*page+off)*w+e
			switch {
			case c.f32v:
				vpool[i] = math.Float32bits(vs[p][e])
			case e%2 == 0:
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			default:
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	round := func(x float32) float64 {
		if c.f32v {
			return float64(x)
		}
		return float64(quant.DecodeHalf(quant.EncodeHalf(x)))
	}
	first := func(r int) int {
		if c.window > 0 {
			return max(0, p0+r+1-c.window)
		}
		return 0
	}
	// end is one past row r's last attended position; runOn off states the
	// run's rows causal, the violation.
	runOn := true
	end := func(r int) int {
		if runOn && r >= in.run[0] && r < in.run[1] {
			return p0 + in.run[1]
		}
		return p0 + r + 1
	}
	// A sink near the log-sum-exp of the first row's scores holds about as
	// much as the positions do, at any length and spread.
	sinkv := make([]float32, c.heads)
	for h := range sinkv {
		var sc []float64
		for p := first(0); p <= p0; p++ {
			var s float64
			for e := range c.dim {
				s += float64(q[h*c.dim+e]) * float64(ks[p][h/(c.heads/c.kvh)*c.dim+e])
			}
			sc = append(sc, s*float64(scale))
		}
		top, sum := slices.Max(sc), 0.0
		for _, x := range sc {
			sum += math.Exp(x - top)
		}
		sinkv[h] = float32(top + math.Log(sum) + rng.NormFloat64())
	}
	// row is the definition at row r over the pools with the row factor qr
	// and, where msq, the expected squared error the flash form's binary16
	// operands add to it (flashErr).
	row := func(r int, qr []float32, sinkOn, msq bool) ([]float64, float64) {
		want, n := make([]float64, qw), end(r)-first(r)
		var mse float64
		raw, wt := make([]float64, 0, n+1), make([]float64, 0, n+1)
		for h := range c.heads {
			kv, qh := h/(c.heads/c.kvh), qr[h*c.dim:(h+1)*c.dim]
			raw = raw[:0]
			for p := first(r); p < end(r); p++ {
				var s float64
				for e, x := range qh {
					s += float64(x) * float64(ks[p][kv*c.dim+e])
				}
				raw = append(raw, s*float64(scale))
			}
			if sinkOn {
				raw = append(raw, float64(sinkv[h]))
			}
			wt = append(wt[:0], raw...)
			oracle.Softmax(wt)
			o := want[h*c.dim : (h+1)*c.dim]
			for d := range o {
				for i := range n {
					o[d] += wt[i] * round(vs[first(r)+i][kv*c.dim+d])
				}
			}
			if !msq {
				continue
			}
			mse += flashErr(qh, func(i int) []float32 { return ks[first(r)+i][kv*c.dim : (kv+1)*c.dim] },
				func(i, d int) float64 { return round(vs[first(r)+i][kv*c.dim+d]) }, wt[:n], raw, o, float64(scale), c.f32v)
		}
		return want, mse
	}
	pages := make([]uint32, keys/page)
	for i := range pages {
		pages[i] = uint32(i)
	}
	run := func(shift uint32, sv, q []float32) []float32 {
		rows, win := make([]uint32, capRows*4), make([]uint32, capRows*4)
		for r := range R {
			p, e := uint32(p0+r), uint32(end(r))
			copy(rows[r*4:], []uint32{0, shift, e, p})
			copy(win[r*4:], []uint32{0, uint32(first(r)) + shift, e, p})
		}
		vals := map[graph.ValueID][]uint32{g.In[0]: f32w(k), g.In[1]: f32w(v), g.In[2]: f32w(q), g.Out[0]: poisoned(capRows * qw)}
		tables := map[string][]uint32{"rows": rows, "rows.win": win, "rows.n": {uint32(R)}, "pages": pages,
			"kv.k": append([]uint32(nil), kpool...), "kv.v": append([]uint32(nil), vpool...)}
		vecs := map[jlm.Role][]uint32{jlm.RoleAttnSinks: f32w(sv)}
		out, err := launchAttend(d, codes, vals, tables, vecs, g.Out[0])
		if err != nil {
			t.Fatal(err)
		}
		return wf32(out)
	}
	nmse := func(got []float32, want []float64, r int) float64 {
		var num, den float64
		for i := r * qw; i < (r+1)*qw; i++ {
			e := float64(got[i]) - want[i]
			num += e * e
			den += want[i] * want[i]
		}
		return num / den
	}
	want := make([]float64, R*qw)
	budget := make([]float64, R)
	// A device whose lowering rounds every operation alone is held to
	// float32's band; a driver free to contract and approximate (SPIR-V's)
	// to a wider one: the V100's Vulkan driver reads 1.4e-7 on the form
	// before the running pass as after it, the laptop's 3e-14.
	band := 1e-8
	if !dev.ExactArith {
		band = 1e-6
	}
	for r := range R {
		w, mse := row(r, q[r*qw:(r+1)*qw], c.sinks, flash == 1)
		copy(want[r*qw:], w)
		budget[r] = band
		if flash == 1 {
			budget[r] = flashBudget(mse, w, band)
		}
	}
	got := run(0, sinkv, q)
	worst, tight := 0.0, 0.0
	for r := range R {
		e := nmse(got, want, r)
		if math.IsNaN(e) || e > budget[r] {
			t.Errorf("row %d at position %d: NMSE %.3e against the definition, past its budget %.3e", r, p0+r, e, budget[r])
		}
		worst, tight = max(worst, e), max(tight, e/budget[r])
	}
	for i := R * qw; i < capRows*qw; i++ {
		if math.Float32bits(got[i]) != guardWord {
			t.Fatalf("output word %d, past the call's %d rows, was written", i, R)
		}
	}
	t.Logf("%d rows of a %d-row form: worst NMSE %.3e against the definition, %.3f of a row's budget at most", R, capRows, worst, tight)
	last := R - 1
	// A peaked softmax hides one position, and cancelling factors' binary16
	// error is as wide as one: half the context is moved.
	shift := uint32(1)
	if spread > 1 || in.cancel {
		shift = uint32(p0 / 2)
	}
	if e := nmse(run(shift, sinkv, q), want, last); !(e > max(1e-6, budget[last])) {
		t.Errorf("with every row's first attended position moved the NMSE is %.3e: the gate cannot see a wrong row", e)
	}
	if c.sinks {
		if e := nmse(run(0, make([]float32, c.heads), q), want, 0); !(e > max(1e-6, budget[0])) {
			t.Errorf("with the sinks zeroed the NMSE is %.3e: the gate cannot see the sinks", e)
		}
	}
	if in.run[1] > 0 {
		runOn = false
		causal := run(0, sinkv, q)
		runOn = true
		r := in.run[0]
		if e := nmse(causal, want, r); !(e > max(1e-6, budget[r])) {
			t.Errorf("with the run's rows causal row %d reads NMSE %.3e: the gate cannot see the run", r, e)
		}
	}
	if flash != 1 {
		return
	}
	// The last row's factor scaled by 1+f, where the definition moves one
	// and a half budgets past the form's own error there: a fault a budget
	// a few times too wide lets through, this one must not.
	lw := want[last*qw : (last+1)*qw]
	target := math.Pow(math.Sqrt(1.5*budget[last])+math.Sqrt(nmse(got, want, last)), 2)
	scaled := func(f float64) []float32 {
		x := slices.Clone(q)
		for i := last * qw; i < (last+1)*qw; i++ {
			x[i] = float32(float64(x[i]) * (1 + f))
		}
		return x
	}
	moved := func(f float64) float64 {
		w, _ := row(last, scaled(f)[last*qw:(last+1)*qw], c.sinks, false)
		var num, den float64
		for i, x := range w {
			num += (x - lw[i]) * (x - lw[i])
			den += lw[i] * lw[i]
		}
		return num / den
	}
	lo, hi := 0.0, 1.0
	for range 40 {
		if mid := (lo + hi) / 2; moved(mid) < target {
			lo = mid
		} else {
			hi = mid
		}
	}
	e := nmse(run(0, sinkv, scaled(hi)), want, last)
	t.Logf("the last row's factor scaled by 1+%.3e moves the definition %.3e: the form reads %.3e, %.2f budgets", hi, moved(hi), e, e/budget[last])
	if !(e > budget[last]) {
		t.Errorf("the last row's factor scaled by 1+%.3e reads NMSE %.3e within its budget %.3e: the budget is wider than the form's error", hi, e, budget[last])
	}
}

// flashErr is the expected squared error binary16 operands add to a head
// of a row -- its factor, the position factors and a float32 history's
// second factor rounded -- each rounding an independent error uniform over a
// binary16 step, carried to first order: an error in score i moves output d
// by w_i (v_id - o_d), so the row factor's, shared by every position, by
// sum_e err_e A_de with A_de = scale sum_i w_i (v_id - o_d) k_ie. The
// exponentials are staged as their halves and their remainders' halves,
// within float32's band, but an exponential is taken in the frame of a
// running maximum, at most the row's, and one that can fall below binary16's
// normal range loses its remainder too: a subnormal step.
func flashErr(qh []float32, k func(int) []float32, v func(i, d int) float64, w, raw, o []float64, scale float64, f32v bool) float64 {
	const normal = 1.0 / (1 << 14)
	D, top := len(qh), slices.Max(raw)
	A, ck := make([]float64, D*D), make([]float64, len(w))
	dv := make([]float64, D)
	for i, wi := range w {
		ki := k(i)
		for e, x := range qh {
			ck[i] += float64(x) * float64(x) * step2(float64(ki[e]))
		}
		for d := range D {
			dv[d] = wi * (v(i, d) - o[d])
		}
		for d, a := range dv {
			row := A[d*D : (d+1)*D]
			for e, x := range ki {
				row[e] += a * float64(x)
			}
		}
	}
	var mse float64
	for d := range D {
		for e, x := range qh {
			mse += step2(float64(x)) * A[d*D+e] * A[d*D+e] * scale * scale
		}
		for i, wi := range w {
			vi := v(i, d)
			mse += wi * wi * (vi - o[d]) * (vi - o[d]) * ck[i] * scale * scale
			if math.Exp(raw[i]-top) < normal {
				mse += step2(0) * vi * vi
			}
			if f32v {
				mse += wi * wi * step2(vi)
			}
		}
	}
	return mse
}

// ulp16 is binary16's step at x: 2^(e-10) on [2^e, 2^(e+1)), and 2^-24
// below the normal range.
func ulp16(x float64) float64 {
	if x == 0 {
		return 0x1p-24
	}
	_, e := math.Frexp(x)
	return math.Ldexp(1, max(e-11, -24))
}

// step2 is the variance of x's rounding to binary16.
func step2(x float64) float64 {
	u := ulp16(x)
	return u * u / 12
}

// flashBudget is a row's NMSE budget: flashTail times the expected squared
// error over the row's own, beside the device's float32 band.
func flashBudget(mse float64, want []float64, band float64) float64 {
	var den float64
	for _, x := range want {
		den += x * x
	}
	return math.Pow(math.Sqrt(flashTail*mse/den)+math.Sqrt(band), 2)
}

// flashTail is a row's squared error's ratio to its expectation that the
// gate admits: a chi-square of eight degrees, the fewest heads a case
// has, exceeds six times its mean with probability 1e-7.
const flashTail = 6

// launchAttend runs codes in order at their full grids, each binding the
// gate's words by kind -- values by id, tables by name, vectors by role --
// and its own scratch poisoned, and reads value out back.
func launchAttend(d backend.Device, codes []composer.Code, vals map[graph.ValueID][]uint32, tables map[string][]uint32,
	vecs map[jlm.Role][]uint32, out graph.ValueID) ([]uint32, error) {
	return launchKeyed(d, codes, vals, tables, vecs, fmt.Sprintf("v%d", out), len(vals[out]))
}

// launchTable is launchAttend reading back table name.
func launchTable(d backend.Device, codes []composer.Code, vals map[graph.ValueID][]uint32, tables map[string][]uint32,
	name string) ([]uint32, error) {
	return launchKeyed(d, codes, vals, tables, nil, "t"+name, len(tables[name]))
}

// launchKeyed is launchAttend reading back the n words bound as outKey.
func launchKeyed(d backend.Device, codes []composer.Code, vals map[graph.ValueID][]uint32, tables map[string][]uint32,
	vecs map[jlm.Role][]uint32, outKey string, n int) ([]uint32, error) {
	bufs := map[string]backend.Buf{}
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	get := func(key string, w []uint32) (backend.Buf, error) {
		if b, ok := bufs[key]; ok {
			return b, nil
		}
		b, err := d.Alloc(max(4*len(w), 4))
		if err != nil {
			return nil, err
		}
		p := make([]byte, 4*len(w))
		for i, x := range w {
			binary.LittleEndian.PutUint32(p[4*i:], x)
		}
		if len(p) > 0 {
			if err := b.Write(p); err != nil {
				return nil, err
			}
		}
		bufs[key] = b
		return b, nil
	}
	for ci, c := range codes {
		k, err := d.Compile(c.GPU)
		if err != nil {
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			var key string
			var w []uint32
			switch {
			case o.Kind == composer.OperandValue && vals[o.Value] != nil:
				key, w = fmt.Sprintf("v%d", o.Value), vals[o.Value]
			case o.Kind == composer.OperandTable && tables[o.Name] != nil:
				key, w = "t"+o.Name, tables[o.Name]
			case o.Kind == composer.OperandVector && vecs[o.Role] != nil:
				key, w = fmt.Sprintf("r%d", o.Role), vecs[o.Role]
			case bd.Words != nil:
				key, w = fmt.Sprintf("c%d.%d", ci, i), bd.Words
			case o.Kind == composer.OperandScratch || o.Kind == composer.OperandValue:
				key, w = fmt.Sprintf("s%d.%s.%d", o.Kind, o.Name, o.Value), poisoned(int(max(bd.Bytes, 4)/4))
			default:
				k.Close()
				return nil, fmt.Errorf("%s binds %+v, which the gate does not hold", c.Name, o)
			}
			b, err := get(key, w)
			if err != nil {
				k.Close()
				return nil, err
			}
			args[i] = b
		}
		err = k.Launch(c.Groups, c.GPU.Group[0], args...)
		k.Close()
		if err != nil {
			return nil, fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	b, ok := bufs[outKey]
	if !ok {
		return nil, fmt.Errorf("no code binds %s", outKey)
	}
	p := make([]byte, 4*n)
	if err := b.Read(p); err != nil {
		return nil, err
	}
	res := make([]uint32, n)
	for i := range res {
		res[i] = binary.LittleEndian.Uint32(p[4*i:])
	}
	return res, nil
}
