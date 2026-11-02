package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUAttendInPasses forms an attention over a paged history read
// in passes (History.Passes) and runs it the way the tier does: the codes
// before the pass once, the pass's once a pass over the rows' descriptors
// clipped to its keys with the running partial folded from one side into the
// other, then the codes after. Every row must read what the form over the
// whole history reads, with the rows at different positions so a pass holds
// keys some rows may not see; a fold that drops a pass must not. A pass
// starts at the first code reading the history and ends at the fold, and
// writes none of the call's values: with a front before the read and a
// residual after the fold, at widths where every kernel is one workgroup,
// the composer must not join either into the pass.
func TestComposedGPUAttendInPasses(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	wide := make([]int, 128)
	for i := range wide {
		wide[i] = 300 + i
	}
	cases := []passCase{
		{name: "rows", pos: []int{40, 70, 100}},
		{name: "chunk", pos: []int{92, 93, 94, 95, 96, 97, 98, 99}, seqs: 1},
		{name: "chunk/wide", pos: wide, seqs: 1},
		{name: "chunk/matrix", pos: wide, seqs: 1, matrix: true},
		{name: "rows/around", pos: []int{40, 70, 100}, around: true},
		{name: "narrow/around", pos: []int{40, 70, 100}, around: true, heads: 2, kvh: 1, dim: 16},
		{name: "narrow/around/chunk", pos: []int{92, 93, 94, 95, 96, 97, 98, 99}, seqs: 1, around: true, heads: 2, kvh: 1, dim: 16},
		{name: "rows/around/one", pos: []int{100}, one: true, around: true},
		{name: "narrow/around/one", pos: []int{100}, one: true, around: true, heads: 2, kvh: 1, dim: 16},
	}
	for _, d := range devs {
		tiles, _ := backend.LowersTile(d, backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32})
		for _, c := range cases {
			if c.matrix && !tiles {
				continue
			}
			t.Run(d.API()+"/"+c.name, func(t *testing.T) { attendInPasses(t, d, c) })
		}
	}
}

// passCase is one shape of the pass gate: the rows' positions, one sequence
// (a prompt chunk) or a sequence a row, whether the device's matrix unit
// forms the attention, whether a front adds to q and k before the attention
// and a residual to its result (around), whether the step is formed for
// its one row (a decode, whose one-workgroup kernels the composer phases
// into one launch), and the heads and head dimension where not 4, 2 and 64.
type passCase struct {
	name            string
	pos             []int
	seqs            int
	matrix          bool
	around          bool
	one             bool
	heads, kvh, dim int
}

func attendInPasses(t *testing.T, d backend.Device, c passCase) {
	pos, seqs := c.pos, c.seqs
	const page = 16
	heads, kvh, dim := 4, 2, 64
	if c.dim > 0 {
		heads, kvh, dim = c.heads, c.kvh, c.dim
	}
	R := len(pos)
	capRows := max(8, R)
	if c.one {
		capRows = R
	}
	keys := 8 * page
	passKeys := 2 * page
	if pos[R-1] >= keys {
		keys, passKeys = 32*page, 8*page
	}
	b := graph.New()
	kIn, vIn, qIn := b.In("k", kvh*dim), b.In("v", kvh*dim), b.In("q", heads*dim)
	kRow, qRow := kIn, qIn
	var kb, qb, resid graph.ValueID
	if c.around {
		kb, qb, resid = b.In("k.b", kvh*dim), b.In("q.b", heads*dim), b.In("resid", heads*dim)
		kRow, qRow = b.Add(kIn, kb), b.Add(qIn, qb)
	}
	n := b.KVAppend(kRow, vIn, 0)
	out := b.Attend(qRow, n, graph.Attrs{NHead: heads, NKVHead: kvh, HeadDim: dim, Scale: 1 / math.Sqrt(float64(dim))})
	if c.around {
		out = b.Add(out, resid)
	}
	b.Out(out)
	g := b.G
	dev := gpuDevice(d)
	if c.matrix {
		dev.Matrix = [3]int{16, 16, 16}
	}
	form := func(passes bool) []composer.Code {
		reg := composer.Default()
		if passes {
			reg = reg.ForPasses(composer.DeviceGPU)
		}
		_, codes, err := reg.Form(composer.Request{Graph: g, Step: composer.Step{Rows: capRows, Page: page, Keys: keys, Seqs: seqs},
			Device: dev, History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}, Passes: passes}})
		if err != nil {
			t.Fatal(err)
		}
		return codes
	}
	whole, inPasses := form(false), form(true)

	rng := rand.New(rand.NewSource(5))
	w, qw := kvh*dim, heads*dim
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	kpool, vpool := poisoned(keys*w), poisoned(keys*w/2)
	hist := rnd(keys * w)
	hvals := rnd(keys * w)
	for p := 0; p < keys; p++ {
		pg, off := p/page, p%page
		for e := range w {
			kpool[(pg*w+e)*page+off] = math.Float32bits(hist[p*w+e])
			hv := uint32(quant.EncodeHalf(hvals[p*w+e]))
			i := (pg*page+off)*w + e
			if e%2 == 0 {
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			} else {
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	k, v, q := make([]float32, capRows*w), make([]float32, capRows*w), rnd(capRows*qw)
	for r := range R {
		copy(k[r*w:], hist[pos[r]*w:(pos[r]+1)*w])
		copy(v[r*w:], hvals[pos[r]*w:(pos[r]+1)*w])
	}
	pages := make([]uint32, keys/page)
	for i := range pages {
		pages[i] = uint32(i)
	}
	rowsDesc := func(lo, hi int) []uint32 {
		rows := make([]uint32, capRows*4)
		for r := range R {
			p := uint32(pos[r])
			s, e := uint32(min(max(0, lo), hi)), uint32(min(max(int(p)+1, lo), hi))
			copy(rows[r*4:], []uint32{0, s, e, p})
		}
		return rows
	}
	vals := map[graph.ValueID][]uint32{kIn: f32w(k), vIn: f32w(v), qIn: f32w(q), g.Out[0]: poisoned(capRows * qw)}
	if c.around {
		vals[kb], vals[qb], vals[resid] = f32w(rnd(capRows*w)), f32w(rnd(capRows*qw)), f32w(rnd(capRows*qw))
	}
	base := map[string][]uint32{"rows.n": {uint32(R)}, "pages": pages, "kv.k": kpool, "kv.v": vpool, "q.off": make([]uint32, capRows*4)}

	runner := newPassRunner(d)
	defer runner.close()
	full := map[string][]uint32{"rows": rowsDesc(0, keys), "rows.win": rowsDesc(0, keys)}
	want, err := runner.run(whole, 0, len(whole), vals, base, full, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantF := wf32(want)

	lo, hi := passSpan(inPasses)
	if lo < 0 || hi <= lo {
		t.Fatalf("no pass between codes %d and %d", lo, hi)
	}
	t.Logf("%d codes: %d before the pass, %d a pass, %d after", len(inPasses), lo, hi-lo, len(inPasses)-hi)
	for ci := lo; ci < hi; ci++ {
		for _, bd := range inPasses[ci].Binds {
			switch {
			case !bd.Writes:
			case bd.Operand.Kind == composer.OperandValue,
				bd.Operand.Kind == composer.OperandTable && bd.Operand.Name != "run.out" && bd.Operand.Name != "kv.k" && bd.Operand.Name != "kv.v":
				t.Errorf("code %d of the pass (%s) writes %s, which a pass run again over cannot", ci, inPasses[ci].Name, bd.Operand.Name)
			}
		}
	}
	var runIn func(int, bool) []float32
	for ci, c := range inPasses {
		var bs []string
		for _, bd := range c.Binds {
			bs = append(bs, fmt.Sprintf("%s:%v/%d", bd.Operand.Name, bd.Writes, bd.Bytes))
		}
		t.Logf("code %d %s passes %v binds %v", ci, c.Name, c.Passes, bs)
	}
	runPasses := func(fold bool) []float32 { return runIn(passKeys, fold) }
	runIn = func(passKeys int, fold bool) []float32 {
		r := newPassRunner(d)
		defer r.close()
		runN := capRows * heads * (dim + 2)
		zero := make([]uint32, runN)
		run := map[string][]uint32{"run.0": zero, "run.1": poisoned(runN), "run.2": poisoned(runN)}
		if _, err := r.run(inPasses, 0, lo, vals, base, full, nil); err != nil {
			t.Fatal(err)
		}
		np := (keys + passKeys - 1) / passKeys
		for j := 0; j < np; j++ {
			in, out := "run.0", fmt.Sprintf("run.%d", 1+j%2)
			if j > 0 && fold {
				in = fmt.Sprintf("run.%d", 1+(j+1)%2)
			}
			desc := rowsDesc(j*passKeys, (j+1)*passKeys)
			ov := map[string][]uint32{"rows": desc, "rows.win": desc}
			alias := map[string]string{"run.in": in, "run.out": out}
			if _, err := r.run(inPasses, lo, hi, vals, base, ov, aliasTables(run, alias)); err != nil {
				t.Fatal(err)
			}
		}
		last := fmt.Sprintf("run.%d", 1+(np-1)%2)
		got, err := r.run(inPasses, hi, len(inPasses), vals, base, full, aliasTables(run, map[string]string{"run.in": last, "run.out": last}))
		if err != nil {
			t.Fatal(err)
		}
		return wf32(got)
	}
	one := runIn(keys, true)
	for r := range R {
		var num, den float64
		for i := r * qw; i < (r+1)*qw; i++ {
			e := float64(one[i]) - float64(wantF[i])
			num, den = num+e*e, den+float64(wantF[i])*float64(wantF[i])
		}
		t.Logf("one pass, row %d: NMSE %.3e; first values %v against %v", r, num/den, one[r*qw:r*qw+3], wantF[r*qw:r*qw+3])
	}
	got := runPasses(true)
	worst := 0.0
	for r := range R {
		var num, den float64
		for i := r * qw; i < (r+1)*qw; i++ {
			e := float64(got[i]) - float64(wantF[i])
			num, den = num+e*e, den+float64(wantF[i])*float64(wantF[i])
		}
		worst = max(worst, num/den)
		t.Logf("row %d at %d: NMSE %.3e against the whole history", r, pos[r], num/den)
	}
	if !(worst < 1e-6) {
		t.Fatalf("read in passes the rows reach NMSE %.3e from the whole history's form", worst)
	}
	bad := runPasses(false)
	var num, den float64
	for i := range R * qw {
		e := float64(bad[i]) - float64(wantF[i])
		num, den = num+e*e, den+float64(wantF[i])*float64(wantF[i])
	}
	if num/den < 1e-3 {
		t.Fatalf("every pass after the first left unfolded reads NMSE %.3e: the gate cannot see a fold", num/den)
	}
}

// passSpan is the codes a pass runs, as the tier finds them: from the first
// that reads the history (its keys or values) to the last that writes the
// running partial.
func passSpan(codes []composer.Code) (lo, hi int) {
	lo, hi = -1, -1
	for ci, c := range codes {
		for _, bd := range c.Binds {
			if bd.Operand.Kind != composer.OperandTable {
				continue
			}
			switch bd.Operand.Name {
			case "kv.k", "kv.v":
				if bd.Read && lo < 0 {
					lo = ci
				}
			case "run.out":
				if bd.Writes {
					hi = ci + 1
				}
			}
		}
	}
	return lo, hi
}

// aliasTables binds each named table to one of run's buffers.
func aliasTables(run map[string][]uint32, alias map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range alias {
		out[k] = v
	}
	for k := range run {
		out[k] = k
	}
	return out
}

// passRunner launches a form's codes over buffers held across calls, so the
// codes before a pass, the passes and the codes after share their values.
type passRunner struct {
	d     backend.Device
	bufs  map[string]backend.Buf
	calls int
}

func newPassRunner(d backend.Device) *passRunner {
	return &passRunner{d: d, bufs: map[string]backend.Buf{}}
}

func (r *passRunner) close() {
	for _, b := range r.bufs {
		b.Free()
	}
}

func (r *passRunner) buf(key string, w []uint32) (backend.Buf, error) {
	if b, ok := r.bufs[key]; ok {
		return b, nil
	}
	b, err := r.d.Alloc(max(4*len(w), 4))
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
	r.bufs[key] = b
	return b, nil
}

// run launches codes [lo, hi): tables from over first, then base, the
// running partial's through runs (a table name to a held buffer's key), and
// returns the output value's words.
func (r *passRunner) run(codes []composer.Code, lo, hi int, vals map[graph.ValueID][]uint32, base, over map[string][]uint32,
	runs map[string]string) ([]uint32, error) {
	var outVal graph.ValueID
	wrote := false
	r.calls++
	for ci := lo; ci < hi; ci++ {
		c := codes[ci]
		k, err := r.d.Compile(c.GPU)
		if err != nil {
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			var key string
			var w []uint32
			switch {
			case o.Kind == composer.OperandTable && runs[o.Name] != "":
				key = "run:" + runs[o.Name]
				w = make([]uint32, int(max(bd.Bytes, 4)/4))
				if runs[o.Name] != "run.0" {
					w = poisoned(len(w))
				}
			case o.Kind == composer.OperandTable && over[o.Name] != nil:
				key, w = fmt.Sprintf("t%s#%d", o.Name, r.calls), over[o.Name]
			case o.Kind == composer.OperandTable && base[o.Name] != nil:
				key, w = "t"+o.Name, base[o.Name]
			case o.Kind == composer.OperandValue && vals[o.Value] != nil:
				key, w = fmt.Sprintf("v%d", o.Value), vals[o.Value]
				if bd.Writes {
					outVal, wrote = o.Value, true
				}
			case bd.Words != nil:
				key, w = fmt.Sprintf("c%d.%d", ci, i), bd.Words
			case o.Kind == composer.OperandScratch || o.Kind == composer.OperandValue:
				key, w = fmt.Sprintf("s%d.%s.%d", o.Kind, o.Name, o.Value), poisoned(int(max(bd.Bytes, 4)/4))
			default:
				k.Close()
				return nil, fmt.Errorf("%s binds %+v, which the gate does not hold", c.Name, o)
			}
			b, err := r.buf(key, w)
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
	if !wrote {
		return nil, nil
	}
	b := r.bufs[fmt.Sprintf("v%d", outVal)]
	p := make([]byte, 4*len(vals[outVal]))
	if err := b.Read(p); err != nil {
		return nil, err
	}
	res := make([]uint32, len(vals[outVal]))
	for i := range res {
		res[i] = binary.LittleEndian.Uint32(p[4*i:])
	}
	return res, nil
}
