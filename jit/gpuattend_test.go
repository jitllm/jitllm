package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// attendRowsCase is an attention a gate forms at a prompt chunk's rows: its
// heads, its window (0 none) and whether it carries sinks.
type attendRowsCase struct {
	heads, kvh, dim, window int
	sinks                   bool
}

// TestComposedGPUAttendRows forms an appended history and the attention
// over it at a prompt chunk's rows on every GPU, the served models' pool
// layout (keys float32 by dimension, values binary16 by position), and runs
// fewer live rows than the form holds, past a workgroup's chunk of
// positions so a causal row's span takes two partitions, starting inside a
// page and crossing into the next: each row attends causally from the history's
// start, or from its window's where the attention slides, with a sink a head
// where it carries them, held row by row to the definition over the history
// as the pools hold it. The form must run its partitions in order (the
// running pass), and its scratch is counted at a key bound far past the live
// positions: no binding a kernel reads or writes may hold more than four
// words a row, head and dimension, where a [rows, heads, keys] buffer at the
// bound is sixty-four of them. Two violations must fail: every row's first
// attended position moved by one, and the sinks zeroed.
func TestComposedGPUAttendRows(t *testing.T) {
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
	cases := []attendRowsCase{
		{8, 2, 64, 0, false},
		{8, 2, 64, 0, true},
		{16, 8, 128, 0, false},
		{64, 8, 64, 16, true},
		{64, 8, 64, 0, true},
		{12, 4, 64, 16, false},
	}
	for _, d := range devs {
		for _, c := range cases {
			name := fmt.Sprintf("%s/%dq%dkv%dd/window%d/sinks%v", d.API(), c.heads, c.kvh, c.dim, c.window, c.sinks)
			t.Run(name, func(t *testing.T) { gpuAttendRows(t, d, c) })
		}
	}
}

func gpuAttendRows(t *testing.T, d backend.Device, c attendRowsCase) {
	const page, p0, R, capRows, keys = 64, 2100, 37, 64, 64 * 64
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
	_, codes, err := jit.Default().Form(jit.Request{Graph: g, Step: jit.Step{Rows: capRows, Page: page, Keys: keys, Seqs: 1},
		Device: dev, History: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: true}}})
	if err != nil {
		t.Fatal(err)
	}
	var scratch int64
	running := 0
	for _, cd := range codes {
		if slices.Contains(cd.Passes, "running") {
			running++
		}
		for _, bd := range cd.Binds {
			if bd.Operand.Kind != jit.OperandScratch || !bd.Read && !bd.Writes {
				continue
			}
			scratch = max(scratch, bd.Bytes)
			if bd.Bytes > int64(4*4*capRows*c.heads*c.dim) {
				t.Errorf("%s binds %s of %d bytes: more than four words a row, head and dimension at a key bound of %d",
					cd.Name, bd.Operand.Name, bd.Bytes, keys)
			}
		}
	}
	if running != 1 {
		t.Errorf("%d kernels run the partitions in order: the gate did not select the running form", running)
	}
	t.Logf("%d kernels, the largest scratch the attention reads or writes %d bytes (a [rows, heads, keys] buffer is %d)",
		len(codes), scratch, 4*capRows*c.heads*keys)

	rng := rand.New(rand.NewSource(11))
	w, qw := c.kvh*c.dim, c.heads*c.dim
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	ks, vs := make([][]float32, p0+R), make([][]float32, p0+R)
	for p := range p0 + R {
		ks[p], vs[p] = rnd(w), rnd(w)
	}
	q := rnd(capRows * qw)
	k, v := make([]float32, capRows*w), make([]float32, capRows*w)
	for r := range R {
		copy(k[r*w:], ks[p0+r])
		copy(v[r*w:], vs[p0+r])
	}
	sinkv := make([]float32, c.heads)
	for h := range sinkv {
		sinkv[h] = float32(rng.NormFloat64()) * 2
	}
	kpool := poisoned(keys / page * page * w)
	vpool := poisoned(keys / page * page * w / 2)
	for p := range p0 {
		pg, off := p/page, p%page
		for e := range w {
			kpool[(pg*w+e)*page+off] = math.Float32bits(ks[p][e])
			hv, i := uint32(quant.EncodeHalf(vs[p][e])), (pg*page+off)*w+e
			if e%2 == 0 {
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			} else {
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	first := func(r int) int {
		if c.window > 0 {
			return max(0, p0+r+1-c.window)
		}
		return 0
	}
	ref := func(sinkOn bool) []float64 {
		want := make([]float64, R*qw)
		for r := range R {
			end := p0 + r + 1
			for h := range c.heads {
				kv := h / (c.heads / c.kvh)
				var sc []float64
				for p := first(r); p < end; p++ {
					var s float64
					for e := range c.dim {
						s += float64(q[r*qw+h*c.dim+e]) * float64(ks[p][kv*c.dim+e])
					}
					sc = append(sc, s*float64(scale))
				}
				if sinkOn {
					sc = append(sc, float64(sinkv[h]))
				}
				oracle.Softmax(sc)
				for e := range c.dim {
					var o float64
					for i, p := 0, first(r); p < end; i, p = i+1, p+1 {
						o += sc[i] * round(vs[p][kv*c.dim+e])
					}
					want[r*qw+h*c.dim+e] = o
				}
			}
		}
		return want
	}
	pages := make([]uint32, keys/page)
	for i := range pages {
		pages[i] = uint32(i)
	}
	run := func(shift uint32, sv []float32) []float32 {
		rows, win := make([]uint32, capRows*4), make([]uint32, capRows*4)
		for r := range R {
			p := uint32(p0 + r)
			copy(rows[r*4:], []uint32{0, shift, p + 1, p})
			copy(win[r*4:], []uint32{0, uint32(first(r)) + shift, p + 1, p})
		}
		vals := map[graph.ValueID][]uint32{g.In[0]: f32w(k), g.In[1]: f32w(v), g.In[2]: f32w(q), g.Out[0]: poisoned(capRows * qw)}
		tables := map[string][]uint32{"rows": rows, "rows.win": win, "rows.n": {R}, "pages": pages,
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
	want := ref(c.sinks)
	got := run(0, sinkv)
	// A device whose lowering rounds every operation alone is held to
	// float32's band; a driver free to contract and approximate (SPIR-V's)
	// to a wider one: the V100's Vulkan driver reads 1.4e-7 on the form
	// before the running pass as after it, the laptop's 3e-14.
	band := 1e-8
	if !dev.ExactArith {
		band = 1e-6
	}
	worst := 0.0
	for r := range R {
		e := nmse(got, want, r)
		if math.IsNaN(e) || e > band {
			t.Errorf("row %d at position %d: NMSE %.3e against the definition", r, p0+r, e)
		}
		worst = max(worst, e)
	}
	for i := R * qw; i < capRows*qw; i++ {
		if math.Float32bits(got[i]) != guardWord {
			t.Fatalf("output word %d, past the call's %d rows, was written", i, R)
		}
	}
	t.Logf("%d rows of a %d-row form: worst NMSE %.3e against the definition", R, capRows, worst)
	bad := run(1, sinkv)
	if e := nmse(bad, want, R-1); !(e > 1e-6) {
		t.Errorf("with every row's first attended position moved the NMSE is %.3e: the gate cannot see a wrong row", e)
	}
	if c.sinks {
		bad := run(0, make([]float32, c.heads))
		if e := nmse(bad, want, 0); !(e > 1e-6) {
			t.Errorf("with the sinks zeroed the NMSE is %.3e: the gate cannot see the sinks", e)
		}
	}
}

// launchAttend runs codes in order at their full grids, each binding the
// gate's words by kind -- values by id, tables by name, vectors by role --
// and its own scratch poisoned, and reads value out back.
func launchAttend(d backend.Device, codes []jit.Code, vals map[graph.ValueID][]uint32, tables map[string][]uint32,
	vecs map[jlm.Role][]uint32, out graph.ValueID) ([]uint32, error) {
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
			case o.Kind == jit.OperandValue && vals[o.Value] != nil:
				key, w = fmt.Sprintf("v%d", o.Value), vals[o.Value]
			case o.Kind == jit.OperandTable && tables[o.Name] != nil:
				key, w = "t"+o.Name, tables[o.Name]
			case o.Kind == jit.OperandVector && vecs[o.Role] != nil:
				key, w = fmt.Sprintf("r%d", o.Role), vecs[o.Role]
			case bd.Words != nil:
				key, w = fmt.Sprintf("c%d.%d", ci, i), bd.Words
			case o.Kind == jit.OperandScratch || o.Kind == jit.OperandValue:
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
	w := vals[out]
	p := make([]byte, 4*len(w))
	if err := bufs[fmt.Sprintf("v%d", out)].Read(p); err != nil {
		return nil, err
	}
	res := make([]uint32, len(w))
	for i := range res {
		res[i] = binary.LittleEndian.Uint32(p[4*i:])
	}
	return res, nil
}
