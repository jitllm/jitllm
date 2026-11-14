package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// historyTables binds every table codes read by its kind and history: the
// rows' and windows' descriptors, the pages, and each history's pool.
func historyTables(codes []composer.Code, rows, wins []uint32, pools map[int][]uint32, pages []uint32) map[string][]uint32 {
	tables := map[string][]uint32{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			o := bd.Operand
			if o.Kind != composer.OperandTable {
				continue
			}
			switch {
			case o.Table == composer.TableRows:
				tables[o.Name] = rows
			case o.Table == composer.TableWindows:
				tables[o.Name] = wins
			case o.Table == composer.TablePages:
				tables[o.Name] = pages
			case o.Table == composer.TableKeys && pools[2*o.Instance] != nil:
				tables[o.Name] = pools[2*o.Instance]
			case o.Table == composer.TableValues && pools[2*o.Instance+1] != nil:
				tables[o.Name] = pools[2*o.Instance+1]
			}
		}
	}
	return tables
}

// TestComposedGPUAttendBoth is TestComposedCPUAttendBoth on every GPU: an
// attention over two histories (DeepSeek V4's rows and its compressed
// entries), a chunk of rows appending to the card's paged history -- keys
// float32 by dimension, values binary16 by position -- each row attending
// its window of them and the entries of a second history, one each Span
// positions and float32 a position, those its position+1 over Span closes,
// in one softmax with a sink a head counted once. Held to the
// concatenation row by row, and the answer without the entries and the one
// without the rows must each fail it. Each shape runs again at rows further
// on over a history read in passes of a page (History.Passes), as the
// tier runs it: one sweep reading both histories, its running partial
// folded pass by pass, held to the same concatenation; the passes left
// unfolded must fail it, and a pass of rows that attend none of it reads
// only poison there.
func TestComposedGPUAttendBoth(t *testing.T) {
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
	for _, d := range devs {
		for _, sh := range []struct {
			heads, dim, win, period int
			sinks                   bool
		}{{4, 8, 0, 4, true}, {4, 8, 0, 4, false}, {2, 6, 5, 3, true}, {3, 4, 6, 8, true}} {
			t.Run(fmt.Sprintf("%s/%dq%dd/win%d/every%d/sinks%v", d.API(), sh.heads, sh.dim, sh.win, sh.period, sh.sinks), func(t *testing.T) {
				gpuAttendBoth(t, d, sh.heads, sh.dim, sh.win, sh.period, sh.sinks, 2, 0)
			})
			t.Run(fmt.Sprintf("%s/%dq%dd/win%d/every%d/sinks%v/passes", d.API(), sh.heads, sh.dim, sh.win, sh.period, sh.sinks), func(t *testing.T) {
				gpuAttendBoth(t, d, sh.heads, sh.dim, sh.win, sh.period, sh.sinks, 14, 8)
			})
		}
	}
}

// gpuAttendBoth runs the shape with its rows from p0 on, and where passKeys
// is set over passes of that many positions too.
func gpuAttendBoth(t *testing.T, d backend.Device, H, D, win, P int, sinks bool, p0, passKeys int) {
	const page, R, capRows = 8, 9, 16
	b := graph.New()
	k := b.In("k", D)
	n := b.KVAppend(k, k, 0)
	a := graph.Attrs{NHead: H, NKVHead: 1, HeadDim: D, Scale: 0.3, Window: win}
	var roles []jlm.Role
	if sinks {
		roles = append(roles, jlm.RoleAttnSinks)
	}
	b.Out(b.AttendBoth(b.In("q", H*D), graph.StateRef{State: graph.StateKV, Version: n},
		graph.StateRef{State: graph.StateKV, Hist: 1}, P, D, a, roles...))
	g := b.G
	ents := composer.History{ValuesInKeys: true, Period: P}
	_, codes, err := composer.Default().Form(composer.Request{Graph: g,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}, More: []composer.History{ents},
		Step: composer.Step{Rows: capRows, Page: page, Keys: 4 * page, Seqs: 1}, Device: gpuDevice(d)})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(13))
	// Every row binary16-exact, so the key and value pools hold it alike.
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(quant.DecodeHalf(quant.EncodeHalf(float32(rng.NormFloat64()))))
		}
		return x
	}
	last := p0 + R
	rows := make([][]float32, last)
	for p := range rows {
		rows[p] = rnd(D)
	}
	entries := make([][]float32, last/P)
	for j := range entries {
		entries[j] = rnd(D)
	}
	q := rnd(capRows * H * D)
	sink := rnd(H)
	kin := make([]float32, capRows*D)
	for r := range R {
		copy(kin[r*D:], rows[p0+r])
	}
	slots := ents.Slots(page)
	kpool, vpool, entPool := poisoned(4*page*D), poisoned(4*page*D/2), poisoned(4*slots*D)
	for p := range p0 {
		for e, x := range rows[p] {
			kpool[(p/page*D+e)*page+p%page] = math.Float32bits(x)
			hv, i := uint32(quant.EncodeHalf(x)), p*D+e
			if i%2 == 0 {
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			} else {
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	for j := range entries {
		tp := (j+1)*P - 1
		for e, x := range entries[j] {
			entPool[(tp/page*slots+tp%page/P)*D+e] = math.Float32bits(x)
		}
	}
	want := func(useRows, useEnts bool) []float64 {
		out := make([]float64, R*H*D)
		for r := range R {
			t := p0 + r
			var keys [][]float32
			if useRows {
				first := 0
				if win > 0 {
					first = max(0, t+1-win)
				}
				keys = append(keys, rows[first:t+1]...)
			}
			if useEnts {
				keys = append(keys, entries[:(t+1)/P]...)
			}
			for h := range H {
				sc := make([]float64, len(keys))
				mx := math.Inf(-1)
				for i, key := range keys {
					var s float64
					for e := range D {
						s += float64(q[(r*H+h)*D+e]) * float64(key[e])
					}
					sc[i] = s * 0.3
					mx = math.Max(mx, sc[i])
				}
				if sinks {
					mx = math.Max(mx, float64(sink[h]))
				}
				var sum float64
				for i := range sc {
					sc[i] = math.Exp(sc[i] - mx)
					sum += sc[i]
				}
				if sinks {
					sum += math.Exp(float64(sink[h]) - mx)
				}
				for e := range D {
					var o float64
					for i, key := range keys {
						o += sc[i] / sum * float64(key[e])
					}
					out[(r*H+h)*D+e] = o
				}
			}
		}
		return out
	}
	desc, wdesc := make([]uint32, capRows*4), make([]uint32, capRows*4)
	for r := range R {
		p := uint32(p0 + r)
		copy(desc[r*4:], []uint32{0, 0, p + 1, p})
		first := uint32(0)
		if win > 0 {
			first = uint32(max(0, int(p)+1-win))
		}
		copy(wdesc[r*4:], []uint32{0, first, p + 1, p})
	}
	tables := historyTables(codes, desc, wdesc, map[int][]uint32{0: kpool, 1: vpool, 2: entPool}, []uint32{0, 1, 2, 3})
	tables["rows.n"], tables["q.off"] = []uint32{R}, make([]uint32, capRows*4)
	vs := map[graph.ValueID][]uint32{g.In[0]: f32w(kin), g.In[1]: f32w(q), g.Out[0]: poisoned(capRows * H * D)}
	vecs := map[jlm.Role][]uint32{}
	if sinks {
		vecs[jlm.RoleAttnSinks] = f32w(sink)
	}
	res, err := launchAttend(d, codes, vs, tables, vecs, g.Out[0])
	if err != nil {
		t.Fatal(err)
	}
	out := wf32(res)
	nmseOf := func(out []float32, w []float64) float64 {
		var num, den float64
		for i := range w {
			e := float64(out[i]) - w[i]
			num += e * e
			den += w[i] * w[i]
		}
		return num / den
	}
	nmse := func(w []float64) float64 { return nmseOf(out, w) }
	e := nmse(want(true, true))
	t.Logf("%d rows over a window of %d and an entry each %d: NMSE %.3e against the concatenation", R, win, P, e)
	if math.IsNaN(e) || e > 1e-9 {
		t.Fatalf("the composed attention over two histories on the card: NMSE %.3e against the concatenation", e)
	}
	if e := nmse(want(true, false)); !(e > 1e-6) {
		t.Errorf("without the entries the NMSE is %.3e: the gate cannot see them", e)
	}
	if e := nmse(want(false, true)); !(e > 1e-6) {
		t.Errorf("without the rows the NMSE is %.3e: the gate cannot see them", e)
	}
	if passKeys == 0 {
		return
	}
	pents := ents
	pents.Passes = true
	_, inPasses, err := composer.Default().ForPasses(composer.DeviceGPU).Form(composer.Request{Graph: g,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}, Passes: true}, More: []composer.History{pents},
		Step: composer.Step{Rows: capRows, Page: page, Keys: passKeys, Seqs: 1, Context: 4 * page}, Device: gpuDevice(d)})
	if err != nil {
		t.Fatal(err)
	}
	sweeps := passSweeps(inPasses)
	if len(sweeps) != 1 {
		t.Fatalf("the form over passes runs sweeps %v: one reads both histories", sweeps)
	}
	lo, hi := sweeps[0][0], sweeps[0][1]
	reads := map[int]bool{}
	for ci := lo; ci < hi; ci++ {
		for _, bd := range inPasses[ci].Binds {
			if o := bd.Operand; o.Kind == composer.OperandTable && o.Table == composer.TableKeys && bd.Read {
				reads[o.Instance] = true
			}
		}
	}
	if !reads[0] || !reads[1] {
		t.Fatalf("the sweep reads the keys of histories %v: a pass reads both", reads)
	}
	base := historyTables(inPasses, desc, wdesc, map[int][]uint32{0: kpool, 1: vpool, 2: entPool}, []uint32{0, 1, 2, 3})
	base["rows.n"], base["q.off"] = []uint32{R}, make([]uint32, capRows*4)
	run := map[string][]uint32{"run.0": nil, "run.1": nil, "run.2": nil}
	np := 4 * page / passKeys
	runPasses := func(fold bool) []float32 {
		r := newPassRunner(d)
		r.vecs = vecs
		defer r.close()
		if _, err := r.run(inPasses, 0, lo, vs, base, nil, nil); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < np; j++ {
			in, out := "run.0", fmt.Sprintf("run.%d", 1+j%2)
			if j > 0 && fold {
				in = fmt.Sprintf("run.%d", 1+(j+1)%2)
			}
			first, end := uint32(j*passKeys), uint32((j+1)*passKeys)
			clip := func(src []uint32) []uint32 {
				d := slices.Clone(src)
				for o := 0; o < len(d); o += 4 {
					d[o+1], d[o+2] = min(max(d[o+1], first), end), min(max(d[o+2], first), end)
				}
				return d
			}
			ov := historyTables(inPasses, clip(desc), clip(wdesc), nil, nil)
			if _, err := r.run(inPasses, lo, hi, vs, base, ov, aliasTables(run, map[string]string{"run.in": in, "run.out": out})); err != nil {
				t.Fatal(err)
			}
		}
		last := fmt.Sprintf("run.%d", 1+(np-1)%2)
		got, err := r.run(inPasses, hi, len(inPasses), vs, base, nil, aliasTables(run, map[string]string{"run.in": last, "run.out": last}))
		if err != nil {
			t.Fatal(err)
		}
		if got == nil {
			t.Fatal("no code after the sweep writes the attention's output")
		}
		return wf32(got)
	}
	pe := nmseOf(runPasses(true), want(true, true))
	t.Logf("over %d passes of %d positions (codes %d to %d a pass of %d): NMSE %.3e against the concatenation", np, passKeys, lo, hi, len(inPasses), pe)
	if math.IsNaN(pe) || pe > 1e-9 {
		t.Fatalf("the composed attention over two histories read in passes: NMSE %.3e against the concatenation", pe)
	}
	if e := nmseOf(runPasses(false), want(true, true)); !(e > 1e-6) {
		t.Errorf("every pass after the first left unfolded reads NMSE %.3e: the gate cannot see a fold", e)
	}
}

// TestComposedGPUPeriodicAppend is TestComposedCPUPeriodicAppend on every
// GPU: an append of one entry each Span positions (OpKVAppend's Span) to
// the card's entry pool, float32 a position: a row closing a window writes
// its entry at its page's slot (position%page)/Span, every other row
// zeros to the page's last slot, and no other word of the pool changes.
func TestComposedGPUPeriodicAppend(t *testing.T) {
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
	for _, d := range devs {
		for _, P := range []int{1, 2, 3, 4, 16} {
			t.Run(fmt.Sprintf("%s/%d", d.API(), P), func(t *testing.T) { gpuPeriodicAppend(t, d, P) })
		}
	}
}

func gpuPeriodicAppend(t *testing.T, d backend.Device, P int) {
	const page, p0, R, capRows, w = 8, 5, 7, 8, 6
	b := graph.New()
	e := b.In("e", w)
	b.KVAppendEvery(1, e, e, 0, P)
	// The append writes nothing a gate reads back but the pool: an output
	// the form must keep is the entry itself.
	b.Out(e)
	g := b.G
	ent := composer.History{ValuesInKeys: true, Period: P}
	_, codes, err := composer.Default().Form(composer.Request{Graph: g,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}, More: []composer.History{ent},
		Step: composer.Step{Rows: capRows, Page: page, Keys: 4 * page, Seqs: 1}, Device: gpuDevice(d)})
	if err != nil {
		t.Fatal(err)
	}
	slots := ent.Slots(page)
	rng := rand.New(rand.NewSource(5))
	in := make([]float32, capRows*w)
	for i := range in {
		in[i] = float32(rng.NormFloat64())
	}
	const sentinel = uint32(0x7fc0beef)
	pool := make([]uint32, 4*slots*w)
	for i := range pool {
		pool[i] = sentinel
	}
	desc := make([]uint32, capRows*4)
	for r := range R {
		p := uint32(p0 + r)
		copy(desc[r*4:], []uint32{0, 0, p + 1, p})
	}
	tables := historyTables(codes, desc, desc, map[int][]uint32{2: pool}, []uint32{0, 1, 2, 3})
	tables["rows.n"], tables["q.off"] = []uint32{R}, make([]uint32, capRows*4)
	name := ""
	for _, c := range codes {
		for _, bd := range c.Binds {
			if o := bd.Operand; o.Kind == composer.OperandTable && o.Table == composer.TableKeys && o.Instance == 1 {
				name = o.Name
			}
		}
	}
	if name == "" {
		t.Fatal("no code binds the entry pool")
	}
	got, err := launchTable(d, codes, map[graph.ValueID][]uint32{g.In[0]: f32w(in)}, tables, name)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]uint32, len(pool))
	for i := range want {
		want[i] = sentinel
	}
	for r := range R {
		p := p0 + r
		closes := (p+1)%P == 0
		slot := slots - 1
		if closes {
			slot = p % page / P
		}
		for x := range w {
			v := math.Float32bits(in[r*w+x])
			if !closes {
				v = 0
			}
			want[((p/page)*slots+slot)*w+x] = v
		}
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("word %d (page %d, slot %d): %#x, want %#x", i, i/(slots*w), i/w%slots, got[i], want[i])
		}
	}
}
