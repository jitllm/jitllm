package composer_test

import (
	"fmt"
	"math"
	"math/rand"
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
// without the rows must each fail it.
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
				gpuAttendBoth(t, d, sh.heads, sh.dim, sh.win, sh.period, sh.sinks)
			})
		}
	}
}

func gpuAttendBoth(t *testing.T, d backend.Device, H, D, win, P int, sinks bool) {
	const page, p0, R, capRows = 8, 2, 9, 16
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
	nmse := func(w []float64) float64 {
		var num, den float64
		for i := range w {
			e := float64(out[i]) - w[i]
			num += e * e
			den += w[i] * w[i]
		}
		return num / den
	}
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
