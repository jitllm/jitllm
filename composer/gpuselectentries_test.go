package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUSelectEntriesInPasses is TestComposedGPUSelectInPasses for
// an attention over two histories whose entries a selection keeps (DeepSeek
// V4's compressed and selected blocks): rows attending their window of the
// first history and the entries of the second, one each period positions,
// that a selection over the entries' indexer keys -- the entry row past its
// value -- keeps, block by block. Read in passes as the tier reads it, every
// row must read what the form over the whole history reads, and each sweep's
// fold dropped must not, in some case of the device's: a case whose rows'
// best blocks lie in one pass reads the same without the selection's fold.
func TestComposedGPUSelectEntriesInPasses(t *testing.T) {
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
		var seen [2]float64
		for _, c := range []passCase{
			{name: "rows", pos: []int{40, 70, 103, 127}},
			{name: "chunk", pos: []int{96, 97, 98, 99, 100, 101, 102, 103}, seqs: 1},
		} {
			t.Run(d.API()+"/"+c.name, func(t *testing.T) {
				s := selectEntriesInPasses(t, d, c)
				seen[0], seen[1] = max(seen[0], s[0]), max(seen[1], s[1])
			})
		}
		for i, name := range []string{"the selection's", "the attention's"} {
			if seen[i] < 1e-3 {
				t.Fatalf("%s: %s passes left unfolded read NMSE %.3e at most: the gate cannot see that fold", d.API(), name, seen[i])
			}
		}
	}
}

// selectEntriesInPasses runs case c on d and returns the NMSE each sweep's
// fold dropped reads, the selection's and the attention's.
func selectEntriesInPasses(t *testing.T, d backend.Device, c passCase) [2]float64 {
	pos, seqs := c.pos, c.seqs
	const page, heads, dim, win, period, ih, idim, span, topk = 16, 4, 8, 8, 4, 2, 8, 2, 2
	R := len(pos)
	capRows := max(8, R)
	keys, passKeys := 8*page, 2*page
	ew := dim + idim
	b := graph.New()
	kIn, qIn, iqIn := b.In("k", dim), b.In("q", heads*dim), b.In("iq", ih*idim)
	n := b.KVAppend(kIn, kIn, 0)
	rowsRef, entsRef := graph.StateRef{State: graph.StateKV, Version: n}, graph.StateRef{State: graph.StateKV, Hist: 1}
	sel := b.Select(iqIn, entsRef, graph.Attrs{NHead: ih, NKVHead: 1, HeadDim: idim, Offset: dim, Width: ew, Span: span, TopK: topk, Local: 1})
	b.Out(b.AttendBothSelected(qIn, rowsRef, entsRef, period, ew, sel, graph.Attrs{NHead: heads, NKVHead: 1, HeadDim: dim, Scale: 0.3, Window: win}))
	g := b.G
	form := func(passes bool) []composer.Code {
		reg, step := composer.Default(), composer.Step{Rows: capRows, Page: page, Keys: keys, Seqs: seqs}
		hist := composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}
		ents := composer.History{ValuesInKeys: true, Period: period}
		if passes {
			reg, step.Keys, step.Context = reg.ForPasses(composer.DeviceGPU), passKeys, keys
			hist.Passes, ents.Passes = true, true
		}
		_, codes, err := reg.Form(composer.Request{Graph: g, Step: step, Device: gpuDevice(d), History: hist, More: []composer.History{ents}})
		if err != nil {
			t.Fatal(err)
		}
		return codes
	}
	whole, inPasses := form(false), form(true)

	rng := rand.New(rand.NewSource(11))
	// Every value binary16-exact, so the key and value pools hold a row alike.
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(quant.DecodeHalf(quant.EncodeHalf(float32(rng.NormFloat64()))))
		}
		return x
	}
	hist0, entv := rnd(keys*dim), rnd(keys/period*ew)
	slots := composer.History{ValuesInKeys: true, Period: period}.Slots(page)
	kpool, vpool, entPool := poisoned(keys*dim), poisoned(keys*dim/2), poisoned(keys/page*slots*ew)
	for p := 0; p < keys; p++ {
		for e := range dim {
			x := hist0[p*dim+e]
			kpool[(p/page*dim+e)*page+p%page] = math.Float32bits(x)
			hv, i := uint32(quant.EncodeHalf(x)), p*dim+e
			if i%2 == 0 {
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			} else {
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	for j := range keys / period {
		tp := (j+1)*period - 1
		for e := range ew {
			entPool[(tp/page*slots+tp%page/period)*ew+e] = math.Float32bits(entv[j*ew+e])
		}
	}
	k := make([]float32, capRows*dim)
	for r := range R {
		copy(k[r*dim:], hist0[pos[r]*dim:(pos[r]+1)*dim])
	}
	pages := make([]uint32, keys/page)
	for i := range pages {
		pages[i] = uint32(i)
	}
	descs := func(lo, hi int) (rows, wins []uint32) {
		rows, wins = make([]uint32, capRows*4), make([]uint32, capRows*4)
		for r := range R {
			p := pos[r]
			clip := func(x int) uint32 { return uint32(min(max(x, lo), hi)) }
			copy(rows[r*4:], []uint32{0, clip(0), clip(p + 1), uint32(p)})
			copy(wins[r*4:], []uint32{0, clip(max(0, p+1-win)), clip(p + 1), uint32(p)})
		}
		return rows, wins
	}
	vals := map[graph.ValueID][]uint32{kIn: f32w(k), qIn: f32w(rnd(capRows * heads * dim)), iqIn: f32w(rnd(capRows * ih * idim)),
		g.Out[0]: poisoned(capRows * heads * dim)}
	pools := map[int][]uint32{0: kpool, 1: vpool, 2: entPool}
	fullRows, fullWins := descs(0, keys)
	baseOf := func(codes []composer.Code) map[string][]uint32 {
		base := historyTables(codes, fullRows, fullWins, pools, pages)
		base["rows.n"], base["q.off"] = []uint32{uint32(R)}, make([]uint32, capRows*4)
		return base
	}
	runner := newPassRunner(d)
	defer runner.close()
	want, err := runner.run(whole, 0, len(whole), vals, baseOf(whole), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantF := wf32(want)
	qw := heads * dim

	sweeps := passSweeps(inPasses)
	if len(sweeps) != 2 {
		t.Fatalf("%d sweeps of passes, want the selection's and the attention's: %v", len(sweeps), sweeps)
	}
	t.Logf("%d codes, sweeps %v", len(inPasses), sweeps)
	base := baseOf(inPasses)
	run := func(dropSel, dropAttend bool) []float32 {
		r := newPassRunner(d)
		defer r.close()
		np := keys / passKeys
		tabs := map[string][]uint32{"run.0": nil, "run.1": nil, "run.2": nil, "sel.0": nil, "sel.1": nil, "sel.2": nil}
		alias := map[string]string{"run.in": "run.0", "run.out": "run.0", "sel.in": "sel.0", "sel.out": "sel.0"}
		at := 0
		for i, sw := range sweeps {
			if _, err := r.run(inPasses, at, sw[0], vals, base, nil, aliasTables(tabs, alias)); err != nil {
				t.Fatal(err)
			}
			pre, drop := "sel", dropSel
			if i == 1 {
				pre, drop = "run", dropAttend
			}
			for j := 0; j < np; j++ {
				in, out := pre+".0", fmt.Sprintf("%s.%d", pre, 1+j%2)
				if j > 0 && !drop {
					in = fmt.Sprintf("%s.%d", pre, 1+(j+1)%2)
				}
				alias[pre+".in"], alias[pre+".out"] = in, out
				rows, wins := descs(j*passKeys, (j+1)*passKeys)
				if _, err := r.run(inPasses, sw[0], sw[1], vals, base, historyTables(inPasses, rows, wins, nil, nil), aliasTables(tabs, alias)); err != nil {
					t.Fatal(err)
				}
			}
			last := fmt.Sprintf("%s.%d", pre, 1+(np-1)%2)
			alias[pre+".in"], alias[pre+".out"] = last, last
			at = sw[1]
		}
		got, err := r.run(inPasses, at, len(inPasses), vals, base, nil, aliasTables(tabs, alias))
		if err != nil {
			t.Fatal(err)
		}
		return wf32(got)
	}
	nmse := func(got []float32, r int) float64 {
		var num, den float64
		for i := r * qw; i < (r+1)*qw; i++ {
			e := float64(got[i]) - float64(wantF[i])
			num, den = num+e*e, den+float64(wantF[i])*float64(wantF[i])
		}
		return num / den
	}
	got := run(false, false)
	worst := 0.0
	for r := range R {
		worst = max(worst, nmse(got, r))
		t.Logf("row %d at %d: NMSE %.3e against the whole history", r, pos[r], nmse(got, r))
	}
	if !(worst < 1e-6) {
		t.Fatalf("read in passes the rows reach NMSE %.3e from the whole history's form", worst)
	}
	var seen [2]float64
	for i, v := range []struct {
		name          string
		sel, attended bool
	}{{"the selection's", true, false}, {"the attention's", false, true}} {
		bad := run(v.sel, v.attended)
		for r := range R {
			seen[i] = max(seen[i], nmse(bad, r))
		}
		t.Logf("%s passes left unfolded: NMSE %.3e", v.name, seen[i])
	}
	return seen
}
