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

// TestComposedGPUHistoryRows is TestComposedCPUHistoryRows on every GPU,
// over the paged history a card holds (keys float32 by dimension): every
// element of every row's window held to the definition exactly. A write
// position moved by one must fail it.
func TestComposedGPUHistoryRows(t *testing.T) {
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
		for _, c := range composer.HistoryRowsCases() {
			t.Run(d.API()+"/"+c.Name(), func(t *testing.T) { gpuHistoryRows(t, d, c) })
		}
	}
}

func gpuHistoryRows(t *testing.T, d backend.Device, c composer.HistoryRowsCase) {
	const page, p0, R, capRows, pages = 8, 13, 7, 8, 4
	g := c.Graph()
	_, codes, err := composer.Default().Form(composer.Request{Graph: g,
		History: composer.History{Keys: composer.Pool{ByDim: true}},
		Step:    composer.Step{Rows: capRows, Page: page, Keys: pages * page, Seqs: 1}, Device: gpuDevice(d)})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	rows := make([][]float32, p0+R)
	for p := range rows {
		rows[p] = make([]float32, c.RW)
		for e := range rows[p] {
			rows[p][e] = float32(quant.DecodeHalf(quant.EncodeHalf(float32(rng.NormFloat64() * 2))))
		}
	}
	k := make([]float32, capRows*c.RW)
	for r := range R {
		copy(k[r*c.RW:], rows[p0+r])
	}
	kpool := poisoned(pages * page * c.RW)
	vpool := poisoned(pages * page * c.RW)
	for p := range p0 {
		for e, x := range rows[p] {
			kpool[(p/page*c.RW+e)*page+p%page] = math.Float32bits(x)
		}
	}
	W := c.Span * c.Width
	run := func(shift uint32) []float32 {
		desc := make([]uint32, capRows*4)
		for r := range R {
			p := uint32(p0+r) + shift
			copy(desc[r*4:], []uint32{0, 0, p + 1, p})
		}
		vs := map[graph.ValueID][]uint32{g.In[0]: f32w(k), g.In[1]: f32w(k), g.Out[0]: poisoned(capRows * W)}
		tables := map[string][]uint32{"rows": desc, "rows.win": desc, "rows.n": {R}, "pages": {0, 1, 2, 3}, "q.off": make([]uint32, capRows*4),
			"kv.k": append([]uint32(nil), kpool...), "kv.v": append([]uint32(nil), vpool...)}
		out, err := launchAttend(d, codes, vs, tables, nil, g.Out[0])
		if err != nil {
			t.Fatal(err)
		}
		return wf32(out)[:R*W]
	}
	want, held, consts := c.Want(rows, p0, R)
	if held == 0 || consts == 0 {
		t.Fatalf("the case reads %d held elements and %d consts: it cannot tell the two apart", held, consts)
	}
	got := run(0)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("row %d position %d element %d: %v, want %v (%d held, %d consts)", i/W, i%W/c.Width, i%c.Width, got[i], want[i], held, consts)
		}
	}
	moved, same := run(1), true
	for i := range want {
		same = same && moved[i] == want[i]
	}
	if same {
		t.Errorf("with the write position moved every element agrees: the gate cannot see the window")
	}
}

// TestComposedGPUColumnReduce is TestComposedCPUColumnReduce on every GPU.
func TestComposedGPUColumnReduce(t *testing.T) {
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
		for _, fn := range []graph.MapFn{graph.MapAdd, graph.MapMax, graph.MapMin} {
			for _, sh := range composer.ColumnReduceShapes() {
				for _, rows := range []int{1, 3} {
					t.Run(fmt.Sprintf("%s/%v/%dx%dx%d/%d", d.API(), fn, sh[0], sh[1], sh[2], rows), func(t *testing.T) {
						gpuColumnReduce(t, d, fn, sh[0], sh[1], sh[2], rows)
					})
				}
			}
		}
	}
}

func gpuColumnReduce(t *testing.T, d backend.Device, fn graph.MapFn, G, S, D, rows int) {
	g, in, y := composer.ColumnReduceGraph(fn, G, S, D, G*D)
	_, codes, err := composer.Default().Form(composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{},
		Step: composer.Step{Rows: rows, Page: 64, Keys: 1}, Device: gpuDevice(d)})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(int64(G*100 + S*10 + D)))
	x := make([]float32, rows*G*S*D)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 3)
	}
	out, err := launchAttend(d, codes, map[graph.ValueID][]uint32{in: f32w(x), y: poisoned(rows * G * D)}, map[string][]uint32{"rows.n": {uint32(rows)}}, nil, y)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range composer.ColumnReduceWant(fn, x, rows, G, S, D) {
		got := float64(math.Float32frombits(out[i]))
		if math.IsNaN(got) || math.Abs(got-w) > 1e-5*(1+math.Abs(w)) {
			t.Fatalf("row %d block %d column %d: %g, want %g", i/(G*D), i%(G*D)/D, i%D, got, w)
		}
	}
}
