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

// TestComposedGPUHistoryPool is TestComposedCPUHistoryPool on every GPU:
// OpHistoryPool (DeepSeek V4's compressed history) over a chunk of rows
// appended to the paged history a card holds -- keys float32 by dimension,
// values binary16 or float32 by position -- each row's window of Span
// positions, and for a two-segment pool the window before it, pooled per
// channel by the softmax of gate plus bias, held to the definition row by
// row. A zeroed bias and a write position moved by one must fail it.
func TestComposedGPUHistoryPool(t *testing.T) {
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
			name  string
			p, w  int
			segs  [][4]uint64
			width int
		}{
			{"one", 8, 6, [][4]uint64{{0, 0, 6, 0}}, 12},
			{"one-ragged", 4, 5, [][4]uint64{{0, 0, 5, 0}}, 10},
			{"overlap", 4, 6, [][4]uint64{{1, 0, 12, 0}, {0, 6, 18, 6}}, 24},
			{"overlap-ragged", 2, 5, [][4]uint64{{1, 0, 10, 0}, {0, 5, 15, 5}}, 20},
		} {
			for _, f32v := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/f32values%v", d.API(), sh.name, f32v), func(t *testing.T) {
					gpuHistoryPool(t, d, sh.p, sh.w, sh.segs, sh.width, f32v)
				})
			}
		}
	}
}

func gpuHistoryPool(t *testing.T, d backend.Device, P, W int, segs [][4]uint64, RW int, f32v bool) {
	const page, p0, R, capRows = 8, 13, 7, 8
	var words []uint64
	bw := 0
	for _, s := range segs {
		words = append(words, s[:]...)
		bw = max(bw, int(s[3])+W)
	}
	b := graph.New()
	n := b.KVAppend(b.In("k", RW), b.In("v", RW), 0)
	b.Out(b.HistoryPool(graph.StateRef{State: graph.StateKV, Version: n}, graph.Attrs{Span: P, Width: RW, Words: words}, jlm.RoleCompAPE, W))
	g := b.G
	_, codes, err := composer.Default().Form(composer.Request{Graph: g,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: !f32v}},
		Step:    composer.Step{Rows: capRows, Page: page, Keys: 4 * page, Seqs: 1}, Device: gpuDevice(d)})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	// Every row binary16-exact, so each pool holds it alike.
	rnd := func(n int, m float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(quant.DecodeHalf(quant.EncodeHalf(float32(rng.NormFloat64() * m))))
		}
		return x
	}
	rows := make([][]float32, p0+R)
	for p := range rows {
		rows[p] = rnd(RW, 2)
	}
	bias := rnd(P*bw, 1)
	k := make([]float32, capRows*RW)
	for r := range R {
		copy(k[r*RW:], rows[p0+r])
	}
	kpool := poisoned(4 * page * RW)
	vpool := poisoned(4 * page * RW / 2)
	if f32v {
		vpool = poisoned(4 * page * RW)
	}
	for p := range p0 {
		for e, x := range rows[p] {
			kpool[(p/page*RW+e)*page+p%page] = math.Float32bits(x)
			hv, i := uint32(quant.EncodeHalf(x)), p*RW+e
			switch {
			case f32v:
				vpool[i] = math.Float32bits(x)
			case i%2 == 0:
				vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
			default:
				vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
			}
		}
	}
	want := func(bias []float32) []float64 {
		out := make([]float64, R*W)
		for r := range R {
			t := p0 + r
			ws := t / P * P
			for ch := range W {
				var sc, val []float64
				for _, s := range segs {
					base := ws - int(s[0])*P
					if base < 0 {
						continue
					}
					for j := range P {
						p := base + j
						if p > t {
							continue
						}
						sc = append(sc, float64(rows[p][int(s[2])+ch])+float64(bias[j*bw+int(s[3])+ch]))
						val = append(val, float64(rows[p][int(s[1])+ch]))
					}
				}
				mx := math.Inf(-1)
				for _, x := range sc {
					mx = math.Max(mx, x)
				}
				var num, den float64
				for i, x := range sc {
					e := math.Exp(x - mx)
					num += e * val[i]
					den += e
				}
				out[r*W+ch] = num / den
			}
		}
		return out
	}
	run := func(bias []float32, shift uint32) []float32 {
		desc := make([]uint32, capRows*4)
		for r := range R {
			p := uint32(p0+r) + shift
			copy(desc[r*4:], []uint32{0, 0, p + 1, p})
		}
		vs := map[graph.ValueID][]uint32{g.In[0]: f32w(k), g.In[1]: f32w(k), g.Out[0]: poisoned(capRows * W)}
		tables := map[string][]uint32{"rows": desc, "rows.win": desc, "rows.n": {R}, "pages": {0, 1, 2, 3}, "q.off": make([]uint32, capRows*4),
			"kv.k": append([]uint32(nil), kpool...), "kv.v": append([]uint32(nil), vpool...)}
		vecs := map[jlm.Role][]uint32{jlm.RoleCompAPE: f32w(bias)}
		out, err := launchAttend(d, codes, vs, tables, vecs, g.Out[0])
		if err != nil {
			t.Fatal(err)
		}
		return wf32(out)
	}
	nmse := func(got []float32, want []float64) float64 {
		var num, den float64
		for i := range want {
			e := float64(got[i]) - want[i]
			num += e * e
			den += want[i] * want[i]
		}
		return num / den
	}
	ref := want(bias)
	e := nmse(run(bias, 0), ref)
	t.Logf("%d rows, windows of %d, %d segments: NMSE %.3e against the definition", R, P, len(segs), e)
	if math.IsNaN(e) || e > 1e-9 {
		t.Fatalf("the composed pool on the card: NMSE %.3e against the definition", e)
	}
	if e := nmse(run(make([]float32, len(bias)), 0), ref); !(e > 1e-6) {
		t.Errorf("with the bias zeroed the NMSE is %.3e: the gate cannot see the bias", e)
	}
	// A row moved past its own reads a word nothing appended: NaN, seen.
	if e := nmse(run(bias, 1), ref); !(e > 1e-6 || math.IsNaN(e)) {
		t.Errorf("with the write position moved the NMSE is %.3e: the gate cannot see the window", e)
	}
}
