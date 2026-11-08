package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUAttendValuesInKeys forms a latent attention as the model
// states one -- every query head over one kv head whose cached row is both
// its key and its value (a history whose values are its keys' rows) -- and
// holds each row's output to the definition over the binary16 rows: the
// cached positions before the step and the step's own, several heads a kv
// head, as the CPU's pass pairing heads that share a kv head may run them.
func TestComposedCPUAttendValuesInKeys(t *testing.T) {
	for _, sh := range []struct{ heads, dim, rows, at int }{{4, 8, 9, 2}, {2, 6, 1, 2}, {8, 12, 5, 2}, {4, 8, 1, 2},
		{4, 8, 1, 0}, {4, 8, 3, 0}, {16, 40, 1, 0}, {16, 40, 6, 0}, {16, 8, 1, 0}, {4, 40, 1, 0}, {2, 40, 1, 0}, {2, 16, 1, 0}, {2, 17, 1, 0}, {2, 20, 1, 0}, {2, 24, 1, 0}, {2, 32, 1, 0}, {2, 33, 1, 0}} {
		t.Run(fmt.Sprintf("%dq%dd/%drows/from%d", sh.heads, sh.dim, sh.rows, sh.at), func(t *testing.T) {
			composedCPUAttendVIK(t, sh.heads, sh.dim, sh.rows, sh.at)
		})
	}
}

func composedCPUAttendVIK(t *testing.T, H, D, R, p0 int) {
	const page, capRows = 8, 16
	b := graph.New()
	k := b.In("k", D)
	n := b.KVAppend(k, k, 0)
	a := graph.Attrs{NHead: H, NKVHead: 1, HeadDim: D, Scale: 0.3}
	b.Out(b.Attend(b.In("q", H*D), n, a))
	_, codes, err := Default().Form(Request{Graph: b.G, History: History{ValuesInKeys: true},
		Step:   Step{Rows: capRows, Page: page, Keys: 4 * page, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(hostArm().tg.Features().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(17))
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	last := p0 + R
	rows := make([][]float32, last)
	for p := range rows {
		rows[p] = rnd(D)
	}
	q := rnd(capRows * H * D)
	kin := make([]float32, capRows*D)
	for r := range R {
		copy(kin[r*D:], rows[p0+r])
	}
	pool := make([]uint32, 4*page*D/2)
	for p := range p0 {
		for d, x := range rows[p] {
			h, i := uint32(quant.EncodeHalf(x)), p*D+d
			if d%2 == 0 {
				pool[i/2] = pool[i/2]&0xFFFF0000 | h
			} else {
				pool[i/2] = pool[i/2]&0xFFFF | h<<16
			}
		}
	}
	round := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	want := make([]float64, R*H*D)
	for r := range R {
		keys := rows[:p0+r+1]
		for h := range H {
			sc := make([]float64, len(keys))
			mx := math.Inf(-1)
			for i, key := range keys {
				var s float64
				for d := range D {
					s += float64(q[(r*H+h)*D+d]) * round(key[d])
				}
				sc[i] = s * 0.3
				mx = math.Max(mx, sc[i])
			}
			var sum float64
			for i := range sc {
				sc[i] = math.Exp(sc[i] - mx)
				sum += sc[i]
			}
			for d := range D {
				var o float64
				for i, key := range keys {
					o += sc[i] / sum * round(key[d])
				}
				want[(r*H+h)*D+d] = o
			}
		}
	}
	out := make([]float32, capRows*H*D)
	desc := make([]uint32, capRows*4)
	for r := range R {
		p := uint32(p0 + r)
		copy(desc[r*4:], []uint32{0, 0, p + 1, p})
	}
	bufs := map[string][]uint32{"k": f32Words(kin), "q": f32Words(q), "rows.n": {uint32(R)}, "q.off": make([]uint32, capRows*4)}
	for _, c := range codes {
		for _, bd := range c.Binds {
			o := bd.Operand
			if o.Kind != OperandTable {
				continue
			}
			switch o.Table {
			case TableRows, TableWindows:
				bufs[o.Name] = desc
			case TablePages:
				bufs[o.Name] = []uint32{0, 1, 2, 3}
			case TableKeys:
				bufs[o.Name] = pool
			}
		}
	}
	g := b.G
	bufs[g.Values[g.Out[0]].Name] = f32Words(out)
	if err := runCPURows(hostArm(), codes, g, bufs, R); err != nil {
		t.Fatal(err)
	}
	for _, bd := range codes[len(codes)-1].Binds {
		if o := bd.Operand; bd.Writes && o.Kind == OperandScratch {
			copy(out, wordsF32(bufs["own:"+o.Name]))
		}
	}
	var num, den float64
	for i, w := range want {
		d := float64(out[i]) - w
		num += d * d
		den += w * w
	}
	t.Logf("%d heads of %d over one kv head, %d rows: NMSE %.3e", H, D, R, num/den)
	if !(num/den < 1e-9) {
		t.Fatalf("%d heads of %d over one kv head, %d rows: NMSE %.3e against the definition", H, D, R, num/den)
	}
}
