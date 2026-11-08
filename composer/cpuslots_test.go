package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// slotRun forms graph g at rows rows on the CPU and runs it over x, the
// step's live rows live and its request req, returning n outputs.
func slotRun(t *testing.T, g *graph.Graph, rows, live int, req []uint32, x []float32, n int) []float32 {
	t.Helper()
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	_, codes, err := r.Form(Request{Graph: g, Step: Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	pos := make([]uint32, rows+1)
	pos[0] = uint32(live)
	for i := range rows {
		pos[1+i] = uint32(i)
	}
	got, err := runCPUBound(hostArm(), codes, x, n, [3][]uint32{}, nil, func(o Operand) []uint32 {
		switch o.Name {
		case "rope.pos":
			return pos
		case "request":
			return req
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// slotRequest is a request whose slot s holds rows[s] (its row r at word
// RequestRows + s*live + r), padded past the last slot by the form's rows.
func slotRequest(live, rows int, slots map[int][]uint32) []uint32 {
	top := 0
	for s := range slots {
		top = max(top, s)
	}
	req := make([]uint32, RequestRows+(top+1)*live+rows)
	for s, v := range slots {
		copy(req[RequestRows+s*live:], v)
	}
	return req
}

func slotRows(R, D int) []float32 {
	x := make([]float32, R*D)
	for i := range x {
		x[i] = float32(math.Sin(0.37*float64(i)+0.2) * (1 + float64(i%3)/2))
	}
	return x
}

// TestComposedCPUSpanPool holds graph.OpSpanPool to its definition on the
// CPU: each live row the mean of the input's rows its span (the request's
// slots 1 and 2) names, an empty span's zero. Every row's span differs and
// one is empty, so a pool that read the wrong slot or the whole run fails.
func TestComposedCPUSpanPool(t *testing.T) {
	const R, D, live = 8, 4, 6
	x := slotRows(R, D)
	b := graph.NewBlock(-1)
	in := b.In("x", D)
	b.Out(b.Emit(graph.OpSpanPool, []graph.ValueID{in}, []graph.Result{{Name: "pool", Width: D}}, nil,
		graph.Attrs{Offset: 1}, nil, nil)[0])
	spans := [][2]uint32{{0, 3}, {2, 6}, {5, 5}, {1, 2}, {4, 6}, {0, 6}}
	lo, hi := make([]uint32, live), make([]uint32, live)
	for i, sp := range spans {
		lo[i], hi[i] = sp[0], sp[1]
	}
	got := slotRun(t, b.G, R, live, slotRequest(live, R, map[int][]uint32{1: lo, 2: hi}), x, R*D)
	worst := 0.0
	for i, sp := range spans {
		for e := range D {
			want := 0.0
			for j := sp[0]; j < sp[1]; j++ {
				want += float64(x[int(j)*D+e])
			}
			if n := sp[1] - sp[0]; n > 0 {
				want /= float64(n)
			}
			d := math.Abs(float64(got[i*D+e]) - want)
			if math.IsNaN(d) || d > 1e-5 {
				t.Fatalf("row %d element %d: %g, want %g (span %v)", i, e, got[i*D+e], want, sp)
			}
			worst = max(worst, d)
		}
	}
	t.Logf("%d spans, an empty one among them, worst %.2e", len(spans), worst)
}

// TestComposedCPUTakeRowsFromASlot holds graph.OpTakeRows reading the
// request's slot 2 to its definition: row r the input's row the slot names.
// The rows named are out of order and repeat, so a take that read rows in
// place or slot 0 fails.
func TestComposedCPUTakeRowsFromASlot(t *testing.T) {
	const R, D, live = 8, 4, 6
	x := slotRows(R, D)
	b := graph.NewBlock(-1)
	in := b.In("x", D)
	b.Out(b.Emit(graph.OpTakeRows, []graph.ValueID{in}, []graph.Result{{Name: "take", Width: D}}, nil,
		graph.Attrs{Offset: 2}, nil, nil)[0])
	at := []uint32{4, 4, 0, 5, 2, 1}
	got := slotRun(t, b.G, R, live, slotRequest(live, R, map[int][]uint32{0: {1, 2, 3, 4, 5, 0}, 2: at}), x, R*D)
	for r, j := range at {
		for e := range D {
			if g, w := got[r*D+e], x[int(j)*D+e]; g != w {
				t.Fatalf("row %d element %d: %g, want row %d's %g", r, e, g, j, w)
			}
		}
	}
}

// TestComposedCPURowsAttendInARange holds graph.OpRowsAttend with a key
// range per row (the request's slots 1 and 2) to the softmax over that range
// alone: each live row of x attending x's rows in its range, two heads of
// four. The ranges differ per row, so attending every live row fails.
func TestComposedCPURowsAttendInARange(t *testing.T) {
	const R, H, hd, live = 8, 2, 4, 6
	const D, scale = H * hd, 0.5
	x := slotRows(R, D)
	b := graph.NewBlock(-1)
	in := b.In("x", D)
	b.Out(b.Emit(graph.OpRowsAttend, []graph.ValueID{in, in, in}, []graph.Result{{Name: "o", Width: D}}, nil,
		graph.Attrs{NHead: H, NKVHead: H, HeadDim: hd, Scale: scale, Offset: 1}, nil, nil)[0])
	ranges := [][2]uint32{{0, 2}, {1, 6}, {3, 4}, {0, 6}, {2, 5}, {5, 6}}
	lo, hi := make([]uint32, live), make([]uint32, live)
	for i, rg := range ranges {
		lo[i], hi[i] = rg[0], rg[1]
	}
	got := slotRun(t, b.G, R, live, slotRequest(live, R, map[int][]uint32{1: lo, 2: hi}), x, R*D)
	worst := 0.0
	for i, rg := range ranges {
		for h := range H {
			s := make([]float64, rg[1]-rg[0])
			m := math.Inf(-1)
			for k := range s {
				j := int(rg[0]) + k
				for e := range hd {
					s[k] += float64(x[i*D+h*hd+e]) * float64(x[j*D+h*hd+e])
				}
				s[k] *= scale
				m = math.Max(m, s[k])
			}
			l := 0.0
			for k := range s {
				s[k] = math.Exp(s[k] - m)
				l += s[k]
			}
			for e := range hd {
				want := 0.0
				for k := range s {
					want += s[k] * float64(x[(int(rg[0])+k)*D+h*hd+e])
				}
				want /= l
				d := math.Abs(float64(got[i*D+h*hd+e]) - want)
				if math.IsNaN(d) || d > 1e-5 {
					t.Fatalf("row %d head %d element %d: %g, want %g (keys %v)", i, h, e, got[i*D+h*hd+e], want, rg)
				}
				worst = max(worst, d)
			}
		}
	}
	t.Logf("%d rows each over a range of its own, worst %.2e", live, worst)
}
