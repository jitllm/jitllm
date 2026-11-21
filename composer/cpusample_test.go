package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// formOnCPU forms graph g on the CPU and runs it over vals, the profile
// with the passes off turned off.
func formOnCPU(t *testing.T, cpuOn cpuArm, g *graph.Graph, vals map[graph.ValueID][]uint32, off ...string) {
	t.Helper()
	reg := Default()
	p := reg.Profiles[DeviceCPU]
	p.Off = map[string]bool{}
	for _, o := range off {
		p.Off[o] = true
	}
	reg.Profiles[DeviceCPU] = p
	_, codes, err := reg.Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(cpuOn), Apart: true})
	if err != nil {
		t.Fatalf("forming: %v", err)
	}
	if err := runCPUValues(cpuOn, codes, g, vals, [3][]uint32{}); err != nil {
		t.Fatal(err)
	}
}

// rankOnCPU is OpRank of x over one group, formed and run on the CPU.
func rankOnCPU(t *testing.T, cpuOn cpuArm, x []float32, off ...string) []uint32 {
	b := graph.New()
	in := b.In("x", len(x))
	r := b.Emit(graph.OpRank, []graph.ValueID{in}, []graph.Result{{Name: "rank", Width: len(x), Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(r)
	xs := make([]uint32, len(x))
	for i, v := range x {
		xs[i] = math.Float32bits(v)
	}
	vals := map[graph.ValueID][]uint32{in: xs}
	formOnCPU(t, cpuOn, b.G, vals, off...)
	return vals[r]
}

// TestComposedCPURank forms OpRank (each element's place, largest first,
// the lower index of two equal first) on the CPU at ragged widths, values
// on a grid of five so that ties are many, held to the definition. The
// ranks with ties to the higher index must part from what it gives.
func TestComposedCPURank(t *testing.T) { gateComposedCPURank(t, hostArm()) }
func gateComposedCPURank(t *testing.T, cpuOn cpuArm) {
	rng := rand.New(rand.NewSource(3))
	ran, parted := 0, 0
	for _, n := range []int{1, 2, 5, 7, 8, 9, 15, 16, 17, 33, 100} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.Intn(5))
		}
		want, high := make([]uint32, n), make([]uint32, n)
		for i := range n {
			for j := range n {
				if x[i] < x[j] || x[i] == x[j] && j < i {
					want[i]++
				}
				if x[i] < x[j] || x[i] == x[j] && j > i {
					high[i]++
				}
			}
		}
		got := rankOnCPU(t, cpuOn, x)
		for i := range n {
			if got[i] != want[i] {
				t.Fatalf("width %d: rank %v, want %v", n, got[:n], want)
			}
		}
		for i := range n {
			if got[i] != high[i] {
				parted++
				break
			}
		}
		ran++
	}
	if ran != 11 {
		t.Fatalf("%d of 11 widths held", ran)
	}
	if parted == 0 {
		t.Fatal("the ranks with ties to the higher index match: the gate cannot see the tie order")
	}
}

// TestComposedCPUTopK forms OpTopK of three (the indices and values of the
// three largest, the lower index of two equal first) on the CPU at ragged
// widths and holds it to the definition; the form with FoldsAside off,
// whose place counter runs a trip ahead, must part.
func TestComposedCPUTopK(t *testing.T) { gateComposedCPUTopK(t, hostArm()) }
func gateComposedCPUTopK(t *testing.T, cpuOn cpuArm) {
	rng := rand.New(rand.NewSource(4))
	const K = 3
	run := func(x []float32, off ...string) ([]uint32, []float32) {
		b := graph.New()
		in := b.In("x", len(x))
		outs := b.Emit(graph.OpTopK, []graph.ValueID{in}, []graph.Result{{Name: "ids", Width: K, Elem: graph.ElemU32}, {Name: "vals", Width: K}}, nil, graph.Attrs{TopK: K}, nil, nil)
		b.Out(outs[0])
		b.Out(outs[1])
		xs := make([]uint32, len(x))
		for i, v := range x {
			xs[i] = math.Float32bits(v)
		}
		vals := map[graph.ValueID][]uint32{in: xs}
		formOnCPU(t, cpuOn, b.G, vals, off...)
		return vals[outs[0]], wordsF32(vals[outs[1]])
	}
	parted := 0
	for _, n := range []int{3, 5, 8, 9, 17, 37, 100} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.Intn(n))
		}
		order := sampleOrder(x)
		ids, vs := run(x)
		for p := range K {
			if int(ids[p]) != order[p] || vs[p] != x[order[p]] {
				t.Fatalf("width %d: place %d holds %d (%g), want %d (%g)", n, p, ids[p], vs[p], order[p], x[order[p]])
			}
		}
		if bi, _ := run(x, "foldsaside"); int(bi[0]) != order[0] || int(bi[1]) != order[1] || int(bi[2]) != order[2] {
			parted++
		}
	}
	if parted == 0 {
		t.Fatal("with FoldsAside off every width holds: the gate cannot see a place counter read a trip ahead")
	}
}

// TestComposedCPUSampleTopK forms the sampler's candidates
// (graph.SampleTopK) on the CPU for every case (SampleCases), the ranked
// form at selectMin and below and the selection form past it, and holds each to
// the sampler's candidates bit for bit in value and id, guard words past
// every output, ties among the kept on some. The violation: the window moved
// to the next id (SampleWindow) parts on every case with a window.
func TestComposedCPUSampleTopK(t *testing.T) { gateComposedCPUSampleTopK(t, hostArm()) }
func gateComposedCPUSampleTopK(t *testing.T, cpuOn cpuArm) {
	ran, ties, want, selected := 0, 0, 0, 0
	for _, c := range SampleCases() {
		if c.V > selectMin {
			selected++
		}
		want++
		s := graph.SampleTopK{NVocab: c.V, K: c.K}
		g := s.Build()
		run := func(shift bool) ([]float32, []uint32) {
			lw := make([]uint32, len(c.Logits))
			for i, v := range c.Logits {
				lw[i] = math.Float32bits(v)
			}
			vals := map[graph.ValueID][]uint32{g.In[0]: lw, g.In[1]: SampleWindow(c, shift), g.In[2]: {math.Float32bits(c.Pen), 0, 0}}
			formOnCPU(t, cpuOn, g, vals)
			return wordsF32(vals[g.Out[0]]), vals[g.Out[1]]
		}
		gv, gi := run(false)
		wv, wi := SampleWant(c)
		if bad := SampleMismatch(gv, gi, wv, wi); bad > 0 {
			t.Errorf("%s: %d of %d places part from the host's; first %v %v, want %v %v", c.Name, bad, len(wv), gv[:min(4, len(gv))], gi[:min(4, len(gi))], wv[:min(4, len(wv))], wi[:min(4, len(wi))])
			continue
		}
		if SampleTieHigh(gi, wv) {
			ties++
		}
		if len(c.Hist) > 0 {
			bv, bi := run(true)
			if SampleMismatch(bv, bi, wv, wi) == 0 {
				t.Errorf("%s: the window moved to the next id matches: the gate cannot see it", c.Name)
			}
		}
		ran++
	}
	if want == 0 || ran != want {
		t.Fatalf("%d of %d cases held", ran, want)
	}
	if ties == 0 {
		t.Fatal("no case keeps two places of one value: the ties to the lower id are not held")
	}
	if selected == 0 || selected == ran {
		t.Fatalf("%d of %d cases past selectMin: both forms are not held", selected, ran)
	}
	t.Logf("%d cases held bit for bit (%d through the selection form), %d with ties among the kept", ran, selected, ties)
}

// TestComposedCPUSampleGreedy forms greedy sampling (graph.Sample, the
// largest logit and the lower id of two equal) on the CPU at ragged widths,
// each row's best placed at its start, middle and end, and tied with an
// earlier one, and holds the token to the definition. The violation: the
// logits negated, whose largest is another id, must part.
func TestComposedCPUSampleGreedy(t *testing.T) { gateComposedCPUSampleGreedy(t, hostArm()) }
func gateComposedCPUSampleGreedy(t *testing.T, cpuOn cpuArm) {
	rng := rand.New(rand.NewSource(5))
	ran, parted := 0, 0
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 33, 100, 1531} {
		for _, at := range []int{0, n / 2, n - 1} {
			l := make([]float32, n)
			for i := range l {
				l[i] = float32(rng.NormFloat64())
			}
			l[at] = 9
			want := at
			if at > 0 {
				l[at/2] = 9
				want = at / 2
			}
			run := func(l []float32) int {
				s := graph.Sample{NVocab: n}
				g := s.Build()
				lw := make([]uint32, n)
				for i, v := range l {
					lw[i] = math.Float32bits(v)
				}
				vals := map[graph.ValueID][]uint32{g.In[0]: lw}
				formOnCPU(t, cpuOn, g, vals)
				return int(vals[g.Out[0]][0])
			}
			if got := run(l); got != want {
				t.Errorf("width %d, best at %d: token %d, want %d", n, at, got, want)
				continue
			}
			neg := make([]float32, n)
			for i, v := range l {
				neg[i] = -v
			}
			if n > 2 && run(neg) != want {
				parted++
			}
			ran++
		}
	}
	if ran != 33 {
		t.Fatalf("%d of 33 rows held", ran)
	}
	if parted == 0 {
		t.Fatal("the negated logits give every token: the gate cannot see a wrong one")
	}
}
