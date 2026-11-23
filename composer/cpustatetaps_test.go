package composer

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// rowTapShapes are ragged state convolutions with per-row taps: taps,
// channels and the groups sharing a row's taps.
var rowTapShapes = [][3]int{{2, 3, 1}, {3, 8, 2}, {4, 7, 7}, {4, 33, 3}, {3, 64, 4}, {5, 9, 3}}

// rowTapGraph is an OpStateConv of C channels over a window in the
// recurrent convolution's state, each row's taps of G groups added to its
// static ones.
func rowTapGraph(T, C, G int) *graph.Builder {
	b := graph.New()
	x := b.In("x", C)
	d := b.In("taps", G*T)
	b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x, d}, []graph.Result{{Name: "conv", Width: C}},
		[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: T, Rows: C}}, graph.Attrs{Taps: T},
		[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0])
	return b
}

// rowTapPlanes is the oracle's plane-major taps of a row: each channel's
// static taps plus its group's of the row's, the group group(c).
func rowTapPlanes(taps []float32, dyn []float32, T, C, G int, group func(c int) int) []float64 {
	planes := make([]float64, C*T)
	for c := range C {
		for k := range T {
			v := float64(taps[c*T+k])
			if dyn != nil {
				v += float64(dyn[group(c)*T+k])
			}
			planes[k*C+c] = v
		}
	}
	return planes
}

// rowTapFaults are the oracle arms a gate holds the form against: the
// truth, the per-row taps dropped, and the groups interleaved over the
// channels in place of contiguous.
type rowTapArm int

const (
	rowTapTrue rowTapArm = iota
	rowTapDropped
	rowTapInterleaved
	rowTapArms
)

func rowTapRef(arm rowTapArm, taps, dyn []float32, T, C, G int) []float64 {
	gw := C / G
	switch arm {
	case rowTapDropped:
		return rowTapPlanes(taps, nil, T, C, G, nil)
	case rowTapInterleaved:
		return rowTapPlanes(taps, dyn, T, C, G, func(c int) int { return c % G })
	}
	return rowTapPlanes(taps, dyn, T, C, G, func(c int) int { return c / gw })
}

func randF32(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// TestComposedCPUStateConvRowTaps forms OpStateConv with per-row taps (each
// group of channels' own taps a row, added to the static ones: a dynamic
// short convolution) on the CPU device over ragged channel counts, taps and
// groups, at a step of one row run four rows on and at a step of eight rows
// in three runs over six live rows. Every row's output and every window
// after are held to the oracle's convolution under the row's summed taps;
// with the per-row taps dropped, or the groups taken interleaved over the
// channels, the oracle must part, and per-row taps that do not divide into
// the channels' groups are refused at forming.
func TestComposedCPUStateConvRowTaps(t *testing.T) {
	cpuOn := hostArm()
	rng := rand.New(rand.NewSource(17))
	parted := [rowTapArms]int{}
	interleavable := 0
	for _, sh := range rowTapShapes {
		T, C, G := sh[0], sh[1], sh[2]
		if G > 1 && C/G > 1 {
			interleavable++
		}
		W := (T - 1) * C
		taps := randF32(rng, C*T)
		wt := unsafe.Slice((*uint32)(unsafe.Pointer(&taps[0])), len(taps))
		b := rowTapGraph(T, C, G)
		formats := map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32}

		_, codes, err := Default().Form(Request{Graph: b.G, Formats: formats, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(cpuOn)})
		if err != nil {
			t.Fatalf("%v at one row: %v", sh, err)
		}
		nan := math.Float32bits(float32(math.NaN()))
		pool := make([]uint32, 3*W)
		for i := range pool {
			pool[i] = nan
		}
		window := randF32(rng, W)
		refs := [rowTapArms][]float64{}
		for a := range refs {
			refs[a] = make([]float64, W)
			for i, v := range window {
				refs[a][i] = float64(v)
			}
		}
		for i, v := range window {
			pool[W+i] = math.Float32bits(v)
		}
		tables := map[string][]uint32{"rec.conv": pool, "rec.conv.out": pool, "rec.rows": {1, 1, 1}}
		worst := [rowTapArms]float64{}
		for row := range 4 {
			xs, dyn := randF32(rng, C), randF32(rng, G*T)
			vals := map[graph.ValueID][]uint32{
				b.G.In[0]: unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), C),
				b.G.In[1]: unsafe.Slice((*uint32)(unsafe.Pointer(&dyn[0])), G*T),
			}
			if err := runCPUTables(cpuOn, codes, b.G, vals, [3][]uint32{wt}, tables); err != nil {
				t.Fatalf("%v row %d: %v", sh, row, err)
			}
			got := wordsF32(vals[b.G.Out[0]])
			x64 := make([]float64, C)
			for i, v := range xs {
				x64[i] = float64(v)
			}
			for a := range rowTapArms {
				want := make([]float64, C)
				oracle.Conv1d(want, refs[a], rowTapRef(a, taps, dyn, T, C, G), x64, T, C)
				for i, w := range want {
					worst[a] = max(worst[a], math.Abs(float64(got[i])-w))
				}
			}
			for i := range W {
				worst[rowTapTrue] = max(worst[rowTapTrue], math.Abs(float64(math.Float32frombits(pool[W+i]))-refs[rowTapTrue][i]))
			}
			for i, w := range pool {
				if (i < W || i >= 2*W) && w != nan {
					t.Fatalf("%v row %d: the form wrote %v at %d, outside its slot", sh, row, math.Float32frombits(w), i)
				}
			}
		}
		if !(worst[rowTapTrue] <= 1e-5) {
			t.Fatalf("%v at one row: the composed convolution is %g from the oracle", sh, worst[rowTapTrue])
		}
		for a := rowTapDropped; a < rowTapArms; a++ {
			if worst[a] > 1e-3 {
				parted[a]++
			}
		}

		for a, w := range runRowTapStep(t, cpuOn, sh, rng) {
			if a == int(rowTapTrue) {
				if !(w <= 1e-5) {
					t.Fatalf("%v at eight rows: the composed convolution is %g from the oracle", sh, w)
				}
				continue
			}
			if w > 1e-3 {
				parted[a]++
			}
		}
	}
	if parted[rowTapDropped] != 2*len(rowTapShapes) {
		t.Fatalf("the per-row taps dropped parted %d of %d times: the gate cannot see them", parted[rowTapDropped], 2*len(rowTapShapes))
	}
	if parted[rowTapInterleaved] < 2*interleavable {
		t.Fatalf("the groups interleaved parted %d of %d times: the gate cannot see the grouping", parted[rowTapInterleaved], 2*interleavable)
	}
	for _, bad := range [][3]int{{3, 8, 0}, {4, 9, 2}} {
		T, C, G := bad[0], bad[1], bad[2]
		b := graph.New()
		x := b.In("x", C)
		d := b.In("taps", G*T+1)
		b.Out(b.Emit(graph.OpStateConv, []graph.ValueID{x, d}, []graph.Result{{Name: "conv", Width: C}},
			[]graph.Binding{{Role: jlm.RoleSSMConv1d, K: T, Rows: C}}, graph.Attrs{Taps: T},
			[]graph.StateRef{{State: graph.StateRecConv}}, []graph.StateRef{{State: graph.StateRecConv, Version: 1}})[0])
		if _, _, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32},
			Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice(cpuOn)}); err == nil {
			t.Fatalf("per-row taps %d wide for %d taps over %d channels formed", G*T+1, T, C)
		}
	}
	t.Logf("%d shapes held to the oracle at one row and eight; the dropped taps parted %d, the interleaved groups %d",
		len(rowTapShapes), parted[rowTapDropped], parted[rowTapInterleaved])
}

// runRowTapStep runs one step of eight rows of the shape's convolution with
// per-row taps in three runs over six live rows, each run's window read from
// its slot of one pool and written to its slot of another, and returns each
// oracle arm's worst distance at the live rows' outputs and the windows
// written.
func runRowTapStep(t *testing.T, cpuOn cpuArm, sh [3]int, rng *rand.Rand) [rowTapArms]float64 {
	const R, live = 8, 6
	runs := [][]int{{1, 3, 0, 4}, {2}, {5}}
	T, C, G := sh[0], sh[1], sh[2]
	W := (T - 1) * C
	b := rowTapGraph(T, C, G)
	_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleSSMConv1d: jlm.TypeF32},
		Step: Step{Rows: R, Page: 64, Keys: 1}, Device: cpuDevice(cpuOn)})
	if err != nil {
		t.Fatalf("%v at %d rows: %v", sh, R, err)
	}
	taps := randF32(rng, C*T)
	wt := unsafe.Slice((*uint32)(unsafe.Pointer(&taps[0])), len(taps))
	nan := math.Float32bits(float32(math.NaN()))
	pool, outPool := make([]uint32, 7*W), make([]uint32, 7*W)
	for i := range pool {
		pool[i], outPool[i] = nan, nan
	}
	in, out := []int{1, 3, 5}, []int{2, 4, 6}
	refs := make([][rowTapArms][]float64, len(runs))
	for j := range runs {
		for i, v := range randF32(rng, W) {
			pool[in[j]*W+i] = math.Float32bits(v)
		}
		for a := range rowTapArms {
			refs[j][a] = make([]float64, W)
			for i := range W {
				refs[j][a][i] = float64(math.Float32frombits(pool[in[j]*W+i]))
			}
		}
	}
	xs, dyn := randF32(rng, R*C), randF32(rng, R*G*T)
	vals := map[graph.ValueID][]uint32{
		b.G.In[0]: unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), len(xs)),
		b.G.In[1]: unsafe.Slice((*uint32)(unsafe.Pointer(&dyn[0])), len(dyn)),
	}
	tables := map[string][]uint32{"rec.conv": pool, "rec.conv.out": outPool, "rec.rows": scanDesc(runs, live, in, out)}
	gr := *b.G
	gr.Values = slices.Clone(b.G.Values)
	for i := range gr.Values {
		gr.Values[i].Width *= R
	}
	if err := runCPUTables(cpuOn, codes, &gr, vals, [3][]uint32{wt}, tables); err != nil {
		t.Fatalf("%v at %d rows: %v", sh, R, err)
	}
	got := wordsF32(vals[b.G.Out[0]])
	var worst [rowTapArms]float64
	for j, r := range runs {
		for _, row := range r {
			x64 := make([]float64, C)
			for i := range C {
				x64[i] = float64(xs[row*C+i])
			}
			for a := range rowTapArms {
				want := make([]float64, C)
				oracle.Conv1d(want, refs[j][a], rowTapRef(a, taps, dyn[row*G*T:(row+1)*G*T], T, C, G), x64, T, C)
				for i, w := range want {
					worst[a] = max(worst[a], math.Abs(float64(got[row*C+i])-w))
				}
			}
		}
		for i := range W {
			worst[rowTapTrue] = max(worst[rowTapTrue], math.Abs(float64(math.Float32frombits(outPool[out[j]*W+i]))-refs[j][rowTapTrue][i]))
		}
	}
	for i := range W {
		if outPool[i] != nan {
			t.Fatalf("%s: the form wrote %v in slot 0, which no run names", fmt.Sprint(sh), math.Float32frombits(outPool[i]))
		}
	}
	return worst
}
