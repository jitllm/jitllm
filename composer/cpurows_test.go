package composer

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
)

// cpuDevice is the CPU as the composer forms for it at the shipping width.
func cpuDevice() Device {
	return Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
}

func wordsF32(w []uint32) []float32 { return unsafe.Slice((*float32)(unsafe.Pointer(&w[0])), len(w)) }

// TestComposedCPUEqual forms MapEqual on the CPU device at ragged widths,
// over two index rows (their integers) and over two float rows (-0 equal to
// +0, a NaN equal to nothing), with guard words past the output; with the
// reference's equality one id off, it must part.
func TestComposedCPUEqual(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	ran, parted := 0, 0
	for _, ints := range []bool{true, false} {
		for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 33} {
			b := graph.New()
			elem := graph.ElemF32
			if ints {
				elem = graph.ElemU32
			}
			x, y := b.In("x", n), b.In("y", n)
			b.G.Values[x].Elem, b.G.Values[y].Elem = elem, elem
			b.Out(b.Emit(graph.OpMap, []graph.ValueID{x, y}, []graph.Result{{Name: "eq", Width: n}}, nil, graph.Attrs{Fn: graph.MapEqual}, nil, nil)[0])
			_, codes, err := Default().Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: cpuDevice()})
			if err != nil {
				t.Fatalf("ints %v width %d: %v", ints, n, err)
			}
			xs, ys := make([]uint32, n), make([]uint32, n)
			for i := range xs {
				xs[i] = uint32(rng.Intn(4))
				ys[i] = uint32(rng.Intn(4))
				if ints {
					// Words a float conversion would round together.
					xs[i] += 1 << 25
					ys[i] += 1 << 25
					continue
				}
				fx := []float32{0, float32(math.Copysign(0, -1)), 1, float32(math.NaN())}
				xs[i], ys[i] = math.Float32bits(fx[xs[i]]), math.Float32bits(fx[ys[i]])
			}
			vals := map[graph.ValueID][]uint32{x: xs, y: ys}
			if err := runCPUValues(codes, b.G, vals, [3][]uint32{}); err != nil {
				t.Fatalf("ints %v width %d: %v", ints, n, err)
			}
			got := wordsF32(vals[b.G.Out[0]])
			for i := range n {
				want := float32(0)
				if ints && xs[i] == ys[i] || !ints && math.Float32frombits(xs[i]) == math.Float32frombits(ys[i]) {
					want = 1
				}
				if got[i] != want {
					t.Fatalf("ints %v width %d: element %d is %v, want %v (%#x, %#x)", ints, n, i, got[i], want, xs[i], ys[i])
				}
				if off := ints && xs[i]+1 == ys[i]; off != (got[i] == 1) {
					parted++
				}
			}
			ran++
		}
	}
	if parted == 0 {
		t.Fatal("the equality one id off agrees with the form everywhere: the gate cannot see it")
	}
	t.Logf("%d widths held on the CPU device; one id off, %d elements part", ran, parted)
}

// TestComposedCPURowsMatVec forms a row-selected matvec (OpMatVec with a
// selection) on the CPU device over every packed type at ragged selection
// widths, an id past the matrix among them (read as its last row), held to
// the oracle's matvec at the selected rows, guard words past the output.
// With the weight's words moved by one, and with the selection one row off,
// it must part.
func TestComposedCPURowsMatVec(t *testing.T) {
	const k, rows = 256, 300
	rng := rand.New(rand.NewSource(9))
	right := 0
	for _, typ := range append([]quant.Type{quant.F32}, quant.PackedTypes...) {
		q, ok := kernels.QuantOf(typ)
		if typ == quant.F32 {
			q, ok = kernels.Float32, true
		}
		jt, jok := jlm.TypeOf(typ)
		if !ok || !jok {
			continue
		}
		w := cpuTestWeights(typ, q, rows, k, rng)
		qs, d, sc, err := kernels.PackWeights(q, w, rows, k)
		if err != nil {
			t.Fatalf("%v: %v", typ, err)
		}
		planes := [3][]uint32{qs, d, sc}
		for _, arm := range []struct {
			name  string
			r     *Registry
			exact bool
		}{{"float", Default().WithoutInt8(DeviceCPU), true}, {"int8", withCPUInt8(), true}, {"int8 inexact division", withCPUInt8(), false}} {
			dev := cpuDevice()
			dev.ExactDiv = arm.exact
			for _, p := range []int{1, 3, 8, 17, 64} {
				name := fmt.Sprintf("%v/%s/%d ids", typ, arm.name, p)
				b := graph.New()
				x := b.In("x", k)
				ids := b.In("ids", p)
				b.G.Values[ids].Elem = graph.ElemU32
				b.Out(b.Emit(graph.OpMatVec, []graph.ValueID{x, ids}, []graph.Result{{Name: "picked", Width: p}},
					[]graph.Binding{{Role: jlm.RoleOutput, K: k, Rows: rows}}, graph.Attrs{}, nil, nil)[0])
				_, codes, err := arm.r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
					Step: Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				xs := make([]float32, k)
				for i := range xs {
					xs[i] = float32(rng.NormFloat64())
				}
				ref := xs
				if arm.r.Profiles[DeviceCPU].Int8 != nil && typ != quant.F32 {
					ref = int8Row(xs, 32)
				}
				full := make([]float32, rows)
				if err := oracle.MatVec(full, typ, w, ref, rows, k); err != nil {
					t.Fatal(err)
				}
				sel := make([]uint32, p)
				for i := range sel {
					sel[i] = uint32(rng.Intn(rows))
				}
				sel[p-1] = rows + 5
				run := func(pl [3][]uint32, s []uint32) []float32 {
					vals := map[graph.ValueID][]uint32{x: f32Words(xs), ids: s}
					if err := runCPUValues(codes, b.G, vals, pl); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					return wordsF32(vals[b.G.Out[0]])
				}
				want := make([]float32, p)
				for i, id := range sel {
					want[i] = full[min(int(id), rows-1)]
				}
				if e := cpuNMSE(run(planes, sel), want); math.IsNaN(e) || !(e < 1e-9) {
					t.Fatalf("%s: NMSE %.3e against the oracle", name, e)
				}
				moved := [3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), d, sc}
				if e := cpuNMSE(run(moved, sel), want); !(e > 1e-6) {
					t.Fatalf("%s: with the weight's words moved by one the NMSE is %.3e", name, e)
				}
				next := make([]uint32, p)
				for i, id := range sel {
					next[i] = (id + 1) % rows
				}
				if e := cpuNMSE(run(planes, next), want); !(e > 1e-6) {
					t.Fatalf("%s: with the selection one row off the NMSE is %.3e", name, e)
				}
				right++
			}
		}
	}
	t.Logf("%d row-selected matvecs held on the CPU device", right)
}
