package composer

import (
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUPosBuckets reads a 7x7 table, each row its own index,
// through graph.OpPosBuckets on the CPU at grids smaller, larger and uneven
// against it, and holds every row to the bucket NaViT's rational names:
// table row floor(7*y/gh)*7 + floor(7*x/gw). The cell's own index, unbucketed,
// is the violation.
func TestComposedCPUPosBuckets(t *testing.T) {
	const S, e = 7, 2
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	tab := make([]float32, S*S*e)
	for i := range S * S {
		tab[i*e], tab[i*e+1] = float32(i), float32(-i)
	}
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&tab[0])), len(tab))
	run := func(gh, gw int) []float32 {
		n := gh * gw
		b := graph.NewBlock(-1)
		b.In("x", e)
		b.Out(b.Emit(graph.OpPosBuckets, nil, []graph.Result{{Name: "pos", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: e, Rows: S * S}}, graph.Attrs{Span: S}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVPosEmbd: jlm.TypeF32},
			Step: Step{Rows: n, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%dx%d: %v", gh, gw, err)
		}
		pos := make([]uint32, n+1)
		pos[0] = uint32(n)
		for i := range n {
			pos[1+i] = uint32(i)
		}
		req := make([]uint32, RequestRows+n)
		req[RequestGrid], req[RequestGridH] = uint32(gw), uint32(gh)
		got, err := runCPUBound(hostArm(), codes, make([]float32, n*e), n*e, [3][]uint32{words}, nil, func(o Operand) []uint32 {
			switch o.Name {
			case "request":
				return req
			case "rope.pos":
				return pos
			}
			return nil
		})
		if err != nil {
			t.Fatalf("%dx%d: %v", gh, gw, err)
		}
		return got
	}
	plain := 0
	for _, g := range [][2]int{{3, 5}, {7, 7}, {10, 13}, {16, 9}, {1, 20}} {
		gh, gw := g[0], g[1]
		got := run(gh, gw)
		for y := range gh {
			for x := range gw {
				i := y*gw + x
				want := (S*y/gh)*S + S*x/gw
				if got[i*e] != float32(want) || got[i*e+1] != float32(-want) {
					t.Fatalf("%dx%d: cell (%d, %d) read table row %v, the bucket's %d", gh, gw, y, x, got[i*e], want)
				}
				if i < S*S && want != i {
					plain++
				}
			}
		}
	}
	t.Logf("five grids, every cell its bucket's row; %d cells where the unbucketed index would read another", plain)
	if plain == 0 {
		t.Fatal("VIOLATION: no cell's bucket differs from its own index")
	}
}
