package composer

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUPosResampleAntialias resizes a 16x16 table of a known
// formula onto grids up, down and unequal through graph.FilterAntialias on
// the CPU, and holds each row to torch's F.interpolate(bilinear,
// antialias=True) (scripts/phi4visiongold.py --model-free). A reduction is
// where the window widens, so the golden must hold one; plain bilinear on
// it is the violation.
func TestComposedCPUPosResampleAntialias(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "phi4-naflex.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v (scripts/phi4visiongold.py --model-free)", err)
	}
	var g struct {
		Grids []struct {
			GH, GW int
			V      []float64
		}
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	const S, e = 16, 2
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	tab := make([]float32, S*S*e)
	for y := range S {
		for x := range S {
			for c := range e {
				tab[(y*S+x)*e+c] = float32(math.Sin(0.37*float64(y) + 0.11*float64(x*c) + float64(c)))
			}
		}
	}
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&tab[0])), len(tab))
	run := func(gh, gw int, filter uint8) []float32 {
		n := gh * gw
		b := graph.NewBlock(-1)
		b.In("x", e)
		b.Out(b.Emit(graph.OpPosResample, nil, []graph.Result{{Name: "pos", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: e, Rows: S * S}},
			graph.Attrs{Span: S, Width: 1, Chain: graph.ChainHalfPixel, Filter: filter}, nil, nil)[0])
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
		req[RequestGrid] = uint32(gw)
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
	reduced, seen := 0, false
	for _, gr := range g.Grids {
		if len(gr.V) != gr.GH*gr.GW*e {
			t.Fatalf("%dx%d: the golden holds %d values", gr.GH, gr.GW, len(gr.V))
		}
		got := run(gr.GH, gr.GW, graph.FilterAntialias)
		worst := 0.0
		for i, w := range gr.V {
			worst = math.Max(worst, math.Abs(float64(got[i])-w))
		}
		t.Logf("16x16 -> %dx%d: worst |d| %.2e against torch", gr.GH, gr.GW, worst)
		if !(worst <= 1e-5) {
			t.Errorf("16x16 -> %dx%d: worst |d| %.2e against torch", gr.GH, gr.GW, worst)
		}
		if gr.GH < S || gr.GW < S {
			reduced++
			bad := run(gr.GH, gr.GW, graph.FilterBilinear)
			for i, w := range gr.V {
				seen = seen || math.Abs(float64(bad[i])-w) > 1e-3
			}
		}
	}
	if reduced == 0 {
		t.Fatal("no golden grid reduces the table, so the antialiasing window was never exercised")
	}
	if !seen {
		t.Fatal("VIOLATION: plain bilinear reduces the table as torch's antialiased resize does")
	}
}
