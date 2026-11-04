package composer

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// windowOrder is transformers' Qwen2_5_VisionTransformerPretrainedModel
// .get_window_index for one picture of gh x gw merge units: the grid padded
// to whole windows of win units a side -- a whole window more where a side is
// already whole, as its pad_h and pad_w are -- read window by window, each
// window's units row by row, the pad dropped; win 0 is raster order.
func windowOrder(gh, gw, win int) []int {
	var out []int
	if win == 0 {
		for u := range gh * gw {
			out = append(out, u)
		}
		return out
	}
	nh, nw := (gh+win-gh%win)/win, (gw+win-gw%win)/win
	for wy := range nh {
		for wx := range nw {
			for y := range win {
				for x := range win {
					if gy, gx := wy*win+y, wx*win+x; gy < gh && gx < gw {
						out = append(out, gy*gw+gx)
					}
				}
			}
		}
	}
	return out
}

// TestComposedCPUPatchifyTiles gathers a picture's patches in nested order
// on the CPU (graph.OpPatchify's Width and Window): each row at its own
// index the patch transformers' order puts there -- the merge windows'
// patches together (Qwen2-VL's processor), the merge windows in Qwen2.5-VL's
// windows with those at the edges cut (get_window_index). The grid stated a
// merge window narrower is the violation it must see.
func TestComposedCPUPatchifyTiles(t *testing.T) {
	const P, M = 2, 2
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	for _, c := range []struct{ gh, gw, win int }{
		{4, 4, 0}, {6, 10, 0}, {2, 2, 4}, {8, 8, 4}, {12, 16, 4}, {6, 10, 4}, {14, 6, 4}, {10, 18, 2},
	} {
		np, k := c.gh*c.gw, 3*P*P
		b := graph.NewBlock(-1)
		px := b.In("pixels", k)
		b.Out(b.Emit(graph.OpPatchify, []graph.ValueID{px}, []graph.Result{{Name: "patches", Width: k}}, nil,
			graph.Attrs{Span: P, Width: M, Window: c.win}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: np, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		pix := make([]float32, np*k)
		for i := range pix {
			pix[i] = float32(i%9973) + 0.5
		}
		pos := make([]uint32, np+1)
		pos[0] = uint32(np)
		for i := range np {
			pos[1+i] = uint32(i)
		}
		gh2, gw2 := c.gh/M, c.gw/M
		want := make([]float32, np*k)
		i := 0
		for _, u := range windowOrder(gh2, gw2, c.win) {
			for j := range M * M {
				py, pxi := u/gw2*M+j/M, u%gw2*M+j%M
				for e := range k {
					ch, ky, kx := e/(P*P), e/P%P, e%P
					want[i*k+e] = pix[((py*P+ky)*c.gw*P+pxi*P+kx)*3+ch]
				}
				i++
			}
		}
		if i != np {
			t.Fatalf("%+v: the reference ordered %d of %d patches", c, i, np)
		}
		run := func(gw int) []float32 {
			req := make([]uint32, RequestRows+np)
			req[RequestGrid] = uint32(gw)
			got, err := runCPUBound(hostArm(), codes, pix, np*k, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "rope.pos":
					return pos
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%+v: %v", c, err)
			}
			return got
		}
		got := run(c.gw)
		for e := range want {
			if math.Float32bits(got[e]) != math.Float32bits(want[e]) {
				t.Fatalf("%+v: element %d (row %d) is %v, want %v", c, e, e/k, got[e], want[e])
			}
		}
		bad, same := run(c.gw-M), true
		for e := range want {
			same = same && bad[e] == want[e]
		}
		if same {
			t.Fatalf("VIOLATION: %+v: a grid stated a merge window narrower gathers the same rows", c)
		}
		t.Logf("%dx%d patches, %dx%d merge windows in windows of %d: every row the reference's patch", c.gh, c.gw, M, M, c.win)
	}
}

// TestComposedCPUUntile puts a picture's merge windows, laid out in
// Qwen2.5-VL's window order (get_window_index), back in raster order on the
// CPU (graph.OpUntile): each output row the input row of the cell at its
// raster index, the windows at the edges cut. Windows stated one narrower,
// which order the rows otherwise, are the violation it must see.
func TestComposedCPUUntile(t *testing.T) {
	const M, E = 2, 3
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	for _, c := range []struct{ gh, gw, win int }{{3, 4, 4}, {6, 8, 4}, {7, 5, 4}, {9, 9, 2}, {1, 1, 4}} {
		n := c.gh * c.gw
		form := func(win int) []Code {
			b := graph.NewBlock(-1)
			x := b.In("cells", E)
			b.G.Values[x].Rows = n
			b.Out(b.Emit(graph.OpUntile, []graph.ValueID{x}, []graph.Result{{Name: "raster", Width: E, Rows: n}}, nil,
				graph.Attrs{Width: M, Window: win}, nil, nil)[0])
			if err := graph.Verify(b.G, graph.Facts{}); err != nil {
				t.Fatal(err)
			}
			_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: n * M * M, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
			if err != nil {
				t.Fatalf("%+v: %v", c, err)
			}
			return codes
		}
		in := make([]float32, n*E)
		for i, cell := range windowOrder(c.gh, c.gw, c.win) {
			for e := range E {
				in[i*E+e] = float32(cell*E+e) + 0.5
			}
		}
		run := func(codes []Code) []float32 {
			req := make([]uint32, RequestRows+n)
			req[RequestGrid] = uint32(c.gw * M)
			pos := make([]uint32, n*M*M+1)
			pos[0] = uint32(n * M * M)
			got, err := runCPUBound(hostArm(), codes, in, n*E, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch o.Name {
				case "request":
					return req
				case "rope.pos":
					return pos
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%+v: %v", c, err)
			}
			return got
		}
		got := run(form(c.win))
		for i, v := range got {
			if want := float32(i) + 0.5; v != want {
				t.Fatalf("%+v: element %d (row %d) is %v, want %v", c, i, i/E, v, want)
			}
		}
		if other := windowOrder(c.gh, c.gw, c.win-1); c.win > 1 && !slices.Equal(other, windowOrder(c.gh, c.gw, c.win)) {
			bad, same := run(form(c.win-1)), true
			for i, v := range bad {
				same = same && v == got[i]
			}
			if same {
				t.Fatalf("VIOLATION: %+v: windows stated one narrower untile the same rows", c)
			}
		}
		t.Logf("%dx%d merge windows in windows of %d: every row back at its raster place", c.gh, c.gw, c.win)
	}
}
