package composer

import (
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// linspace is torch.linspace(0, end, n) in float32 as its kernel computes
// it: the first half from the start, the second back from the end.
func linspace(end float32, n int) []float32 {
	out := make([]float32, n)
	if n == 1 {
		return out
	}
	step := end / float32(n-1)
	for i := range n {
		if i < n/2 {
			out[i] = step * float32(i)
		} else {
			out[i] = end - step*float32(n-1-i)
		}
	}
	return out
}

// posInterp is transformers' Qwen3VLVisionModel.fast_pos_embed_interpolate
// for one picture of gh x gw patches over an S x S table of e columns: each
// patch's four table rows at the floors and ceilings of linspace(0, S-1, n)
// on each axis, weighted by the fractions' products and summed in that
// order, the patches in raster order.
func posInterp(tab []float32, S, e, gh, gw int) []float32 {
	hs, ws := linspace(float32(S-1), gh), linspace(float32(S-1), gw)
	out := make([]float32, gh*gw*e)
	for y := range gh {
		for x := range gw {
			fh, fw := int(hs[y]), int(ws[x])
			ch, cw := min(fh+1, S-1), min(fw+1, S-1)
			dh, dw := hs[y]-float32(fh), ws[x]-float32(fw)
			idx := [4]int{fh*S + fw, fh*S + cw, ch*S + fw, ch*S + cw}
			wt := [4]float32{(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw}
			for k := range e {
				var s float32
				for j := range 4 {
					s += tab[idx[j]*e+k] * wt[j]
				}
				out[(y*gw+x)*e+k] = s
			}
		}
	}
	return out
}

// TestComposedCPUPosResampleFollowsTheReference resamples a learned
// position table to pictures' grids on the CPU (graph.OpPosResample, the
// aligned chain through the linear taps) with the rows in the patch
// gather's nested order (Width, Window), each row held to Qwen3-VL's own
// interpolation (posInterp) at the cell that order puts there: a grid
// smaller than the table on one axis and larger on the other, a side of one
// patch, and Qwen2.5-VL's windows. A grid stated a merge window narrower is
// the violation it must see.
func TestComposedCPUPosResampleFollowsTheReference(t *testing.T) {
	const S, e = 8, 16
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	tab := make([]float32, S*S*e)
	for i := range tab {
		tab[i] = float32(i%89)*0.25 - 11
	}
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&tab[0])), len(tab))
	for _, c := range []struct{ gh, gw, m, win int }{
		{6, 10, 2, 0}, {12, 4, 2, 0}, {2, 2, 2, 0}, {5, 7, 1, 0}, {1, 9, 1, 0}, {16, 20, 2, 4}, {10, 6, 2, 2},
	} {
		n := c.gh * c.gw
		b := graph.NewBlock(-1)
		b.In("x", e)
		b.Out(b.Emit(graph.OpPosResample, nil, []graph.Result{{Name: "pos", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: e, Rows: S * S}},
			graph.Attrs{Span: S, Width: c.m, Window: c.win, Chain: graph.ChainAligned, Filter: graph.FilterLinear}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVPosEmbd: jlm.TypeF32},
			Step: Step{Rows: n, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		pos := make([]uint32, n+1)
		pos[0] = uint32(n)
		for i := range n {
			pos[1+i] = uint32(i)
		}
		run := func(gw int) []float32 {
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
				t.Fatalf("%+v: %v", c, err)
			}
			return got
		}
		raster := posInterp(tab, S, e, c.gh, c.gw)
		want := make([]float32, n*e)
		gh2, gw2, i := c.gh/c.m, c.gw/c.m, 0
		for _, u := range windowOrder(gh2, gw2, c.win) {
			for j := range c.m * c.m {
				py, px := u/gw2*c.m+j/c.m, u%gw2*c.m+j%c.m
				copy(want[i*e:(i+1)*e], raster[(py*c.gw+px)*e:])
				i++
			}
		}
		got := run(c.gw)
		worst := 0.0
		for k := range want {
			d := math.Abs(float64(got[k] - want[k]))
			if !(d <= 1e-5*(1+math.Abs(float64(want[k])))) {
				t.Fatalf("%+v: row %d element %d is %v, want %v", c, k/e, k%e, got[k], want[k])
			}
			worst = max(worst, d)
		}
		if c.m > 1 {
			bad, same := run(c.gw-c.m), true
			for k := range want {
				same = same && bad[k] == got[k]
			}
			if same {
				t.Fatalf("VIOLATION: %+v: a grid stated a merge window narrower resamples the same rows", c)
			}
		}
		t.Logf("%dx%d patches (merge %d, windows %d): every row the reference's, max |d| %.2g", c.gh, c.gw, c.m, c.win, worst)
	}
}

// cubicTaps is torch's grid_sampler bicubic along one axis for cell i of n
// over a side of S (align_corners false): the source coordinate, its floor
// and the four taps' clamped indices (padding_mode border) and weights
// (get_cubic_upsampling_coefficients, A = -0.75), in float32.
func cubicTaps(i, n, S int) ([4]int, [4]float32) {
	x := (float32(i)+0.5)/float32(n)*2 - 1
	ix := ((x+1)*float32(S) - 1) / 2
	fl := float32(math.Floor(float64(ix)))
	t := ix - fl
	const A = -0.75
	c1 := func(x float32) float32 { return ((A+2)*x-(A+3))*x*x + 1 }
	c2 := func(x float32) float32 { return ((A*x-5*A)*x+8*A)*x - 4*A }
	w := [4]float32{c2(t + 1), c1(t), c1(1 - t), c2(1 - t + 1)}
	var idx [4]int
	for k := range 4 {
		idx[k] = min(max(int(fl)-1+k, 0), S-1)
	}
	return idx, w
}

// posCubic is transformers' Glm4vVisionEmbeddings' table for gh x gw patches
// over an S x S table of e columns: F.grid_sample bicubic, each row of taps
// along the width, then the four along the height, the patches in raster
// order.
func posCubic(tab []float32, S, e, gh, gw int) []float32 {
	out := make([]float32, gh*gw*e)
	for y := range gh {
		iy, wy := cubicTaps(y, gh, S)
		for x := range gw {
			ix, wx := cubicTaps(x, gw, S)
			for k := range e {
				var acc float32
				for a := range 4 {
					var row float32
					for b := range 4 {
						row += tab[(iy[a]*S+ix[b])*e+k] * wx[b]
					}
					acc += row * wy[a]
				}
				out[(y*gw+x)*e+k] = acc
			}
		}
	}
	return out
}

// TestComposedCPUPosResampleCubic resamples a learned position table to
// pictures' grids bicubic on the CPU (graph.OpPosResample, the half-pixel
// chain through the cubic taps), the rows in the merge windows' order, each
// held to GLM-4V's own (posCubic): grids smaller and larger than the table,
// a side of one window. A grid stated a window narrower is the violation.
func TestComposedCPUPosResampleCubic(t *testing.T) {
	const S, e = 8, 16
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	tab := make([]float32, S*S*e)
	for i := range tab {
		tab[i] = float32(i%83)*0.25 - 10
	}
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&tab[0])), len(tab))
	for _, c := range []struct{ gh, gw, m int }{{6, 10, 2}, {12, 4, 2}, {2, 2, 2}, {16, 20, 2}, {5, 7, 1}} {
		n := c.gh * c.gw
		b := graph.NewBlock(-1)
		b.In("x", e)
		b.Out(b.Emit(graph.OpPosResample, nil, []graph.Result{{Name: "pos", Width: e}},
			[]graph.Binding{{Role: jlm.RoleVPosEmbd, K: e, Rows: S * S}},
			graph.Attrs{Span: S, Width: c.m, Chain: graph.ChainHalfPixel, Filter: graph.FilterCubic}, nil, nil)[0])
		_, codes, err := r.Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVPosEmbd: jlm.TypeF32},
			Step: Step{Rows: n, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		pos := make([]uint32, n+1)
		pos[0] = uint32(n)
		for i := range n {
			pos[1+i] = uint32(i)
		}
		run := func(gw int) []float32 {
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
				t.Fatalf("%+v: %v", c, err)
			}
			return got
		}
		raster := posCubic(tab, S, e, c.gh, c.gw)
		want := make([]float32, n*e)
		gh2, gw2, i := c.gh/c.m, c.gw/c.m, 0
		for _, u := range windowOrder(gh2, gw2, 0) {
			for j := range c.m * c.m {
				py, px := u/gw2*c.m+j/c.m, u%gw2*c.m+j%c.m
				copy(want[i*e:(i+1)*e], raster[(py*c.gw+px)*e:])
				i++
			}
		}
		got := run(c.gw)
		worst := 0.0
		for k := range want {
			d := math.Abs(float64(got[k] - want[k]))
			if !(d <= 1e-4*(1+math.Abs(float64(want[k])))) {
				t.Fatalf("%+v: row %d element %d is %v, want %v", c, k/e, k%e, got[k], want[k])
			}
			worst = max(worst, d)
		}
		if c.m > 1 {
			bad, same := run(c.gw-c.m), true
			for k := range want {
				same = same && bad[k] == got[k]
			}
			if same {
				t.Fatalf("VIOLATION: %+v: a grid stated a window narrower resamples the same rows", c)
			}
		}
		t.Logf("%dx%d patches (merge %d): every row grid_sample's bicubic, max |d| %.2g", c.gh, c.gw, c.m, worst)
	}
}
