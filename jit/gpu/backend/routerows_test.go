package backend_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestRouteRowsMatchOneRow routes R tokens in one launch of each router kernel
// (MoERoute.Rows) and each token alone, and demands the same selection and the
// same weights bit for bit -- every route shape, including the V3 family's
// grouped one, whose two extra planes (group scores, masked selection) are per
// row. A batched mixture prefill (tier's moegroup.go) is exactly the Rows arm.
//
// The rows differ on purpose. Row r's logits are drawn independently, so a
// kernel that ranked every row against row 0's group scores -- the plane
// stride left out -- selects different experts and fails here at once, where
// identical rows would pass it.
func TestRouteRowsMatchOneRow(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	routes := []oldkernels.MoERoute{
		{NExpert: 64, K: 6, Norm: true}, // qwen3moe, olmoe-like
		{NExpert: 64, K: 6},             // DeepSeek-V2-Lite: softmax, no renorm
		{NExpert: 64, K: 6, Sigmoid: true, Bias: true, Norm: true, Scale: 2.446}, // ungrouped V3
		{NExpert: 64, K: 6, Sigmoid: true, Bias: true, NGroup: 8, NGroupUsed: 4, Norm: true, Scale: 2.5},
		{NExpert: 32, K: 4, Sigmoid: true, NGroup: 4, NGroupUsed: 2, Norm: true},
		{NExpert: 16, K: 2, SparseMixer: 0.01}, // Phi-3.5-MoE
	}
	const R = 24
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for i, rt := range routes {
				t.Run(fmt.Sprintf("route%d", i), func(t *testing.T) {
					routeRowsCase(t, d, rt, R, rand.New(rand.NewSource(int64(20260927+i))))
				})
			}
		})
	}
}

func routeRowsCase(t *testing.T, d backend.Device, rt oldkernels.MoERoute, R int, rng *rand.Rand) {
	n, k := rt.NExpert, rt.K
	l := make([]float32, R*n)
	for i := range l {
		l[i] = float32(rng.NormFloat64() * 2)
	}
	// A few exact ties inside a row, which the lexicographic rule decides.
	l[3], l[5] = l[7], l[7]
	bias := make([]float32, n)
	for i := range bias {
		bias[i] = float32(rng.NormFloat64() * 0.1)
	}
	g := newGPU(t, d)
	defer g.free()
	compile := func(mk func() (*ir.Kernel, error)) backend.Kernel {
		t.Helper()
		kk, err := mk()
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(kk)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { c.Close() })
		return c
	}
	launch := func(c backend.Kernel, grid, width int, args ...backend.Buf) {
		t.Helper()
		if err := c.Launch(grid, width, args...); err != nil {
			t.Fatal(err)
		}
	}
	// route runs the whole router over `rows` tokens of logits and returns the
	// k selected ids and k weights of each, as raw bytes.
	route := func(rows int, logits []float32) (sel, w []byte) {
		r := rt
		r.Rows = rows
		bL := g.up(f32bytes(logits))
		bB := g.up(f32bytes(bias))
		bSel := g.up(make([]byte, rows*(k+1)*4))
		bTop := g.up(make([]byte, rows*(k+1)*4))
		bW := g.up(make([]byte, rows*k*4))
		blocks := func(threads int) int { return (threads + 127) / 128 }
		switch {
		case r.Grouped():
			bGS := g.up(make([]byte, rows*r.NGroup*4))
			bBM := g.up(make([]byte, rows*n*4))
			gs := compile(func() (*ir.Kernel, error) { return oldkernels.ExpertGroupScore(r) })
			gm := compile(func() (*ir.Kernel, error) { return oldkernels.ExpertGroupMask(r) })
			if r.Bias {
				launch(gs, blocks(rows*r.NGroup), 128, bL, bGS, bB)
				launch(gm, blocks(rows*n), 128, bL, bGS, bBM, bB)
			} else {
				launch(gs, blocks(rows*r.NGroup), 128, bL, bGS)
				launch(gm, blocks(rows*n), 128, bL, bGS, bBM)
			}
			launch(compile(func() (*ir.Kernel, error) { return oldkernels.ExpertRank(r) }),
				blocks(rows*n), 128, bL, bSel, bTop, bBM)
		case r.Bias:
			launch(compile(func() (*ir.Kernel, error) { return oldkernels.ExpertRank(r) }),
				blocks(rows*n), 128, bL, bSel, bTop, bB)
		default:
			launch(compile(func() (*ir.Kernel, error) { return oldkernels.ExpertRank(r) }),
				blocks(rows*n), 128, bL, bSel, bTop)
		}
		wk := compile(func() (*ir.Kernel, error) { return oldkernels.ExpertWeights(r) })
		grid, width := 1, 1
		if rows > 1 {
			grid, width = (rows+63)/64, 64
		}
		if !r.Norm && !r.Sigmoid {
			launch(wk, grid, width, bTop, bW, bL)
		} else {
			launch(wk, grid, width, bTop, bW)
		}
		selAll, w := make([]byte, rows*(k+1)*4), make([]byte, rows*k*4)
		if err := bSel.Read(selAll); err != nil {
			t.Fatal(err)
		}
		if err := bW.Read(w); err != nil {
			t.Fatal(err)
		}
		// Slot k of each row is ExpertRank's bin, raced by the unselected
		// threads; only the k real slots are compared.
		for r := 0; r < rows; r++ {
			sel = append(sel, selAll[r*(k+1)*4:(r*(k+1)+k)*4]...)
		}
		return sel, w
	}
	selR, wR := route(R, l)
	for r := 0; r < R; r++ {
		sel1, w1 := route(1, l[r*n:(r+1)*n])
		if !bytes.Equal(selR[r*k*4:(r+1)*k*4], sel1) {
			t.Fatalf("row %d: batched selection %v, alone %v", r, u32s(selR[r*k*4:(r+1)*k*4]), u32s(sel1))
		}
		if !bytes.Equal(wR[r*k*4:(r+1)*k*4], w1) {
			t.Fatalf("row %d: batched weights differ from the row alone", r)
		}
	}
}

func u32s(b []byte) []uint32 {
	v := make([]uint32, len(b)/4)
	for i := range v {
		v[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return v
}
