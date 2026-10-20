package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestComposedLatentWriteMatchesTheMenu holds a latent row's write into its
// paged history, composed as two row groups (the latent at the row's first
// columns; the positional key turned by the rotary and stored after it), to
// the menu's three launches (RoPERows into a staging row, then PagedCopyRows
// for each part) bit for bit on PTX and within a few ulps where a driver
// contracts at its own whim, in NORM and NEOX pairs, over rows whose
// positions fall on different pages and a pool row wider than the two parts
// (an indexer's key after them). Its violation turns the key in the other
// pair layout.
func TestComposedLatentWriteMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, lat, rot, width, page, npage = 3, 64, 16, 96, 64, 6
	latv, kpe, cs := make([]float32, rows*lat), make([]float32, rows*rot), make([]float32, rows*rot)
	for i := range latv {
		latv[i] = float32(math.Sin(float64(i)*0.37)) * 2
	}
	for i := range kpe {
		kpe[i] = float32(math.Cos(float64(i)*0.29)) * 3
	}
	for r := 0; r < rows; r++ {
		for p := 0; p < rot/2; p++ {
			a := float64(r+5) / math.Pow(10000, float64(2*p)/rot)
			cs[r*rot+2*p], cs[r*rot+2*p+1] = float32(math.Cos(a)), float32(math.Sin(a))
		}
	}
	u32 := func(v ...uint32) []byte {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], x)
		}
		return b
	}
	tab := u32(5, 2, 4, 0, 3, 1)
	desc := u32(0, 7, 71, 70, 0, 9, 9, 9, 2, 10, 140, 139)
	part := func(w, col int, rotate, neox bool) *ir.Kernel {
		g := &compose.Row{Width: w, Rows: rows}
		in := g.Buf(compose.BufRow)
		x := -1
		var c int
		if rotate {
			c = g.Buf(compose.BufVec)
			x = g.Rotary(in, c, rot, rot, neox, false)
		}
		d, tb := g.Buf(compose.BufU32), g.Buf(compose.BufU32)
		pid, off := g.PagedPos(d, tb, page)
		if x < 0 {
			x = g.Load(in)
		}
		g.StoreAt(g.Buf(compose.BufOut), x, g.UBin(compose.OpAdd, g.PagedRow(pid, off, page, width, col), g.Column()))
		k, err := g.Emit(compose.Schedule{Group: 128})
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			gp := newGPU(t, d)
			defer gp.free()
			pool := npage * page * width * 4
			launch := func(ker *ir.Kernel, groups int, args ...backend.Buf) {
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				if err := c.Launch(groups, 128, args...); err != nil {
					t.Fatal(err)
				}
			}
			read := func(b backend.Buf) []byte {
				raw := make([]byte, pool)
				if err := b.Read(raw); err != nil {
					t.Fatal(err)
				}
				return raw
			}
			for _, neox := range []bool{false, true} {
				// The menu: the key turned into staging at r*rot, then each part
				// copied into the pool.
				dst, stage := gp.up(make([]byte, pool)), gp.up(make([]byte, rows*rot*4))
				offs := u32(0, rot, 2*rot)
				rope, err := mainkernels.RoPERows(1, rot, rot, neox, rows)
				if err != nil {
					t.Fatal(err)
				}
				launch(rope, (rows*rot/2+127)/128, gp.up(f32bytes(kpe)), gp.up(f32bytes(cs)), gp.up(offs), stage)
				cl, err := mainkernels.PagedCopyRows(lat, rows, page, width, 0, false)
				if err != nil {
					t.Fatal(err)
				}
				launch(cl, (rows*lat+127)/128, gp.up(f32bytes(latv)), dst, gp.up(u32(0)), gp.up(tab), gp.up(desc))
				cp, err := mainkernels.PagedCopyRows(rot, rows, page, width, lat, false)
				if err != nil {
					t.Fatal(err)
				}
				launch(cp, (rows*rot+127)/128, stage, dst, gp.up(u32(0)), gp.up(tab), gp.up(desc))
				want := read(dst)
				composed := func(layout bool) []byte {
					out := gp.up(make([]byte, pool))
					launch(part(lat, 0, false, layout), rows, gp.up(f32bytes(latv)), gp.up(desc), gp.up(tab), out)
					launch(part(rot, lat, true, layout), rows, gp.up(f32bytes(kpe)), gp.up(f32bytes(cs)), gp.up(desc), gp.up(tab), out)
					return read(out)
				}
				// Bit for bit where the backend contracts as the menu does (PTX);
				// within a few ulps of the inputs' largest magnitude (3) where a
				// driver fuses a multiply and an add at its own whim (SPIR-V), as
				// TestComposedRotaryMatchesRoPERows.
				worst := func(got []byte) float64 {
					w := 0.0
					for i := 0; i < len(got); i += 4 {
						a := math.Float32frombits(binary.LittleEndian.Uint32(got[i:]))
						b := math.Float32frombits(binary.LittleEndian.Uint32(want[i:]))
						w = math.Max(w, math.Abs(float64(a-b)))
					}
					return w
				}
				bound := 0.0
				if d.API() != "ptx" {
					bound = 4 * 3 * 1.2e-7
				}
				if w := worst(composed(neox)); w > bound {
					t.Errorf("neox=%v: the composed latent write is %.3g from the menu's (bound %.3g)", neox, w, bound)
				}
				if w := worst(composed(!neox)); w <= bound {
					t.Errorf("neox=%v: the key turned in the other pair layout is within %.3g", neox, bound)
				}
			}
		})
	}
}
