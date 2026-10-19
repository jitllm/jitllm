package backend_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestComposedPagedWritesMatchTheMenu holds the paged history's row writes,
// composed (PagedPos and an indexed store), to PagedCopyRowsT (K, transposed)
// and PagedCopyRows (V, float32 and packed binary16) bit for bit, over rows
// whose write positions fall on different pages of a scattered table; a
// write position read from the descriptor's start word is its violation.
func TestComposedPagedWritesMatchTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, width, page, npage = 3, 96, 64, 6
	src := make([]float32, rows*width)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.23)) * 3
	}
	u32 := func(v ...uint32) []byte {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], x)
		}
		return b
	}
	tab := u32(5, 2, 4, 0, 3, 1)
	// tab offset, start, end, write position a row: positions on pages 1, 0
	// and 2 of the row's table.
	desc := u32(0, 7, 71, 70, 0, 9, 9, 9, 2, 10, 140, 139)
	build := func(f16 bool, kpool bool, wordFrom int) *ir.Kernel {
		w := width
		if f16 {
			w = width / 2
		}
		g := &compose.Row{Width: w, Rows: rows}
		in, d, tb := g.Buf(compose.BufRow), g.Buf(compose.BufU32), g.Buf(compose.BufU32)
		kind := compose.BufOut
		if f16 {
			kind = compose.BufOutU32
		}
		out := g.Buf(kind)
		pid, off := g.PagedPos(d, tb, page)
		if wordFrom != kernels.PRowWrite {
			base := g.UBin(compose.OpMul, g.RowIdx(), g.UConst(kernels.PRowWords))
			pos := g.LoadAt(d, g.UBin(compose.OpAdd, base, g.UConst(wordFrom)))
			first := g.LoadAt(d, g.UBin(compose.OpAdd, base, g.UConst(kernels.PRowTab)))
			pid = g.LoadAt(tb, g.UBin(compose.OpAdd, first, g.UBin(compose.OpDiv, pos, g.UConst(page))))
			off = g.UBin(compose.OpRem, pos, g.UConst(page))
		}
		switch {
		case kpool:
			g.StoreAt(out, g.Load(in), g.PagedK(pid, off, width, page))
		case f16:
			row := g.UBin(compose.OpMul, g.RowIdx(), g.UConst(width))
			e := g.UBin(compose.OpAdd, row, g.UBin(compose.OpMul, g.Column(), g.UConst(2)))
			x := g.Bin(compose.OpPackF16, g.LoadAt(in, e), g.LoadAt(in, g.UBin(compose.OpAdd, e, g.UConst(1))))
			at := g.UBin(compose.OpAdd, g.UBin(compose.OpDiv, g.PagedRow(pid, off, page, width, 0), g.UConst(2)), g.Column())
			g.StoreAt(out, x, at)
		default:
			g.StoreAt(out, g.Load(in), g.UBin(compose.OpAdd, g.PagedRow(pid, off, page, width, 0), g.Column()))
		}
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
			run := func(ker *ir.Kernel, groups int, menu bool) []byte {
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				dst := gp.up(make([]byte, pool))
				args := []backend.Buf{gp.up(f32bytes(src)), dst, gp.up(u32(0)), gp.up(tab), gp.up(desc)}
				if !menu {
					args = []backend.Buf{gp.up(f32bytes(src)), gp.up(desc), gp.up(tab), dst}
				}
				if err := c.Launch(groups, 128, args...); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, pool)
				if err := dst.Read(raw); err != nil {
					t.Fatal(err)
				}
				return raw
			}
			for _, c := range []struct {
				name       string
				f16, kpool bool
			}{{"k", false, true}, {"v", false, false}, {"v-f16", true, false}} {
				var menu *ir.Kernel
				var err error
				n := rows * width
				if c.kpool {
					menu, err = kernels.PagedCopyRowsT(width, rows, page)
				} else {
					menu, err = kernels.PagedCopyRows(width, rows, page, width, 0, c.f16)
					if c.f16 {
						n /= 2
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				want := run(menu, (n+127)/128, true)
				if got := run(build(c.f16, c.kpool, kernels.PRowWrite), rows, false); !bytes.Equal(got, want) {
					t.Errorf("%s: the composed paged write parts from the menu's", c.name)
				}
				if got := run(build(c.f16, c.kpool, kernels.PRowStart), rows, false); bytes.Equal(got, want) {
					t.Errorf("%s: a write at the row's start position matched", c.name)
				}
			}
		})
	}
}
