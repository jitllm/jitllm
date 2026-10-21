//go:build amd64 && linux

package cpu

import (
	"math/rand"
	"syscall"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// Guard-page gates: every input span a kernel reads is placed so its last byte
// is the last byte before a PROT_NONE page. A kernel that reads one byte past
// a row, a plane or an activation faults here -- a crash, which is a red test
// -- where on a Go heap slice the same over-read returns whatever the next
// allocation holds and every numeric gate stays green.
//
// The tail is where an SSE port over-reads: a MOVUPS that loads four floats to
// use one reads twelve bytes past the row and still computes the right
// answer, so only a fault can see it.

// guardAlloc returns n bytes whose end abuts a PROT_NONE page, and the unmap.
func guardAlloc(t *testing.T, n int) []byte {
	t.Helper()
	pg := syscall.Getpagesize()
	body := (n + pg - 1) / pg * pg
	if body == 0 {
		body = pg
	}
	m, err := syscall.Mmap(-1, 0, body+pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(m[body:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { syscall.Munmap(m) })
	return m[body-n : body : body]
}

func guardF32(t *testing.T, src []float32) []float32 {
	b := guardAlloc(t, 4*len(src))
	f := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(src))
	copy(f, src)
	return f
}

func guardI8(t *testing.T, src []int8) []int8 {
	b := guardAlloc(t, len(src))
	f := unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), len(src))
	copy(f, src)
	return f
}

func guardBytes(t *testing.T, src []byte) []byte {
	b := guardAlloc(t, len(src))
	copy(b, src)
	return b
}

// TestSSEFloatMatVecReadsNothingPastItsRow runs the float matvec with its
// weights and its activation each ending at a guard page, at every k shape.
func TestSSEFloatMatVecReadsNothingPastItsRow(t *testing.T) {
	for _, typ := range sseFloatTypes {
		code := sseFloatKernel(t, typ)
		c := mustMap(t, code)
		es := int(typ.BlockBytes())
		for _, k := range sseFloatKs {
			const rows = 3
			rng := rand.New(rand.NewSource(int64(k)))
			w, _ := floatRows(rng, typ, rows, k, false)
			x := floatActs(rng, k, false)
			gw, gx := guardBytes(t, w), guardF32(t, x)
			out := make([]float32, rows)
			c.Call(&Args{Out: &out[0], W: &gw[0], A: (*int8)(unsafe.Pointer(&gx[0])),
				Rows: rows, K: int64(k), RowStr: int64(k * es)})
		}
		c.Close()
	}
}

// TestSSEPackedReadsNothingPastItsSpans runs the tile, tail and fused kernels
// over the last rows of a tensor whose payload, scale planes, activations,
// scratch and constant block each end at a guard page.
func TestSSEPackedReadsNothingPastItsSpans(t *testing.T) {
	for _, g := range quant.PackedTypes {
		tile, tail, fused := ssePackedKernels(t, g)
		p := newSSEPackFix(t, g, ssePackRows, ssePackK, 5, false)
		p.qb, p.db = guardBytes(t, p.qb), guardBytes(t, p.db)
		if len(p.scb) > 0 {
			p.scb = guardBytes(t, p.scb)
		}
		p.q8, p.pairs, p.half = guardI8(t, p.q8), guardF32(t, p.pairs), guardF32(t, p.half)
		p.scr = guardBytes(t, p.scr)

		ref := p.oracle(t)
		check := func(name string, got []float32, r0 int) {
			t.Helper()
			if nmse := nmseOf(t, got, ref[r0:r0+len(got)]); nmse > 1e-3 {
				t.Fatalf("%s %s: NMSE %.3e on the last rows", g, name, nmse)
			}
		}
		// The last 64-row tile, then the two 8-row tails that end the tensor.
		r := p.nrows/PackedRows*PackedRows - PackedRows
		check("tile", p.runTile(t, tile, PackedRows, r, 1), r)
		r = p.nrows / PackedRows * PackedRows
		check("tail", p.runTile(t, tail, PackedTail, r, (p.nrows-r)/PackedTail), r)

		// The fused kernel's last two groups, with its scratch at a guard too.
		c := mustMap(t, fused)
		grp := PackedFusedGroupOf(g)
		r = p.nrows - 2*grp
		out := make([]float32, 2*grp)
		scratch, q32 := guardF32(t, make([]float32, 2*grp)), guardF32(t, make([]float32, 2*grp))
		a := p.args(out, r, 2, scratch, q32)
		c.Call(&a)
		c.Close()
		check("fused", out, r)
	}
}
