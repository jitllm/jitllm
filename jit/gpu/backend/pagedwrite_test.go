package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestPagedWriters holds the paged K/V writers to the page layout: rows
// written at positions on either side of page edges, through shuffled tables,
// a padded row into the dummy page, K through PagedCopyRowsT and then
// PagedRoPERowsT over a partial rotary (the tail first, as the tier does), V
// as f32 and as packed f16. Every pool word starts NaN and every word the
// writers were not meant to touch must still be NaN afterwards, so a stray
// write fails as surely as a wrong one. The same writes through an identity
// table must land somewhere else.
func TestPagedWriters(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			for _, page := range []int{64, 256} {
				for _, geo := range []struct {
					heads, hd, nRot int
					neox            bool
				}{{2, 64, 64, true}, {4, 32, 16, false}, {1, 128, 32, true}} {
					for _, f16 := range []bool{false, true} {
						// binary16 K (FlashShape.F16K) beside f32 K, on the
						// other parity of V.
						f16k := f16 != geo.neox
						t.Run(fmt.Sprintf("P%d/h%d/d%d/rot%d/neox%v/f16%v/f16k%v", page, geo.heads, geo.hd, geo.nRot, geo.neox, f16, f16k), func(t *testing.T) {
							// Positions around page edges, a first and a last slot,
							// and a padded row into the dummy page.
							rows := []pagedRow{
								{0, 0, page - 1}, {0, 0, page}, {0, 0, 3*page + 7}, {0, 0, 0},
								{0, 0, 2*page - 1}, {5, 5, 5},
							}
							if err := pagedWriteCheck(t, d, page, geo.heads, geo.hd, geo.nRot, geo.neox, f16, f16k, rows); err != nil {
								t.Fatal(err)
							}
						})
					}
				}
			}
		})
	}
}

// f16k writes K as binary16 pairs (PagedCopyRowsTF16, PagedRoPERowsTF16).
func pagedWriteCheck(t *testing.T, d backend.Device, page, heads, hd, nRot int, neox, f16, f16k bool, rows []pagedRow) error {
	t.Helper()
	kvRow := heads * hd
	R := len(rows)
	rng := rand.New(rand.NewSource(int64(page*kvRow + nRot)))
	// Tables: every row's pages shuffled, the padded row (ks == ke == wp) at
	// the dummy page.
	need := 0
	for _, r := range rows {
		need += r.wp/page + 1
	}
	pages := need + 2
	perm := rng.Perm(pages)
	dummy, next := perm[0], 1
	var tab, desc []uint32
	for _, r := range rows {
		off := len(tab)
		if r.ks == r.ke && r.wp == r.ks && r.ks > 0 {
			// A padded row: one table entry, the dummy, and a slot in it.
			tab = append(tab, uint32(dummy))
			desc = append(desc, uint32(off), uint32(r.ks), uint32(r.ke), uint32(r.wp%page))
			continue
		}
		for j := 0; j <= r.wp/page; j++ {
			tab = append(tab, uint32(perm[next]))
			next++
		}
		desc = append(desc, uint32(off), uint32(r.ks), uint32(r.ke), uint32(r.wp))
	}
	// The wrong table: every entry moved to the next page id.
	wrong := make([]uint32, len(tab))
	for i, x := range tab {
		wrong[i] = (x + 1) % uint32(pages)
	}
	src := make([]float32, R*kvRow)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	cs := make([]float32, R*nRot)
	for r := 0; r < R; r++ {
		for p := 0; p < nRot/2; p++ {
			a := rng.Float64() * 6.28
			cs[r*nRot+2*p], cs[r*nRot+2*p+1] = float32(math.Cos(a)), float32(math.Sin(a))
		}
	}
	// The expected K row per source row: the copy, then the rotation.
	kWant := append([]float32(nil), src...)
	for r := 0; r < R; r++ {
		for h := 0; h < heads; h++ {
			for p := 0; p < nRot/2; p++ {
				i0, i1 := h*hd+2*p, h*hd+2*p+1
				if neox {
					i0, i1 = h*hd+p, h*hd+p+nRot/2
				}
				c, s := cs[r*nRot+2*p], cs[r*nRot+2*p+1]
				x0, x1 := src[r*kvRow+i0], src[r*kvRow+i1]
				kWant[r*kvRow+i0], kWant[r*kvRow+i1] = x0*c-x1*s, x0*s+x1*c
			}
		}
	}
	run := func(tab []uint32) (kp, vp []float32) {
		g := newGPU(t, d)
		defer g.free()
		nw := pages * page * kvRow
		poison := make([]float32, nw)
		for i := range poison {
			poison[i] = nan32
		}
		kb, vb := g.up(f32bytes(poison)), g.up(f32bytes(poison))
		sb, csb := g.up(f32bytes(src)), g.up(f32bytes(cs))
		off := g.up(u32bytes([]uint32{0xFFFFFFFF}))
		tb, rb := g.up(u32bytes(tab)), g.up(u32bytes(desc))
		launch := func(k func() (*ir.Kernel, error), n int, args ...backend.Buf) {
			kk, err := k()
			if err != nil {
				t.Fatal(err)
			}
			if kk.Params[len(kk.Params)-2].Name != "pTab" || kk.Params[len(kk.Params)-1].Name != "pRow" {
				t.Fatalf("%s is not a paged kernel: %v", kk.Name, kk.Params)
			}
			c, err := d.Compile(kk)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if err := c.Launch((n+127)/128, 128, args...); err != nil {
				t.Fatal(err)
			}
		}
		if f16k {
			words := R * heads * nRot / 2
			if neox {
				words /= 2
			}
			launch(func() (*ir.Kernel, error) { return oldkernels.PagedCopyRowsTF16(kvRow, R, page) }, R*kvRow/2, sb, kb, off, tb, rb)
			launch(func() (*ir.Kernel, error) { return oldkernels.PagedRoPERowsTF16(heads, hd, nRot, neox, R, page) }, words, sb, csb, off, kb, tb, rb)
		} else {
			launch(func() (*ir.Kernel, error) { return oldkernels.PagedCopyRowsT(kvRow, R, page) }, R*kvRow, sb, kb, off, tb, rb)
			launch(func() (*ir.Kernel, error) { return oldkernels.PagedRoPERowsT(heads, hd, nRot, neox, R, page) }, R*heads*nRot/2, sb, csb, off, kb, tb, rb)
		}
		per := kvRow
		if f16 {
			per = kvRow / 2
		}
		launch(func() (*ir.Kernel, error) { return oldkernels.PagedCopyRows(kvRow, R, page, kvRow, 0, f16) }, R*per, sb, vb, off, tb, rb)
		return readF32raw(t, kb, nw), readF32raw(t, vb, nw)
	}
	check := func(kp, vp []float32, tab []uint32) error {
		kSet, vSet := map[int]bool{}, map[int]bool{}
		for r := range rows {
			off, wp := int(desc[r*4]), int(desc[r*4+3])
			pid := int(tab[off+wp/page])
			for e := 0; e < kvRow; e++ {
				ki := pid*page*kvRow + e*page + wp%page
				got, want := kp[ki], kWant[r*kvRow+e]
				tol := 1e-5 * (1 + math.Abs(float64(want)))
				if f16k {
					// The pair's word; binary16 within 2^-10, as V is held.
					ki = pid*page*kvRow/2 + e/2*page + wp%page
					got = float32(quant.DecodeHalf(uint16(math.Float32bits(kp[ki]) >> (16 * (e % 2)))))
					tol = math.Abs(float64(want))/1024 + 1e-7
				}
				kSet[ki] = true
				if !(math.Abs(float64(got-want)) <= tol) {
					return fmt.Errorf("row %d K[%d] = %g, want %g", r, e, got, want)
				}
				vi := (pid*page+wp%page)*kvRow + e
				want = src[r*kvRow+e]
				if f16 {
					w := math.Float32bits(vp[vi/2])
					got = float32(quant.DecodeHalf(uint16(w >> (16 * (vi % 2)))))
					vSet[vi/2] = true
					// The pack's rounding is the driver's (packHalf2x16 leaves
					// it to the implementation: one Vulkan driver truncates), so
					// within binary16's 2^-10, as packf16_test holds it.
					if !(math.Abs(float64(got-want)) <= math.Abs(float64(want))/1024) {
						return fmt.Errorf("row %d V[%d] = %g, want %g within 2^-10", r, e, got, want)
					}
					continue
				}
				got = vp[vi]
				vSet[vi] = true
				if got != want {
					return fmt.Errorf("row %d V[%d] = %g, want %g", r, e, got, want)
				}
			}
		}
		for i, x := range kp {
			if !kSet[i] && !math.IsNaN(float64(x)) {
				return fmt.Errorf("K word %d written (%g) and no row owns it", i, x)
			}
		}
		for i, x := range vp {
			if !vSet[i] && !math.IsNaN(float64(x)) {
				return fmt.Errorf("V word %d written (%g) and no row owns it", i, x)
			}
		}
		return nil
	}
	kp, vp := run(tab)
	if err := check(kp, vp, tab); err != nil {
		return err
	}
	// The padded row wrote its slot of the dummy page (and check says nothing
	// else of it).
	dk := dummy*page*kvRow + 5
	if f16k {
		dk = dummy*page*kvRow/2 + 5
	}
	if math.IsNaN(float64(kp[dk])) {
		return fmt.Errorf("the padded row did not write the dummy page")
	}
	// Through the wrong table the writes land elsewhere: checked against the
	// real one, that must fail.
	kp, vp = run(wrong)
	if err := check(kp, vp, tab); err == nil {
		t.Fatal("writes through a wrong table matched the shuffled one: the table is not read")
	} else {
		t.Logf("wrong table: %v", err)
	}
	return nil
}

func readF32raw(t *testing.T, b backend.Buf, n int) []float32 {
	t.Helper()
	raw := make([]byte, n*4)
	if err := b.Read(raw); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}
