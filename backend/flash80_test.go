package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestPagedFlashPrefill80 holds FlashPrefill80 (m16n8k16, sm_80 on) and its
// m16n8k8 twin (FlashPrefill80K8, sm_75 on) to the oracle as TestPagedFlashPrefill holds FlashPrefill70: every prefill case
// (causal, windowed, chunked, bidirectional runs, rows at a later base, a
// history across pages), GQA, one KV head, packed f16 V, a softcap and a
// sink, head widths 64, 96, 128 and 256 (its dims split over two warps), at
// no split, one and more than the keys need; to FlashPrefill70's paged
// output on the same pool where the card runs both; and demands failure through a wrong table and keyEnds one
// short. A head wider than 256 is refused by name.
func TestPagedFlashPrefill80(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	if why := kernels.FlashPrefill80WhyNot(kernels.FlashPrefill70Shape{Heads: 8, KVHeads: 8, Dim: 320, Rows: 64, Page: 64}); why == "" {
		t.Fatal("a 320-wide head was taken; FlashPrefill80 declines it by name")
	}
	shapes := append(flashPrefillShapes[:len(flashPrefillShapes):len(flashPrefillShapes)],
		struct {
			fs   kernels.FlashPrefill70Shape
			sink bool
		}{kernels.FlashPrefill70Shape{Heads: 12, KVHeads: 4, Dim: 96, Scale: .102}, false},
		// Past 128 the head's dims split over two warps (gemma-2b's 256).
		struct {
			fs   kernels.FlashPrefill70Shape
			sink bool
		}{kernels.FlashPrefill70Shape{Heads: 8, KVHeads: 4, Dim: 256, Scale: .0625}, false},
		struct {
			fs   kernels.FlashPrefill70Shape
			sink bool
		}{kernels.FlashPrefill70Shape{Heads: 8, KVHeads: 1, Dim: 256, Scale: .0625, Softcap: 50, F16: true}, true})
	ran := 0
	vdevs := mmaDevices(t, devs, ir.MMAVolta())
	for _, fk := range []struct {
		form flashForm
		k    int
	}{{flash80Form, 16}, {flash80K8Form, 8}} {
		form := fk.form
		for _, d := range mmaDevices(t, devs, ir.MMAShape{M: 16, N: 8, K: fk.k, Kind: ir.MMAF16}) {
			both := false
			for _, v := range vdevs {
				both = both || v == d
			}
			t.Run(fmt.Sprintf("k%d/%s/%s", fk.k, d.API(), d.Name()), func(t *testing.T) {
				for _, page := range []int{64, 256} {
					for _, sh := range shapes {
						for _, c := range prefillCases(page, 64) {
							rows := prefillRows(c)
							for _, splits := range []int{0, 1, 3, 40} {
								if sh.sink && splits == 0 {
									continue
								}
								fs := sh.fs
								fs.Page, fs.Splits = page, splits
								t.Run(fmt.Sprintf("P%d/%s/h%d-%d/d%d/f16%v/cap%g/sink%v/S%d", page, c.name, fs.Heads, fs.KVHeads, fs.Dim, fs.F16, fs.Softcap, sh.sink, splits), func(t *testing.T) {
									p := newPrefillPool(rand.New(rand.NewSource(int64(page+fs.Dim+c.pos0))), page, fs.KVHeads*fs.Dim, fs.F16, false, false, rows)
									got, nmse, err := pagedFlashPrefill(t, d, fs, sh.sink, form, p, p.tab, p.desc, rows)
									if err != nil || nmse > 1e-5 {
										t.Fatalf("shuffled pages: NMSE %.3g, %v", nmse, err)
									}
									if _, bad, err := pagedFlashPrefill(t, d, fs, sh.sink, form, p, p.wrong, p.desc, rows); err == nil && bad < 1e-4 {
										t.Fatalf("a wrong table passed (NMSE %.3g): the table is not read", bad)
									}
									if _, bad, err := pagedFlashPrefill(t, d, fs, sh.sink, form, p, p.tab, shortDesc(p.desc), rows); err == nil && bad < 1e-4 {
										t.Fatalf("keyEnds one short passed (NMSE %.3g)", bad)
									}
									msg := fmt.Sprintf("NMSE %.3g", nmse)
									if both && (fs.Dim == 64 || fs.Dim == 128) {
										ref, _, err := pagedFlashPrefill(t, d, fs, sh.sink, flash70Form, p, p.tab, p.desc, rows)
										if err != nil {
											t.Fatal(err)
										}
										n := realRowsNMSE(got, ref, rows)
										if math.IsNaN(n) || n > 1e-5 {
											t.Fatalf("against FlashPrefill70: NMSE %.3g", n)
										}
										msg += fmt.Sprintf(", against FlashPrefill70 %.3g", n)
									}
									t.Log(msg)
								})
								ran++
							}
						}
					}
				}
			})
		}
	}
	if ran == 0 {
		t.Skip("no device lowers m16n8k16 or m16n8k8")
	}
}
