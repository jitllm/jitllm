//go:build jitllmfault

package backend_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestPagedPrefillGatesDiscriminate arms each generation fault the paged
// prefill kernels share (mainkernels.SetPagedFault) and demands the prefill gates
// FAIL on it, on every path and every device that runs the path, and pass
// again disarmed (RULE 10):
//
//	"(t-1)/P"    the first slot of every page read from the page before
//	"unaligned"  the partition base and the flash key tiles at keyStart, not
//	             a multiple of 64 or 32, so a tile crosses a page edge while
//	             its page id is read once
//	"vmask"      V read without the tile's mask: the poisoned slots below the
//	             chunk's first key and past its last reach the sums
//
//	go test -tags jitllmfault ./jit_old/gpu/backend/ -run TestPagedPrefillGatesDiscriminate
func TestPagedPrefillGatesDiscriminate(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	defer mainkernels.SetPagedFault("")
	const page = 64
	// "later" crosses page edges and ends mid-page; "window" starts mid-page,
	// so the aligned tile below its first key reads poisoned slots.
	cases := map[string]prefillCase{}
	for _, c := range prefillCases(page, 64) {
		cases[c.name] = c
	}
	faults := []struct{ fault, cs string }{{"(t-1)/P", "later"}, {"unaligned", "window"}, {"vmask", "window"}}
	type path struct {
		name  string
		mma   *ir.MMAShape
		tile  bool
		flash bool
		f     prefillForm
	}
	sh16 := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	paths := []path{
		{name: "tiled", f: prefillForm{scoresQT: 2, scoresKT: 2, accQT: 2, smLanes: 32}},
		{name: "mma", mma: &sh16, f: prefillForm{mmaNT: 2, accQT: 4, smLanes: 32}},
		{name: "volta", mma: &ir.MMAVolta(), f: prefillForm{voltaMT: 1, voltaNT: 2, smLanes: 32}},
		{name: "flash70", mma: &ir.MMAVolta(), flash: true},
		{name: "flashtile", tile: true, flash: true},
	}
	ran := 0
	for _, pa := range paths {
		var pdevs []backend.Device
		switch {
		case pa.tile:
			for _, d := range devs {
				if d.API() == "msl" {
					pdevs = append(pdevs, d)
				}
			}
		case pa.mma != nil:
			pdevs = mmaDevices(t, devs, *pa.mma)
		default:
			pdevs = devs
		}
		for _, d := range pdevs {
			for _, fc := range faults {
				t.Run(fmt.Sprintf("%s/%s/%s/%s", pa.name, d.API(), d.Name(), fc.fault), func(t *testing.T) {
					rows := prefillRows(cases[fc.cs])
					run := func(fault string) (float64, error) {
						mainkernels.SetPagedFault(fault)
						defer mainkernels.SetPagedFault("")
						if pa.flash {
							fs := mainkernels.FlashPrefill70Shape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: page, Splits: 2}
							p := newPrefillPool(rand.New(rand.NewSource(7)), page, fs.KVHeads*fs.Dim, false, false, false, rows)
							_, n, err := pagedFlashPrefill(t, d, fs, false, formOf(pa.tile), p, p.tab, p.desc, rows)
							return n, err
						}
						s := mainkernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: page, Chunk: 64}
						s.Splits = prefillSplits(rows, s.Chunk, 1)
						p := newPrefillPool(rand.New(rand.NewSource(7)), page, s.KVHeads*s.Dim, false, false, false, rows)
						_, n, err := pagedPrefillStaged(t, d, s, pa.f, p, p.tab, p.desc, rows)
						return n, err
					}
					bad, berr := run(fc.fault)
					if berr == nil && bad < 1e-4 {
						t.Fatalf("violation %q passed the gate: NMSE %.3g", fc.fault, bad)
					}
					good, gerr := run("")
					if gerr != nil || good > 1e-5 {
						t.Fatalf("disarmed: NMSE %.3g, %v", good, gerr)
					}
					t.Logf("violation %q: NMSE %.3g %v; clean %.3g", fc.fault, bad, berr, good)
				})
				ran++
			}
		}
	}
	if ran == 0 {
		t.Fatal("no prefill path ran")
	}
}
