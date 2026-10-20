package backend_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// prefillValue is the width a shape's accumulate runs at.
func prefillValue(s mainkernels.FlashShape) int {
	if s.MLA > 0 {
		return s.MLA
	}
	return s.Dim
}

// TestPagedPrefillMMA is TestPagedPrefillStaged for the matrix-instruction
// forms: PagedAttnScoresMMA (m16n8k16, sm_80 and later) with the tiled
// accumulate and with PagedAttnAccMMA (m16n8k8), and the Volta pair
// PagedAttnScoresMMA70 and PagedAttnAccMMA70
// (m8n8k4, which sm_86 runs too), on every device that lowers each. Binary16
// operands and float32 sums, so the bound is FlashPrefill70's 1e-5 against
// the float64 oracle. MLA's row-major region runs on the Volta pair, as the
// contiguous tier runs it.
func TestPagedPrefillMMA(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, pass := range []struct {
		sh    ir.MMAShape
		forms []prefillForm
	}{
		{ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}, []prefillForm{
			{name: "mma1", mmaNT: 1, accQT: 1, smLanes: 32},
			{name: "mma4", mmaNT: 4, accQT: 4, smLanes: 1, groupMerge: true},
			// PagedAttnAccMMA (m16n8k8, sm_75 on) behind the m16n8k16 scores.
			{name: "mma2acc4x2", mmaNT: 2, accMMA: [2]int{4, 2}, smLanes: 32},
			{name: "mma1acc1x1", mmaNT: 1, accMMA: [2]int{1, 1}, smLanes: 1, groupMerge: true},
		}},
		{ir.MMAVolta(), []prefillForm{
			{name: "v1x2", voltaMT: 1, voltaNT: 2, smLanes: 32},
			{name: "v2x1", voltaMT: 2, voltaNT: 1, smLanes: 32, groupMerge: true},
			{name: "v2x4", voltaMT: 2, voltaNT: 4, voltaAccMT: 1, smLanes: 1}, // the tier's
		}},
	} {
		for _, d := range mmaDevices(t, devs, pass.sh) {
			t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
				for _, page := range []int{64, 256} {
					for _, sh := range prefillShapes {
						for fi, f := range pass.forms {
							if f.mmaNT > 0 && (sh.MLA > 0 || sh.Dim%16 != 0) ||
								f.accMMA[0] > 0 && prefillValue(sh)%(16*f.accMMA[0]) != 0 ||
								f.voltaMT > 0 && (sh.Dim%4 != 0 || prefillValue(sh)%(32*f.accMT()) != 0) {
								continue
							}
							for _, c := range prefillCases(page, 32) {
								rows := prefillRows(c)
								for _, extra := range []int{-1, 0, 2} {
									s := sh
									s.Page, s.Chunk = page, []int{64, 256}[fi%2]
									if extra < 0 {
										s.Chunk = directChunk(rows) // the direct form
									} else {
										s.Splits = prefillSplits(rows, s.Chunk, extra)
									}
									chunk := s.Chunk
									t.Run(fmt.Sprintf("P%d/%s/h%d-%d/d%d/mla%d/f16%v/sink%v/%s/C%d/S%d", page, c.name, s.Heads, s.KVHeads, s.Dim, s.MLA, s.F16, s.Sink, f.name, chunk, s.Splits), func(t *testing.T) {
										seed := int64(page + s.Dim + c.pos0)
										p := newPrefillPool(rand.New(rand.NewSource(seed)), page, s.KVHeads*s.Dim, s.F16, s.MLA > 0, false, rows)
										_, nmse, err := pagedPrefillStaged(t, d, s, f, p, p.tab, p.desc, rows)
										if err != nil || nmse > 1e-5 {
											t.Fatalf("shuffled pages: NMSE %.3g, %v", nmse, err)
										}
										if _, bad, err := pagedPrefillStaged(t, d, s, f, p, p.wrong, p.desc, rows); err == nil && bad < 1e-4 {
											t.Fatalf("a wrong table passed (NMSE %.3g): the table is not read", bad)
										}
										if _, bad, err := pagedPrefillStaged(t, d, s, f, p, p.tab, shortDesc(p.desc), rows); err == nil && bad < 1e-4 {
											t.Fatalf("keyEnds one short passed (NMSE %.3g)", bad)
										}
										t.Logf("NMSE %.3g", nmse)
									})
									ran++
								}
							}
						}
					}
				}
			})
		}
	}
	if ran == 0 {
		t.Skip("no device lowers either matrix instruction")
	}
}
