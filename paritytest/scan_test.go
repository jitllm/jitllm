package paritytest

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestScanIsMains holds the composer's recurrent update to main's fused
// scans: Mamba-2's selective update, Mamba-1's, and the gated delta rule
// with a decay a head and a decay a channel (bounded and not), at a
// subgroup's lanes and at one.
func TestScanIsMains(t *testing.T) {
	type form struct {
		name string
		sc   compose.Scan
		d    mainkernels.DeltaScan
		f    mainkernels.DeltaFuse
	}
	var forms []form
	for _, lanes := range []int{32, 1} {
		for _, rows := range []int{1, 16} {
			ssd := compose.Scan{VHeads: 8, VDim: 16, K: 64, KHeads: 1, Chans: 8*16 + 2*64}
			forms = append(forms, form{"ssd", ssd,
				mainkernels.DeltaScan{VHeads: 8, VDim: 16, KDim: 64, Rep: 8, Rows: rows, Lanes: lanes},
				mainkernels.DeltaFuse{Chans: ssd.Chans, SSD: true}})
			m1 := compose.Scan{VHeads: 64, VDim: 1, K: 16, KHeads: 1, Separate: true}
			forms = append(forms, form{"mamba1", m1,
				mainkernels.DeltaScan{VHeads: 64, VDim: 1, KDim: 16, Rep: 64, Rows: rows, Lanes: lanes},
				mainkernels.DeltaFuse{Mamba1: true}})
			eps, scale := float32(1e-6/128), float32(1/math.Sqrt(128))
			dl := compose.Scan{VHeads: 8, VDim: 32, K: 128, KHeads: 4, Chans: 2*4*128 + 8*32, Delta: true, BARep: 2, Eps: eps, Scale: scale}
			forms = append(forms, form{"delta", dl,
				mainkernels.DeltaScan{VHeads: 8, VDim: 32, KDim: 128, Rep: 2, Rows: rows, Lanes: lanes},
				mainkernels.DeltaFuse{Chans: dl.Chans, Eps: eps, Scale: scale, BARep: 2}})
			tiled := dl
			tiled.KeyTiled = true
			forms = append(forms, form{"delta_tiled", tiled,
				mainkernels.DeltaScan{VHeads: 8, VDim: 32, KDim: 128, Rep: 2, TiledK: 4, Rows: rows, Lanes: lanes},
				mainkernels.DeltaFuse{Chans: dl.Chans, Eps: eps, Scale: scale, BARep: 2}})
			for _, bound := range []float32{0, -5} {
				kda := compose.Scan{VHeads: 4, VDim: 64, K: 64, KHeads: 4, Chans: 3 * 4 * 64, Delta: true, ChanDecay: true, Eps: eps, Scale: scale, Bound: bound}
				forms = append(forms, form{fmt.Sprintf("kda_b%v", bound), kda,
					mainkernels.DeltaScan{VHeads: 4, VDim: 64, KDim: 64, Rep: 1, PerChan: true, Rows: rows, Lanes: lanes},
					mainkernels.DeltaFuse{Chans: kda.Chans, Eps: eps, Scale: scale, Bound: bound}})
			}
		}
	}
	for _, c := range forms {
		name := fmt.Sprintf("%s_l%d_r%d", c.name, c.d.Lanes, c.d.Rows)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.GatedDeltaFused(c.d, c.f)
			if err != nil {
				t.Fatal(err)
			}
			L := c.d.ScanLanes()
			sc := c.sc
			sc.Slot = sc.VHeads * sc.VDim * sc.K
			g := &compose.Row{Width: L, Rows: sc.VHeads * sc.VDim}
			g.Scan(sc, g.ScanBuffers(sc))
			tg := sm70
			if c.d.Lanes == 1 {
				tg.Lanes = 0
			}
			pl, err := g.Plan(compose.Facts{Target: tg})
			if err != nil {
				t.Fatal(err)
			}
			if len(pl) != 1 || pl[0].S.Lanes != L {
				t.Fatalf("planned %d kernels, lanes %d", len(pl), pl[0].S.Lanes)
			}
			same(t, name, []*ir.Kernel{mk}, emitPlanned(t, pl[0]))
		})
	}
}

// TestLaneNormIsMains holds the composer's lane-row RMSNorm to main's
// HeadNorm (a weight one head wide) and GroupNormRows (a weight a group's
// slice), a subgroup a head and a thread a head.
func TestLaneNormIsMains(t *testing.T) {
	for _, c := range []struct{ groups, width, rows, lanes int }{
		{1, 128, 8, 32}, {1, 64, 1, 32}, {4, 64, 2, 32}, {8, 32, 16, 32},
		{1, 128, 8, 1}, {1, 48, 3, 1}, {4, 64, 2, 1},
	} {
		name := fmt.Sprintf("g%d_w%d_r%d_l%d", c.groups, c.width, c.rows, c.lanes)
		t.Run(name, func(t *testing.T) {
			var mk *ir.Kernel
			var err error
			n := c.groups * c.rows
			if c.groups == 1 {
				mk, err = mainkernels.HeadNorm(c.rows, c.width, 1e-5, c.lanes)
				n = c.rows
			} else {
				mk, err = mainkernels.GroupNormRows(c.groups, c.width, 1e-5, c.rows, c.lanes)
			}
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: c.lanes, Rows: n}
			x, w, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufOut)
			g.LaneRMSNorm(x, w, out, c.width, c.groups, float32(1e-5))
			f := compose.Facts{Target: sm70}
			if c.lanes == 1 {
				f.Target.Lanes = 0
			} else {
				f.RowGroup = 32
			}
			pl, err := g.Plan(f)
			if err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, emitPlanned(t, pl[0]))
		})
	}
}
