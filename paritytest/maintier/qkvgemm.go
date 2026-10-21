package maintier

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// qkvSeg is a prompt chunk's q/k/v grid for one set of shapes: the
// GemmVoltaSegments kernel, its launch, which of q (0), k (1) and v (2) it
// carries, and the binary16 conversion they all read.
type qkvSeg struct {
	kern           backend.Kernel
	threads, width int
	segs           []int
	act            backend.Kernel
	actN           int
}

// has says the grid carries projection i.
func (q *qkvSeg) has(i int) bool {
	for _, s := range q.segs {
		if s == i {
			return true
		}
	}
	return false
}

// qkvGemm is block l's q, k and v as one GemmVoltaSegments grid for an R-row
// prompt chunk, or false where it does not apply: a ragged step, Config.QKVGemm off,
// or fewer than two of the three on a binary16 GEMM that reads q's operand
// layout. Each projection's own tile is kept, unsplit: a k or v projection
// that alone split k to fill the card and reduced its partials rides q's
// grid instead. A projection that reads another layout (Q6_K's v beside
// Q4_K's q) runs alone. Compiled once per set of shapes and tiles.
func (g *devTier) qkvGemm(l *layer, R int, ragged bool) (*qkvSeg, bool) {
	if !g.QKVGemm || ragged || R < 2 || l.vFromK || l.clampB != nil {
		return nil, false
	}
	var bms [3]mv
	for i, m := range [3]mv{l.mvq, l.mvk, l.mvv} {
		bm, ok := g.batchMV(m, R, g.bs.tok)
		if !ok || bm.act == nil || bm.tail != nil || bm.vt.MT == 0 || kernels.IsFloat(bm.q) {
			if i == 0 {
				return nil, false
			}
			continue
		}
		bms[i] = bm
	}
	q := bms[0]
	var segs []int
	var shapes []kernels.MatVecShape
	var tiles []kernels.VoltaTile
	key := fmt.Sprintf("%d", R)
	for i, bm := range bms {
		if bm.kern == nil || bm.act != q.act || bm.vt.Threads() != q.vt.Threads() || bm.vt.F16K != q.vt.F16K {
			continue
		}
		segs = append(segs, i)
		vt := bm.vt
		shapes = append(shapes, kernels.MatVecShape{Center: g.center, T: bm.q, K: bm.k, Rows: bm.rows, NTok: R,
			Bias: bm.bias != nil})
		tiles = append(tiles, vt)
		key += fmt.Sprintf("/%d:%v:%d:%d:%v:%+v", i, bm.q, bm.rows, bm.k, bm.bias != nil, vt)
	}
	if len(segs) < 2 {
		return nil, false
	}
	if g.qkvK == nil {
		g.qkvK = map[string]*qkvSeg{}
	}
	if s, ok := g.qkvK[key]; ok {
		return s, s != nil
	}
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	var s *qkvSeg
	if blocks, nth, err := kernels.GemmVoltaSegmentsGrid(shapes, tiles); err == nil {
		if ker, err := kernels.GemmVoltaSegments(shapes, tiles); err == nil {
			if c, err := g.dev.Compile(ker); err == nil {
				// The segment guard holds a value or two across the body, and a
				// tile already at the register cap (sm_70's m4n4 Q4_K, 255)
				// spills a word for it: qkvSpillMax admits that much and no
				// more (the measurement is in gpu-kernels.md).
				if r, ok := backend.KernelResourcesOf(c); ok && r.SpillBytes > qkvSpillMax && !g.SpillingTiles {
					c.Close()
					g.SpillRefused++
				} else {
					total := 0
					for _, n := range blocks {
						total += n
					}
					s = &qkvSeg{kern: c, threads: total * nth, width: nth, segs: segs, act: q.act, actN: q.actN}
				}
			} else {
				g.LastErr = err.Error()
			}
		}
	}
	g.qkvK[key] = s
	return s, s != nil
}

// qkvSpillMax is the local-memory spill a q/k/v grid may carry.
const qkvSpillMax = 16

// qkvGemmSwapped hands the grid's k segment v's weights and v's k's (where
// they are one format and shape): the violation the q/k/v gate must see
// (SwapQKVGemmForTest, jitllmtest builds only).
var qkvGemmSwapped bool
