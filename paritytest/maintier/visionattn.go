package maintier

import (
	"encoding/binary"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// A vision block's attention in query chunks.
//
// A vision block is one submission over the whole image (visrows.go), and
// its score and probability planes are rows x NHead x stride floats each. For
// Gemma 3's SigLIP that is 4096 x 16 x 4100 x 4 bytes, 1.07 GB a plane, which
// no 4 GB card holds beside anything else. The keys are the whole image either
// way, so the queries are what can be cut: each chunk's rows are gathered into
// a chunk-sized q, run through the same score, softmax and accumulate kernels
// built for the chunk's row count, and their output scattered back -- the same
// arithmetic per row, only fewer rows at a time. Nothing else in the block
// changes, and a tower whose planes fit runs exactly as before.

// visionPlaneBudget is the most the two score planes of one vision block may
// take before its attention goes in chunks.
const visionPlaneBudget = 128 << 20

// planeBudget is the score-plane budget this device chunks vision attention
// at: Config.VisionPlaneBudget, or visionPlaneBudget.
func (g *devTier) planeBudget() int {
	b := visionPlaneBudget
	if g.VisionPlaneBudget > 0 {
		b = g.VisionPlaneBudget
	}
	if g.planeCap > 0 {
		b = min(b, g.planeCap)
	}
	return b
}

// minPlaneBudget is the least a tight budget narrows the score planes to
// (resizeTower): below it the passes are too many for what they save.
const minPlaneBudget = 4 << 20

// visionChunk is how a vision block's attention takes its query rows: all of
// them when the two score planes fit budget, else passes of arows -- a
// multiple of 16 and of both kernels' tiles -- over a scratch of capRows rows.
// arows is the largest such chunk that fits; a divisor of rows is preferred
// when one is at least half that, and otherwise the scratch's rows are rounded
// up to whole chunks. The rows past the run are capacity: a pass writes their
// output and nothing reads it, and the key counts and windows cover them
// (keyruns.go). Over budget the attention always goes in chunks, whatever rows
// factors into -- a capacity of four times a prime had no halving that divides
// it, and its planes were allocated whole.
func visionChunk(rows, heads, sstride, qtile, atile, budget int) (arows, capRows int) {
	per := 2 * heads * sstride * 4 // the two planes' bytes per query row
	if rows*per <= budget {
		return rows, rows
	}
	unit := lcmInt(16, lcmInt(max(qtile, 1), max(atile, 1)))
	fit := max(unit, budget/per/unit*unit)
	for r := fit; r >= unit && 2*r >= fit; r -= unit {
		if rows%r == 0 {
			return r, rows
		}
	}
	return fit, (rows + fit - 1) / fit * fit
}

func lcmInt(a, b int) int {
	x, y := a, b
	for y != 0 {
		x, y = y, x%y
	}
	return a / x * b
}

// allocAttnChunks allocates a chunked vision block's gather and scatter: the
// chunk's q and output, one offset per chunk (written once, since a chunk's
// place in q never changes), and the two copy kernels.
func (g *devTier) allocAttnChunks(bs *blockScratch, qdim int, al func(*backend.Buf, int)) {
	n := bs.arows * qdim
	al(&bs.qc, n*4)
	al(&bs.xc, n*4)
	bs.qoffs = make([]backend.Buf, bs.rows/bs.arows)
	for c := range bs.qoffs {
		al(&bs.qoffs[c], 4)
		if bs.qoffs[c] != nil {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], uint32(c*n))
			if err := bs.qoffs[c].Write(b[:]); err != nil {
				g.LastErr = err.Error()
			}
		}
	}
	comp := func(dst *backend.Kernel, build func(int) (*ir.Kernel, error)) {
		k, err := build(n)
		if err == nil {
			*dst, err = g.dev.Compile(k)
		}
		if err != nil {
			g.LastErr = err.Error()
		}
	}
	comp(&bs.qGather, kernels.CopySpan)
	comp(&bs.xScatter, kernels.CopyAt)
}
