package backend_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGemmTileTailCoversThePrompt holds the tier's wide-block-plus-tail form
// (tileMV) to the reference: a 64x64 block over the leading tokens and a
// 64x32 block at a token offset over the rest, into one poisoned output, so a
// tail that does not run or lands at the wrong offset leaves a NaN.
// JITLLM_VIOLATE_TILE_TAIL=1 skips the tail launch, which must fail.
func TestGemmTileTailCoversThePrompt(t *testing.T) {
	gpuLock(t)
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if ok, _ := backend.LowersTile(d, tile8); !ok {
			continue
		}
		for _, q := range []mainkernels.Quant{mainkernels.Q4_K, mainkernels.Q6_K, mainkernels.Q8_0, mainkernels.Q4_0} {
			for _, c := range []struct{ rows, k, ntok int }{{128, 512, 96}, {192, 1024, 160}} {
				sub, _, _, _ := mainkernels.Layout(q)
				kb := max(32/sub, 1)
				lead := c.ntok - c.ntok%64
				wide := mainkernels.TileGemm{MT: 4, NT: 4, WM: 2, WN: 2, KB: kb}
				tail := mainkernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: kb, TokOff: lead}
				full := mainkernels.MatVecShape{T: q, K: c.k, Rows: c.rows, NTok: c.ntok}
				sw, st := full, full
				sw.NTok, st.NTok = lead, c.ntok-lead
				kw, err := mainkernels.GemmTile(sw, wide)
				if err != nil {
					t.Fatal(err)
				}
				kt, err := mainkernels.GemmTile(st, tail)
				if err != nil {
					t.Fatal(err)
				}
				cw, err := d.Compile(kw)
				if err != nil {
					t.Fatal(err)
				}
				ct, err := d.Compile(kt)
				if err != nil {
					t.Fatal(err)
				}
				ran++
				mma70Case(t, d, cw, full, fmt.Sprintf("gemmtile 64x64 over %d + tail at %d", lead, lead),
					mainkernels.GemmTileGroups(sw, wide), wide.Threads(), true,
					func(args ...backend.Buf) error {
						if os.Getenv("JITLLM_VIOLATE_TILE_TAIL") != "" {
							return nil
						}
						return ct.Launch(mainkernels.GemmTileGroups(st, tail), tail.Threads(), args...)
					})
				cw.Close()
				ct.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the collective tile GEMM here")
	}
}
