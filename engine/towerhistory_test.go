package engine

import (
	"slices"
	"testing"
)

// TestTowerHoldsOneBlocksHistory encodes a picture through stated towers on
// the CPU device -- Qwen2-VL's, Qwen2.5-VL's windowed blocks and Qwen3-VL's
// tapped ones -- and holds the run to one block's history at a time: each
// block's pages go once its call is done (kvCache.Spent), so the most the
// run held is the control's history over its blocks.
// The control keeps every block's pages, as a run did before; the rows of the
// two arms are the same bits.
func TestTowerHoldsOneBlocksHistory(t *testing.T) {
	for _, name := range []string{"synth-qwen2vl", "synth-qwen25vl", "synth-qwen3vl"} {
		t.Run(name, func(t *testing.T) {
			m, g, opts := openQVLTower(t, name)
			t.Cleanup(func() { m.Close() })
			img := rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
			encode := func(spent bool) ([]float32, *kvCache) {
				s := testInstance(t, m, 64, opts...)
				t.Cleanup(func() { s.Close() })
				if s.tower == nil || s.tower.s.kv == nil {
					t.Fatalf("the instance holds no tower run: %v", s.towerErr)
				}
				kv := s.tower.s.kv
				kv.spent = spent
				rows, err := s.EncodePicture(img)
				if err != nil {
					t.Fatal(err)
				}
				return slices.Clone(rows), kv
			}
			rows, kv := encode(true)
			ctl, ckv := encode(false)
			nb := len(m.Tower.Blocks)
			held := ckv.peak
			if held == 0 || nb < 2 {
				t.Fatalf("the control's %d tower blocks hold %d bytes of history: the gate sees nothing", nb, held)
			}
			if !slices.Equal(rows, ctl) {
				t.Fatal("the tower's rows differ from the control's, which keeps every block's history")
			}
			if kv.peak == 0 || kv.peak*uint64(nb) > held {
				t.Fatalf("the run held at most %d bytes, past one of %d blocks' share of the control's %d",
					kv.peak, nb, held)
			}
			t.Logf("%d blocks: at most %d bytes of history held, the control %d", nb, kv.peak, held)
		})
	}
}
