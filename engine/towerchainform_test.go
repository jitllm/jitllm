package engine

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/composer"
)

// TestTowerChainBlocksFormForACard forms every block of Gemma 3n's chained
// tower at its stated rows for a card, as placement does, and holds the
// longest kernel to a size a card's compiler takes at once. A chain that
// computed the pointwise expansion again at every tap of the depthwise
// convolution after it held 239,445 instructions, 9.2 MB of PTX the V100's
// driver compiled for over ten minutes.
func TestTowerChainBlocksFormForACard(t *testing.T) {
	m := openGemma3nV(t)
	defer m.Close()
	tg, rec := m.Tower, m.Container.Vision().Tower
	if tg == nil || !tg.Chain || len(tg.Blocks) != len(rec.Chain) {
		t.Fatal("synth-gemma3nv holds no chained tower: this gate proved nothing")
	}
	// Four lanes a thread, as a card composes, is where the chain formed.
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 30720, Lanes: 4, Units: 80, Regs: 255}
	most, at := 0, ""
	for i, g := range tg.Blocks {
		rows := int(rec.Chain[i].Rows)
		start := time.Now()
		_, codes, err := composer.Default().Form(composer.Request{Graph: g, Formats: m.TowerFormats(g),
			Step: composer.Step{Rows: rows, Page: 64, Keys: 1}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("block %d at %d rows: %v", i, rows, err)
		}
		for _, c := range codes {
			if c.GPU == nil {
				t.Fatalf("block %d: %s has no card code", i, c.Name)
			}
			if n := len(c.GPU.Ops); n > most {
				most, at = n, c.Name
			}
		}
		t.Logf("block %d at %d rows: %d kernels in %v", i, rows, len(codes), time.Since(start).Round(time.Millisecond))
	}
	t.Logf("the longest kernel: %s, %d instructions", at, most)
	if most > 1<<15 {
		t.Fatalf("%s holds %d instructions: a chain computed a reduction again at every place it is read", at, most)
	}
}
