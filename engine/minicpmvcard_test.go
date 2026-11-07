package engine

import (
	"fmt"
	"path/filepath"
	"testing"
)

// TestMiniCPMVTowerRunsOnTheCard places MiniCPM-V 4's tower on each device,
// every block and half of them, and holds the card's rows to the checkpoint's
// own (the 384 square, an overview alone: <image>'s marker row, the
// resampler's 64 rows, </image>'s) at the tower's bound. The CPU device's
// rows are not the yardstick here: SigLIP-so400m's ~2000-wide residual
// channels make the int8 activations' rounding order-sensitive, so a card
// and the CPU part by more than the generic band while both sit inside the
// reference's.
func TestMiniCPMVTowerRunsOnTheCard(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	g, dir := readMcv(t, "quad-and-disc")
	if len(g.Pieces) != 1 {
		t.Fatalf("the square is %d pieces, want 1", len(g.Pieces))
	}
	D := int(m.Container.Vision().ProjDim)
	marks := markerTable(t, m, D)
	emb := readF32n(t, filepath.Join(dir, fmt.Sprintf("%s.p0.emb", "quad-and-disc")), m.Tower.Fixed*D)
	ref := append(append(append([]float32(nil), marks[:D]...), emb...), marks[D:2*D]...)
	towerRunsOnTheCard(t, m, decodePNG(t, filepath.Join("testdata", "quad-and-disc.png")), ref, 3e-2)
}
