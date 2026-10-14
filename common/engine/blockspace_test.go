//go:build linux

package engine

import (
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// The memory map draws the model's whole block space. A vision tower's blocks
// are blocks of the model, placed and paged like the text blocks, so a map
// that stopped at the text blocks would hide them.
func TestTheMemoryMapCoversTheTowersBlocks(t *testing.T) {
	fresh.Must(t, visionModel)
	if _, err := os.Stat(visionModel); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	sh := newTestShell()
	e := New(sh, sh.state())
	defer e.Close()
	e.Load(visionModel)
	pump(t, sh, 180*time.Second, "the model to load", func() bool { return sh.Store.Loaded.Get() })
	text, all := e.m.Cfg.NLayer, e.m.Blocks()
	if all <= text {
		t.Fatalf("%s has %d blocks against %d text blocks: no tower blocks to draw, so this proves nothing", visionModel, all, text)
	}
	if got := len(sh.Store.BlockMap.Get()); got != all {
		t.Fatalf("the memory map draws %d blocks, the model has %d (%d of them text)", got, all, text)
	}
}
