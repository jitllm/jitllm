package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestVideoPastAnAspectOf200IsRefusedAsTheReference: a video's resize is
// transformers' smart_resize step for step (videoGeometry), and that
// function raises for frames whose long side passes 200 times the short
// one, after lifting a side below the patch factor. jitllm refuses the same
// videos, with the reference's reason, by the user's decision: it has no
// output for them to be held to. A strip of 100:1 still resizes, and fewer
// frames sized together than the processor's temporal patch are refused too.
func TestVideoPastAnAspectOf200IsRefusedAsTheReference(t *testing.T) {
	// factor 32 (patch 16, merge 2), pixel bounds, temporal patch 2.
	g := videoGeometry(32, 128*32*32, 16384*32*32, 2, false, 16)
	if _, _, err := model.FrameSize(g, 4, 6400, 30); err == nil || !strings.Contains(err.Error(), "refuses") {
		t.Fatalf("6400x30 frames (213:1 once the short side is lifted to 32) resized (%v): the reference refuses them", err)
	}
	if w, h, err := model.FrameSize(g, 4, 4000, 40); err != nil || w <= 0 || h <= 0 {
		t.Fatalf("4000x40 frames (100:1) did not resize: %dx%d, %v", w, h, err)
	}
	if _, _, err := model.FrameSize(g, 1, 64, 64); err == nil {
		t.Fatal("one frame sized together as a pair resized: the reference refuses it")
	}
	if w, h, err := model.FrameSize(videoGeometry(28, 100352, 602112, 2, true, 14), 1, 64, 64); err != nil || w <= 0 || h <= 0 {
		t.Fatalf("a frame sized alone did not resize: %dx%d, %v", w, h, err)
	}
}
