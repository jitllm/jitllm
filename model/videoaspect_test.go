package model

import (
	"strings"
	"testing"
)

// TestVideoPastAnAspectOf200IsRefusedAsTheReference: a video's resize is
// transformers' smart_resize step for step (videoSize), and that function
// raises for frames whose long side passes 200 times the short one, after
// lifting a side below the patch factor. jitllm refuses the same videos,
// with the reference's reason, by the user's decision: it has no output for
// them to be held to. A strip of 100:1 still resizes.
func TestVideoPastAnAspectOf200IsRefusedAsTheReference(t *testing.T) {
	// factor 32 (patch 16, merge 2), pixel bounds, temporal patch 2.
	r := []int32{32, 128 * 32 * 32, 16384 * 32 * 32, 0, 0, 2}
	if _, _, err := videoSize(4, 6400, 30, r); err == nil || !strings.Contains(err.Error(), "aspect of 200") {
		t.Fatalf("6400x30 frames (213:1 once the short side is lifted to 32) resized (%v): the reference refuses them", err)
	}
	if w, h, err := videoSize(4, 4000, 40, r); err != nil || w <= 0 || h <= 0 {
		t.Fatalf("4000x40 frames (100:1) did not resize: %dx%d, %v", w, h, err)
	}
}
