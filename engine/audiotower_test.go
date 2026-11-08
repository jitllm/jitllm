package engine

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestD1OmniStatesItsAudioTower converts d1-omni with its mmproj: the
// container opens, its vision tower lowers, and its audio tower reads back
// stated -- an entry framing the clip, a chained block for each block, a
// head -- with the longest clip the reference reads.
func TestD1OmniStatesItsAudioTower(t *testing.T) {
	path := testmodels.Path("d1-omni/d1-omni-600M-F16.gguf")
	mm := testmodels.Path("d1-omni/mmproj-d1-omni-600M-F16.gguf")
	if !sourcePresent(path, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	m, err := model.Open(jlmOfPair(t, path, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.Container.Vision().Audio
	switch {
	case m.Tower == nil:
		t.Fatal("Open lowered no vision tower")
	case a == nil || a.Tower == nil:
		t.Fatal("the container states no audio tower")
	case a.Tower.Entry.Ops[0].Code != jlm.TOpFrames || len(a.Tower.Chain) != int(a.NLayer) || a.MaxSamples != 30*a.SampleRate:
		t.Fatalf("an audio entry from op %d, %d chained blocks of %d, clips of %d samples",
			a.Tower.Entry.Ops[0].Code, len(a.Tower.Chain), a.NLayer, a.MaxSamples)
	}
}
