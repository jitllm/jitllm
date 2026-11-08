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
// head -- with the longest clip the reference reads, and lowers: the entry
// over the clip's 3001 frames into the subsampling's 376 rows of the
// tower's width, each block over them, the head into rows of the text
// model's.
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
	ag := m.Audio
	if ag == nil || !ag.Chain || len(ag.Blocks) != int(a.NLayer) || ag.Head == nil {
		t.Fatalf("Open lowered the audio tower to %+v", ag)
	}
	e := ag.Classes[0].Entry[1]
	eo, in := e.Values[e.Out[0]], ag.Blocks[0].Values[ag.Blocks[0].In[0]]
	ho := ag.Head.Values[ag.Head.Out[0]]
	if ag.Rows != 3001 || eo.Rows != 376 || eo.Width != int(a.NEmbd) || in.Rows != 376 || ho.Rows != 376 || ho.Width != int(a.ProjDim) {
		t.Fatalf("the entry writes %d rows of %d from %d frames, the blocks read %d, the head writes %d of %d",
			eo.Rows, eo.Width, ag.Rows, in.Rows, ho.Rows, ho.Width)
	}
	if got, want := ag.Blocks[0].Block, m.TowerBase+int(m.Container.Vision().NLayer); got != want {
		t.Fatalf("the audio's first block is block %d of the model, want %d (after the vision tower's)", got, want)
	}
}
