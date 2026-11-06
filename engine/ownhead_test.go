package engine

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestACardReadsTheHeadIntoItsOwnFrames: a card offered the head reads it
// from the model's file into frames of its own, not from the CPU device's
// copy. The CPU's copy is poisoned before the card places the head, and the
// card's decode is still a clean instance's, bit for bit; a card uploading
// from the CPU's bytes would project through the poison. The model's head is
// not its embedding (tinyllama ties none), so the poison reaches nothing the
// CPU runs.
func TestACardReadsTheHeadIntoItsOwnFrames(t *testing.T) {
	src := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	path, ok := existingModel(src)
	if !ok {
		t.Fatalf("model not present: %s (nor its container)", src)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.TiedEmbd {
		t.Fatal("the model ties its head to its embedding: poisoning the head poisons the CPU's embedding too, and the gate proves nothing")
	}
	ids := []int32{1, 450, 7483, 310, 3444, 338, 9, 13}
	g := swaTier(t)
	defer g.Close()
	place := func(in *Instance) {
		t.Helper()
		in.SetDevice(g)
		if in.GPULayers() != m.Cfg.NLayer || !in.HeadOnDevice() {
			t.Fatalf("placed %d of %d blocks, head on the card %v: %q", in.GPULayers(), m.Cfg.NLayer, in.HeadOnDevice(), g.Err())
		}
	}

	clean := testInstance(t, m, len(ids)+1)
	place(clean)
	want := teacherForced(t, clean, ids)
	clean.Close()

	x := testInstance(t, m, len(ids)+1)
	defer x.Close()
	out := x.dense.Output
	if out.E == nil || len(out.Data) == 0 {
		t.Fatal("the instance holds no bound head of its own: nothing to poison")
	}
	planes := [][]byte{out.Data}
	if out.Packed != nil {
		planes = append(planes, out.Packed.QS, out.Packed.D, out.Packed.SC)
	}
	saved := make([][]byte, len(planes))
	for i, p := range planes {
		saved[i] = slices.Clone(p)
	}
	for _, p := range planes {
		for i := range p {
			p[i] = 0xff
		}
	}
	place(x)
	got := teacherForced(t, x, ids)
	for i, p := range planes {
		copy(p, saved[i])
	}
	if at := sameRows(got, want); at >= 0 {
		t.Errorf("with the CPU's copy of the head poisoned the card's decode parts from a clean one at position %d: the card read the head from the CPU device's memory", at)
	}
}
