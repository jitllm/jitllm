package engine

import (
	"math"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestReusedPromptScratchIsNotRead is the gate on a closed session's prompt
// rows (promptBuf) and history pages (kvPagePool) serving the next instance
// on the same model.
//
// The next session's prompt and decode must be bit-identical to the first
// instance's, which ran on fresh memory -- with every handed-back float set to
// NaN first, so a read of anything the new instance did not write shows up as a
// NaN or a changed logit rather than as a plausible number. And it checks the
// reuse was SELECTED (RULE 10): the new instance's scratch is the poisoned array
// and its history came out of the pool, or the comparison proves nothing.
func TestReusedPromptScratchIsNotRead(t *testing.T) {
	paths := []string{benchModel(t)}
	if p, ok := existingModel(testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf")); ok {
		paths = append(paths, p)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) { reusedScratchGate(t, path) })
	}
}

func reusedScratchGate(t *testing.T, path string) {
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// One CPU device: what a closed instance hands back is the device's, for
	// the next instance of the model on it.
	cd, err := OpenCPU()
	if err != nil {
		t.Fatal(err)
	}
	defer cd.Close()
	// The device keeps the model, and what its instances hand back, while an
	// instance holds it there: the last one's Close lets it all go. One held
	// for the whole gate stands for the server's loaded model.
	held := testInstance(t, m, 256, WithCPU(cd))
	defer held.Close()
	hs := cd.Attached(m.Id, func() any { return &hostShared{} }).(*hostShared)
	short := m.Vocab.Encode("The capital of France is Paris, and the capital of Germany is", true)
	long := m.Vocab.Encode("Every spring the river rose until the bridges were islands, and the "+
		"clerk on the upper floor counted barrels of salt by lamplight while the "+
		"boats knocked against the stairs. The capital of Spain is", true)
	const ctx, steps = 256, 12

	run := func(ids []int32) [][]float32 {
		s := testInstance(t, m, ctx, WithCPU(cd))
		defer s.Close()
		lg, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for i := 0; i < steps; i++ {
			lg, err = s.Forward(argmaxID(lg))
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}

	want := run(short) // fresh memory: nothing handed back yet
	run(long)          // a different, longer history fills the scratch

	// Poison everything a closed instance handed back.
	hs.mu.Lock()
	if hs.spare == nil || hs.spare.bx == nil {
		hs.mu.Unlock()
		t.Fatal("no prompt scratch was handed back to the CPU device")
	}
	for _, b := range [][]float32{hs.spare.bx, hs.spare.bcs, hs.spare.bcsSWA} {
		for i := range b {
			b[i] = float32(math.NaN())
		}
	}
	poisoned := unsafe.SliceData(hs.spare.bx)
	hs.mu.Unlock()
	pooled := cd.KVPool(m.Id).Held(func(b []float32) {
		for i := range b {
			b[i] = float32(math.NaN())
		}
	})

	s := testInstance(t, m, ctx, WithCPU(cd))
	lg, err := s.Prefill(short)
	if err != nil {
		t.Fatal(err)
	}
	if unsafe.SliceData(s.bx) != poisoned {
		t.Fatal("the new session's prompt scratch is not the handed-back set: the reuse was not selected")
	}
	if pooled == 0 {
		// The composed CPU holds a layer's history in an arena it unmaps at
		// Close: no page comes back to be read stale, and a new history is
		// fresh memory.
		for li := s.kv.lo; li < len(s.kv.layers); li++ {
			if l := &s.kv.layers[li]; l.p > 0 && l.arena() == nil {
				t.Fatalf("block %d's history is in pages, and no page was handed back to the CPU device", li)
			}
		}
	} else if held := cd.KVPool(m.Id).Held(nil); held >= pooled {
		t.Fatalf("the pool still holds %d of %d bytes: the new history did not come from it", held, pooled)
	}
	got := [][]float32{append([]float32(nil), lg...)}
	for i := 0; i < steps; i++ {
		if lg, err = s.Forward(argmaxID(lg)); err != nil {
			t.Fatal(err)
		}
		got = append(got, append([]float32(nil), lg...))
	}
	s.Close()
	for step := range want {
		for i := range want[step] {
			if math.Float32bits(got[step][i]) != math.Float32bits(want[step][i]) {
				t.Fatalf("step %d logit %d: %v on reused memory, %v on fresh", step, i, got[step][i], want[step][i])
			}
		}
	}
	t.Logf("%d steps bit-identical on reused, poisoned scratch and history (%d pooled bytes)", len(want), pooled)
}
