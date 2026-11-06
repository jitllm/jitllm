package engine

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAnInstancesBlocksAreItsOwn holds two instances of one model to their
// own bindings: one runs every block on the CPU, the other places every block
// on a card. The card's instance gives up its own binding of each block it
// placed and leaves the other's alone, the page stays for the instance that
// runs the block from host memory (jlm.File.Keep), and that instance decodes
// as an instance alone does.
func TestAnInstancesBlocksAreItsOwn(t *testing.T) {
	src := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	path, ok := existingModel(src)
	if !ok {
		t.Fatalf("model not present: %s (nor its container)", src)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := []int32{1, 45, 79, 33, 21, 9, 65, 108}
	run := func(in *Instance, ids []int32) [][]float32 {
		t.Helper()
		var out [][]float32
		for _, id := range ids {
			lg, err := in.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(lg))
		}
		return out
	}
	ref := testInstance(t, m, 64)
	want := run(ref, prompt)
	ref.Close()

	// One CPU device under both instances: the card's must leave the page
	// the CPU's runs from where it is, in the frames they share.
	shared := sharedCPU(t)
	cpu := testInstance(t, m, 64, shared)
	defer cpu.Close()
	got := run(cpu, prompt[:4])
	g := swaTier(t)
	defer g.Close()
	card := testInstance(t, m, 64, shared)
	defer card.Close()
	card.SetDevice(g)
	if card.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("placed %d of %d blocks: %q", card.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	for li := 0; li < m.Cfg.NLayer; li++ {
		if len(card.layer(li).Wq.Data) != 0 {
			t.Errorf("block %d: the card's instance kept its own binding of a block it placed", li)
		}
		if len(cpu.layer(li).Wq.Data) == 0 {
			t.Errorf("block %d: placing it on the card unbound the CPU's instance", li)
		}
		if !cpu.fr.Kept(li) || !m.BlockHeld(cpu.fr, li) {
			t.Errorf("block %d: its page left (kept %v, resident %v) while an instance runs it from host memory",
				li, cpu.fr.Kept(li), m.BlockHeld(cpu.fr, li))
		}
	}
	got = append(got, run(cpu, prompt[4:])...)
	for p := range want {
		if !slices.Equal(got[p], want[p]) {
			e, _ := nmseOf(got[p], want[p])
			t.Errorf("position %d: the CPU's instance beside the card's decodes %.3e from it alone (NMSE)", p, e)
		}
	}
	// The card's instance runs too: placing gave nothing of the CPU's away.
	run(card, prompt)
}
