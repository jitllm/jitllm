package engine

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAnInstancesBlocksAreItsOwn holds two instances of one model to their
// own bindings: one runs every block on the CPU, the other places every block
// on a card. The card's instance gives up its own binding of each block it
// placed and leaves the other's alone, the page stays for the instance that
// runs the block from host memory (jlm.Frames.Keep) -- a mixture's experts
// with it -- and that instance decodes as an instance alone does, reading
// what one alone reads: a page given back under it would be read again.
func TestAnInstancesBlocksAreItsOwn(t *testing.T) {
	for _, src := range []string{testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), models()["qwen3moe"]} {
		path, ok := existingModel(src)
		if !ok {
			t.Fatalf("model not present: %s (nor its container)", src)
		}
		t.Run(filepath.Base(path), func(t *testing.T) { instancesBlocksAreTheirOwn(t, path) })
	}
}

func instancesBlocksAreTheirOwn(t *testing.T, path string) {
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
	f0 := m.PageFaults()
	ref := testInstance(t, m, 64)
	want := run(ref, prompt)
	ref.Close()
	alone := m.PageFaults() - f0

	// One CPU device under both instances: the card's must leave the page
	// the CPU's runs from where it is, in the frames they share.
	shared := sharedCPU(t)
	cpu := testInstance(t, m, 64, shared)
	defer cpu.Close()
	f0 = m.PageFaults()
	got := run(cpu, prompt[:4])
	read := m.PageFaults() - f0
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
	f0 = m.PageFaults()
	got = append(got, run(cpu, prompt[4:])...)
	read += m.PageFaults() - f0
	for p := range want {
		if !slices.Equal(got[p], want[p]) {
			e, _ := nmseOf(got[p], want[p])
			t.Errorf("position %d: the CPU's instance beside the card's decodes %.3e from it alone (NMSE)", p, e)
		}
	}
	if alone == 0 {
		t.Fatal("an instance alone read no page: the count proves nothing")
	}
	if read != alone {
		t.Errorf("the CPU's instance read %d pages beside the card's, one alone %d: the card's placement gave back pages it runs", read, alone)
	}
	// The card's instance runs too: placing gave nothing of the CPU's away.
	run(card, prompt)
}
