package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/nn"
)

// TestAHeadRefusedForRoomIsAskedAgain: a projection refused for room under a
// budget is taken once the budget is raised, not answered by the refusal
// resident() kept. The reserved scratch is off and the CPU unlisted, so the
// refusal is resident()'s and not the per-row logits' or a block slot's.
func TestAHeadRefusedForRoomIsAskedAgain(t *testing.T) {
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff),
		WithConfig(func(c *Config) { c.NoScratchReserve = true }))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	p := fakePlan()
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	if !g.PrepLayer(0, p, denseWeights(p, q4)) {
		t.Fatalf("the block was refused an empty tier: %s", g.Err())
	}
	const rows = 1 << 17
	w := q4(rows, p.NEmbd)
	head := func() *nn.Head {
		return headGraphs(&nn.Head{Model: p.Model, Vectors: []nn.RoleVector{{Role: jlm.RoleOutputNorm, V: make([]float32, p.NEmbd)}},
			W: w, Logits: make([]float32, rows), NoCPU: true}, p, jlm.RoleOutput)
	}
	if _, err := g.SetBudget(g.Bytes() + uint64(len(w.Data))/2); err != nil {
		t.Fatal(err)
	}
	if g.PrepHead(head()) {
		t.Fatal("the projection was taken in half its bytes")
	}
	if _, err := g.SetBudget(1 << 40); err != nil {
		t.Fatal(err)
	}
	if !g.PrepHead(head()) {
		t.Fatalf("with the room there the projection was refused: %s", g.Err())
	}
	if !g.HeadResident() {
		t.Fatal("the projection was taken and is not resident")
	}
}
