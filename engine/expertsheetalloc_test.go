package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestExpertSheetHandoverAllocatesNothing holds the path a device block's
// expert sheets travel -- the bank's Weight, a sheet held in its page and
// released, the selected pages warmed -- at zero heap allocations once warm.
// Every streamed or hybrid block takes it on every token and every prompt
// chunk, so an allocation here is one per sheet per step in the engine's
// budget (the principle: a warm step allocates nothing).
func TestExpertSheetHandoverAllocatesNothing(t *testing.T) {
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	li := -1
	for i := range m.Layers {
		if e := m.Layers[i].ExpBank().E; e != nil && m.Container.ExpertPaged(e) {
			li = i
			break
		}
	}
	if li < 0 {
		t.Fatal("no block of the model keeps its experts in expert pages: this gate proved nothing")
	}
	l := &m.Layers[li]
	sel := []uint32{0, 1}

	bank := func() {
		m.BankWeight(m.Frames, l.Gate)
		m.BankWeight(m.Frames, l.Up)
		m.BankWeight(m.Frames, l.Down)
	}
	bank()
	if n := testing.AllocsPerRun(50, bank); n != 0 {
		t.Errorf("bankWeight of a paged bank allocates %v times a call", n)
	}

	w := m.BankWeight(m.Frames, l.Down)
	if w.Packed == nil || w.Packed.Sheet == nil {
		t.Fatal("a paged packed bank hands the device no Sheet: this gate proved nothing")
	}
	sheet := func() {
		for x := 0; x < 2; x++ {
			p, release, err := w.Packed.Sheet(x)
			if err != nil || len(p.QS) == 0 {
				t.Fatalf("sheet %d: %v, %d bytes", x, err, len(p.QS))
			}
			release()
			release() // a second release is a no-op, as a Lease's is
		}
	}
	sheet()
	if n := testing.AllocsPerRun(50, sheet); n != 0 {
		t.Errorf("handing over a sheet allocates %v times a pair", n)
	}
	// Two sheets held at once are two holds, however often one was released.
	p0, r0, err := w.Packed.Sheet(0)
	if err != nil {
		t.Fatal(err)
	}
	p1, r1, err := w.Packed.Sheet(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(p0.QS) == 0 || len(p1.QS) == 0 {
		t.Fatal("an empty sheet")
	}
	r0()
	r1()

	warm := func() {
		if err := m.EnsureBlockSelected(m.Frames, li, sel, false); err != nil {
			t.Fatal(err)
		}
	}
	warm()
	if n := testing.AllocsPerRun(50, warm); n != 0 {
		t.Errorf("warming %d selected experts of a resident block allocates %v times", len(sel), n)
	}
}
