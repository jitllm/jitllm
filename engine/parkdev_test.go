package engine

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestParkTakesTheCardHistoryOffAndBack: a session with every block on
// the device parks (its history comes home; the instance keeps its blocks on
// the card), resumes onto the device, and decodes the tokens its solo run on the
// device decodes.
func TestParkTakesTheCardHistoryOffAndBack(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if src := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Skipf("MODEL MISSING: %v -- this gate proved nothing", err)
	}
	defer m.Close()
	g := swaTier(t)
	defer g.Close()
	prompt := m.Vocab.Encode("Once upon a time", true)
	const half, n = 20, 40
	start := func() *Instance {
		st := testInstance(t, m, 256)
		st.SetDevice(g)
		if st.GPULayers() != m.Cfg.NLayer {
			st.Close()
			t.Skipf("CARD TOO SMALL: placed %d of %d blocks (%s) -- this gate proved nothing here",
				st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		return st
	}
	solo := start()
	lg, err := solo.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := decodeIDs(t, solo, Greedy(lg), n)
	solo.Close()

	st := start()
	defer st.Close()
	if lg, err = st.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	got, next := decodeIDs(t, st, Greedy(lg), half)
	ps, err := st.Park()
	if err != nil {
		t.Fatal(err)
	}
	// The session's history comes home; the instance's blocks stay on the
	// card, for every session leasing it.
	if ps.Blocks != m.Cfg.NLayer || heldOnDevice(st) != 0 || st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("park brought %d blocks' history home, left %d bytes of it on the device and %d of %d blocks placed",
			ps.Blocks, heldOnDevice(st), st.GPULayers(), m.Cfg.NLayer)
	}
	if _, err := st.Resume(); err != nil {
		t.Fatal(err)
	}
	if heldOnDevice(st) == 0 {
		t.Fatalf("resume left the session's history off the device (%v)", st.DeviceDeclines())
	}
	rest, _ := decodeIDs(t, st, next, n-half)
	got = append(got, rest...)
	if !slices.Equal(got, want) {
		t.Fatalf("parked and resumed on the device %v\nsolo %v", got, want)
	}
	t.Logf("%d blocks home and back, %d pages out; %d tokens equal to the solo run", ps.Blocks, ps.PagesOut, n)
}

// heldOnDevice is what st's session holds on its device now.
func heldOnDevice(st *Instance) uint64 {
	if d, ok := st.ld.(nn.Session); ok {
		return d.HeldBytes()
	}
	return 0
}
