package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestParkTakesTheCardHistoryOffAndBack: a session with every block on
// the device parks (its KV pages come home and the device's blocks are given
// back), resumes onto the device, and decodes the tokens its solo run on the
// device decodes.
func TestParkTakesTheCardHistoryOffAndBack(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if src := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	m, err := Open(p)
	if err != nil {
		t.Skipf("MODEL MISSING: %v -- this gate proved nothing", err)
	}
	defer m.Close()
	g := swaTier(t)
	defer g.Close()
	prompt := m.Vocab.Encode("Once upon a time", true)
	const half, n = 20, 40
	start := func() *State {
		st := m.NewState(256)
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
	if ps.Blocks != m.Cfg.NLayer || st.GPULayers() != 0 {
		t.Fatalf("park brought %d blocks home and left %d on the device", ps.Blocks, st.GPULayers())
	}
	if _, err := st.Resume(); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("resume put %d of %d blocks back on the device", st.GPULayers(), m.Cfg.NLayer)
	}
	rest, _ := decodeIDs(t, st, next, n-half)
	got = append(got, rest...)
	if !slices.Equal(got, want) {
		t.Fatalf("parked and resumed on the device %v\nsolo %v", got, want)
	}
	t.Logf("%d blocks home and back, %d pages out; %d tokens equal to the solo run", ps.Blocks, ps.PagesOut, n)
}
