package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestTypedHistoryHeadsOnTheCard: a quantized history whose heads are not
// whole blocks of 32 -- synth-gemma4's sliding layers, heads of 16 -- is
// written on a card as on the CPU device, which writes each head padded to
// its blocks: the card narrows a head at a time (typedAppends). Two CPU
// instances prefill a prompt under the type, one moves to the card with
// every block and its history, and both step on the same tokens: each
// step's logits stand from the CPU twin's within ten times what the same
// arms read over an f32 history, under q8_0 and q4_0. A card appending
// floats there reads each new row's quantization error.
func TestTypedHistoryHeadsOnTheCard(t *testing.T) {
	m, _, io := openGemma4(t, "synth-gemma4")
	defer m.Close()
	odd := 0
	for li := range m.Cfg.NLayer {
		if m.Cfg.HeadDimAt(li)%32 != 0 {
			odd++
		}
	}
	if odd == 0 {
		t.Fatal("synth-gemma4 has no layer whose heads are not whole blocks of 32: the gate sees nothing")
	}
	devs := stepDevices()
	if len(devs) == 0 {
		t.Fatal("no card listed: the gate sees nothing")
	}
	ids := make([]int32, 48)
	for i := range ids {
		ids[i] = int32((i*7919 + 13) % 50000)
	}
	toks := []int32{11, 2027, 9, 501, 77, 3141, 26, 404}
	run := func(kt model.KVType) float64 {
		opts := append(append([]Option(nil), io...), WithKVType(kt), withCPUFloat())
		card, cpu := testInstance(t, m, len(ids)+len(toks)+4, opts...), testInstance(t, m, len(ids)+len(toks)+4, opts...)
		defer card.Close()
		defer cpu.Close()
		if _, err := card.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		if _, err := cpu.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		g := openCentredCard(t, devs[0], 0)
		defer g.Close()
		if err := card.SetDeviceLayers(g, m.Cfg.NLayer); err != nil {
			t.Fatal(err)
		}
		if card.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d blocks: %s", card.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		worst := 0.0
		for _, tok := range toks {
			a, err := card.Forward(tok)
			if err != nil {
				t.Fatal(err)
			}
			b, err := cpu.Forward(tok)
			if err != nil {
				t.Fatal(err)
			}
			nm, _ := nmseOf(a, b)
			if math.IsNaN(nm) {
				return math.Inf(1)
			}
			worst = max(worst, nm)
		}
		return worst
	}
	band := run(model.KVF32)
	for _, kt := range []model.KVType{model.KVQ8_0, model.KVQ4_0} {
		got := run(kt)
		t.Logf("%v: the card against its CPU twin from one history, worst logit NMSE %.3e; over an f32 history %.3e (%d of %d layers' heads not whole blocks)",
			kt, got, band, odd, m.Cfg.NLayer)
		if got > 10*max(band, 1e-9) {
			t.Errorf("%v: the card stands at NMSE %.3e from its CPU twin, past ten times the f32 history's %.3e", kt, got, band)
		}
	}
}
