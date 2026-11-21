package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestCutBlocksKeepATypedHistory: TestRoutedExpertsRunOnTheCPUBesideTheCard
// under a q8_0 and a q4_0 history -- every mixture block cut at its routed
// experts (WithExperts("cpu")), its first half on the card from the typed
// graph cut, its experts on the CPU device -- held to the CPU device's own
// decode under the same history within four times what the whole model on
// the card reads under it. A first half appending floats where the CPU
// device appends the type's values reads the type's loss.
func TestCutBlocksKeepATypedHistory(t *testing.T) {
	ids := []int32{1, 5, 9, 13, 2}
	toks := []int32{3, 7, 11, 4}
	for _, kt := range []model.KVType{model.KVQ8_0, model.KVQ4_0} {
		for _, dev := range stepDevices() {
			t.Run(kt.String()+"/"+dev, func(t *testing.T) {
				m, _, openOpts := openMoE(t, "synth-qwen2moe")
				defer m.Close()
				io := append(slices.Clip(openOpts), WithKVType(kt))
				want := cpuDecode(t, m, ids, toks, io...)
				card, _, _ := composedDecode(t, m, slices.Clip(io), dev, ids, toks, nn.ComposedBlock, false, false)
				g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				s := testInstance(t, m, len(ids)+len(toks)+4, append(slices.Clip(io), WithExperts("cpu"), WithStreamTrial(false))...)
				defer s.Close()
				s.SetDeviceFallback(false)
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				mixtures := 0
				for li := range m.Cfg.NLayer {
					if m.Cfg.MoEAt(li) {
						if !s.cutAt(li) {
							t.Fatalf("mixture block %d is not cut (card: %q, declines: %v)", li, g.Err(), s.DeviceDeclines())
						}
						mixtures++
					}
				}
				if mixtures == 0 {
					t.Fatal("the fixture has no mixture block")
				}
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				var got [][]float32
				for _, tok := range toks {
					got = append(got, append([]float32(nil), l...))
					if l, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
				}
				got = append(got, append([]float32(nil), l...))
				worst, cardWorst := 0.0, 0.0
				for i := range want {
					e, c := logitNMSE(want[i], got[i]), logitNMSE(want[i], card[i])
					if math.IsNaN(e) || math.IsInf(e, 0) {
						t.Fatalf("step %d: non-finite logits (NMSE %v)", i, e)
					}
					worst, cardWorst = max(worst, e), max(cardWorst, c)
				}
				t.Logf("%v: %d mixtures cut, worst NMSE %.3g from the CPU device's logits under the history, the whole card's %.3g",
					kt, mixtures, worst, cardWorst)
				if worst > 4*max(cardWorst, 1e-7) {
					t.Fatalf("%v: the cut blocks part from the CPU device by %.3g, past 4x the whole card's %.3g", kt, worst, cardWorst)
				}
			})
		}
	}
}
