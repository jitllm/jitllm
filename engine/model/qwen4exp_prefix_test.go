package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestPrefixCacheRestoresQwen4Exp: a prompt restored from the store goes on
// exactly as the one that computed it, on the host and with every block on
// each device. What a restore must bring back here is more than KV pages:
// each linear block's state slot, which on the n-gram block carries the
// tokens the next n-gram hashes and the convolution's history, and the QSA
// blocks' rows with their raw indexer keys. The prompt crosses the
// end-of-sequence reset and selects (past position 12); the continuation's
// logits are compared, not only its tokens.
func TestPrefixCacheRestoresQwen4Exp(t *testing.T) {
	m, g := openQ4E(t, "synth-qwen4exp")
	defer m.Close()
	const page, gen = 8, 6
	prompt := g.IDs[:len(g.IDs)-gen]
	arms := append([]string{"host"}, stepDevices()...)
	for _, arm := range arms {
		var gpu *tier.GPU
		if arm != "host" {
			var err error
			gpu, err = openTier(t, append([]tier.Option{tier.WithDevices(arm),
				tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
			if err != nil || gpu == nil {
				t.Logf("%s: not present (%v)", arm, err)
				continue
			}
		}
		t.Run(arm, func(t *testing.T) {
			if gpu != nil {
				defer gpu.Close()
			}
			fs, err := NewFileStore(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			run := func() ([][]float32, int) {
				defer m.setKVPageForTest(page)()
				s := m.NewState(len(g.IDs) + 1)
				defer s.Close()
				if gpu != nil {
					s.SetDeviceLayers(gpu, -1)
					if s.GPULayers() != m.Cfg.NLayer {
						t.Fatalf("placed %d of %d blocks: %v", s.GPULayers(), m.Cfg.NLayer, gpu.Err())
					}
				}
				s.SetKVStore(fs)
				mustKey(t, s, "qwen4exp/"+arm)
				lg, err := s.PrefillCached(prompt)
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), lg...)}
				for _, id := range g.IDs[len(prompt):] {
					if lg, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out, s.KVRestored()
			}
			want, first := run()
			if first != 0 {
				t.Fatalf("the first pass restored %d positions from an empty store", first)
			}
			got, restored := run()
			if restored == 0 {
				t.Fatal("the second pass restored nothing: this gate would test the prefill twice")
			}
			worst := 0.0
			for p := range want {
				var num, den float64
				for i := range want[p] {
					d := float64(got[p][i] - want[p][i])
					num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
				}
				worst = max(worst, num/den)
			}
			if !(worst < 1e-9) {
				t.Fatalf("restored %d positions, then the continuation's logits are %.3e from the computed run's",
					restored, worst)
			}
			t.Logf("restored %d of %d positions; the continuation's logits within %.3e", restored, len(prompt), worst)
		})
	}
}
