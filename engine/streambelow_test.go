package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAMixtureBlockStreamsTheBlocksBelowIt places synth-ernie45moe under a
// budget 32 KB past what every block placed streamed takes, where the last
// block does not fit resident: streamed alone it pays the streamed banks'
// shared bank by itself and misses too, so the mixture block below it streams
// and both stay on the card, rather than the last going to the CPU (it did, 3
// of 4 blocks on the card, at margins of 16 to 64 KB). Placed one block at a
// time the card falls 12.8 KB short of the last block's composed buffers at
// the all-streamed bytes exactly, hence the margin. The run's experts are read
// through the streamed banks: fills, not a resident bank.
func TestAMixtureBlockStreamsTheBlocksBelowIt(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("synth-ernie45moe.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is a city that", true)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			open := func(budget uint64, stream bool) (*gpu.GPU, *Instance) {
				opts := []gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
					gpu.WithConfig(func(c *gpu.Config) {
						c.PromptRows = 32
						if stream {
							c.StreamExperts, c.StreamCacheSlots = true, -1
						}
					})}
				if budget > 0 {
					opts = append(opts, gpu.WithBudget(budget))
				}
				g, err := openGPU(t, append(opts, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				s := testInstance(t, m, len(ids)+8, WithStreamTrial(false), WithHeadPlacement(false), WithKVPrewarm(false))
				s.SetDeviceFallback(false)
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				return g, s
			}
			g, s := open(0, true)
			all := g.Stats()
			s.Close()
			g.Close()
			if all.StreamBlocks != mixtureBlocks(m) {
				t.Fatalf("%d of %d mixture blocks streamed with every expert streamed", all.StreamBlocks, mixtureBlocks(m))
			}
			budget := all.BudgetUsed + all.Slack + 32<<10
			g, s = open(budget, false)
			defer g.Close()
			defer s.Close()
			streamed := 0
			for li := range m.Cfg.NLayer {
				if s.streamedAt(li) {
					streamed++
				}
			}
			if s.GPULayers() != m.Cfg.NLayer || streamed < 2 {
				t.Fatalf("under %d bytes, where every block streamed holds %d, the card took %d of %d blocks, %d streamed: %s",
					budget, all.BudgetUsed, s.GPULayers(), m.Cfg.NLayer, streamed, g.Err())
			}
			if _, err := s.Prefill(ids); err != nil {
				t.Fatal(err)
			}
			if _, err := s.Forward(ids[0]); err != nil {
				t.Fatal(err)
			}
			if st := g.Stats(); st.StreamFills == 0 || s.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("%d fills, %d of %d blocks on the card after a step: the streamed banks did not run", st.StreamFills, s.GPULayers(), m.Cfg.NLayer)
			}
			t.Logf("under %d bytes the card holds %d blocks, %d streamed, %d fills", budget, s.GPULayers(), streamed, g.Stats().StreamFills)
		})
	}
}
