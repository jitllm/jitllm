package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// passesCtx is the context the gate's sessions open, the eviction gate's:
// a history whose form over passes at the default 512-row reserved width
// needs more than the budgets swept hold beside the blocks.
const passesCtx = 449

// TestAPlacedBlockHoldsItsFormOverPasses sweeps a budget over
// synth-ernie45moe in the default configuration -- the 512-row reserved
// width -- and holds every placement to two things. Every block on the card
// holds an instance of its form over passes, the form a step over evicted
// history runs, at the decode scratch's rows and at the reserved width's
// (Stats.PassesUnheld): a block without one composes at a token, or fails
// there. And holding them costs no block its place: every budget from what
// the blocks and their forms take at the narrowest reserved width (32 rows,
// measured unbounded) places every block, the reserved width narrowing to
// make the room rather than a block being declined for it.
func TestAPlacedBlockHoldsItsFormOverPasses(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("synth-ernie45moe.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.Cfg.NLayer
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			// rows 0 is the default reserved width, budget 0 the card's own.
			place := func(budget uint64, rows int) (gpu.Stats, int) {
				opts := []gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
					gpu.WithConfig(func(c *gpu.Config) {
						c.KVPage = 64
						if rows > 0 {
							c.PromptRows = rows
						}
					})}
				if budget > 0 {
					opts = append(opts, gpu.WithBudget(budget))
				}
				g, err := openGPU(t, append(opts, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				s := testInstance(t, m, passesCtx, WithStreamTrial(false), WithHeadPlacement(false), WithKVPrewarm(false))
				defer s.Close()
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				return g.Stats(), s.GPULayers()
			}
			all, placed := place(0, batchGrainRows)
			if placed != n || all.ReservedRows != batchGrainRows || all.PassesUnheld != 0 {
				t.Fatalf("unbounded at %d reserved rows: %d of %d blocks placed, %d reserved rows, %d without their form over passes: %s",
					batchGrainRows, placed, n, all.ReservedRows, all.PassesUnheld, all.LastErr)
			}
			need := all.BudgetUsed + all.Slack
			step := need / 64
			lowest := uint64(0)
			for b := need * 5 / 8; b <= need*5/4; b += step {
				st, placed := place(b, 0)
				t.Logf("budget %d: %d of %d blocks placed, %d streamed, %d reserved rows, %d without their form over passes, %d used",
					b, placed, n, st.StreamBlocks, st.ReservedRows, st.PassesUnheld, st.BudgetUsed)
				if st.PassesUnheld != 0 {
					t.Errorf("under %d bytes %d of the %d blocks placed hold no instance of their form over passes at %d reserved rows: "+
						"a step over evicted history composes at a token (%s)", b, st.PassesUnheld, placed, st.ReservedRows, st.LastErr)
				}
				if placed == n && lowest == 0 {
					lowest = b
				}
				if placed != n && b >= need {
					t.Errorf("under %d bytes, past the %d every block and its forms take at %d reserved rows, %d of %d blocks placed at %d reserved rows: %s",
						b, need, batchGrainRows, placed, n, st.ReservedRows, st.LastErr)
				}
			}
			if lowest == 0 {
				t.Logf("no budget swept placed every block; the blocks and their forms take %d at %d reserved rows", need, batchGrainRows)
			} else {
				t.Logf("every block placed from %d bytes; the blocks and their forms take %d at %d reserved rows", lowest, need, batchGrainRows)
			}
		})
	}
}

// batchGrainRows is the narrowest reserved width (the device's batchGrain).
const batchGrainRows = 32
