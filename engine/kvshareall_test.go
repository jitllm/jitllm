package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestKVSharingOnEveryBlockKind gives a latent-attention model, two hybrids
// and a mixture a KV-sharing last block (Gemma 4's E models' feature): it
// reads the last earlier block that attends instead of writing a history of
// its own. The shared run must move the logits (the block read another
// block's rows), name an attending source, and agree between the CPU and a
// card that holds every block as closely as the unshared model's two arms
// agree.
func TestKVSharingOnEveryBlockKind(t *testing.T) {
	const gen = 6
	ran := 0
	for _, name := range []string{"synth-deepseek-yarn.gguf", "synth-falconh1.gguf", "synth-qwen35-hybrid-mtp.gguf",
		"synth-qwen2moe.gguf"} {
		t.Run(name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s", testmodels.Path(name))
			}
			path := jlmOf(t, src)
			run := func(share bool, g *gpu.GPU) [][]float32 {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				c := m.Cfg
				// The last block that attends with an earlier one that does.
				last, first := -1, -1
				for li := range c.NLayer {
					if c.LayerKind(li).Attends() {
						if first < 0 {
							first = li
						} else {
							last = li
						}
					}
				}
				if last < 0 {
					t.Fatalf("%s has no two blocks that attend", name)
				}
				if share {
					c.NKVShared = c.NLayer - last
					// A sharing block stores no key, value or latent of its own.
					m.Front.Restate = func(block int32, inv map[jlm.Role]graph.Shape) {
						if int(block) == last {
							for _, r := range []jlm.Role{jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnKBias, jlm.RoleAttnVBias,
								jlm.RoleAttnKNorm, jlm.RoleAttnKVA, jlm.RoleAttnKVANorm} {
								delete(inv, r)
							}
						}
					}
					m.Front.Restated++
					m.Front.Set = nil
					src := c.KVSource(last)
					if !c.KVShared(last) || src < 0 || src >= last || !c.LayerKind(src).Attends() {
						t.Fatalf("block %d shares %v from %d", last, c.KVShared(last), src)
					}
				}
				ids := m.Vocab.Encode("The capital of France is", true)
				s := testInstance(t, m, len(ids)+gen+1, WithKVF16(false))
				defer s.Close()
				if g != nil {
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if s.GPULayers() != c.NLayer {
						t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), c.NLayer, g.Err())
					}
				}
				lg, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), lg...)}
				for range gen {
					if lg, err = s.Forward(Greedy(lg)); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out
			}
			worst := func(a, b [][]float32) float64 {
				w := 0.0
				for i := range a {
					w = max(w, logitNMSE(a[i], b[i]))
				}
				return w
			}
			plain, shared := run(false, nil), run(true, nil)
			if moved := worst(shared, plain); moved == 0 {
				t.Fatal("the shared block's logits are the unshared model's: it read its own history")
			}
			ran++
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					band := worst(run(false, g), plain)
					got := worst(run(true, g), shared)
					t.Logf("shared: %.3e from the CPU's, unshared %.3e", got, band)
					if got > 10*max(band, 1e-9) {
						t.Fatalf("the shared model's card parts from its CPU by %.3e, the unshared model's by %.3e", got, band)
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no model ran")
	}
}
