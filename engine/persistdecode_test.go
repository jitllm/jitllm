package engine

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestPersistentDecodeMatchesPerOp decodes on the card three ways -- the
// block's launches one by one, as one persistent kernel whose phases grid
// barriers separate, and as one whose launches dependency counters order --
// teacher-forced on the CPU device's tokens, every block placed, every block
// with no graph, and half of them: the persistent arms give the per-op
// logits bit for bit, and all three stay coherent with the CPU. Its
// violations are TestPersistentDecodeSeesABrokenBarrier and
// TestDepDecodeSeesABrokenWait.
func TestPersistentDecodeMatchesPerOp(t *testing.T) {
	ran := 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf"} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %s", name)
			}
		}
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
			const steps = 24
			var toks []int32
			run := func(g *gpu.GPU, layers int) ([][]float32, int) {
				s := testInstance(t, m, len(ids)+steps+4)
				defer s.Close()
				if g != nil {
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if layers >= 0 {
						s.SetGPULayers(layers)
					}
				}
				placed := s.GPULayers()
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				var out [][]float32
				for i := range steps {
					out = append(out, append([]float32(nil), l...))
					if g == nil {
						toks = append(toks, argmaxID(l))
					}
					if l, err = s.Forward(toks[i]); err != nil {
						t.Fatal(err)
					}
				}
				if g != nil {
					s.SetGPULayers(0)
				}
				return out, placed
			}
			host, _ := run(nil, -1)
			for _, c := range []struct {
				seam    int
				noGraph bool
			}{{-1, false}, {-1, true}, {m.Cfg.NLayer / 2, false}} {
				var arms [3][][]float32
				for i, mode := range []int{1, 2, 0} {
					persist, deps := mode > 0, mode == 2
					g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff), gpu.WithConfig(func(cf *gpu.Config) {
						cf.Persist, cf.PersistDeps, cf.NoGraph = persist, deps, c.noGraph
					}))
					if err != nil || g == nil {
						t.Skipf("no GPU: %v", err)
					}
					got, placed := run(g, c.seam)
					want := m.Cfg.NLayer
					if c.seam >= 0 {
						want = c.seam
					}
					st := g.Stats()
					why := g.Err()
					g.Close()
					if placed != want {
						t.Fatalf("seam %d: the card took %d blocks, want %d: %s", c.seam, placed, want, why)
					}
					switch {
					case !persist && st.PersistLaunches != 0:
						t.Fatalf("the per-op arm launched %d persistent kernels", st.PersistLaunches)
					case persist && st.PersistLaunches == 0:
						t.Fatalf("seam %d: no persistent kernel was launched (%s): the per-op path answered", c.seam, st.PersistWhy)
					case deps && st.PersistDepLaunches == 0:
						t.Fatalf("seam %d: the dependency-counter arm launched no dep kernel (%d persistent)", c.seam, st.PersistLaunches)
					case !deps && st.PersistDepLaunches != 0:
						t.Fatalf("seam %d: an arm without PersistDeps launched %d dep kernels", c.seam, st.PersistDepLaunches)
					}
					if persist {
						t.Logf("seam %d graph %v deps %v: %d persistent launches (%d with counters) ran %d per-op launches in %d phases (%d built, %d fell back)",
							c.seam, !c.noGraph, deps, st.PersistLaunches, st.PersistDepLaunches, st.PersistParts, st.PersistPhases, st.PersistCompiles, st.PersistFallbacks)
					}
					arms[i] = got
				}
				for a, name := range []string{"grid-barrier", "dependency-counter"} {
					for i := range arms[a] {
						for j := range arms[a][i] {
							if math.Float32bits(arms[a][i][j]) != math.Float32bits(arms[2][i][j]) {
								t.Fatalf("seam %d step %d logit %d: %s %v, per-op %v -- the composition is not its parts",
									c.seam, i, j, name, arms[a][i][j], arms[2][i][j])
							}
						}
					}
				}
				for i := range arms[2] {
					nm, _ := nmseOf(arms[2][i], host[i])
					ga, ha := argmaxID(arms[2][i]), argmaxID(host[i])
					if math.IsNaN(nm) || nm > 2e-2 || ga != ha && float64(host[i][ha]-host[i][ga]) > tieMargin {
						t.Fatalf("seam %d step %d: NMSE %.3e against the CPU, argmax %d vs %d", c.seam, i, nm, ga, ha)
					}
				}
				t.Logf("seam %d graph %v: %d steps bit-identical grid-barrier, dependency-counter and per-op, coherent with the CPU", c.seam, !c.noGraph, steps)
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}
