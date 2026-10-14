package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPersistentDecodeMatchesPerOp decodes real models on a device twice --
// with Config.Persist, which composes each decode block's launches into
// persistent kernels (tier/persist.go, ir.Persist), and without -- and holds
// the two to the same logits bit for bit and so the same token ids: a
// persistent kernel runs the per-op kernels' own bodies, each output by the
// same thread in the same order, so any difference is a composition or
// grid-barrier bug. Both arms run with every block placed and with a partial
// seam (half the blocks home), with the captured graph and without it, and
// both are held to the host by the coherence bar the other device gates use.
//
// Stats().PersistLaunches is the selection check: the persistent arm must
// have launched composed kernels and the per-op arm none.
func TestPersistentDecodeMatchesPerOp(t *testing.T) {
	ran := 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf",
		"gemma-2b.gguf"} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, p), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
			const steps = 24
			// run greedily decodes after the prompt and returns every step's
			// logits; layers < 0 places every block.
			// The host chooses the tokens and every arm is fed them
			// (teacher-forced), so a near-tie the device breaks the other way
			// cannot send the arms down different sequences.
			var toks []int32
			run := func(g *tier.GPU, layers int) ([][]float32, int) {
				s := m.NewState(len(ids) + steps + 4)
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
				for i := 0; i < steps; i++ {
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
				seam := c.seam
				// Arms: grid barriers, dependency counters, per-op (the last
				// is the reference).
				var arms [3][][]float32
				for i, mode := range []int{1, 2, 0} {
					persist, deps := mode > 0, mode == 2
					g, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff),
						tier.WithConfig(func(cf *tier.Config) {
							cf.Persist, cf.PersistDeps, cf.NoGraph = persist, deps, c.noGraph
						}))
					if err != nil || g == nil {
						t.Skipf("no GPU: %v", err)
					}
					got, placed := run(g, seam)
					want := m.Cfg.NLayer
					if seam >= 0 {
						want = seam
					}
					if placed != want {
						g.Close()
						t.Fatalf("seam %d: the device took %d blocks, want %d: %s", seam, placed, want, g.Err())
					}
					st := g.Stats()
					g.Close()
					if !persist && st.PersistLaunches != 0 {
						t.Fatalf("the per-op arm launched %d persistent kernels", st.PersistLaunches)
					}
					if persist && st.PersistLaunches == 0 {
						if st.PersistWhy != "" {
							t.Skipf("the device declines the persistent path: %s", st.PersistWhy)
						}
						t.Fatalf("seam %d: no persistent kernel was launched: the per-op path answered", seam)
					}
					if deps && st.PersistDepLaunches == 0 {
						t.Fatalf("seam %d: the dependency-counter arm launched no dep kernel (%d persistent)", seam, st.PersistLaunches)
					}
					if !deps && st.PersistDepLaunches != 0 {
						t.Fatalf("seam %d: an arm without PersistDeps launched %d dep kernels", seam, st.PersistDepLaunches)
					}
					if persist {
						t.Logf("seam %d graph %v deps %v: %d persistent launches (%d with counters, %d tile edges) ran %d per-op launches in %d phases (%d built, %d fell back)",
							seam, !c.noGraph, deps, st.PersistLaunches, st.PersistDepLaunches, st.PersistDepTiles, st.PersistParts, st.PersistPhases, st.PersistCompiles, st.PersistFallbacks)
					}
					arms[i] = got
				}
				for a, name := range []string{"grid-barrier", "dependency-counter"} {
					for i := range arms[a] {
						for j := range arms[a][i] {
							if math.Float32bits(arms[a][i][j]) != math.Float32bits(arms[2][i][j]) {
								t.Fatalf("seam %d step %d logit %d: %s %v, per-op %v -- the composition is not its parts",
									seam, i, j, name, arms[a][i][j], arms[2][i][j])
							}
						}
					}
				}
				for i := range arms[0] {
					var num, den float64
					for j := range host[i] {
						d := float64(arms[0][i][j] - host[i][j])
						num += d * d
						den += float64(host[i][j]) * float64(host[i][j])
					}
					ga, ha := argmaxID(arms[0][i]), argmaxID(host[i])
					if num/den > 2e-2 || ga != ha && float64(host[i][ha]-host[i][ga]) > tieMargin {
						t.Fatalf("seam %d step %d: NMSE %.3e against the host, argmax %d vs %d", seam, i, num/den, ga, ha)
					}
				}
				t.Logf("seam %d graph %v: %d steps bit-identical grid-barrier, dependency-counter and per-op, coherent with the host", seam, !c.noGraph, steps)
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}
