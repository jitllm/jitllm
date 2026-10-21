package model

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestFusedDecodeMatchesPerOp decodes real models on a device twice -- with
// the fused kernels (kernels.RoPEQKVPaged) and with Config.NoFuse, which
// issues their parts -- and holds the two to the same logits bit for bit and
// so the same token ids: a fused kernel is its parts' emitted bodies, so any
// difference is a wiring bug. Both arms run with every block placed and with a
// partial seam (half the blocks home), and both are held to the host by the
// coherence bar the other device gates use.
//
// Stats().FusedLaunches is the selection check: the fused arm must have
// encoded the fused kernel and the per-op arm never.
func TestFusedDecodeMatchesPerOp(t *testing.T) {
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
			for _, seam := range []int{-1, m.Cfg.NLayer / 2} {
				var arms [2][][]float32
				exact := false
				for i, noFuse := range []bool{false, true} {
					g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff),
						tier.WithConfig(func(c *tier.Config) {
							// In-group splits in both arms: the operand prologue is
							// built only where few workgroups share a row (actInPays).
							c.NoFuse, c.ForceGroupSplit = noFuse, true
							// Forced: TuneOff runs no prologue tuner. 11 is q/k/v,
							// the attention output and down.
							c.ActInSites = 11
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
					fl, ai := g.Stats().FusedLaunches, g.Stats().ActInLaunches
					why := fmt.Sprintf("%s; %d prologue kernels declined at their width", g.Err(), g.Stats().ActInDeclined)
					exact = strings.Contains(g.Name(), "[ptx]") && g.Devices() == 1
					g.Close()
					if noFuse && (fl != 0 || ai != 0) {
						t.Fatalf("NoFuse encoded %d fused launches and %d operand prologues", fl, ai)
					}
					if !noFuse && ai == 0 {
						t.Fatalf("seam %d: no matvec built its own operand: the per-op path answered (%s)", seam, why)
					}
					if !noFuse && fl == 0 {
						t.Fatalf("seam %d: no fused launch was encoded: the per-op path answered", seam)
					}
					arms[i] = got
				}
				for i := range arms[0] {
					var fn, fd float64
					for j := range arms[0][i] {
						if !exact {
							d := float64(arms[0][i][j] - arms[1][i][j])
							fn, fd = fn+d*d, fd+float64(arms[1][i][j])*float64(arms[1][i][j])
							continue
						}
						if math.Float32bits(arms[0][i][j]) != math.Float32bits(arms[1][i][j]) {
							t.Fatalf("seam %d step %d logit %d: fused %v, per-op %v -- the fused kernel is not its parts",
								seam, i, j, arms[0][i][j], arms[1][i][j])
						}
					}
					if fn > 1e-6*fd || argmaxID(arms[0][i]) != argmaxID(arms[1][i]) {
						t.Fatalf("seam %d step %d: fused against per-op NMSE %.3e, argmax %d vs %d", seam, i, fn/fd,
							argmaxID(arms[0][i]), argmaxID(arms[1][i]))
					}
					if nmse, ok := coherentWithTheHost(host[i], arms[0][i]); !ok {
						t.Fatalf("seam %d step %d: NMSE %.3e against the host, argmax %d vs %d", seam, i, nmse,
							argmaxID(arms[0][i]), argmaxID(host[i]))
					}
				}
				t.Logf("seam %d: %d steps bit-identical fused and per-op, coherent with the host", seam, steps)
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}
