// The collective-tile prompt path (kernels.GemmTile, kernels.FlashPrefillTile)
// runs wherever a device lowers the ir tile ops (tier.tilesOK): Metal's
// simdgroup_matrix, and mma.sync on sm_75 and later.

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestTilePrefillMatchesTheHost prefills real models with every block placed,
// through the collective tiles -- kernels.GemmTile for the matvecs and
// kernels.FlashPrefillTile for attention -- and holds the prompt's logits AND
// the decode that reads the chunk's KV to the host.
//
// The models cover the formats and features the tile must carry: Q4_K/Q6_K,
// Q3_K/Q4_K/Q5_K/Q6_K, biased q/k/v, Q4_0/Q8_0 and a mixture. Stats().TileMV
// is the selection check, since the dp4a twin also answers correctly. CUDA
// runs its own m16n8 GEMMs and FlashPrefill80 ahead of the tiles, so the gate
// pins PreferTiles, the arm that selects the tiles there and changes nothing
// on Metal.
//
// The host uses int8 activations and the tile binary16, so the bar is NMSE
// under 2e-2 and an argmax that agrees or is a tie (tieMargin).
func TestTilePrefillMatchesTheHost(t *testing.T) {
	// One tier per model: a tier's block scratch is built from the first
	// model offered to it.
	open := func(t *testing.T) *tier.GPU {
		g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.PreferTiles = true }))
		if err != nil || g == nil {
			t.Skipf("no GPU: %v", err)
		}
		return g
	}
	ran, noTiles := 0, 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "tinyllama-1.1b-q3_K_M.gguf",
		"Qwen2-1.5B-Instruct-Q4_K_M.gguf", "gemma-2b.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf",
		"Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			text := strings.Repeat("The river rose every spring until the bridges were islands, "+
				"and the clerk counted barrels of salt on the upper floor. ", 40)
			ids := m.Vocab.Encode(text, true)[:200]
			drive := []int32{ids[3], ids[17], ids[40], ids[7]}
			run := func(s *State) [][]float32 {
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), l...)}
				for _, tok := range drive {
					if l, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), l...))
				}
				return out
			}
			host := m.NewState(len(ids) + 8)
			want := run(host)
			host.Close()

			g := open(t)
			defer g.Close()
			dev := m.NewState(len(ids) + 8)
			dev.SetDevice(g)
			if dev.GPULayers() != m.Cfg.NLayer {
				dev.Close()
				t.Fatalf("the device took %d of %d blocks: %s", dev.GPULayers(), m.Cfg.NLayer, g.Err())
			}
			got := run(dev)
			demoted, placed := dev.DeviceDemotions(), dev.GPULayers()
			dev.SetGPULayers(0)
			dev.Close()
			if demoted != 0 || placed != m.Cfg.NLayer {
				t.Fatalf("%d demotion(s), %d of %d blocks on the device: the host answered",
					demoted, placed, m.Cfg.NLayer)
			}
			for i := range want {
				var num, den float64
				for j := range want[i] {
					v := float64(got[i][j])
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Fatalf("step %d: logit %d is %v on the device", i, j, v)
					}
					d := v - float64(want[i][j])
					num += d * d
					den += float64(want[i][j]) * float64(want[i][j])
				}
				ga, wa := argmaxID(got[i]), argmaxID(want[i])
				t.Logf("step %d: logit NMSE %.3e, argmax %d (host %d)", i, num/den, ga, wa)
				if num/den > 2e-2 {
					t.Fatalf("step %d: logit NMSE %.3e against the host", i, num/den)
				}
				if ga != wa && float64(want[i][wa]-want[i][ga]) > tieMargin {
					t.Fatalf("step %d: argmax %d against the host's %d on a margin of %.3f",
						i, ga, wa, want[i][wa]-want[i][ga])
				}
			}
			// The selection check comes AFTER the comparison so that a run with
			// the tile switched off still prints the dp4a arm's numbers.
			st := g.Stats()
			if st.TileMV == 0 {
				// sm_70 and older have no m16n8 matrix instruction to lower the
				// tiles on (sm_70's m8n8k4 is GemmVolta's, and runs ahead of
				// them): the device is not asked, which is not a refusal.
				preTile := strings.Contains(g.Name(), "(sm_70)") || strings.Contains(g.Name(), "(sm_6") ||
					strings.Contains(g.Name(), "(sm_5") || strings.Contains(g.Name(), "(sm_3")
				if (strings.Contains(g.Name(), "[msl]") || strings.Contains(g.Name(), "[ptx]")) && !preTile {
					t.Fatalf("no GemmTile was built on %s (refused: %q): the dp4a twin answered", g.Name(), st.TileWhy)
				}
				noTiles++
				t.Skipf("%s does not lower the collective tiles: %s", g.Name(), st.TileWhy)
			}
			// FlashTileFor's 32-row blocking exists up to head width 128;
			// wider heads (gemma's 256) keep the three-kernel attention.
			// And only where the device multiplies float32 tiles: sm_75 has
			// the f16 tiles GemmTile needs and not these (tf32 is sm_80).
			sm75 := strings.Contains(g.Name(), "(sm_75)")
			if ft, ok := kernels.FlashTileFor(int(m.Cfg.HeadDim)); ok && ft.SG >= 4 && !sm75 && st.FlashPrefills == 0 {
				t.Fatalf("no one-kernel prompt attention ran (%d tile matvecs): FlashPrefillTile was not selected", st.TileMV)
			}
			// 200 tokens are a 224-row chunk on Metal, which no 64-token block
			// divides: the widest block runs over 192 and a tail launch over
			// 32. A device that runs the prompt in an exact grain width
			// (ExactPrompts) may not need one.
			if st.TileTail == 0 && st.ExactPrompts == 0 {
				t.Fatalf("no GemmTile ran as a wide block plus a tail (%d tile matvecs)", st.TileMV)
			}
			// A mixture's experts are the grouped GemmTile, counted apart.
			if m.Cfg.NExpert > 0 && (st.GroupedMoE == 0 || st.GroupedVolta == 0) {
				t.Fatalf("%d grouped mixture block(s), %d expert matvecs on the tile: the dp4a grouped matvec answered",
					st.GroupedMoE, st.GroupedVolta)
			}
			ran++
		})
	}
	if ran == 0 && noTiles > 0 {
		t.Skip("the device lowers no collective tiles: nothing to select")
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}
