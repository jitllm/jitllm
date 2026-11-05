package engine

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStagedGemmPrefillMatchesTheHost prefills real models on CUDA with every
// block placed and holds the prompt's logits AND the decode that reads the
// chunk's KV to the host, in two arms: the composer's prompt forms on the
// card's matrix unit (the default where the card lowers a tile), and with
// none on it (Config.NoMMA), whose projections read int8 activations as the
// host's do.
//
// The models cover the formats and features the forms carry: Q4_K/Q6_K,
// Q3_K beside Q4_K/Q5_K/Q6_K, Q8_0, biased q/k/v, Q4_0, the hybrid's
// projections and a mixture's dense ones. The selection checks are every
// block of the prompt served by the composer and the matrix unit's launches
// (Stats.MatrixPart): some in the default arm on a card that lowers a tile,
// none in the other.
//
// The bar is the tile's: NMSE under 2e-2 and an argmax that agrees or is a
// tie (tieMargin). The split tuner is pinned off (RULE 11c).
func TestStagedGemmPrefillMatchesTheHost(t *testing.T) {
	ran := 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf",
		"tinyllama-1.1b-q3_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "gemma-2b.gguf",
		"Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		t.Run(name, func(t *testing.T) {
			instOpts := []Option{WithKVF16(false)}
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			text := strings.Repeat("The river rose every spring until the bridges were islands, "+
				"and the clerk counted barrels of salt on the upper floor. ", 40)
			ids := m.Vocab.Encode(text, true)[:200]
			drive := []int32{ids[3], ids[17], ids[40], ids[7]}
			run := func(s *Instance) [][]float32 {
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
			host := testInstance(t, m, len(ids)+8, instOpts...)
			want := run(host)
			host.Close()
			for _, arm := range []struct {
				name     string
				noMatrix bool
			}{{"matrix unit", false}, {"no matrix unit", true}} {
				t.Run(arm.name, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithAPI("ptx"), gpu.WithDeviceTune(gpu.TuneOff),
						gpu.WithConfig(func(c *gpu.Config) { c.NoMMA = arm.noMatrix }))
					if err != nil || g == nil {
						noDevice(t, "cuda", err)
					}
					defer g.Close()
					if !strings.Contains(g.Name(), "[ptx]") {
						t.Skipf("the device is %s; this gate is CUDA's", g.Name())
					}
					stagedGemmArm(t, m, g, len(ids)+8, run, want, arm.noMatrix, instOpts...)
				})
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}

// stagedGemmArm is one arm of TestStagedGemmPrefillMatchesTheHost on g.
func stagedGemmArm(t *testing.T, m *model.Model, g *gpu.GPU, seq int, run func(*Instance) [][]float32, want [][]float32, noMatrix bool, io ...Option) {
	t.Helper()
	s0 := g.Stats()
	dev := testInstance(t, m, seq, io...)
	dev.SetDevice(g)
	if dev.GPULayers() != m.Cfg.NLayer {
		dev.Close()
		t.Skipf("CARD TOO SMALL: the device took %d of %d blocks (%s) -- this model proved nothing here",
			dev.GPULayers(), m.Cfg.NLayer, g.Err())
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
	// After the comparison, so an arm that selected nothing still prints the
	// other's numbers.
	st := g.Stats()
	prompt, matrix := servedSince(s0, st, nn.ComposedPrompt), st.MatrixPart(nn.EveryPart)-s0.MatrixPart(nn.EveryPart)
	sm := ptxTargetSM(g.Name())
	t.Logf("%s: %d composed prompt block steps, %d launches on the matrix unit", g.Name(), prompt, matrix)
	switch {
	case prompt < m.Cfg.NLayer:
		t.Fatalf("%d composed prompt block steps for %d blocks: the prompt took another path", prompt, m.Cfg.NLayer)
	case noMatrix && matrix != 0:
		t.Fatalf("no matrix unit asked and %d launches on it: the arm is not the int8 one", matrix)
	case !noMatrix && sm >= 70 && matrix == 0:
		t.Fatalf("%s lowers the binary16 tile and the composer formed nothing on it", g.Name())
	}
}

// ptxTargetSM reads the target out of a CUDA device's name, "... (sm_86)
// [ptx]", and is 0 when there is none.
func ptxTargetSM(name string) int {
	i := strings.LastIndex(name, "(sm_")
	if i < 0 {
		return 0
	}
	sm := 0
	for _, c := range name[i+4:] {
		if c < '0' || c > '9' {
			break
		}
		sm = sm*10 + int(c-'0')
	}
	return sm
}
