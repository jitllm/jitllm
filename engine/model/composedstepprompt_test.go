package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedPromptRunsOnTheMatrixUnit prefills a two-chunk prompt on a
// device whose composer serves a step's parts and the prompt's
// (gpu.ComposedStep, the jitllmcomposed build's default): the prompt's
// projections run on the matrix unit -- a dense model's, and a mixture's
// routed experts grouped by expert -- at least one matrix launch a block of
// every chunk (Stats.MatrixPart of the prompt) -- and its logits sit from
// the host's no further than bandRatio times the tier's own prompt's, with
// the same argmax.
func TestComposedPromptRunsOnTheMatrixUnit(t *testing.T) {
	for _, name := range []string{"Qwen3-0.6B-Q8_0", "Qwen3-MOE-4x0.6B-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			m, ids, want := composedPromptModel(t, name)
			defer m.Close()
			got, s0, s1 := composedPrompt(t, m, ids, gpu.ComposedStep)
			n := s1.MatrixPart(gpu.ComposedPrompt) - s0.MatrixPart(gpu.ComposedPrompt)
			chunks := (len(ids) + nn.MaxDevicePrefillChunk - 1) / nn.MaxDevicePrefillChunk
			if n < chunks*m.Cfg.NLayer {
				t.Fatalf("the prompt's blocks launched %d kernels on the matrix unit, fewer than one a block of each of its %d chunks (%d)",
					n, chunks, chunks*m.Cfg.NLayer)
			}
			e := promptNMSE(t, got, want)
			tierGot, _, _ := composedPrompt(t, m, ids, gpu.ComposedStep&^gpu.ComposedPrompt)
			te := promptNMSE(t, tierGot, want)
			if !(e <= bandRatio*te) || Greedy(got) != Greedy(want) {
				t.Fatalf("the prompt's logits NMSE %.3e against the host, the tier's %.3e (band %.1fx), argmax %d against %d",
					e, te, bandRatio, Greedy(got), Greedy(want))
			}
			t.Logf("%d matrix launches, %d chunks of %d blocks, logits NMSE %.3e against the host, the tier's %.3e", n, chunks, m.Cfg.NLayer, e, te)
			for k := 2; k <= matrixTileCount(); k++ {
				got, s0, s1 := composedPromptAt(t, m, ids, gpu.ComposedStep, k)
				tiled := s1.MatrixTiled
				n := s1.MatrixPart(gpu.ComposedPrompt) - s0.MatrixPart(gpu.ComposedPrompt)
				if tiled == 0 || n < chunks*m.Cfg.NLayer {
					t.Fatalf("tile %d: %d parts formed at it, %d matrix launches", k, tiled, n)
				}
				e := promptNMSE(t, got, want)
				if !(e <= bandRatio*te) || Greedy(got) != Greedy(want) {
					t.Fatalf("tile %d: the prompt's logits NMSE %.3e against the host, the tier's %.3e (band %.1fx), argmax %d against %d",
						k, e, te, bandRatio, Greedy(got), Greedy(want))
				}
				t.Logf("tile %d: %d parts formed at it, %d matrix launches, logits NMSE %.3e", k, tiled, n, e)
			}
		})
	}
}

// composedPromptModel opens a checkpoint, a prompt of 700 tokens and the
// host's logits after it.
func composedPromptModel(t *testing.T, name string) (*Model, []int32, []float32) {
	t.Helper()
	src := testmodels.Path(name + ".gguf")
	if !fileExists(src) {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", src)
	}
	m, err := Open(jlmOf(t, src), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 40)
	ids := m.Vocab.Encode(text, true)[:700]
	host := m.NewInstance(len(ids) + 4)
	want, err := host.Prefill(ids)
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()
	return m, ids, want
}

// composedPrompt prefills ids with every block on a device the composer
// serves parts of, and the logits and the device's stats before and after.
func composedPrompt(t *testing.T, m *Model, ids []int32, parts gpu.Parts) ([]float32, gpu.Stats, gpu.Stats) {
	t.Helper()
	return composedPromptAt(t, m, ids, parts, 0)
}

// matrixTileCount is the GPU profile's matrix tiles, its own among them.
func matrixTileCount() int {
	for _, p := range composer.Default().Profiles[composer.DeviceGPU].Passes {
		if mx, ok := p.(composer.Matrix); ok {
			return 1 + len(mx.Tiles)
		}
	}
	return 0
}

// composedPromptAt is composedPrompt with the matrix tile forced
// (gpu.Config.MatrixTile).
func composedPromptAt(t *testing.T, m *Model, ids []int32, parts gpu.Parts, tile int) ([]float32, gpu.Stats, gpu.Stats) {
	t.Helper()
	g, err := gpu.OpenWith(gpu.WithDeviceTune(gpu.TuneOff), gpu.WithConfig(func(c *gpu.Config) { c.MatrixTile = tile }))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	t.Cleanup(g.Close)
	st := m.NewInstance(len(ids) + 4)
	t.Cleanup(func() { st.Close() })
	st.SetDevice(g)
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("placed %d of %d blocks: %s", st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	s0 := g.Stats()
	got, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if st.DeviceDemotions() != 0 {
		t.Fatalf("the device demoted: %s", g.Err())
	}
	return append([]float32(nil), got...), s0, g.Stats()
}

// promptNMSE is a prompt's logits' NMSE against the host's, every one
// finite.
func promptNMSE(t *testing.T, got, want []float32) float64 {
	t.Helper()
	var num, den float64
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("logit %d is %g", i, got[i])
		}
		d := float64(got[i] - want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	return num / den
}
