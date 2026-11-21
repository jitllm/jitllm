package engine

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// longBidirModel is Gemma 3 4B, whose pictures are bidirectional runs.
func longBidirModel(t *testing.T) *model.Model {
	t.Helper()
	text := testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")
	if !sourcePresent(text) {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", text)
	}
	m, err := model.Open(jlmOf(t, text))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// longBidirSpans is a prompt of text, a run of text past one submission's
// rows marked bidirectional (as a picture's span is), and text after it.
func longBidirSpans(t *testing.T, m *model.Model, bidir bool) []model.Span {
	t.Helper()
	pre := m.Vocab.Encode("Read the notes below and answer the question after them.", true)
	run := m.Vocab.Encode(strings.Repeat("The ledger lists the parcels by weight, the weights by day, and the days by the "+
		"van that carried them; a parcel that missed its van waits on the shelf by the door. ", 26), false)
	post := m.Vocab.Encode(" Which parcels waited on the shelf?", false)
	if len(run) <= nn.MaxDevicePrefillChunk || len(run) > 2*nn.MaxDevicePrefillChunk {
		t.Fatalf("the run is %d rows: the gate wants one past a %d-row submission and inside two", len(run), nn.MaxDevicePrefillChunk)
	}
	return []model.Span{{Tokens: pre}, {Tokens: run, Bidir: bidir}, {Tokens: post}}
}

// longBidirLogits is the last row's logits of the prompt, layers of its
// blocks on g where g is not nil, the CPU on its float forms over an f32
// history (withCPUFloat): the band is the reduction order's.
func longBidirLogits(t *testing.T, m *model.Model, g *gpu.GPU, layers int, spans []model.Span) []float32 {
	t.Helper()
	in := testInstance(t, m, model.SpanPositions(spans, m.Cfg.NEmbd)+1, WithKVF16(false), withCPUFloat())
	defer in.Close()
	if g != nil {
		if err := in.SetDeviceLayers(g, layers); err != nil {
			t.Fatal(err)
		}
		if in.GPULayers() != layers {
			t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), layers, g.Err())
		}
	}
	lg, err := in.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), lg...)
}

// runPhases is the phase calls of runs wider than a submission g's devices
// ran.
func runPhases(g *gpu.GPU) int {
	n := 0
	for _, s := range g.DevStats() {
		n += s.RunPhases
	}
	return n
}

// longBidirBound is how many times the causal band the run's logits may
// stand from the CPU's.
const longBidirBound = 3.0

// longBidirArm is the CPU's logits of the prompt with the run and causal.
type longBidirArm struct {
	m              *model.Model
	cpuRun, cpuCau []float32
}

func newLongBidirArm(t *testing.T, m *model.Model) *longBidirArm {
	return &longBidirArm{m: m, cpuRun: longBidirLogits(t, m, nil, 0, longBidirSpans(t, m, true)),
		cpuCau: longBidirLogits(t, m, nil, 0, longBidirSpans(t, m, false))}
}

// TestLongBidirRunMatchesTheCPU prefills Gemma 3 4B's prompt holding a run
// of about 950 rows attending both ways on the CPU, where the run is one
// chunk, and with every block on each card, where a submission holds 512
// rows: the card runs the blocks one at a time, every piece's keys and
// values first and then every piece's attention over the whole run
// (gpu.GPU.phased); and with half the blocks there, the rest on the CPU.
// The card's logits are held to the CPU's within three times the band the
// same prompt read causally on both shows, at the same seam (measured 1.7x
// on CUDA, under 1x on Vulkan); the phases ran;
// and the run is load-bearing -- its logits stand far from the causal
// prompt's. The violation (the run cut at the submission) is
// TestLongBidirRunCutIsSeen, under jitllmfault.
func TestLongBidirRunMatchesTheCPU(t *testing.T) {
	m := longBidirModel(t)
	defer m.Close()
	a := newLongBidirArm(t, m)
	load, _ := nmseOf(a.cpuCau, a.cpuRun)
	t.Logf("CPU: the run against the causal prompt, logit NMSE %.3e", load)
	if !(load > 1e-3) {
		t.Fatalf("the run reads %.3e from the causal prompt on the CPU: it is not load-bearing", load)
	}
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			for _, layers := range []int{m.Cfg.NLayer, m.Cfg.NLayer / 2} {
				band, _ := nmseOf(longBidirLogits(t, m, g, layers, longBidirSpans(t, m, false)), a.cpuCau)
				before := runPhases(g)
				got := longBidirLogits(t, m, g, layers, longBidirSpans(t, m, true))
				phases := runPhases(g) - before
				nm, worst := nmseOf(got, a.cpuRun)
				t.Logf("%s, %d of %d blocks: the run against the CPU's, logit NMSE %.3e (max|d| %.4f); causal band %.3e; %d phase calls",
					spec, layers, m.Cfg.NLayer, nm, worst, band, phases)
				if phases < 2*layers {
					t.Fatalf("%d phase calls for %d blocks: the run did not go in phases", phases, layers)
				}
				if math.IsNaN(nm) || nm > longBidirBound*max(band, 1e-6) {
					t.Errorf("the run on %s, %d blocks, reads NMSE %.3e against the CPU's, past %gx the causal band %.3e", spec, layers, nm, longBidirBound, band)
				}
			}
		})
	}
}
