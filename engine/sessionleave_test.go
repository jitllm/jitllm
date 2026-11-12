package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestAnInstanceLeavingTheCardLeavesAnothersBlocks places two instances of a
// model on one card, every block of each on it, and after the second's first
// token the first leaves the card -- closing, or moving every block to the
// CPU: the second keeps every block it placed and decodes bit for bit as a
// control whose first instance stays. Gemma 3n's and Gemma 4's layers that
// read an earlier layer's keys hold no history of their own, and a card that
// counted a block's sessions by their history freed them under the second
// when the first moved off.
func TestAnInstanceLeavingTheCardLeavesAnothersBlocks(t *testing.T) {
	for _, name := range []string{"synth-gemma3n.gguf", "synth-gemma4-e.gguf"} {
		path := jlmOf(t, testmodels.Path(name))
		t.Run(name, func(t *testing.T) {
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					for _, leave := range []string{"close", "cpu"} {
						t.Run(leave, func(t *testing.T) { leavingLeavesAnothersBlocks(t, path, spec, leave) })
					}
				})
			}
		})
	}
}

func leavingLeavesAnothersBlocks(t *testing.T, path, spec, leave string) {
	g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const ntok = 6
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	open := func() *Instance {
		t.Helper()
		s := testInstance(t, m, len(prompt)+ntok+1)
		s.SetDeviceFallback(false)
		if err := s.SetDevice(g); err != nil {
			s.Close()
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			s.Close()
			t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		return s
	}
	// decode is s's logits over the prompt and ntok greedy tokens, then is
	// called between the first token and the second.
	decode := func(s *Instance, then func()) [][]float32 {
		t.Helper()
		lg, err := s.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{slices.Clone(lg)}
		for j := range ntok {
			if j == 1 {
				then()
			}
			if lg, err = s.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(lg))
		}
		return out
	}
	// The control: the same two instances, the first staying. Two instances
	// on a card need not compute as one alone does (Gemma 3n's and Gemma 4's
	// part by a reduction order's rounding), so the arms differ by the leave
	// alone.
	c0, c := open(), open()
	if _, err := c0.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	ctl := decode(c, func() {})
	c.Close()
	c0.Close()

	x, y := open(), open()
	defer y.Close()
	defer x.Close()
	held := y.placedOn(spec)
	if len(held) != m.Cfg.NLayer {
		t.Fatalf("the second instance names %d of its %d blocks on %s", len(held), m.Cfg.NLayer, spec)
	}
	if _, err := x.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	got := decode(y, func() {
		switch leave {
		case "close":
			x.Close()
		case "cpu":
			if _, err := x.MoveBlocks(x.lo, x.hi, nn.CPU); err != nil || x.GPULayers() != 0 {
				t.Fatalf("the first instance kept %d blocks on %s: %v", x.GPULayers(), spec, err)
			}
		}
		if now := y.placedOn(spec); !slices.Equal(now, held) {
			t.Fatalf("after the first instance left (%s) the second holds blocks %v on %s, %v before", leave, now, spec, held)
		}
	})
	for j := range ctl {
		for k, v := range ctl[j] {
			if math.Float32bits(v) != math.Float32bits(got[j][k]) {
				t.Fatalf("token %d: logit %d is %v, its control's %v", j, k, got[j][k], v)
			}
		}
	}
	t.Logf("%s: %d blocks kept on %s; %d tokens bit for bit", leave, len(held), spec, len(ctl))
}
