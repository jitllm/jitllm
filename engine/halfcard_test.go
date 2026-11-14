package engine

import (
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestTwoSessionsStepOnAHalfCard places Llama-3.2-1B on a card whose budget
// holds half its weights and then some, at eight sizes a quarter of a block
// apart: the card is listed alone and pages the blocks it cannot hold. At
// each size it keeps every block through three tokens and through two
// sessions stepped together (stepsTogetherOnCard: nothing formed at the
// step, every block served on the card).
//
// Two refusals lived here. A reserved width narrowed for the head's per-row
// logits that did not build left the card none, and the two sessions' step
// fell to a width whose head nothing formed; and a step whose history
// reservation found no room, on a card with nowhere to move a block, was
// refused before its own pages could stream blocks for the room.
func TestTwoSessionsStepOnAHalfCard(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	probe, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	total, layers := probe.WeightBytes(), uint64(probe.Cfg.NLayer)
	probe.Close()
	for k := range uint64(8) {
		budget := total/2 + k*total/layers/4
		t.Run(fmt.Sprintf("budget %d", budget), func(t *testing.T) {
			spec := testdevices.Like(t, "cuda:0") + "=" + strconv.FormatUint(budget, 10)
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
			st := cardInstance(t, m, g)
			defer st.Close()
			for i := range 3 {
				lg, err := st.Forward(int32(1 + i))
				if err != nil {
					t.Fatalf("Forward on the card: %v (%s)", err, g.Err())
				}
				finite(t, "Forward on the card", lg)
			}
			stayedOnCard(t, st, m, g)
			stepsTogetherOnCard(t, st, m, g)
			stayedOnCard(t, st, m, g)
		})
	}
}
