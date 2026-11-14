package engine

import (
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestInputRowsAFullHeadCardCannotGather places block 0 on one card and the
// head on a second, which gathers a chunk's input rows and reads them back for
// the first (GPU.embedRows), and caps the second at what the placement holds
// plus a MiB: no room for the rows' buffer. The card must decline the rows,
// counted, and the host's input rows run in their place, the logits within
// three times the band the placement with room shows. It answered for rows
// nothing wrote: the first chunk ran on zeros, every history page of the
// second card's blocks zero, NMSE 1.46 against the host.
func TestInputRowsAFullHeadCardCannotGather(t *testing.T) {
	p := jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm"))
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var prompt []int32
	for len(prompt) < 1100 {
		prompt = append(prompt, m.Vocab.Encode("The capital of France is Paris. Water boils at 100 degrees. ", false)...)
	}
	host := testInstance(t, m, len(prompt)+2)
	want, err := host.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()
	second := testdevices.Within([]string{"cuda:1", "vulkan:1"})
	if len(second) == 0 {
		noDevice(t, "second device", nil)
	}
	testdevices.Require(t, "cuda:0", second[0])
	// The last block on the CPU, the head on the second card.
	place := Placement{Blocks: map[int]Place{}, Head: &Place{On: second[0]}, Strict: true}
	for li := 0; li < m.Cfg.NLayer; li++ {
		place.Blocks[li] = Place{On: second[0]}
		if li < 3 {
			place.Blocks[li] = Place{On: "cuda:0"}
		}
	}
	place.Blocks[m.Cfg.NLayer-1] = Place{On: nn.CPU}
	// What the second card holds placed, measured on a probe.
	probe, err := openGPU(t, gpu.WithDevices("cuda:0,"+second[0]), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || probe == nil {
		noDevice(t, "cuda:0,"+second[0], err)
	}
	pi, err := NewInstance(m, len(prompt)+2, WithPlacement(place))
	if err != nil {
		t.Fatal(err)
	}
	if err := pi.SetDevice(probe); err != nil {
		t.Fatal(err)
	}
	held := probe.Budgets()[1].Used + 1<<20
	// The band: the placement with room, its rows gathered on the card.
	free, err := pi.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	band, _ := nmseOf(free, want)
	pi.Close()
	probe.Close()
	g, err := openGPU(t, gpu.WithDevices("cuda:0,"+second[0]+"="+strconv.FormatUint(held, 10)), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "capped "+second[0], err)
	}
	defer g.Close()
	in, err := NewInstance(m, len(prompt)+2, WithPlacement(place))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	in.SetRelocate(false)
	if err := in.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	got, err := in.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	// The rows were declined for room: a run whose card had room proves
	// nothing about the decline.
	_, declined := in.Account(second[0]).Steps.Of(nn.ComposedEmbed | nn.PromptEmbed)
	if declined[nn.DeclineNoRoom] == 0 {
		t.Fatalf("%s gathered every chunk's input rows under a cap of %d: the gate did not reach the decline", second[0], held)
	}
	nm, _ := nmseOf(got, want)
	t.Logf("%d chunks' input rows declined on %s; NMSE %.3e against the host, the placement with room %.3e; greedy %d vs %d",
		declined[nn.DeclineNoRoom], second[0], nm, band, Greedy(got), Greedy(want))
	if math.IsNaN(nm) || nm > 3*band {
		t.Errorf("NMSE %.3e against the host, past three times the placement with room's %.3e: the chunks ran on rows nothing wrote", nm, band)
	}
}
