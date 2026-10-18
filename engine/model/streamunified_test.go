package model

import (
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// unifiedCandidates are the devices that may share the host's memory, in the
// order they are tried: Metal, then an integrated GPU through Vulkan.
var unifiedCandidates = []string{"metal", "vulkan:1", "vulkan:0"}

// TestStreamedContainerOnAUnifiedDevice is the command line's streamed
// placement on a device whose heap is the host's: every block streamed
// through a third of the model's bytes, with the packed arena given a budget
// as cmd/jitllm gives it one. The arena packs GGUF bytes and must not be asked to
// pack the container's own spans, which no quantized format survives: every
// block would be declined and the "streamed" model run on the host. The gate
// is the placement, the first token against the host, and an arena never asked.
func TestStreamedContainerOnAUnifiedDevice(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	probe, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	third, nl := probe.WeightBytes()/3, probe.Cfg.NLayer
	probe.Close()

	var g *tier.GPU
	var name string
	for _, c := range testdevices.Within(unifiedCandidates) {
		d, err := openTier(t, tier.WithDevices(c+"="+strconv.FormatUint(third, 10)),
			tier.WithDeviceTune(tier.TuneOff), tier.WithArena(1<<30))
		if err != nil || d == nil {
			continue
		}
		if b := d.Budgets(); len(b) == 1 && b[0].Host {
			g, name = d, c
			break
		}
		d.Close()
	}
	if g == nil {
		t.Skipf("NO UNIFIED DEVICE among %v -- this gate proved nothing here; run it on the M4", unifiedCandidates)
	}
	defer g.Close()
	t.Logf("device %s, a %d-byte budget against %d of weights", name, third, 3*third)

	m, err := Open(path, WithPlacement(Placement{Rest: &Place{On: name, Stream: true}}))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.NewState(64)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.devCount() != nl {
		t.Fatalf("streamed %d of %d blocks on a unified device: %s", st.devCount(), nl, g.Err())
	}
	if as := g.ArenaStats(); as.Packs+as.Hits+as.Declines != 0 {
		t.Fatalf("the arena was asked to pack container weights: %d packs, %d hits, %d declines",
			as.Packs, as.Hits, as.Declines)
	}
	host := m.NewState(64)
	defer host.Close()
	ids := m.Vocab.Encode("The capital of France is", true)
	a, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	b, err := host.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if Greedy(a) != Greedy(b) {
		t.Fatalf("first token %d streamed on %s, %d on the host", Greedy(a), name, Greedy(b))
	}
	if _, err := st.Forward(Greedy(a)); err != nil {
		t.Fatal(err)
	}
	if ps := g.Stats(); ps.PageIns <= nl || ps.PageOuts == 0 {
		t.Fatalf("%d page-in(s), %d page-out(s): the blocks did not stream", ps.PageIns, ps.PageOuts)
	}
	t.Logf("%d of %d blocks streamed, %d page-ins, first token %d as on the host",
		st.devCount(), nl, g.Stats().PageIns, Greedy(a))
}
