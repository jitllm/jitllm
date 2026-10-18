package model

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPlacement holds WithPlacement to what it promises: blocks and the head
// where the map names them, best effort and strict telling a miss apart, a
// device that is not attached refused in both, and a pin that seam moves, a
// budget shrink and a detach cannot shift.
func TestPlacement(t *testing.T) {
	testdevices.Require(t, "cuda:0", "vulkan:1")
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	attach := func(t *testing.T, spec string, pl Placement) (*Model, *State, *tier.GPU, error) {
		t.Helper()
		g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		m, err := Open(path, WithPlacement(pl))
		if err != nil {
			g.Close()
			t.Fatal(err)
		}
		st := m.NewState(64)
		err = st.SetDevice(g)
		t.Cleanup(func() { st.Close(); g.Close(); m.Close() })
		return m, st, g, err
	}
	on := func(st *State, li int) string {
		if n, ok := st.ld.(nn.NamedDevice).DeviceOf(li); ok {
			return n
		}
		return "host"
	}

	t.Run("where-named", func(t *testing.T) {
		m, st, _, err := attach(t, "cuda:0,vulkan:1", Placement{Strict: true,
			Blocks: map[int]Place{0: {On: "host"}, 1: {On: "vulkan:1"}, 2: {On: "cuda:0"},
				3: {On: "host"}, 4: {On: "cuda:0"}, 5: {On: "vulkan:1"}},
			Head: &Place{On: "vulkan:1"}})
		if err != nil {
			t.Fatal(err)
		}
		for li, want := range map[int]string{0: "host", 1: "vulkan:1", 2: "cuda:0", 3: "host", 4: "cuda:0", 5: "vulkan:1"} {
			if got := on(st, li); got != want {
				t.Errorf("block %d on %s, named %s", li, got, want)
			}
		}
		if n, ok := st.ld.(nn.NamedDevice).HeadDeviceName(); !ok || n != "vulkan:1" {
			t.Errorf("head on %q (%v), named vulkan:1", n, ok)
		}
		// And it computes the model: the same greedy tokens as the host.
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
			t.Errorf("first token %d on the placement, %d on the host", Greedy(a), Greedy(b))
		}
	})

	// A streaming placement runs every block on a card that holds a third of
	// them, paging weights through its slots, and computes the model; the same
	// card without it keeps the blocks that fit and leaves the rest home, which
	// is what makes the streamed arm a test of the placement and not of the
	// budget.
	t.Run("stream", func(t *testing.T) {
		probe, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		spec := "cuda:0=" + strconv.FormatUint(probe.WeightBytes()/3, 10)
		nl := probe.Cfg.NLayer
		probe.Close()
		_, ctl, _, err := attach(t, spec, Placement{})
		if err != nil {
			t.Fatal(err)
		}
		if ctl.devCount() == 0 || ctl.devCount() == nl {
			t.Fatalf("a third of the model placed %d of %d blocks resident: the budget does not bite, "+
				"so the streamed arm would prove nothing", ctl.devCount(), nl)
		}
		m, st, g, err := attach(t, spec, Placement{Rest: &Place{On: "cuda:0", Stream: true}})
		if err != nil {
			t.Fatal(err)
		}
		if st.devCount() != nl {
			t.Fatalf("streamed %d of %d blocks: %s", st.devCount(), nl, g.Err())
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
			t.Errorf("first token %d streamed, %d on the host", Greedy(a), Greedy(b))
		}
		if _, err := st.Forward(Greedy(a)); err != nil {
			t.Fatal(err)
		}
		if ps := g.Stats(); ps.PageIns <= ctl.devCount() || ps.PageOuts == 0 {
			t.Errorf("%d page-in(s), %d page-out(s): the blocks did not stream through the slots",
				ps.PageIns, ps.PageOuts)
		}
		t.Logf("resident placement: %d of %d blocks; streamed: %d of %d, %d page-ins",
			ctl.devCount(), nl, st.devCount(), nl, g.Stats().PageIns)
	})

	t.Run("head-on-host", func(t *testing.T) {
		_, st, _, err := attach(t, "cuda:0", Placement{Head: &Place{On: "host"}})
		if err != nil {
			t.Fatal(err)
		}
		if st.devCount() == 0 || st.HeadOnDevice() {
			t.Errorf("%d blocks placed, head on a device %v: want blocks placed and the head home",
				st.devCount(), st.HeadOnDevice())
		}
	})

	// A device the tier did not open is an error in both modes, and nothing
	// stays placed.
	for _, strict := range []bool{false, true} {
		t.Run(map[bool]string{false: "unknown-device-best-effort", true: "unknown-device-strict"}[strict], func(t *testing.T) {
			_, st, _, err := attach(t, "cuda:0", Placement{Strict: strict,
				Blocks: map[int]Place{3: {On: "cuda:7"}}})
			if err == nil || !strings.Contains(err.Error(), "cuda:7") {
				t.Fatalf("naming a device that is not attached: err %v", err)
			}
			if st.devCount() != 0 {
				t.Fatalf("%d blocks left placed after the error", st.devCount())
			}
		})
	}

	// A card too small for the block: strict fails and places nothing, best
	// effort puts the block where the engine would.
	small := "cuda:0=1M,vulkan:1"
	t.Run("miss-strict", func(t *testing.T) {
		_, st, _, err := attach(t, small, Placement{Strict: true, Blocks: map[int]Place{2: {On: "cuda:0"}}})
		if err == nil {
			t.Fatal("a block its card cannot hold was placed strictly without an error")
		}
		if st.devCount() != 0 {
			t.Fatalf("%d blocks left placed after a strict miss", st.devCount())
		}
	})
	t.Run("miss-best-effort", func(t *testing.T) {
		_, st, _, err := attach(t, small, Placement{Blocks: map[int]Place{2: {On: "cuda:0"}}})
		if err != nil {
			t.Fatal(err)
		}
		if got := on(st, 2); got == "cuda:0" {
			t.Fatal("a 1 MiB card took the block")
		}
		if st.devCount() == 0 {
			t.Fatal("best effort placed nothing")
		}
	})

	t.Run("pin", func(t *testing.T) {
		_, st, g, err := attach(t, "cuda:0", Placement{Blocks: map[int]Place{5: {On: "cuda:0", Pin: true}}})
		if err != nil {
			t.Fatal(err)
		}
		if on(st, 5) != "cuda:0" {
			t.Fatalf("pinned block 5 on %s", on(st, 5))
		}
		st.SetGPULayers(0)
		if on(st, 5) != "cuda:0" {
			t.Error("SetGPULayers(0) moved a pinned block")
		}
		if _, err := g.SetBudget(1 << 20); err == nil {
			t.Error("a budget below the pinned block was not refused")
		}
		if on(st, 5) != "cuda:0" {
			t.Error("a budget shrink moved a pinned block")
		}
		if err := st.SetDevice(nil); err == nil {
			t.Error("detaching the device a block is pinned to was not refused")
		}
	})
}

// TestParsePlacementStreams reads the stream mark and the rest selector, in
// either order with a pin, and refuses what cannot stream.
func TestParsePlacementStreams(t *testing.T) {
	p, err := ParsePlacement("0=host,1-2=cuda:0~!,3=cuda:0!~,*=cuda:0~", false)
	if err != nil {
		t.Fatal(err)
	}
	for li, want := range map[int]Place{0: {On: "host"}, 1: {On: "cuda:0", Pin: true, Stream: true},
		3: {On: "cuda:0", Pin: true, Stream: true}} {
		if got := p.Blocks[li]; got != want {
			t.Errorf("block %d reads %+v, want %+v", li, got, want)
		}
	}
	if p.Rest == nil || *p.Rest != (Place{On: "cuda:0", Stream: true}) {
		t.Errorf("* reads %+v, want every other block streamed on cuda:0", p.Rest)
	}
	if !p.Streams() {
		t.Error("a placement that streams reports it does not")
	}
	if q, err := ParsePlacement("0=cuda:0,1=host", false); err != nil || q.Streams() {
		t.Errorf("a placement that streams nothing reports it does (%v)", err)
	}
	for _, bad := range []string{"0=host~", "head=cuda:0~", "*=cuda:0,*=host"} {
		if _, err := ParsePlacement(bad, false); err == nil {
			t.Errorf("%q parsed; it streams through the host, streams the head, or names * twice", bad)
		}
	}
}
