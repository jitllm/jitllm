package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestTailPruneOnADeviceHoldingOnlyTheGroup: the front blocks stay on the host
// and the KV-sharing group (its sources and readers, which place together)
// runs on the device, so the pruned tail is a device run behind host blocks.
// A seam inside the group is refused by design (kvSpan, fixKVGroups), so this
// is the placement that splits the model around the tail.
func TestTailPruneOnADeviceHoldingOnlyTheGroup(t *testing.T) {
	spec := testdevices.One("cuda:0")
	name := spec
	path := tailModelPath(t, "synth-gemma4-e")
	probe, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	cfg := probe.Cfg
	probe.Close()
	glo, _ := (&State{c: cfg}).kvSpan(cfg.NLayer - 1)
	front := map[int]Place{}
	for li := 0; li < glo; li++ {
		front[li] = Place{On: "host"}
	}
	if len(front) == 0 {
		t.Fatal("the KV-sharing group starts at block 0: nothing is left on the host")
	}
	run := func(prune bool) tailRun {
		m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithTailPrune(prune),
			WithJITOptions(nn.WithPrefillChunk(7), nn.WithTune(nn.TuneOff)),
			WithPlacement(Placement{Blocks: front, Rest: &Place{On: name}, Head: &Place{On: name}}))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		g, err := tier.OpenWith(tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Fatalf("%s: %v", spec, err)
		}
		defer g.Close()
		st := m.NewState(64)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.devAt(0) || !st.devAt(m.Cfg.NLayer-1) {
			t.Fatalf("placement not as asked: block 0 on device %v, last on device %v", st.devAt(0), st.devAt(m.Cfg.NLayer-1))
		}
		var r tailRun
		lg, err := st.Prefill(tailPrompt(m, 40))
		if err != nil {
			t.Fatal(err)
		}
		r.pre = append(r.pre, lg...)
		for i := 0; i < 6; i++ {
			if lg, err = st.Forward(int32(11 + i)); err != nil {
				t.Fatal(err)
			}
			r.dec = append(r.dec, lg...)
		}
		r.would, r.ran = st.TailPruneRowBlocks()
		return r
	}
	a, b := run(false), run(true)
	T := int(cfg.NKVShared)
	if b.would != 40*T || b.ran != T {
		t.Errorf("row-blocks %d -> %d, want %d -> %d", b.would, b.ran, 40*T, T)
	}
	maxd := 0.0
	for i := range a.dec {
		maxd = max(maxd, float64(max(a.dec[i]-b.dec[i], b.dec[i]-a.dec[i])))
	}
	if maxd > 1e-3 {
		t.Errorf("decode logits differ by %.3g", maxd)
	}
}
