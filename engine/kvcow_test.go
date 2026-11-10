package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestCardCopiesASharedPageItWrites holds the card's copy-on-write
// load-bearing. A ends inside a page; B and D each continue it differently,
// restored at A's end and writing on past it into the page they share; B
// again, one token on, restores B's history. Copied on a write, B2 reads B's
// rows and answers as a cold run does; on a tier whose writes land in a named
// page in place (gpu.Config.KVCowFault), B2 reads D's rows there and its
// logits part. The prompts are ids rather than text so the parting lands
// inside a page whatever the card's page size and chunk.
func TestCardCopiesASharedPageItWrites(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		t.Fatalf("no card: %v", err)
	}
	defer g.Close()
	gv, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff), gpu.WithConfig(func(c *gpu.Config) { c.KVCowFault = true }))
	if err != nil || gv == nil {
		t.Fatalf("the violation's tier: %v", err)
	}
	defer gv.Close()
	ids := func(from, step, k int) []int32 {
		out := make([]int32, k)
		for i := range out {
			out[i] = int32((from + i*step) % m.Cfg.NVocab)
		}
		return out
	}
	a := ids(1000, 7, 500)
	b := append(append([]int32(nil), a...), ids(2000, 3, 300)...)
	d := append(append([]int32(nil), a...), ids(3000, 5, 300)...)
	b2 := append(append([]int32(nil), b...), 42)
	open := func(on *gpu.GPU, store *SharedStore) *Instance {
		s := testInstance(t, m, 4096)
		if err := s.SetDeviceLayers(on, -1); err != nil {
			t.Fatal(err)
		}
		if n := s.GPULayers(); n != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d blocks: %v", n, m.Cfg.NLayer, s.DeviceDeclines())
		}
		if store != nil {
			mustKey(t, s, "kv-cow")
			if err := s.ShareKV(store); err != nil {
				t.Fatal(err)
			}
		}
		return s
	}
	// run is B2's largest logit difference from a cold run's on the same
	// tier, after A, B and D went through one shared store, and the
	// positions they restored.
	run := func(on *gpu.GPU) (float64, int, []float32) {
		cold := open(on, nil)
		want, err := cold.Prefill(b2)
		if err != nil {
			t.Fatal(err)
		}
		want = append([]float32(nil), want...)
		cold.Close()
		s := open(on, NewSharedStore())
		defer s.Close()
		var got []float32
		restored := 0
		for _, p := range [][]int32{a, b, d, b2} {
			s.Reset()
			lg, err := s.PrefillCached(p)
			if err != nil {
				t.Fatal(err)
			}
			restored += s.KVRestored()
			got = append(got[:0], lg...)
		}
		worst := 0.0
		for i := range want {
			worst = max(worst, math.Abs(float64(got[i]-want[i])))
		}
		return worst, restored, got
	}
	ctl, rc, kept := run(g)
	worst, restored, moved := run(gv)
	apart, scale := 0.0, 0.0
	for i := range kept {
		apart = max(apart, math.Abs(float64(moved[i]-kept[i])))
		scale = max(scale, math.Abs(float64(kept[i])))
	}
	t.Logf("copied on a write: %d positions restored, worst logit %.3g from cold; written in place: %d restored, %.3g; "+
		"the two part by %.3g of a logit scale of %.3g", rc, ctl, restored, worst, apart, scale)
	if rc == 0 || restored == 0 {
		t.Fatal("nothing was restored: no session wrote into a shared page")
	}
	if math.IsNaN(ctl) || math.IsNaN(worst) {
		t.Fatalf("non-finite logits: %.3g copied, %.3g in place", ctl, worst)
	}
	if apart < 0.01*scale || worst <= ctl {
		t.Error("writing into a shared page in place moved no logit past a copy's: the gate cannot see it")
	}
}
