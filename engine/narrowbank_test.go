//go:build linux

package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestANarrowStreamedBankMatchesTheResidentOne is the streamed bank's gate on
// the small mixture, whose router the composer fused into the kernel that
// reads the bank: a bank in pages needs its selection written by a kernel
// before it, which the card reads to fault the pages in, so that form did not
// build. Each block streamed under a budget that evicts, the logits are the
// resident bank's bit for bit, and the streamed arm filled banks.
func TestANarrowStreamedBankMatchesTheResidentOne(t *testing.T) {
	probe, _ := hybridModelOpt(t, hyOpt{moe: true})
	page := probe.Container.H.PageSize
	probe.Close()
	m, opts := hybridModelOpt(t, hyOpt{moe: true, budget: 2 * page})
	defer m.Close()
	if !budgetEvicts(m, 2*page) {
		t.Fatalf("a budget of two pages of %d evicts nothing: no sheet would be read", page)
	}
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	run := func(stream bool) ([][]float32, gpu.Stats) {
		t.Helper()
		gopts := []gpu.Option{gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff)}
		if stream {
			gopts = append(gopts, gpu.WithConfig(func(c *gpu.Config) { c.StreamExperts = true }))
		}
		g, err := openGPU(t, gopts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := testInstance(t, m, 16, opts...)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("stream=%v: %d of %d blocks placed: %s", stream, s.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatalf("stream=%v: %v (%s)", stream, err, g.Err())
			}
			out[i] = append([]float32(nil), l...)
		}
		return out, g.Stats()
	}
	want, _ := run(false)
	got, st := run(true)
	if st.StreamBlocks == 0 || st.StreamFills == 0 {
		t.Fatalf("the streamed arm placed %d streamed blocks and filled %d times: it streamed nothing", st.StreamBlocks, st.StreamFills)
	}
	for i := range want {
		for j := range want[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
				t.Fatalf("token %d logit %d: streamed %g, resident %g", i, j, got[i][j], want[i][j])
			}
		}
	}
	t.Logf("%d blocks streamed, %d fills; %d tokens bit for bit", st.StreamBlocks, st.StreamFills, len(ids))
}
