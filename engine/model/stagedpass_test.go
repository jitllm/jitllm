package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// stagedPassRun is one arm of the staged-pass gate: the model wholly on the
// card, the staged decode forced, its plan built for pass keys (0: the
// default, wider than the whole history), and the eviction gate's prompt
// and teacher-forced decode.
func (eg *evictGate) stagedPassRun(t *testing.T, dev string, pass int) ([][]float32, tier.Stats) {
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) {
			c.KVPage = 64
			c.StagedDecode = true
			c.StagedPassKeys = pass
		})}, eg.extra...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	t.Cleanup(g.Close)
	st := eg.m.NewState(eg.seq())
	t.Cleanup(func() { st.Close() })
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if n := st.devCount(); n != eg.m.Cfg.NLayer {
		t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this gate proved nothing here",
			n, eg.m.Cfg.NLayer, g.Err())
	}
	if _, err := st.Prefill(eg.prompt); err != nil {
		t.Fatal(err)
	}
	var out [][]float32
	for _, id := range eg.ids {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatalf("step %d: %v (%s)", len(out), err, g.Err())
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, g.Stats()
}

// TestStagedDecodePassesPastItsWidth: a staged decode plan is built for at
// most Config.StagedPassKeys keys, since its score planes are rows x heads x
// keys and a plan for the whole context held 8 GiB of a V100 that 64
// sessions' histories needed. A row deeper than the width attends in passes
// over its pages on the card, folded, and every token's logits stay with an
// arm whose plan covers the whole history and with the host's.
// Stats.StagedPasses is the selection check: the narrow arm must have run
// passes and the wide arm none. Its violation (passfold) is
// TestStagedPassGateDiscriminates.
func TestStagedDecodePassesPastItsWidth(t *testing.T) {
	eg := newEvictGate(t)
	devs := testdevices.Specs("cuda:0", "vulkan:0")
	for _, dev := range devs {
		t.Run(dev, func(t *testing.T) {
			wide, ws := eg.stagedPassRun(t, dev, 0)
			narrow, ns := eg.stagedPassRun(t, dev, 128)
			if ws.StagedPasses != 0 {
				t.Fatalf("the wide arm ran %d staged passes: its plan did not cover the history", ws.StagedPasses)
			}
			if ns.StagedPasses == 0 || ns.KVStreamPasses != 0 || ns.KVEvictions != 0 {
				t.Fatalf("the narrow arm: %d staged passes, %d upload passes, %d evictions: want passes over pages on the card",
					ns.StagedPasses, ns.KVStreamPasses, ns.KVEvictions)
			}
			got, at := worstStep(narrow, wide)
			host, hat := worstStep(narrow, eg.host)
			ctl, _ := worstStep(wide, eg.host)
			t.Logf("%d tokens, passes of 128 keys: %d staged passes; worst logit NMSE %.3e against the wide plan (step %d), "+
				"%.3e against the host (step %d; the wide plan reads %.3e); scratch %d MiB narrow, %d MiB wide",
				len(eg.ids), ns.StagedPasses, got, at, host, hat, ctl, ns.ScratchBytes>>20, ws.ScratchBytes>>20)
			if !(got < evictNMSE) {
				t.Fatalf("worst logit NMSE %.3e against the wide plan at step %d", got, at)
			}
			if !(host < evictNMSE) {
				t.Fatalf("worst logit NMSE %.3e against the host at step %d", host, hat)
			}
		})
	}
}
