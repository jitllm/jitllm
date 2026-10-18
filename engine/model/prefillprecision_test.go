package model

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPrefillPrecisionArms is a probe (JITLLM_PREC_MODEL=<file>) for reading
// `jitllm verify -prefill`'s batched-vs-per-token max|dlogit| on a CUDA card.
// The two arms use different arithmetic (per-token: int8 activations; batched:
// binary16 via GemmVolta, m8n8k4 on sm_70 and m16n8 from sm_75, and binary16
// attention), so each is compared against
// a more precise third arm (binary16 activations, float32 attention) and the
// host. A batched-path defect shows as batched-vs-reference far above
// per-token-vs-reference.
//
// Every arm prefills the same natural prompt and then decodes eight
// teacher-forced tokens, as verify does, and splits are pinned to one so the
// device arms differ only in the arithmetic named.
func TestPrefillPrecisionArms(t *testing.T) {
	name := os.Getenv("JITLLM_PREC_MODEL")
	if name == "" {
		t.Skip("set JITLLM_PREC_MODEL")
	}
	m, err := Open(jlmOf(t, testmodels.Resolve(name)), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const para = "The city grew along the river, and every spring the water rose " +
		"until the bridges were islands. Merchants kept their ledgers on the upper " +
		"floors. A clerk named Aldis counted barrels of salt, wool, and dried fish, " +
		"and wrote the totals in a small hand that nobody else could read. In winter " +
		"the ice held long enough for carts to cross, and the tolls were collected " +
		"on the far bank by a man who never gave change. "
	ids := m.Vocab.Encode(strings.Repeat(para, 30), true)[:336]
	drive := make([]int32, 8)
	for i := range drive {
		drive[i] = ids[(i*29)%len(ids)]
	}
	run := func(s *State) [][]float32 {
		l, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), l...)}
		for _, tok := range drive {
			if l, err = s.Forward(tok); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}
	host := func() [][]float32 {
		s := m.NewState(len(ids) + 32)
		defer s.Close()
		return run(s)
	}
	dev := func(noBatch bool, opts ...tier.Option) [][]float32 {
		opts = append(opts, tier.WithDeviceTune(tier.TuneOff), tier.WithSplit(1), tier.WithAttnAccSplit(1))
		g, err := openTier(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		g.NoBatch = noBatch
		s := m.NewState(len(ids) + 32)
		defer s.Close()
		s.SetDevice(g)
		if s.GPULayers() != m.Cfg.NLayer {
			t.Skipf("the device took %d of %d blocks", s.GPULayers(), m.Cfg.NLayer)
		}
		out := run(s)
		if s.DeviceDemotions() != 0 {
			t.Fatalf("the device was demoted; this arm is the host")
		}
		st := g.Stats()
		t.Logf("arm noBatch=%v: VoltaGemm %d, GemmF16 %d, attention on the matrix unit %v", noBatch, st.VoltaGemm, st.GemmF16, st.AttnMMA)
		return out
	}
	noVolta := tier.WithConfig(func(c *tier.Config) { c.NoVolta = true })
	all := map[string][][]float32{
		"reference":     dev(false, tier.WithoutAttnMMA(true)),
		"batched-volta": dev(false),
		"batched-dp4a":  dev(false, noVolta),
		"per-token":     dev(true),
		"host":          host(),
	}
	diff := func(x, y [][]float32) string {
		var worst, nmse float64
		flips := 0
		for i := range x {
			var num, den float64
			for j := range x[i] {
				d := float64(x[i][j] - y[i][j])
				num += d * d
				den += float64(y[i][j]) * float64(y[i][j])
				worst = math.Max(worst, math.Abs(d))
			}
			nmse = math.Max(nmse, num/den)
			if argmaxID(x[i]) != argmaxID(y[i]) {
				flips++
			}
		}
		return fmt.Sprintf("max|dlogit| %6.3f  worst NMSE %.2e  %d/%d argmax flips", worst, nmse, flips, len(x))
	}
	for _, a := range []string{"batched-volta", "batched-dp4a", "per-token", "host"} {
		t.Logf("%-14s vs reference  %s", a, diff(all[a], all["reference"]))
	}
	t.Logf("%-14s vs per-token  %s   <- what verify prints", "batched-volta", diff(all["batched-volta"], all["per-token"]))
	t.Logf("%-14s vs per-token  %s", "batched-dp4a", diff(all["batched-dp4a"], all["per-token"]))
	t.Logf("%-14s vs host       %s", "per-token", diff(all["per-token"], all["host"]))
}
