package model

import (
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// hostEmbed hides nn.EmbedDevice, so the same device prefills with the rows
// looked up on the host -- the arm the device gather replaces.
type hostEmbed struct{ nn.LayerDevice }

// Detach forwards the session's release. Embedding nn.LayerDevice alone hid
// nn.Session too, so State.Close never detached this arm and its pages stayed
// held: the second arm's pool doubled, and on a shared card it had
// to relocate or evict where the first arm had not, which the gate's rounding
// bound read as a 1.4e-3 difference between the arms.
func (h hostEmbed) Detach() {
	if s, ok := h.LayerDevice.(nn.Session); ok {
		s.Detach()
	}
}

// HeldBytes completes nn.Session, so the forwarding above is reached at all.
func (h hostEmbed) HeldBytes() uint64 {
	if s, ok := h.LayerDevice.(nn.Session); ok {
		return s.HeldBytes()
	}
	return 0
}

// TestPrefillEmbedsOnTheDevice prefills tied models on the device twice, once
// with the prompt's rows gathered on the card from the resident head
// (nn.EmbedDevice) and once looked up on the host, and demands the same logits
// and the same greedy continuation.
//
// The selection check is Stats.EmbedLaunches, which must move on the device arm
// only. The models cover Q6_K and Q8_0 heads and gemma's embedding scale, which
// the gather applies itself. The NoBatch arm matters as much: there no
// submission's x is the promised slice, so the tier must make the rows real on
// the host before the first row uploads.
func TestPrefillEmbedsOnTheDevice(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0", "gemma-2-2b-it-Q4_K_M"} {
		t.Run(name, func(t *testing.T) { embedsOnTheDevice(t, name) })
	}
	t.Run("Llama-3.2-1B-Instruct-Q4_K_M/nobatch", func(t *testing.T) {
		embedsOnTheDevice(t, "Llama-3.2-1B-Instruct-Q4_K_M", tier.WithConfig(func(c *tier.Config) { c.NoBatch = true }))
	})
}

func embedsOnTheDevice(t *testing.T, name string, opts ...tier.Option) {
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Cfg.TiedEmbd {
		t.Fatalf("%s is not tied: the gather has no head to read from", name)
	}
	g, err := openTier(t, append([]tier.Option{tier.WithDeviceTune(tier.TuneOff)}, opts...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 20)
	ids := m.Vocab.Encode(text, true)[:300]

	// logits holds the prompt's logits and then each decode step's: the
	// decode after a gathered prompt reads the cache the gathered rows built.
	run := func(onDevice bool) (logits [][]float32, toks []int32, launches int) {
		st := m.NewState(len(ids) + 16)
		defer st.Close()
		st.SetDevice(g)
		if gs := g.Stats(); (st.GPULayers() < m.Cfg.NLayer || !st.HeadOnDevice()) && (gs.Declined+gs.NoRoom > 0 || backend.AllocRefused() > 0) {
			// The gather reads the resident head, so a card too small for the
			// blocks and the head has nothing to gather from.
			t.Skipf("CARD TOO SMALL: %d of %d blocks and head %v placed (budget or device memory: %s) "+
				"-- this gate proved nothing on this device; run it on one that holds both", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
		}
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Fatalf("placed %d of %d blocks, head %v: %s (declined %d: %s)", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err(), g.Stats().Declined, g.Stats().DeclineWhy)
		}
		if !onDevice {
			st.setLD(hostEmbed{st.ld})
		}
		before := g.Stats().EmbedLaunches
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		logits = append(logits, append([]float32(nil), lg...))
		next := Greedy(lg)
		for i := 0; i < 8; i++ {
			toks = append(toks, next)
			if lg, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
			logits = append(logits, append([]float32(nil), lg...))
			next = Greedy(lg)
		}
		launches = g.Stats().EmbedLaunches - before
		return logits, toks, launches
	}
	want, wantToks, hostLaunches := run(false)
	got, gotToks, devLaunches := run(true)
	if hostLaunches != 0 {
		t.Fatalf("the host arm gathered on the device %d time(s): the arms are not two arms", hostLaunches)
	}
	if devLaunches == 0 {
		t.Fatalf("the device arm never gathered: %s", g.Err())
	}
	nmse := 0.0
	for s := range want {
		var num, den float64
		for i := range want[s] {
			if math.IsNaN(float64(got[s][i])) || math.IsInf(float64(got[s][i]), 0) {
				t.Fatalf("step %d logit %d is %v", s, i, got[s][i])
			}
			d := float64(got[s][i] - want[s][i])
			num += d * d
			den += float64(want[s][i]) * float64(want[s][i])
		}
		nmse = math.Max(nmse, num/den)
	}
	// The bound is rounding, not a band: the only difference is whether
	// scale*q - bias was fused on the card, one ulp of an embedding value.
	if !(nmse < 1e-6) {
		t.Fatalf("worst logits NMSE %.3e (prompt and 8 decode steps) between the device gather and the host lookup", nmse)
	}
	for i := range wantToks {
		if gotToks[i] != wantToks[i] {
			t.Fatalf("token %d: device gather %v, host lookup %v", i, gotToks, wantToks)
		}
	}
	t.Logf("%d chunk(s) gathered on the device; worst logits NMSE %.3e over the prompt and 8 decode steps; tokens identical %v", devLaunches, nmse, gotToks)
}
