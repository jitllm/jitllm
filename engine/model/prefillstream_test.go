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

// prefillStreamNMSE is the bound a device arm's logits are held to against
// the host's: the paged gates' order-only band on this model is ~2e-3, and a
// block whose history did not come home reads O(1).
const prefillStreamNMSE = 1e-2

// TestPrefillStreamsHostBlocks runs a three-and-a-half-chunk prompt on a card that holds
// half of the model: forced on, the prompt's host blocks stream through the
// card and come home after it, and the prompt's logits and eight decoded tokens
// -- which read the history the streamed blocks wrote on the card and brought
// home -- stay with the host's; forced off nothing streams; on auto exactly one
// decision is taken.
func TestPrefillStreamsHostBlocks(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	probe, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	// Half the model: the head and a few blocks resident, the rest home.
	spec := testdevices.Like(t, "cuda:0") + "=" + strconv.FormatUint(probe.WeightBytes()/2, 10)
	// Three device chunks and part of a fourth: auto measures the first two.
	n := 3*nn.MaxDevicePrefillChunk + nn.MaxDevicePrefillChunk/2
	prompt := probe.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog, and then it runs. ", n/10), true)[:n]
	const gen = 8
	host := probe.NewState(len(prompt) + gen + 1)
	want := []([]float32){}
	lg, err := host.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want = append(want, append([]float32(nil), lg...))
	ids := make([]int32, 0, gen)
	for i := 0; i < gen; i++ {
		id := Greedy(lg)
		ids = append(ids, id)
		if lg, err = host.Forward(id); err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), lg...))
	}
	host.Close()
	probe.Close()

	run := func(t *testing.T, mode PrefillStream) (*State, *tier.GPU, int, [][]float32) {
		g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		m, err := Open(path, WithPrefillStream(mode))
		if err != nil {
			g.Close()
			t.Fatal(err)
		}
		st := m.NewState(len(prompt) + gen + 1)
		t.Cleanup(func() { st.Close(); g.Close(); m.Close() })
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		resident := st.devCount()
		if resident == 0 || resident == m.Cfg.NLayer {
			t.Fatalf("%d of %d blocks resident: the card does not leave blocks on the host, so nothing "+
				"could stream", resident, m.Cfg.NLayer)
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		got := [][]float32{append([]float32(nil), lg...)}
		// No streamed block stays on the device. Fewer than before is the
		// history outgrowing the card, which relocation answers as it always
		// has.
		if st.devCount() > resident {
			t.Fatalf("after the prompt %d blocks are on the device, %d before it: the streamed blocks "+
				"did not come home", st.devCount(), resident)
		}
		for _, id := range ids {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), lg...))
		}
		return st, g, resident, got
	}
	check := func(t *testing.T, got [][]float32) {
		t.Helper()
		for i := range want {
			if e := logitNMSE(got[i], want[i]); !(e < prefillStreamNMSE) {
				t.Fatalf("step %d: logit NMSE %.3e against the host, bound %.0e", i, e, prefillStreamNMSE)
			}
		}
	}

	t.Run("on", func(t *testing.T) {
		st, g, resident, got := run(t, PrefillStreamOn)
		if s, k := st.PrefillStreams(); s != 1 || k != 0 {
			t.Fatalf("%d prompt(s) streamed, %d kept on the host: the forced stream did not run", s, k)
		}
		if ps := g.Stats(); ps.PageIns == 0 {
			t.Fatal("no page-in: nothing streamed through the card")
		}
		check(t, got)
		t.Logf("%d blocks resident, the rest streamed for a %d-token prompt: %d page-ins, back to %d after",
			resident, len(prompt), g.Stats().PageIns, st.devCount())
	})
	t.Run("off", func(t *testing.T) {
		st, _, _, got := run(t, PrefillStreamOff)
		if s, k := st.PrefillStreams(); s != 0 || k != 0 {
			t.Fatalf("forced off, %d prompt(s) streamed and %d measured", s, k)
		}
		check(t, got)
	})
	t.Run("auto", func(t *testing.T) {
		st, _, _, got := run(t, PrefillStreamAuto)
		s, k := st.PrefillStreams()
		if s+k != 1 {
			t.Fatalf("auto took %d decision(s) on one prompt (%d streamed, %d kept)", s+k, s, k)
		}
		check(t, got)
		t.Logf("auto: streamed %d, kept on the host %d", s, k)
	})
}
