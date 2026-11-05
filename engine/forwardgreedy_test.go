package engine

import (
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestForwardGreedyMatchesGreedyOverForward decodes the same prompt twice with
// every block and the head on the device, once through ForwardGreedy and once
// through Forward and Greedy, and demands the same tokens -- and that the
// device actually served the token (nn.Head.Token), because a greedy call that
// quietly fell back to the host arm is Greedy(Forward) compared with itself.
//
// gemma-2-2b is the second model because it has a final softcap, which the
// device applies before its argmax (nn.Head.Softcap), so the capped model must
// be served on the card too.
func TestForwardGreedyMatchesGreedyOverForward(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "gemma-2-2b-it-Q4_K_M"} {
		t.Run(name, func(t *testing.T) { forwardGreedyMatches(t, name) })
	}
}

func forwardGreedyMatches(t *testing.T, name string) {
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if strings.HasPrefix(name, "gemma-2") && m.Cfg.FinalSoftcap == 0 {
		t.Fatal("gemma-2 carries no final softcap: this subtest would test nothing it is here for")
	}
	ids := m.Vocab.Encode("The capital of France is", true)
	const n = 24
	run := func(greedy bool) (out []int32, served int) {
		g := swaTier(t)
		defer g.Close()
		st := testInstance(t, m, 256)
		defer st.Close()
		st.SetDevice(g)
		// A card the desktop shares, or an iGPU's quarter of a capped host,
		// can hold every block and not the head (gemma-2-2b's is 466 MiB):
		// that is a card too small for this gate, as every placement gate
		// says it, not a wrong answer.
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Skipf("CARD TOO SMALL: placed %d of %d blocks, head on the device %v (%s) -- this gate proved nothing here",
				st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
		}
		logits, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		next := Greedy(logits)
		for i := 0; i < n; i++ {
			out = append(out, next)
			if greedy {
				// -1 before every call: the host arm never writes Token, so a
				// stale value would count a host-served token as the device's.
				if st.head != nil {
					st.head.Token = -1
				}
				if next, err = st.ForwardGreedy(next); err != nil {
					t.Fatal(err)
				}
				if st.head != nil && st.head.Token >= 0 {
					served++
				}
			} else {
				if logits, err = st.Forward(next); err != nil {
					t.Fatal(err)
				}
				next = Greedy(logits)
			}
		}
		return out, served
	}
	want, _ := run(false)
	got, served := run(true)
	if served != n {
		t.Fatalf("the device served %d of %d greedy tokens: the rest were the host arm", served, n)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: ForwardGreedy %d, Greedy(Forward) %d\ngot  %v\nwant %v", i, got[i], want[i], got, want)
		}
	}
	t.Logf("%d tokens identical, all %d served on the device: %v", n, served, got)
}
