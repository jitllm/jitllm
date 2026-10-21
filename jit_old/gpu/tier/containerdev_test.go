package tier_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestContainerAgreesHostAndDevice runs the same container and prompt on the
// host and the device and compares token ids. It goes through Prefill on
// purpose: a scale-layout change once reached the decode matvec but not the
// device GEMM that runs prefill, and the kernel and model packages, neither of
// which opens a device, stayed green.
func TestContainerAgreesHostAndDevice(t *testing.T) {
	// Whichever model is present, in order of preference, so the gate runs on
	// hosts holding different files.
	var src string
	names := []string{
		"tinyllama-1.1b-q3_K_M.gguf",
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
		"Qwen3-MOE-4x0.6B-Q4_K_M.gguf",
		"gemma-2b.gguf",
		"olmoe-1b-7b-Q4_K_M.gguf",
	}
	// JITLLM_DEV_MODEL=<file> names one, so a graph feature only a later
	// candidate has (olmoe's wide q/k norm) can be put on the device here.
	if v := os.Getenv("JITLLM_DEV_MODEL"); v != "" {
		names = []string{v}
	}
	for _, name := range names {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err == nil {
			src = p
			break
		}
	}
	if src == "" {
		t.Skip("MODEL MISSING: none of the candidates are in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	t.Logf("model %s", filepath.Base(src))
	dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("convert: %v", err)
	}

	const prompt = "The capital of France is"
	const n = 12

	// A near-tie is not a disagreement: device and host reduce in different
	// orders, and a mixture has shown margins of 0.03 on logits of 18.6. A
	// divergence counts only where the host was not choosing between two
	// almost-equal tokens.
	const tieMargin = 0.25

	run := func(dev bool) ([]int32, []float32, int) {
		m, err := model.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		ids := m.Vocab.Encode(prompt, true)
		st := m.NewState(len(ids) + n + 1)
		defer st.Close()
		placed := 0
		if dev {
			g, err := tier.OpenWith(tier.WithDevices("gpu:0"))
			if err != nil {
				t.Skipf("no accelerator: %v -- this gate proved nothing", err)
			}
			defer g.Close()
			st.SetDeviceLayers(g, -1)
			placed = st.GPULayers()
			if placed == 0 {
				t.Skip("the device took no block -- this gate proved nothing")
			}
		}
		// Prefill, not a Forward loop: only the batched path reaches the
		// GEMM.
		logits, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, n)
		margins := make([]float32, 0, n)
		for i := 0; i < n; i++ {
			best, second, bi := float32(-1e30), float32(-1e30), int32(0)
			for j, v := range logits {
				if v > best {
					best, second, bi = v, best, int32(j)
				} else if v > second {
					second = v
				}
			}
			out = append(out, bi)
			margins = append(margins, best-second)
			if logits, err = st.Forward(bi); err != nil {
				t.Fatal(err)
			}
		}
		return out, margins, placed
	}

	host, margin, _ := run(false)
	dev, _, placed := run(true)
	ties := 0
	for i := range host {
		if host[i] == dev[i] {
			continue
		}
		if margin[i] < tieMargin {
			ties++
			// A tie flips the whole continuation, so there is nothing to
			// compare after it.
			t.Logf("token %d: host %d, device %d -- a tie, margin %.4f", i, host[i], dev[i], margin[i])
			break
		}
		t.Fatalf("token %d: host %d, device %d, host margin %.4f (%d block(s) placed)\n"+
			"  host   %v\n  device %v", i, host[i], dev[i], margin[i], placed, host, dev)
	}
	t.Logf("%d ids compared, %d tie(s), %d block(s) on the device", len(host), ties, placed)
}
