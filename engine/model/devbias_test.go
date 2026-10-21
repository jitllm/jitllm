package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// gemma3Prompt is natural text, so the model is on-distribution at every
// position the gemma3 arms compare.
const gemma3Prompt = "The capital of France is Paris, and the capital of Germany is"

// TestDeviceMatchesHostOnABiasedModel runs one block of a model that has a
// feature on the device against the host (AGENTS.md RULE 11c): kernel gates
// were green while qwen2's attention biases never reached the kernel and phi3's
// fused attn_qkv was uploaded whole. The bar is a bound on the logits with one
// block placed, not token equality: f32 reduction order makes greedy chains
// part at near-ties, while a dropped weight moves the logits by orders of
// magnitude.
func TestDeviceMatchesHostOnABiasedModel(t *testing.T) {
	// Qwen2 carries attention biases; Phi-3.5 carried a fused attn_qkv.
	for _, c := range []struct {
		name      string
		path      string
		wantTaken bool
		// prompt, when set, replaces the single id-1 forward with the model's
		// own BOS followed by this text, compared at every position.
		prompt string
	}{
		// The control comes first: tinyllama has neither feature and matches
		// the host token for token, so its NMSE is the device's f32 band.
		{"tinyllama-control", testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"), true, ""},
		{"qwen2-biases", testmodels.Path("Qwen2-1.5B-Instruct-Q4_K_M.gguf"), true, ""},
		// phi3 must run: convert.unfuse splits its q/k/v and gate/up at
		// conversion. A container that carried the fusion again would put k
		// and v back on q's rows.
		{"phi3-unfused-qkv", testmodels.Path("Phi-3.5-mini-instruct-Q4_K_M.gguf"), true, ""},
		// The gemma family: gemma2's blocks once ran on the device without
		// their post-norms (fluent repetition). gemma3 adds two rotary bases
		// (local 1e4, global 1e6). The 1b arm covers Q5_0 attention weights.
		//
		// gemma3 is fed its own BOS and a sentence, not id 1: id 1 is gemma's
		// <eos>, and at position 0 it is so ill-conditioned that two host runs
		// differing only in KV cache width disagree at NMSE ~1.7e-1 (see
		// docs/engineering-history/model-correctness.md). The prompt also runs
		// the placed block at positions where the rotary base is not the
		// identity.
		{"gemma3-postnorm-splitrope", testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf"), true, gemma3Prompt},
		{"gemma3-1b-q5_0", testmodels.Path("gemma-3-1b-it-Q4_K_M.gguf"), true, gemma3Prompt},
		// bartowski's conversion of the same model, so the bound is not fitted
		// to one converter's rounding.
		{"gemma3-1b-bartowski", testmodels.Path("google_gemma-3-1b-it-Q4_K_M.gguf"), true, gemma3Prompt},
		{"qwen3-qknorm", testmodels.Path("Qwen3-1.7B-Q4_K_M.gguf"), true, ""},
		// gemma2's attention logit softcap is a device kernel
		// (kernels.Softcap), so this arm must run.
		{"gemma2-softcap", testmodels.Path("gemma-2-2b-it-Q4_K_M.gguf"), true, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(c.path); err != nil {
				t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", c.path)
			}
			m, err := Open(jlmOf(t, c.path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()

			nmseOf := func(got, want []float32) (float64, float64) {
				var num, den, worst float64
				for i := range want {
					d := float64(got[i] - want[i])
					num += d * d
					den += float64(want[i]) * float64(want[i])
					if a := math.Abs(d); a > worst {
						worst = a
					}
				}
				return num / den, worst
			}
			// The token stream: id 1 alone, or the model's BOS and a sentence.
			// Both tiers see the same ids, so each position is compared.
			toks := []int32{1}
			if c.prompt != "" {
				if m.Vocab == nil || m.Vocab.BOS < 0 {
					t.Fatalf("%s: the prompt arm needs a vocabulary with a BOS (%v)", c.name, m.TokErr)
				}
				toks = m.Vocab.Encode(c.prompt, true)
				if len(toks) < 4 || toks[0] != m.Vocab.BOS {
					t.Fatalf("%s: %q encoded to %v, want BOS and at least three tokens",
						c.name, c.prompt, toks)
				}
			}
			run := func(st *State) [][]float32 {
				var out [][]float32
				for _, tk := range toks {
					l, err := st.Forward(tk)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), l...))
				}
				return out
			}
			hostRun := func() [][]float32 {
				st := m.NewState(32)
				defer st.Close()
				return run(st)
			}
			// worstOf is the largest NMSE over the positions, with the largest
			// max|d| beside it.
			worstOf := func(got, want [][]float32) (float64, float64, int) {
				var nm, wd float64
				at := 0
				for p := range want {
					n, w := nmseOf(got[p], want[p])
					if n > nm {
						nm, at = n, p
					}
					wd = math.Max(wd, w)
				}
				return nm, wd, at
			}

			// The A/A self-control, run first: two host runs are not
			// bit-identical (each State's JIT tunes its own pack width), so the
			// bound must sit above this floor.
			want := hostRun()
			base, baseWorst, baseAt := worstOf(hostRun(), want)
			t.Logf("%s: host-vs-host NMSE %.3e at position %d of %d, max|d| %.4f",
				c.name, base, baseAt, len(toks), baseWorst)

			g, err := tier.Open(0)
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()

			dev := m.NewState(32)
			defer dev.Close()
			// One block, so a difference is that block's and not accumulated
			// tie-breaking.
			dev.SetDeviceLayers(g, 1)
			taken := dev.GPULayers()
			if got := taken > 0; got != c.wantTaken {
				t.Fatalf("the device took %d blocks, want taken=%v (%s)",
					taken, c.wantTaken, g.Err())
			}
			if !c.wantTaken {
				// Nothing ran on the device; the assertion for this arm is the
				// decline and its reason.
				t.Logf("declined, as it must be: %s", g.Err())
				return
			}
			nmse, worst, at := worstOf(run(dev), want)
			t.Logf("%s: %d device block(s), logit NMSE %.3e at position %d of %d (max|d| %.4f) "+
				"against a host-vs-host floor of %.3e", c.name, taken, nmse, at, len(toks), worst, base)

			// The bound is derived from both sides with one block placed: working
			// arms read ~5e-4 to ~4.5e-3, while dropping qwen2's biases reads
			// 3.6e-1 and dropping a gemma3 post-norm reads 3.0e-1 or more. Logits
			// sit after the remaining host blocks and the head, hence far looser
			// than a residual bound.
			//
			// This gate does not discriminate gemma3's split rope (one block over
			// fourteen positions stays inside the band);
			// TestSlidingWindowRunsOnTheDevice does.
			lim := 5e-2
			if nmse > lim {
				t.Fatalf("logit NMSE %.3e with %d device block(s), limit %.3e (the bug this "+
					"gate exists for measured 3.599e-01): the device is not computing "+
					"this block", nmse, taken, lim)
			}
		})
	}
}
