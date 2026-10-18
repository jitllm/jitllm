package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// lcppLogProbs runs llama-perplexity from dir on gguf over text with a context
// of nctx, saving its log-probabilities (--kl-divergence-base), and returns the
// chunk's tokens and, for positions nctx/2 .. nctx-2, each position's
// log-softmax over the vocabulary. llama.cpp stores them quantized to 16 bits
// over the top 16 nats; an entry below that window comes back as -Inf.
func lcppLogProbs(t *testing.T, dir, gguf, text string, nctx int, ngl int) ([]int32, [][]float32) {
	t.Helper()
	bin := filepath.Join(dir, "llama-perplexity")
	if _, err := os.Stat(bin); err != nil {
		t.Skipf("ORACLE MISSING: %s (%v) -- this gate proved nothing", bin, err)
	}
	tmp := t.TempDir()
	in := filepath.Join(tmp, "text.txt")
	// llama-perplexity wants two contexts of tokens even for one chunk.
	if err := os.WriteFile(in, []byte(strings.Repeat(text+" ", 3)), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(tmp, "base.kld")
	cmd := exec.Command(bin, "-m", gguf, "-f", in, "-c", fmt.Sprint(nctx), "-b", fmt.Sprint(nctx),
		"--chunks", "1", "-ngl", fmt.Sprint(ngl), "-t", "6", "--kl-divergence-base", out)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+dir)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%s: %v\n%s", bin, err, b)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 20 || string(b[:8]) != "_logits_" {
		t.Fatalf("%s: not a logits file", out)
	}
	le := binary.LittleEndian
	n := int(int32(le.Uint32(b[8:])))
	nv := int(int32(le.Uint32(b[12:])))
	nch := int(int32(le.Uint32(b[16:])))
	if n != nctx || nch < 1 {
		t.Fatalf("logits file: n_ctx %d chunks %d", n, nch)
	}
	o := 20
	toks := make([]int32, n)
	for i := range toks {
		toks[i] = int32(le.Uint32(b[o+4*i:]))
	}
	o += 4 * n * nch
	row := 2*((nv+1)/2) + 4
	first := n / 2
	var lp [][]float32
	for i := first; i < n-1; i++ {
		r := b[o : o+2*row]
		o += 2 * row
		scale := math.Float32frombits(le.Uint32(r[0:]))
		minLP := math.Float32frombits(le.Uint32(r[4:]))
		v := make([]float32, nv)
		for j := range v {
			q := le.Uint16(r[8+2*j:])
			if q == 0 {
				v[j] = float32(math.Inf(-1))
			} else {
				v[j] = minLP + float32(q)*scale
			}
		}
		lp = append(lp, v)
	}
	return toks, lp
}

// logSoftmax is lg's log-softmax.
func logSoftmax(lg []float32) []float32 {
	mx := lg[0]
	for _, x := range lg {
		mx = max(mx, x)
	}
	var s float64
	for _, x := range lg {
		s += math.Exp(float64(x - mx))
	}
	ls := float32(math.Log(s))
	out := make([]float32, len(lg))
	for i, x := range lg {
		out[i] = x - mx - ls
	}
	return out
}

// lpDiff is the largest |a-b| over the entries both hold above the top 16 nats
// of b (llama.cpp's window), and whether their argmaxes differ.
func lpDiff(a, b []float32) (float64, bool) {
	mb := float32(math.Inf(-1))
	ia, ib := 0, 0
	for i := range b {
		if b[i] > mb {
			mb, ib = b[i], i
		}
		if a[i] > a[ia] {
			ia = i
		}
	}
	d := 0.0
	for i := range b {
		if b[i] < mb-15.5 || math.IsInf(float64(b[i]), -1) || math.IsInf(float64(a[i]), -1) {
			continue
		}
		d = max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d, ia != ib
}

const postNormText = "The history of the city begins with a small fishing village on the northern bank of the " +
	"river. Over the following centuries the settlement grew into a market town, and by the time the " +
	"railway arrived it had become the most important port in the region. Merchants built warehouses " +
	"along the waterfront, and the old wooden houses near the harbour were replaced with stone buildings " +
	"three or four storeys high. The cathedral, which still dominates the skyline, was completed after " +
	"almost two hundred years of construction, interrupted by fires, wars and the plague. Today the city " +
	"is known for its university, its museums and the festival held every summer in the main square, " +
	"when musicians from all over the country come to play in the streets until the early morning."

// TestPostNormDeviceErrorIsAmplification settles whether the post-norm
// models' device logits, which sit above the 0.2-0.94 band against the host on
// real Q4_K checkpoints, are a device defect or the model amplifying the
// reduction-order difference every device has -- by RULE 11c's method and
// against llama.cpp, teacher-forced over one chunk of text:
//
//   - ONE block on the device, moved in depth: a defect in the block shows the
//     same at every depth, amplification tracks the blocks below it. The last
//     block, with nothing below, must sit inside the band, and the first must
//     read several times the last.
//   - what two independent engines differ by on the same file: llama.cpp's
//     own CUDA build against its CPU build, and jitllm's host against
//     llama.cpp's CPU. jitllm's device against its host may not exceed the
//     larger by half. llama.cpp's cross-backend figure alone moves with its
//     build (OLMo 2: 1.157 from an sm_86 build, 0.693 from an sm_70
//     one) while jitllm's Vulkan arm reads 1.106 on both, as far
//     from the host as the host is from llama.cpp (0.977).
//   - jitllm's host against itself with only the KV cache's width changed
//     (f16 V against f32): a perturbation with no device in it at all.
//
// Every device arm runs an f32 KV cache against an f32-KV host, so the width
// is not counted twice. The violation multiplies the last block's post-FFN
// norm by 1.05 on the host arm alone, which is what a defect in that block's
// device wiring would look like at the depth where nothing amplifies it, and
// the last-block bound must catch it.
func TestPostNormDeviceErrorIsAmplification(t *testing.T) {
	models := []string{"newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf", "newarch/EXAONE-4.0-1.2B-Q4_K_M.gguf"}
	if v := os.Getenv("JITLLM_POSTNORM_MODELS"); v != "" {
		models = strings.Split(v, ",")
	}
	specs := testdevices.Specs("cuda:0", "vulkan:0", "metal")
	// The oracles: a CPU build of llama.cpp (JITLLM_LLAMACPP_CPU, a directory
	// holding llama-perplexity and its libraries) and, optionally, the CUDA
	// build in JITLLM_LCPP.
	cpuDir := os.Getenv("JITLLM_LLAMACPP_CPU")
	if cpuDir == "" {
		t.Skip("JITLLM_LLAMACPP_CPU is unset: it names the directory of a CPU build of " +
			"llama.cpp (llama-perplexity and its libraries), the oracle this gate " +
			"compares against -- this gate proved nothing")
	}
	cudaDir := os.Getenv("JITLLM_LCPP")
	const nctx = 128
	// lastBand is the upper end of the band one block's reduction order moves
	// the logits by with nothing below it to amplify (measured 0.072-0.075).
	const lastBand = 0.25
	for _, name := range models {
		t.Run(filepath.Base(name), func(t *testing.T) {
			gguf := testmodels.Path(name)
			if _, err := os.Stat(gguf); err != nil {
				t.Skipf("MODEL MISSING: %v -- this gate proved nothing", err)
			}
			toks, lc := lcppLogProbs(t, cpuDir, gguf, postNormText, nctx, 0)
			lcGPU := -1.0
			if _, err := os.Stat(filepath.Join(cudaDir, "libggml-cuda.so")); cudaDir != "" && err == nil {
				_, lg := lcppLogProbs(t, cudaDir, gguf, postNormText, nctx, 99)
				lcGPU, _ = lpWorst(lg, lc)
				t.Logf("%-38s max|dlogp| %.3f", "llama.cpp cuda vs its cpu", lcGPU)
			}
			path := jlmOf(t, gguf)
			c, err := Open(path)
			if err != nil {
				t.Fatal(err)
			}
			nl, arch, shared := c.Cfg.NLayer, c.Cfg.Arch, c.Cfg.NKVShared
			// free is the first block a KV-sharing group holds: a block below it
			// can be placed alone.
			free := nl
			for li := nl - shared; li < nl; li++ {
				free = min(free, c.Cfg.KVSource(li))
			}
			// The method is RULE 11c's and holds for any block; the violation
			// bends the last block's post-FFN norm, so that is what a model
			// needs (Gemma 4 carries pre- and post-norms both).
			if c.layers[nl-1].postFFNNorm == nil {
				c.Close()
				t.Fatalf("%s's last block has no post-FFN norm for the violation to bend", arch)
			}
			c.Close()
			// decode runs the chunk teacher-forced on a model opened with opts,
			// on g when it is set, an f32 KV cache when f32, and the last
			// block's post-FFN norm scaled by bend; it returns the log-softmax
			// of the scored half.
			decode := func(g *tier.GPU, f32 bool, bend float32, opts ...Option) [][]float32 {
				m, err := Open(path, append([]Option{noTune}, opts...)...)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				if f32 {
					m.SetKVF16(false)
				}
				tk := append([]int32(nil), toks...)
				if m.Vocab.AddBOS && m.Vocab.BOS >= 0 {
					tk[0] = int32(m.Vocab.BOS)
				}
				s := m.NewState(nctx)
				defer s.Close()
				if g != nil {
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
				}
				if bend != 1 {
					w := m.layers[nl-1].postFFNNorm
					if w == nil {
						t.Fatal("the last block has no post-FFN norm to bend")
					}
					orig := append([]float32(nil), w...)
					for i := range w {
						w[i] *= bend
					}
					defer copy(w, orig)
				}
				var out [][]float32
				for i := 0; i < nctx-1; i++ {
					lgt, err := s.Forward(tk[i])
					if err != nil {
						t.Fatal(err)
					}
					if i >= nctx/2 {
						out = append(out, logSoftmax(lgt))
					}
				}
				return out
			}
			report := func(what string, a, b [][]float32) float64 {
				worst, flips := lpWorst(a, b)
				t.Logf("%-38s max|dlogp| %.3f, %d of %d argmax(es) differ", what, worst, flips, len(b))
				return worst
			}
			host := decode(nil, false, 1)
			host32 := decode(nil, true, 1)
			// engines is what two independent implementations of this model
			// differ by: llama.cpp's CUDA build against its CPU build, and
			// jitllm's host against llama.cpp's CPU. The larger is the band a
			// device is held to.
			engines := report("jitllm host vs llama.cpp cpu", host, lc)
			who := "jitllm's host against llama.cpp's cpu"
			if lcGPU > engines {
				engines, who = lcGPU, "llama.cpp's own cuda against its cpu"
			}
			report("jitllm host, f32 KV vs f16 KV", host32, host)
			bent := decode(nil, true, 1.05)
			onHost := &Place{On: "host"}
			for _, spec := range specs {
				t.Run(spec, func(t *testing.T) {
					g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
						tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					one := func(d int) [][]float32 {
						pl := Placement{Strict: true, Blocks: map[int]Place{d: {On: spec}}, Rest: onHost, Head: onHost}
						return decode(g, true, 1, WithPlacement(pl))
					}
					all := decode(g, true, 1)
					report(spec+" vs llama.cpp cpu", all, lc)
					whole := report(spec+" vs host", all, host32)
					if shared > 0 {
						// A KV-sharing reader runs only beside its source
						// (fixKVGroups), so the last block cannot go alone and
						// nothing here sits with no depth below it: the depths
						// are the ones a block can take alone, and the bounds
						// are left to a model without sharing.
						for _, d := range []int{0, free / 2, free - 1} {
							report(fmt.Sprintf("%s block %d of %d alone vs host", spec, d, nl), one(d), host32)
						}
						t.Logf("%d of %d blocks read another's KV: the depth bounds need a model without sharing",
							shared, nl)
						return
					}
					var at []float64
					for _, d := range []int{0, nl / 2, nl - 1} {
						at = append(at, report(fmt.Sprintf("%s block %d of %d alone vs host", spec, d, nl), one(d), host32))
					}
					last, first := at[len(at)-1], at[0]
					if last > lastBand {
						t.Errorf("the last block alone moves the logits by %.3f, past the %.2f band: with nothing "+
							"below it to amplify, that is the block", last, lastBand)
					}
					if first < 4*last {
						t.Errorf("block 0 alone reads %.3f against the last block's %.3f: the error does not track "+
							"the depth below the seam, so it is not amplification", first, last)
					}
					if whole > 1.5*engines {
						t.Errorf("the device reads %.3f from the host, more than half again what two independent "+
							"engines differ by on this model (%s, %.3f)", whole, who, engines)
					}
					// The violation: the host's last block bent, the device's not.
					v := report(spec+" last block alone vs a bent host", one(nl-1), bent)
					if v <= lastBand {
						t.Errorf("a 5%% error in the last block's post-FFN norm reads %.3f, inside the %.2f band: "+
							"the gate cannot see a defect in one block", v, lastBand)
					}
				})
			}
		})
	}
}

// lpWorst is lpDiff over every scored position: the largest difference and
// how many argmaxes differ.
func lpWorst(a, b [][]float32) (float64, int) {
	worst, flips := 0.0, 0
	for i := range b {
		d, f := lpDiff(a[i], b[i])
		worst = max(worst, d)
		if f {
			flips++
		}
	}
	return worst, flips
}
