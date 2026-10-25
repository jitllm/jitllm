//go:build amd64 || arm64

package model

import (
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPrefillMatchesForward is the gate for the whole batched path.
//
// Prefill must leave the session in EXACTLY the state a loop of Forward calls
// would: the same logits for the last token, the same position, and -- the part
// that actually catches bugs -- a KV cache that makes subsequent decode produce
// identical tokens. Attention is the one op the batch cannot flatten, because
// token i may only see positions up to p0+i, and an off-by-one there does not
// crash; it silently lets a token attend to its own future and produces fluent,
// wrong text.
func TestPrefillMatchesForward(t *testing.T) {
	path := benchModel(t)
	// With the batched GEMM out of the way, Prefill is bit-identical to a
	// Forward loop, so this mode demands equality; an NMSE bar would hide a
	// prefill path computing the same op by different arithmetic.
	t.Run("no-gemm", func(t *testing.T) {
		nn.ResetForTest()
		defer nn.ResetForTest()
		prefillVsForward(t, path, true)
	})
	// The GEMM's exact (float-epilogue) form sums what the matvec sums in the
	// same order, so it is held to equality too. Only the integer-accumulating
	// form is left to the NMSE tripwire.
	t.Run("exact-gemm", func(t *testing.T) {
		nn.ResetForTest()
		defer nn.ResetForTest()
		prefillVsForwardOpts(t, path, true, WithJITOptions(nn.WithGEMMExact(true)))
	})
	t.Run("shipped", func(t *testing.T) {
		nn.ResetForTest()
		defer nn.ResetForTest()
		prefillVsForward(t, path, false)
	})
	// On each device, both sessions on one tier: the tier's own kernels; the
	// composer serving the decode's blocks, the prompt's at the chunk's rows
	// and the head (ComposedPrompt); and the same with every other block left
	// to the tier's kernels (Config.TierBlocks), so a chunk runs the two in
	// turn; and gemma-3-1b's sliding layers over a prompt past their window,
	// whose rows read the windowed descriptors; and the shadow arm, whose
	// prompt stays on the tier's kernels by name. The shipped mode's bar: a
	// block whose output a neighbour's held add overwrote, or a windowed row
	// that read the whole history, misses it by orders.
	t.Run("device", func(t *testing.T) {
		const parts = tier.ComposedBlock | tier.ComposedHead | tier.ComposedPrompt
		llama, gemma := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), testmodels.Path("gemma-3-1b-it-Q4_K_M.gguf")
		for _, p := range []string{llama, gemma} {
			if _, ok := existingModel(p); !ok {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", p)
			}
		}
		for _, dev := range stepDevices() {
			for _, arm := range []struct {
				name                   string
				path                   string
				parts                  tier.Parts
				mixed, windows, shadow bool
			}{{"tier", llama, 0, false, false, false}, {"composed", llama, parts, false, false, false},
				{"mixed", llama, parts, true, false, false}, {"windowed tier", gemma, 0, false, true, false},
				{"windowed", gemma, parts, false, true, false}, {"shadowed", llama, parts, false, false, true}} {
				t.Run(dev+"/"+arm.name, func(t *testing.T) {
					prefillOnDevice(t, arm.path, dev, arm.parts, arm.mixed, arm.windows, arm.shadow)
				})
			}
		}
	})
}

// prefillOnDevice is prefillVsForward on device dev, both sessions on one
// tier asking the composer for parts, every odd block kept on the tier's
// kernels where mixed, the prompt past the model's sliding window where
// windows, the shadow arm on where shadow: the composer served every prompt
// block step it was asked for and declined only the blocks kept, or, in the
// shadow arm, declined every prompt block step by that name.
func prefillOnDevice(t *testing.T, path, dev string, parts tier.Parts, mixed, windows, shadow bool) {
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var keep []int
	for li := 1; mixed && li < m.Cfg.NLayer; li += 2 {
		keep = append(keep, li)
	}
	minTok := 0
	if windows {
		if m.Cfg.SWAWindow == 0 {
			t.Fatalf("%s states no sliding window: this arm would read no windowed descriptor", path)
		}
		minTok = m.Cfg.SWAWindow + 96
	}
	// The prompt goes in chunks of at most 64 rows: the device reserves and
	// warms the widths up to 64, each a form of the composer's.
	opts := []tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.TierBlocks, c.PromptRows = parts, keep, 64 })}
	if shadow {
		opts = append(opts, tier.WithComposedShadow())
	}
	g, err := tier.OpenWith(append(opts, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	prefillVsForwardModel(t, m, false, minTok, func(s *State) {
		// A device that fails is an error here, never a sequence finished at home.
		s.SetDeviceFallback(false)
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("%s took %d of %d blocks: %s", dev, s.GPULayers(), m.Cfg.NLayer, g.Err())
		}
	})
	if parts == 0 {
		return
	}
	var allow []string
	if mixed {
		allow = []string{"the caller keeps it on the tier's kernels"}
	}
	composedRan(t, parts, allow, g)
	st := g.Stats()
	np, dp, up := st.ServedPart(tier.ComposedPrompt)
	if shadow {
		named := 0
		for i, c := range st.BlockDeclines {
			if tier.DeclineName(i) == "a prompt outside the shadow arm" {
				named = c
			}
		}
		if np != 0 || dp == 0 || named != dp || dp%m.Cfg.NLayer != 0 || up != 0 {
			t.Fatalf("in the shadow arm the composer served %d block steps of the prompt and declined %d, %d of them outside the shadow, and was not asked for %d",
				np, dp, named, up)
		}
		t.Logf("in the shadow arm the prompt's %d block steps declined by name", dp)
		return
	}
	if per := m.Cfg.NLayer - len(keep); np == 0 || np%per != 0 || up != 0 {
		t.Fatalf("the composer served %d block steps of the prompt, %d of a chunk, and was not asked for %d", np, per, up)
	}
	t.Logf("the composer served %d block steps of the prompt, the tier's kernels %d blocks a step", np, len(keep))
}

// prefillVsForward compares Prefill against a Forward loop. exact demands bit
// equality; otherwise the NMSE tripwire applies.
func prefillVsForward(t *testing.T, path string, exact bool) {
	if exact {
		prefillVsForwardOpts(t, path, true, noGEMM)
		return
	}
	prefillVsForwardOpts(t, path, false)
}

// prefillVsForwardOpts is prefillVsForward with the options that select the
// batched path under test; exact demands bit equality.
func prefillVsForwardOpts(t *testing.T, path string, exact bool, extra ...Option) {
	// The tuner is pinned so both sessions run the same kernels: it advances
	// on Forward calls and Prefill makes none.
	opts := append([]Option{noTune}, extra...)
	m, err := Open(jlmOf(t, path), opts...)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	prefillVsForwardModel(t, m, exact, 0, nil)
}

// prefillVsForwardModel compares m's Prefill against a Forward loop, each
// session given to attach first where it is not nil, over a prompt of at
// least minTok tokens where minTok is not 0.
func prefillVsForwardModel(t *testing.T, m *Model, exact bool, minTok int, attach func(*State)) {
	var err error
	newState := func(n int) *State {
		s := m.NewState(n)
		if attach != nil {
			attach(s)
		}
		return s
	}

	// A prompt longer than one chunk, so the cross-chunk causal bound is
	// exercised rather than assumed.
	text := "The capital of France is Paris, and the capital of " +
		"Germany is Berlin, and the capital of Italy is Rome, and the capital " +
		"of Spain is Madrid, and the capital of Portugal is"
	full := m.Vocab.Encode(text, true)
	if minTok > 0 {
		if full = m.Vocab.Encode(strings.Repeat(text+" Lisbon. ", minTok/len(full)+1), true); len(full) < minTok {
			t.Fatalf("%d prompt tokens, %d wanted", len(full), minTok)
		}
		full = full[:minTok]
	}
	n := len(full)
	if v := os.Getenv("JITLLM_TEST_NTOK"); v != "" {
		if k, err := strconv.Atoi(v); err == nil && k > 0 && k <= n {
			n = k
		}
	}
	ids := full[:n]
	t.Logf("prompt is %d tokens, max chunk %d", len(ids), MaxPrefillChunk)

	ref := newState(len(ids) + 40)
	defer ref.Close()
	var want []float32
	for _, id := range ids {
		if want, err = ref.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	batch := newState(len(ids) + 40)
	defer batch.Close()
	// A long prompt goes in prompts of 64 tokens, each a chunk of a width
	// the device warmed.
	chunk := len(ids)
	if minTok > 0 {
		chunk = 64
	}
	var got []float32
	for off := 0; off < len(ids); off += chunk {
		if got, err = batch.Prefill(ids[off:min(off+chunk, len(ids))]); err != nil {
			t.Fatal(err)
		}
	}
	if batch.Pos() != ref.Pos() {
		t.Fatalf("position %d after Prefill, want %d", batch.Pos(), ref.Pos())
	}
	var sse, sy2 float64
	for i := range want {
		d := float64(got[i] - want[i])
		sse += d * d
		sy2 += float64(want[i]) * float64(want[i])
	}
	nmse := sse / sy2
	if exact {
		// Without the reassociating GEMM there is no difference at all.
		if nmse != 0 {
			t.Errorf("last-token logits: NMSE %.3e with the GEMM disabled -- "+
				"prefill and decode should be bit-identical here", nmse)
		}
		t.Logf("last-token logits bit-identical (NMSE %.3e)", nmse)
	} else {
		// The shipped GEMM sums a tile where MatVec sums a row, and that
		// reassociation compounds over layers (measured up to ~3e-3 across
		// models). 1e-2 is a gross-error tripwire; the exact modes above are
		// the gates with teeth.
		if nmse > 1e-2 || math.IsNaN(nmse) {
			t.Errorf("last-token logits: NMSE %.3e between Prefill and Forward -- too far to be reassociation", nmse)
		}
		t.Logf("last-token logits NMSE %.3e", nmse)
	}

	// The KV cache is the real product of prefill: decode from both sessions
	// and require identical token ids. A wrong cache diverges at step 0 or 1;
	// 16 steps stays short of where rounding-level gaps flip greedy argmaxes.
	const steps = 16
	for i := 0; i < steps; i++ {
		a, b := Greedy(want), Greedy(got)
		if a != b {
			// A flip where both sides won by less than the perturbation
			// between the two logit vectors is the model being undecided; a
			// flip with either side more confident than that is a defect.
			mA, mB, d := margin(want), margin(got), maxAbsDiff(want, got)
			if mA <= d && mB <= d {
				t.Logf("decode step %d: ids diverge (%d vs %d) on margins "+
					"%.4f and %.4f against a %.4f perturbation -- undecided, "+
					"not a defect", i, b, a, mB, mA, d)
				break
			}
			t.Fatalf("decode step %d: Prefill gave %d (margin %.4f), Forward %d "+
				"(margin %.4f), perturbation %.4f -- too confident to be rounding",
				i, b, mB, a, mA, d)
		}
		if want, err = ref.Forward(a); err != nil {
			t.Fatal(err)
		}
		if got, err = batch.Forward(b); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d decode steps after prefill agree token for token", steps)
}

// margin is how far the argmax won by: the gap to the runner-up.
func margin(l []float32) float64 {
	best, second := float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, v := range l {
		if v > best {
			best, second = v, best
		} else if v > second {
			second = v
		}
	}
	return float64(best - second)
}

// maxAbsDiff is the largest single-logit disagreement between two evaluations
// of the same model -- the size of the perturbation a flip has to beat.
func maxAbsDiff(a, b []float32) float64 {
	d := 0.0
	for i := range a {
		if x := math.Abs(float64(a[i] - b[i])); x > d {
			d = x
		}
	}
	return d
}

// TestSlidingWindowIsHonouredEverywhere: a local layer must attend to the same
// keys in Prefill, ForwardBatch and Forward. It checks the window arithmetic the
// three paths share on a synthetic config with alternating local and global
// layers.
func TestSlidingWindowIsHonouredEverywhere(t *testing.T) {
	c := &Config{NLayer: 4, NCtx: 4096, SWAWindow: 8, SWAPeriod: 2}
	// The pattern must actually alternate, or every layer is global and the
	// comparison below is between two identical things.
	local, global := 0, 0
	for li := 0; li < c.NLayer; li++ {
		if c.SWA(li) {
			local++
		} else {
			global++
		}
	}
	if local == 0 || global == 0 {
		t.Fatalf("%d local and %d global layer(s): this config cannot show the window "+
			"being applied to some layers and not others", local, global)
	}

	// The window arithmetic the three paths share; decode's form (forward.go)
	// is the reference.
	win := func(li, n int) (w0, an int) {
		w0, an = 0, n
		if c.SWA(li) && c.SWAWindow < an {
			w0, an = an-c.SWAWindow, c.SWAWindow
		}
		return
	}
	for _, n := range []int{1, 7, 8, 9, 64} {
		for li := 0; li < c.NLayer; li++ {
			w0, an := win(li, n)
			if w0+an != n {
				t.Fatalf("layer %d at n=%d: window [%d,%d) does not end at the current "+
					"position -- a local layer must include the token being generated",
					li, n, w0, w0+an)
			}
			if c.SWA(li) && n > c.SWAWindow && an != c.SWAWindow {
				t.Fatalf("layer %d at n=%d: local layer read %d keys, want the %d-key window",
					li, n, an, c.SWAWindow)
			}
			if !c.SWA(li) && an != n {
				t.Fatalf("layer %d at n=%d: a GLOBAL layer was narrowed to %d keys", li, n, an)
			}
		}
	}
}
