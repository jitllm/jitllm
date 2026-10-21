package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestPrefillFoldsTheHead prefills with every block on the device twice: once
// with the output projection riding the last chunk's submission (the fold),
// once with the separate tail call it replaces, and demands the same logits
// and greedy continuation.
//
// The last row is the whole risk, so the prompt ends partway into a chunk: 700
// tokens is a full 512-row chunk and a 188-row one whose scratch is wider than
// its rows. Stats.HeadFolds must move on the folded arm only. gemma-2-2b
// carries a final softcap, which the fold must apply too.
func TestPrefillFoldsTheHead(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0", "gemma-2-2b-it-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) { foldsTheHead(t, name, spec) })
			}
		})
	}
}

func foldsTheHead(t *testing.T, name, spec string) {
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
	g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 40)
	ids := m.Vocab.Encode(text, true)
	if len(ids) < 700 {
		t.Fatalf("the prompt is %d tokens; the gate needs 700", len(ids))
	}
	ids = ids[:700]

	run := func(fold bool) (logits []float32, toks []int32, folds int) {
		st := m.NewState(len(ids) + 16)
		defer st.Close()
		st.SetDevice(g)
		if gs := g.Stats(); (st.GPULayers() < m.Cfg.NLayer || !st.HeadOnDevice()) && gs.Declined+gs.NoRoom > 0 {
			// The fold is the resident head riding the chunk over every block,
			// so a card that holds the blocks but not the head -- or the head
			// and not every block, which is where a card shared with a desktop
			// lands -- has nothing to fold.
			t.Skipf("CARD TOO SMALL: %d of %d blocks and head %v placed (%s) -- this gate "+
				"proved nothing on this device; run it on one that holds both", st.GPULayers(), m.Cfg.NLayer,
				st.HeadOnDevice(), g.Err())
		}
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Fatalf("placed %d of %d blocks, head %v: %s", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
		}
		st.noHeadFold = !fold
		before := g.Stats().HeadFolds
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		folds = g.Stats().HeadFolds - before
		if st.DeviceDemotions() != 0 {
			t.Fatalf("the device demoted during the prefill: %s", g.Err())
		}
		logits = append([]float32(nil), lg...)
		next := Greedy(lg)
		for i := 0; i < 8; i++ {
			toks = append(toks, next)
			if lg, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
			next = Greedy(lg)
		}
		return logits, toks, folds
	}
	want, wantToks, tailFolds := run(false)
	got, gotToks, folds := run(true)
	if tailFolds != 0 || folds == 0 {
		// The fold rides a batched chunk, whose scratch the device reserves at
		// placement (tier/scratch.go): a card the model nearly fills
		// once refused it at the prompt with out of memory, and the chunk went
		// a row at a time with nothing to fold.
		t.Fatalf("folds: %d on the tail-call arm, %d on the folded one -- the arms are not two arms "+
			"(%d chunk(s) split, %d bytes of scratch for %d rows: %s)", tailFolds, folds,
			g.Stats().PromptSplits, g.Stats().ScratchBytes, g.Stats().ReservedPrompt, g.Err())
	}
	var num, den float64
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("logit %d is %v", i, got[i])
		}
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	nmse := num / den
	// The same row through the same kernels: bit equality is the expectation,
	// and the bound only tolerates a reduction the tuner could reorder.
	if !(nmse < 1e-10) {
		t.Fatalf("prefill logits NMSE %.3e between the folded head and the tail call", nmse)
	}
	for i := range wantToks {
		if gotToks[i] != wantToks[i] {
			t.Fatalf("token %d: folded %v, tail call %v", i, gotToks, wantToks)
		}
	}
	t.Logf("%d fold(s); logits NMSE %.3e; 8 tokens identical %v", folds, nmse, gotToks)
}
