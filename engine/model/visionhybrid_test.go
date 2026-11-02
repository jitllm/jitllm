package model

import (
	"image"
	"image/color"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestHybridImageChatRestoresAtItsEnd is gate c”” of
// docs/design/vision-as-blocks.md: a hybrid seals its recurrent summary at the
// prompt's exact end and restores only there, and with every row named -- a
// picture's rows by the picture -- a prompt holding a picture restores there
// as a text prompt does: every position, no tower block, the tokens of a cold
// run. Against the violation (the picture's rows unnamed, so the page keys stop
// at the picture) it restores nothing. The fixture is synth-qwen35-hybrid-mtp
// with synth-qwen2vl's tower converted at its width: no implemented VLM is a
// hybrid, and convert takes any mmproj whose projection is the text width.
func TestHybridImageChatRestoresAtItsEnd(t *testing.T) {
	text := testmodels.Path("synth-qwen35-hybrid-mtp.gguf")
	proj := testmodels.Path("mmproj-synth-qwen2vl.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (set JITLLM_MODELS) -- run scripts/mtpgold.py and "+
			"scripts/qwenvlgold.py (RULE 11)", text, proj)
	}
	m, err := Open(jlmOfPair(t, text, proj), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Cfg.Hybrid() || m.Tower() == nil {
		t.Fatalf("the fixture is not a hybrid with a tower (hybrid %v, tower %v)", m.Cfg.Hybrid(), m.Tower() != nil)
	}
	img := image.NewRGBA(image.Rect(0, 0, 64, 48))
	for y := range 48 {
		for x := range 64 {
			img.Set(x, y, color.RGBA{uint8(4 * x), uint8(5 * y), uint8(x * y), 255})
		}
	}
	pre := m.Vocab.Encode("Look at this:", true)
	post := m.Vocab.Encode(" What is it?", false)
	type result struct {
		restored, n int
		blocks      int64
		toks        []int32
	}
	run := func(store KVStore, named bool) result {
		st := m.NewInstance(256)
		defer st.Close()
		if store != nil {
			st.SetKVStore(store)
			mustKey(t, st, "vision-as-blocks/hybrid")
		}
		pc, err := st.Picture(img)
		if err != nil {
			t.Fatal(err)
		}
		sp := Span{Picture: pc}
		if !named {
			vs, err := st.Vision()
			if err != nil {
				t.Fatal(err)
			}
			rows, err := vs.encodePicture(pc)
			if err != nil {
				t.Fatal(err)
			}
			sp = Span{Embd: rows, Grid: pc.Grid}
		}
		spans := []Span{{Tokens: pre}, sp, {Tokens: post}}
		lg, err := st.PrefillCachedMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		var r result
		if st.visState != nil {
			r.blocks = st.visState.vis.hostBlocks + st.visState.vis.devBlocks
		}
		r.restored, r.n = st.KVRestored(), SpanPositions(spans, m.Cfg.NEmbd)
		for range 6 {
			id := Greedy(lg)
			r.toks = append(r.toks, id)
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return r
	}
	cold := run(nil, true)
	store := NewMemStore()
	first := run(store, true)
	again := run(store, true)
	if again.restored != again.n {
		t.Fatalf("the hybrid restored %d of the picture prompt's %d positions: it restores at the exact end "+
			"or nowhere, and every row of this prompt is named", again.restored, again.n)
	}
	if again.blocks != 0 {
		t.Fatalf("the restored prompt ran %d tower blocks", again.blocks)
	}
	if !slices.Equal(again.toks, cold.toks) || !slices.Equal(first.toks, cold.toks) {
		t.Fatalf("restored %v, stored %v, cold %v", again.toks, first.toks, cold.toks)
	}
	unnamed := run(store, false)
	if unnamed.restored != 0 {
		t.Fatalf("unnamed picture rows restored %d positions of a hybrid", unnamed.restored)
	}
	t.Logf("a hybrid's %d-position picture prompt restored whole at its end with no tower block, tokens %v = cold; "+
		"unnamed rows restored %d", again.n, again.toks, unnamed.restored)
}
