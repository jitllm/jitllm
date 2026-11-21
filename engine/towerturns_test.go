package engine

import (
	"image"
	"image/draw"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestTwoTurnImageChatRestoresPastThePicture: a second turn about the same
// picture restores every whole chunk of the first turn's prompt -- the text
// before the picture, its rows and the text after them -- encodes no picture,
// and decodes the tokens a cold run does. Against the violations nothing past
// the picture restores: rows with no name (the gap the prefix cache stops at)
// and a picture one pixel apart.
func TestTwoTurnImageChatRestoresPastThePicture(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	other := image.NewNRGBA(img.Bounds())
	draw.Draw(other, other.Bounds(), img, img.Bounds().Min, draw.Src)
	other.Pix[0] ^= 0x40
	turn1 := []model.ChatMessage{{Role: "user", Content: "What is in this picture?", Images: 1}}
	turn2 := append(append([]model.ChatMessage(nil), turn1...),
		model.ChatMessage{Role: "assistant", Content: "A picture of colours."},
		model.ChatMessage{Role: "user", Content: "What colour is the disc?"})
	type result struct {
		restored, imgAt, imgEnd, n, encodes int
		toks                                []int32
	}
	run := func(store KVStore, msgs []model.ChatMessage, pic image.Image, named bool) result {
		t.Helper()
		st := testInstance(t, m, 2048)
		defer st.Close()
		st.SetKVStore(store)
		mustKey(t, st, "tower/two-turn")
		sp, err := m.PictureSpan(pic)
		if err != nil {
			t.Fatal(err)
		}
		if !named {
			rows, err := st.EncodePicture(pic)
			if err != nil {
				t.Fatal(err)
			}
			sp = model.Span{Embd: slices.Clone(rows)}
		}
		spans, err := m.ChatSpans(msgs, []model.Span{sp}, true)
		if err != nil {
			t.Fatal(err)
		}
		var r result
		for _, s := range spans {
			if s.Rows() {
				r.imgEnd = r.imgAt + s.N(m.Cfg.NEmbd)
				break
			}
			r.imgAt += s.N(m.Cfg.NEmbd)
		}
		before := 0
		if st.tower != nil {
			before = st.tower.encodes
		}
		lg, err := st.PrefillCachedMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		r.encodes = st.tower.encodes - before
		r.restored, r.n = st.KVRestored(), model.SpanPositions(spans, m.Cfg.NEmbd)
		for range 8 {
			id := Greedy(lg)
			r.toks = append(r.toks, id)
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return r
	}
	cold := run(NewMemStore(), turn2, img, true)
	if cold.encodes != 1 {
		t.Fatalf("the cold turn encoded %d pictures, want its one", cold.encodes)
	}
	store := NewMemStore()
	t1 := run(store, turn1, img, true)
	t2 := run(store, turn2, img, true)
	if want := t1.n / kvChunk * kvChunk; t2.restored < want || t2.restored <= t1.imgEnd {
		t.Fatalf("turn 2 restored %d positions of turn 1's %d (want %d, past the picture at %d..%d)",
			t2.restored, t1.n, want, t1.imgAt, t1.imgEnd)
	}
	if t2.encodes != 0 {
		t.Fatalf("turn 2 restored the picture and still encoded %d", t2.encodes)
	}
	if !slices.Equal(t2.toks, cold.toks) {
		t.Fatalf("the restored turn decodes %v, a cold one %v", t2.toks, cold.toks)
	}
	diff := run(store, turn2, other, true)
	if diff.restored > diff.imgAt {
		t.Fatalf("VIOLATION: a picture a pixel apart restored %d positions, past its start at %d", diff.restored, diff.imgAt)
	}
	unnamed := run(store, turn2, img, false)
	if unnamed.restored > unnamed.imgAt {
		t.Fatalf("VIOLATION: unnamed rows restored %d positions, past the picture's start at %d", unnamed.restored, unnamed.imgAt)
	}
	t.Logf("turn 1 %d positions (picture %d..%d); turn 2 restored %d of %d with no picture encoded, tokens %v = cold; "+
		"a picture a pixel apart restored %d, unnamed rows %d", t1.n, t1.imgAt, t1.imgEnd, t2.restored, t2.n, t2.toks,
		diff.restored, unnamed.restored)
}
