package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestHistorySpansRebuildTheHistory: the spans a session's history gives -- a
// chat prompt with a picture's rows in it, and the tokens decoded after it --
// rebuild it on a fresh session, so the next token's logits there are the
// session's own, within the band of the prompt forms against the decode
// forms (hostStepNMSE), on SmolVLM; the history with the picture's rows
// zeroed reads past it. A picture whose rows were given no name, and a
// lookup that knows no picture, have no history to give.
func TestHistorySpansRebuildTheHistory(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	pic, err := m.PictureSpan(img)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []model.ChatMessage{{Role: "user", Content: "What is in this picture?", Images: 1}}
	chat := func(sp model.Span) []model.Span {
		t.Helper()
		spans, err := m.ChatSpans(msgs, []model.Span{sp}, true)
		if err != nil {
			t.Fatal(err)
		}
		return spans
	}
	given := func(k model.RowsKey) (model.Span, bool) { return pic, pic.Key != nil && k == *pic.Key }
	a := testInstance(t, m, 1024)
	defer a.Close()
	lg, err := a.PrefillMixed(chat(pic)...)
	if err != nil {
		t.Fatal(err)
	}
	for range 4 {
		if lg, err = a.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
	}
	next := Greedy(lg)
	hist, err := a.HistorySpans(given)
	if err != nil {
		t.Fatal(err)
	}
	rows := 0
	for _, sp := range hist {
		if sp.Rows() {
			rows++
		}
	}
	if n := model.SpanPositions(hist, m.Cfg.NEmbd); n != a.Pos() || rows != 1 {
		t.Fatalf("the history gives %d positions, %d picture(s), for a session at %d", n, rows, a.Pos())
	}
	want, err := a.Forward(next)
	if err != nil {
		t.Fatal(err)
	}
	b := testInstance(t, m, 1024)
	defer b.Close()
	if _, err := b.PrefillMixed(hist...); err != nil {
		t.Fatal(err)
	}
	got, err := b.Forward(next)
	if err != nil {
		t.Fatal(err)
	}
	e, worst := nmseOf(got, want)
	if !(e <= hostStepNMSE) {
		t.Fatalf("the rebuilt history's next logits are %.3g NMSE (worst %.3g) from the session's (bound %.0e)", e, worst, hostStepNMSE)
	}
	if _, err := a.HistorySpans(func(model.RowsKey) (model.Span, bool) { return model.Span{}, false }); err == nil {
		t.Fatal("a lookup that knows no picture gave a history")
	}
	c := testInstance(t, m, 1024)
	defer c.Close()
	enc, err := c.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.PrefillMixed(chat(model.Span{Embd: append([]float32(nil), enc...)})...); err != nil {
		t.Fatal(err)
	}
	if _, err := c.HistorySpans(given); err == nil {
		t.Fatal("rows nothing named gave a history")
	}
	bad := append([]model.Span(nil), hist...)
	for i := range bad {
		if bad[i].Rows() {
			bad[i] = model.Span{Embd: make([]float32, len(enc))}
		}
	}
	d := testInstance(t, m, 1024)
	defer d.Close()
	if _, err := d.PrefillMixed(bad...); err != nil {
		t.Fatal(err)
	}
	wrong, err := d.Forward(next)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := nmseOf(wrong, want); !(v > hostStepNMSE) {
		t.Fatalf("the history with the picture's rows zeroed reads %.3g NMSE from the session's: the band cannot see a wrong history", v)
	}
	t.Logf("%d spans rebuild %d positions; the next logits are %.3g NMSE apart", len(hist), b.Pos(), e)
}
