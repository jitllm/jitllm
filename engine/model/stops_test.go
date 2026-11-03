package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok"
)

// TestChatStopsAtTheTurnsEnd: under its chat template HunyuanOCR closes its
// answer with <｜hy_Assistant｜>, the eot its GGUF states beside an eos of
// its own, and a loop that stops where Vocab.IsEOG says -- as `jitllm run`,
// the server's generate paths and the UI do -- ends its reply there. With
// the container's stated stops dropped, which is what conversion did before
// it carried them, the same greedy loop runs on past the turn.
func TestChatStopsAtTheTurnsEnd(t *testing.T) {
	src := testmodels.Path("hunyuanvl/HunyuanOCR-Q8_0.gguf")
	if !sourcePresent(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (internal/testmodels/fetch.sh fetches it) -- this gate would prove nothing (RULE 11)", src)
	}
	m, err := Open(jlmOf(t, src), noTune())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	eot, ok := m.Vocab.ID("<｜hy_Assistant｜>")
	if !ok {
		t.Fatal("no <｜hy_Assistant｜> in the vocabulary")
	}
	ids, err := m.ChatIDs([]ChatMessage{{Role: "user", Content: "What is the capital of France?"}}, true)
	if err != nil {
		t.Fatal(err)
	}
	const n = 64
	// gen is a generation loop: greedy, ended by stop or n tokens, the stop
	// token not emitted.
	gen := func(stop func(int32) bool) []int32 {
		st := m.NewInstance(len(ids) + n + 1)
		defer st.Close()
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		for len(out) < n {
			next := Greedy(lg)
			if stop(next) {
				break
			}
			out = append(out, next)
			if lg, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	// The reply up to the turn's end, read with no stop at all.
	whole := gen(func(int32) bool { return false })
	at := slices.Index(whole, eot)
	if at < 1 {
		t.Fatalf("the greedy reply %q never closes its turn with <｜hy_Assistant｜> in %d tokens (at %d) -- the gate needs one that does",
			m.Vocab.Decode(whole), n, at)
	}
	got := gen(m.Vocab.IsEOG)
	if !slices.Equal(got, whole[:at]) {
		t.Errorf("the reply stopped after %d tokens (%q), want the %d before <｜hy_Assistant｜> (%q)",
			len(got), m.Vocab.Decode(got), at, m.Vocab.Decode(whole[:at]))
	}
	t.Logf("stopped at <｜hy_Assistant｜> after %d tokens: %q", len(got), m.Vocab.Decode(got))

	// The violation: the container's stated set dropped.
	vc := *m.container.Vocab()
	vc.Stop = nil
	dropped, err := tok.New(&vc)
	if err != nil {
		t.Fatal(err)
	}
	past := gen(dropped.IsEOG)
	if len(past) <= at {
		t.Errorf("with the stated stops dropped the reply still stopped after %d tokens: the gate cannot see them", len(past))
	}
	t.Logf("with them dropped: %d tokens, on past the turn: %q", len(past), m.Vocab.Decode(past))
}
