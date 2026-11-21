package server

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok"
)

// streamText is where a streaming shim actually goes wrong, so it is gated on
// its own rather than only through the HTTP surface.

type fakeVocab struct{ pieces map[int32]string }

// TestStopStringIsNeverLeakedAcrossTokens is the gate for the held-back tail.
// With the hold-back removed (hold = 0) it fails with `leaked "<" before the
// stop string completed`; a check on the final text alone cannot see this.
func TestStopStringIsNeverLeakedAcrossTokens(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{
		1: "hello", 2: "<", 3: "|end", 4: "|>", 5: "tail",
	}}
	st := newStreamText(v.next, []string{"<|end|>"})

	var emitted strings.Builder
	var stopped bool
	var match string
	ids := []int32{}
	for _, id := range []int32{1, 2, 3, 4, 5} {
		ids = append(ids, id)
		chunk, hit, m := st.push(ids)
		emitted.WriteString(chunk)
		if !hit && strings.Contains(emitted.String(), "<") {
			t.Fatalf("leaked %q before the stop string completed: emitted %q after ids %v",
				"<", emitted.String(), ids)
		}
		if hit {
			stopped, match = true, m
			break
		}
	}
	if !stopped {
		t.Fatalf("the stop string never matched; emitted %q", emitted.String())
	}
	if match != "<|end|>" {
		t.Fatalf("matched %q, want %q", match, "<|end|>")
	}
	if got := emitted.String(); got != "hello" {
		t.Fatalf("emitted %q, want %q -- the stop string itself is never part of the completion", got, "hello")
	}
}

// TestFlushReleasesTheHeldTailWhenNoStopMatched proves the hold-back is not a
// silent truncation: without flush() the last len(stop)-1 bytes vanish and it
// fails with `after flush got "hell", want "hello"`.
func TestFlushReleasesTheHeldTailWhenNoStopMatched(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{1: "hel", 2: "lo"}}
	st := newStreamText(v.next, []string{"<|end|>"})
	var b strings.Builder
	ids := []int32{}
	for _, id := range []int32{1, 2} {
		ids = append(ids, id)
		chunk, hit, _ := st.push(ids)
		if hit {
			t.Fatal("no stop string is present, yet one matched")
		}
		b.WriteString(chunk)
	}
	if b.String() == "hello" {
		t.Fatalf("nothing was held back, so this gate cannot see a missing flush; "+
			"hold is %d", st.hold)
	}
	b.WriteString(st.flush())
	if got := b.String(); got != "hello" {
		t.Fatalf("after flush got %q, want %q", got, "hello")
	}
}

// TestAMultiByteRuneIsNeverSplitAcrossTwoWrites: a BPE token can split a
// rune. Without truncPartialRune in push() it fails with `chunk 0 is not
// valid UTF-8`.
func TestAMultiByteRuneIsNeverSplitAcrossTwoWrites(t *testing.T) {
	// "€" is e2 82 ac; the tokenizer splits it across two ids.
	v := fakeVocab{pieces: map[int32]string{1: "\xe2\x82", 2: "\xac", 3: "!"}}
	st := newStreamText(v.next, nil)
	var b strings.Builder
	ids := []int32{}
	for i, id := range []int32{1, 2, 3} {
		ids = append(ids, id)
		chunk, _, _ := st.push(ids)
		for _, r := range chunk {
			if r == '�' {
				t.Fatalf("chunk %d is not valid UTF-8: %q", i, chunk)
			}
		}
		b.WriteString(chunk)
	}
	b.WriteString(st.flush())
	if got := b.String(); got != "€!" {
		t.Fatalf("reassembled %q, want %q", got, "€!")
	}
}

// TestNoStopStringMeansNoHoldBack: the common case must not be delayed by a
// window it does not need.
func TestNoStopStringMeansNoHoldBack(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{1: "a", 2: "b"}}
	st := newStreamText(v.next, nil)
	if st.hold != 0 {
		t.Fatalf("hold is %d with no stop strings; a request with none must stream every byte immediately", st.hold)
	}
	chunk, _, _ := st.push([]int32{1})
	if chunk != "a" {
		t.Fatalf("first chunk %q, want %q", chunk, "a")
	}
}

func (v fakeVocab) next(id int32) string { return v.pieces[id] }

// oldStreamText is streamText as it was: the whole id list re-decoded on
// every push, the whole text searched for every stop. It is the reference
// the incremental form is held to, byte for byte.
type oldStreamText struct {
	v       *tok.Vocab
	stops   []string
	hold    int
	decoded string
	emitted int
}

func (t *oldStreamText) push(ids []int32) (string, bool, string) {
	t.decoded = t.v.DecodeChat(ids)
	for _, s := range t.stops {
		if s == "" {
			continue
		}
		if i := strings.Index(t.decoded, s); i >= 0 {
			chunk := ""
			if i > t.emitted {
				chunk = t.decoded[t.emitted:i]
			}
			t.emitted = len(t.decoded)
			return truncPartialRune(chunk), true, s
		}
	}
	safe := len(t.decoded) - t.hold
	if safe <= t.emitted {
		return "", false, ""
	}
	chunk := truncPartialRune(t.decoded[t.emitted:safe])
	t.emitted += len(chunk)
	return chunk, false, ""
}

func (t *oldStreamText) flush() string {
	chunk := t.decoded[t.emitted:]
	t.emitted = len(t.decoded)
	return chunk
}

// TestStreamTextIsTheWholeListDecode: on real vocabularies, over a long
// completion of multi-byte text, every push returns what re-decoding the
// whole list returned -- chunk, stop and match -- with stop strings that
// half-match over and over before one completes, and none at all.
//
// VIOLATION SIGNATURE. Start push's stop search at before rather than
// before-len(s)+1 and a stop straddling two tokens is missed:
// `push N: got (..., false, "") want (..., true, ...)`.
func TestStreamTextIsTheWholeListDecode(t *testing.T) {
	para := "Once upon a time, a café in 東京 served naïve “crème brûlée” 🙂 — 日本語。 "
	for _, stem := range []string{"stories15M-q8_0.jlm", "Llama-3.2-1B-Instruct-Q4_K_M.jlm"} {
		c, err := jlm.Open(modelPath(t, stem))
		if err != nil {
			t.Fatal(err)
		}
		jv := c.Vocab()
		c.Close()
		v, err := tok.New(jv)
		if err != nil {
			t.Fatal(err)
		}
		ids := v.Encode(strings.Repeat(para, 40), false)
		ids = append(ids, v.Encode("The END marker 日本語。", false)...)
		if len(ids) < 1000 {
			t.Fatalf("%s: %d ids is not a long completion", stem, len(ids))
		}
		for _, stops := range [][]string{nil, {"END marker"}, {"brûlée” 🙂 — 日本語。 X", "日本語。"}, {"zzz", "naïve “crème"}} {
			hold := 0
			for _, s := range stops {
				hold = max(hold, len(s))
			}
			ref := &oldStreamText{v: v, stops: stops, hold: max(hold-1, 0)}
			st := newStreamText(v.NewChatStream().Next, stops)
			stopped := false
			for i := range ids {
				gc, gh, gm := st.push(ids[:i+1])
				wc, wh, wm := ref.push(ids[:i+1])
				if gc != wc || gh != wh || gm != wm {
					t.Fatalf("%s %q: push %d: got (%q, %v, %q) want (%q, %v, %q)", stem, stops, i, gc, gh, gm, wc, wh, wm)
				}
				if gh {
					stopped = true
					break
				}
			}
			if len(stops) > 0 && !stopped {
				t.Fatalf("%s %q: no stop matched, so the stop arm proved nothing", stem, stops)
			}
			if !stopped {
				if g, w := st.flush(), ref.flush(); g != w {
					t.Fatalf("%s: flush %q, want %q", stem, g, w)
				}
			}
		}
	}
}

// TestStreamTextDecodesEachTokenOnce: the work a push does does not grow with
// the completion. Counted, not timed (RULE 4): over 2000 pushes the decoder is
// asked for 2000 ids. Re-decoding the list, as streamText once did, asks for
// n(n+1)/2.
//
// VIOLATION SIGNATURE. Decode ids rather than ids[t.n:] in push and this
// fails with `decoded 2001000 ids for 2000 tokens`.
func TestStreamTextDecodesEachTokenOnce(t *testing.T) {
	decoded := 0
	next := func(id int32) string {
		decoded++
		return "x"
	}
	st := newStreamText(next, []string{"<|end|>"})
	ids := make([]int32, 0, 2000)
	for i := 0; i < 2000; i++ {
		ids = append(ids, int32(i))
		st.push(ids)
	}
	if decoded != len(ids) {
		t.Fatalf("decoded %d ids for %d tokens", decoded, len(ids))
	}
}

// TestAnInvalidByteDoesNotStallTheStream: a byte that can never become part
// of a rune is shown at once, and every byte after it streams too. Holding
// back until the chunk is valid UTF-8 held back the rest of the generation.
func TestAnInvalidByteDoesNotStallTheStream(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{1: "abc", 2: "\x80", 3: "def", 4: "\xe2\x82"}}
	st := newStreamText(v.next, nil)
	var got string
	ids := []int32{}
	for _, id := range []int32{1, 2, 3, 4} {
		ids = append(ids, id)
		chunk, _, _ := st.push(ids)
		got += chunk
	}
	if want := "abc\x80def"; got != want {
		t.Fatalf("streamed %q before the end, want %q (the unfinished rune held back)", got, want)
	}
}
