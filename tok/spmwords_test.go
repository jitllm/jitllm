package tok

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// spmOneHeap is llama.cpp's SentencePiece merge as it stands there: the text
// cut at its special tokens, then one queue per run seeded with every adjacent
// pair of it. Encode runs one heap per word; this is the oracle that says the
// two produce the same ids.
func spmOneHeap(v *Vocab, text string) []int32 {
	var out []int32
	prev := true
	for _, run := range v.partitionSpecial(text, false) {
		if run.id >= 0 {
			out, prev = append(out, run.id), true
			continue
		}
		out, prev = append(out, spmOneHeapRun(v, run.text, prev)...), false
	}
	return out
}

func spmOneHeapRun(v *Vocab, text string, prefix bool) []int32 {
	if text == "" {
		return nil
	}
	if v.AddSpacePrefix && prefix {
		text = " " + text
	}
	work := strings.ReplaceAll(text, " ", spaceMark)
	var syms []symbol
	for off := 0; off < len(work); {
		n := min(utf8Len(work[off]), len(work)-off)
		i := len(syms)
		next := i + 1
		if off+n >= len(work) {
			next = -1
		}
		syms = append(syms, symbol{off: off, n: n, prev: i - 1, next: next})
		off += n
	}
	h := &bigramHeap{}
	add := func(left, right int) {
		if left == -1 || right == -1 {
			return
		}
		s := work[syms[left].off : syms[right].off+syms[right].n]
		if id, ok := v.ids.find(s); ok {
			h.push(bigram{left: left, right: right, score: v.score[id], size: len(s)})
		}
	}
	for i := 1; i < len(syms); i++ {
		add(i-1, i)
	}
	for len(*h) > 0 {
		b := h.pop()
		l, r := &syms[b.left], &syms[b.right]
		if l.n == 0 || r.n == 0 || l.n+r.n != b.size {
			continue
		}
		l.n += r.n
		r.n = 0
		l.next = r.next
		if r.next >= 0 {
			syms[r.next].prev = b.left
		}
		add(l.prev, b.left)
		add(b.left, l.next)
	}
	var out []int32
	for i := 0; len(syms) > 0 && i != -1; i = syms[i].next {
		out = v.emitSymbol(work[syms[i].off:syms[i].off+syms[i].n], out)
	}
	return out
}

// TestSPMWordHeapsMatchOneHeap holds Encode's per-word SentencePiece merge to
// the one-heap algorithm on every SentencePiece container in the model directory.
//
// The inputs are the places a word cut could be wrong: runs of spaces, which
// every vocabulary here merges into "▁▁▁" tokens across what looks like a word
// boundary, and gemma-3's ">▁</", the one token with a space mark after another
// character. A cut before every space mark passes on prose and fails on both.
func TestSPMWordHeapsMatchOneHeap(t *testing.T) {
	paths, err := filepath.Glob(testmodels.Path("*.jlm"))
	if err != nil {
		t.Fatal(err)
	}
	doc, err := os.ReadFile("../AGENTS.md")
	if err != nil {
		t.Fatal(err)
	}
	inputs := []string{
		string(doc[:min(len(doc), 60000)]),
		"a b  c   d    e     f      g       h        i         j",
		strings.Repeat(" ", 1) + "x" + strings.Repeat(" ", 17) + "y" + strings.Repeat(" ", 40),
		"<p>text</p> <b>bold</b>  </b>   </div> > </a> x> </y",
		"\t\ttabs\t and\n\n  newlines \r\n  mixed  \t ",
		"日本語 の テキスト  中文  🎉 🚀👍🏽  ÀÉÎ ﬁ ſ K",
		"bad \xff\xfe utf8 \xe3\x81 end \xe3\x81",
		"   leading and trailing   ",
	}
	checked := 0
	for _, p := range paths {
		f, err := jlm.Open(p)
		if err != nil {
			continue
		}
		vc := f.Vocab()
		f.Close()
		if vc == nil {
			continue
		}
		v, err := New(vc)
		// A score merge only: not byte-level BPE, not Gemma 4's rank merge.
		if err != nil || v.bpe != nil || v.spmBPE != nil {
			continue
		}
		checked++
		for i, in := range inputs {
			got, want := v.Encode(in, false), spmOneHeap(v, in)
			if !slicesEqual(got, want) {
				t.Errorf("%s input %d: per-word %d ids, one heap %d ids; first difference at %d",
					filepath.Base(p), i, len(got), len(want), firstDiff(got, want))
			}
		}
	}
	if checked == 0 {
		testmodels.Missing(t, "%s", "no SentencePiece container found under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory): this gate proved nothing")
	}
	t.Logf("%d SentencePiece vocabularies, %d inputs each: per-word merges match one heap", checked, len(inputs))
}

func slicesEqual(a, b []int32) bool {
	return firstDiff(a, b) < 0
}

func firstDiff(a, b []int32) int {
	for i := 0; i < min(len(a), len(b)); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}
