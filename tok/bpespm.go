package tok

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/jitllm/jitllm/format/jlm"
)

// SentencePiece-style BPE (jlm.VocabBPESPM), Gemma 4's tokenizer. Its
// tokenizer.json is three stages, read off the artefact:
//
//	normalizer     Replace " " -> "▁"
//	pre_tokenizer  Split on " ", MergedWithPrevious -- after the normalizer
//	               there is no " " left, so the whole run is one piece
//	model          BPE by merge rank over the run's characters, with
//	               byte_fallback: a character the vocabulary lacks becomes
//	               its bytes' <0xNN> tokens
//
// It is spm.go's escaping and byte fallback with bpe.go's rank-driven merge,
// and it is neither: SPM merges any adjacent pair whose concatenation is a
// token, by score, where this merges only the pairs the merge list names.
//
// llama.cpp splits the run on newlines first ("[^\n]+|[\n]+", llama-vocab.cpp,
// LLAMA_VOCAB_PRE_TYPE_GEMMA4), because its BPE asserts no newline inside a
// word. HF tokenizers does not split, and the model's own tokenizer is the
// authority (RULE 7m): this runs the whole run. The two could differ only
// where a merge joins a newline to another character, and Gemma 4's 262144
// tokens carry none (every token with a newline in it is newlines alone), so
// on this vocabulary they agree on every input.

// spmBPE is the merge table of a VocabBPESPM vocabulary.
type spmBPE struct {
	pairs map[uint64]bpeMerge
}

func newSPMBPE(vc *jlm.Vocab, v *Vocab) error {
	if vc.Merges.Len() == 0 {
		return fmt.Errorf("tok: a SentencePiece-style BPE tokenizer with no merges")
	}
	b := &spmBPE{pairs: make(map[uint64]bpeMerge, vc.Merges.Len())}
	for i := range vc.Merges.Len() {
		m := vc.Merges.At(i)
		l, lok := v.ids.find(m.Left)
		r, rok := v.ids.find(m.Right)
		id, ok := v.ids.find2(m.Left, m.Right)
		if !lok || !rok || !ok {
			return fmt.Errorf("tok: merge %d (%q + %q) names a string that is not a token", i, m.Left, m.Right)
		}
		// The first rule for a pair wins, as HF tokenizers keeps the lowest
		// rank.
		if _, dup := b.pairs[pairKey(l, r)]; !dup {
			b.pairs[pairKey(l, r)] = bpeMerge{rank: int32(i), id: id}
		}
	}
	v.spmBPE = b
	return nil
}

// encodeSPMBPERun merges one run of text with no special token in it.
func (v *Vocab) encodeSPMBPERun(text string, out []int32) []int32 {
	if text == "" {
		return out
	}
	if v.AddSpacePrefix {
		text = " " + text
	}
	work := strings.ReplaceAll(text, " ", spaceMark)
	sc := bpeScratches.Get().(*bpeScratch)
	defer sc.release()
	// One symbol per character: its token, or -1 for a character the
	// vocabulary lacks, which no merge names and which falls back to bytes.
	syms := sc.syms[:0]
	offs := make([]int32, 0, len(work)+1)
	for i := 0; i < len(work); {
		_, n := utf8.DecodeRuneInString(work[i:])
		id, ok := v.ids.find(work[i : i+n])
		if !ok {
			id = -1
		}
		syms = append(syms, bpeSym{id: id, prev: int32(len(syms) - 1), next: int32(len(syms) + 1)})
		offs = append(offs, int32(i))
		i += n
	}
	offs = append(offs, int32(len(work)))
	syms[len(syms)-1].next = -1
	sc.syms = syms

	q := &sc.q
	*q = (*q)[:0]
	push := func(l, r int32) {
		if l < 0 || r < 0 {
			return
		}
		lid, rid := syms[l].id, syms[r].id
		if lid < 0 || rid < 0 {
			return
		}
		if m, ok := v.spmBPE.pairs[pairKey(lid, rid)]; ok {
			q.push(bpeBigram{left: l, right: r, rank: m.rank, lid: lid, rid: rid, id: m.id})
		}
	}
	for i := int32(1); i < int32(len(syms)); i++ {
		push(i-1, i)
	}
	for len(*q) > 0 {
		bg := q.pop()
		l, r := &syms[bg.left], &syms[bg.right]
		// Stale entries are skipped, as in mergeWord: a symbol's id and its
		// text determine each other.
		if l.id != bg.lid || r.id != bg.rid || l.next != bg.right {
			continue
		}
		l.id = bg.id
		r.id = deadSym
		l.next = r.next
		if r.next >= 0 {
			syms[r.next].prev = bg.left
		}
		push(l.prev, bg.left)
		push(bg.left, l.next)
	}
	for i := int32(0); i != -1; i = syms[i].next {
		if id := syms[i].id; id >= 0 {
			out = append(out, id)
			continue
		}
		// An unmerged character the vocabulary lacks: its bytes.
		out = v.emitSymbol(work[offs[i]:offs[i+1]], out)
	}
	return out
}

// encodeSPMBPE is Vocab.Encode for a VocabBPESPM vocabulary. Every control
// and user-defined token is matched literally first, as HF tokenizers matches
// added tokens before the normalizer.
func (v *Vocab) encodeSPMBPE(text string, addSpecial bool) []int32 {
	out := make([]int32, 0, len(text)/3+2)
	if addSpecial && v.AddBOS && v.BOS >= 0 {
		out = append(out, v.BOS)
	}
	for _, run := range v.partitionSpecial(text, true) {
		if run.id >= 0 {
			out = append(out, run.id)
			continue
		}
		out = v.encodeSPMBPERun(run.text, out)
	}
	return out
}
