package tok

import (
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/pretok"
)

// Byte-level BPE, which is what every model newer than llama2 carries
// (tokenizer.ggml.model == "gpt2"): llama3, qwen2, qwen3, phi, gpt-oss.
//
// It is a different algorithm from spm.go, not a variant: SPM merges by score
// and never reads the merges; BPE merges by the rank of a pair in the merges
// array and never reads scores. Mixing them produces fluent, wrong output.
//
// Three stages, in this order, and the order is load-bearing:
//
//  1. Split off special tokens (<|im_start|> and friends) by longest match.
//     They are never subject to the regex or to merging.
//  2. Split the remaining text with the architecture's pre-tokenizer pattern.
//  3. Byte-encode each piece, then merge inside it. Merges never cross a piece
//     boundary, which is the whole reason step 2 exists.

// bpeState is the BPE half of a Vocab.
type bpeState struct {
	// pairs is the merge table keyed by the two halves' token ids, the value
	// being the merge's rank (lower wins) and the id of what it makes. It is HF
	// tokenizers' table and keeps merging free of strings.
	pairs map[uint64]bpeMerge
	// byteID is the token of each byte's byte-level rune: a word's symbols
	// before any merge.
	byteID [256]int32
	pre    string
	// ignoreMerges emits a pre-token that is already a vocabulary entry whole
	// instead of merging it up. It is not an optimisation: the merge sequence
	// can arrive at a different split, so a llama3 vocabulary without it gives
	// different ids for ordinary words.
	ignoreMerges bool
}

// gpt2 byte<->rune tables; pretok.ByteToRune is the definition.
var (
	byteToRune = pretok.ByteToRune()
	runeToByte = map[rune]byte{}
)

func init() {
	for c := 0; c < 256; c++ {
		runeToByte[byteToRune[c]] = byte(c)
	}
}

func byteDecode(s string, out *strings.Builder) {
	for _, r := range s {
		if b, ok := runeToByte[r]; ok {
			out.WriteByte(b)
			continue
		}
		// Not from the byte alphabet: a literal, such as an added token that
		// was stored as plain text. Pass it through.
		out.WriteRune(r)
	}
}

// newBPE builds the BPE half. The caller has already filled text/ids/kind.
//
// override is the pipeline the caller asserted with WithTokenizer, or nil. It
// wins over everything else: the model's own tokenizer.json knows something
// tokenizer.ggml.pre cannot say.
func newBPE(vc *jlm.Vocab, v *Vocab, override []pretok.Op) error {
	if len(vc.Merges) == 0 {
		return fmt.Errorf("tok: a byte-level BPE tokenizer with no merges")
	}
	b := &bpeState{pairs: make(map[uint64]bpeMerge, len(vc.Merges))}
	b.pre = vc.PreName
	// The stages come from the container, resolved at conversion. The
	// generated pretok table is consulted only for a container written before
	// stages were stored; a name in neither falls through to the legacy family
	// splitters. A caller-supplied tokenizer.json outranks all of them.
	ops, ok := opsFromContainer(vc.Pre)
	if !ok {
		ops, ok = pretok.PreOps(b.pre)
	}
	if len(override) > 0 {
		ops, ok = override, true
	}
	if ok {
		pipe, err := buildPipeline(ops)
		if err != nil {
			return fmt.Errorf("tok: pre-tokenizer %q: %w", b.pre, err)
		}
		v.pipe = pipe
	}
	// The pair is explicit in the container (the GGUF's "left right" string is
	// ambiguous when a half contains a space). A merge whose halves or result
	// are not tokens is refused, as HF tokenizers refuses it; an id table cannot
	// represent such a merge anyway.
	for i, m := range vc.Merges {
		l, lok := v.ids[m.Left]
		r, rok := v.ids[m.Right]
		id, ok := v.ids[m.Left+m.Right]
		if !lok || !rok || !ok {
			return fmt.Errorf("tok: merge %d (%q + %q) names a string that is not a token", i, m.Left, m.Right)
		}
		b.pairs[pairKey(l, r)] = bpeMerge{rank: int32(i), id: id}
	}
	for c := 0; c < 256; c++ {
		b.byteID[c] = -1
		if id, ok := v.ids[string(byteToRune[c])]; ok {
			b.byteID[c] = id
		}
	}
	// Stored, not inferred from the name family: it is a property of the
	// vocabulary, and the two do not always agree.
	b.ignoreMerges = vc.IgnoreMerges
	v.indexSpecial()
	v.bpe = b
	return nil
}

// encodeBPE is Vocab.Encode for a byte-level BPE vocabulary.
func (v *Vocab) encodeBPE(text string, addSpecial bool) []int32 {
	// Sized for 3 bytes a token: English prose and code run 3.3-3.5 here, so
	// their ids are written once rather than copied through every doubling on
	// the way up; denser text (CJK, symbols, 2-2.7) grows once.
	out := make([]int32, 0, len(text)/3+2)
	sc := bpeScratches.Get().(*bpeScratch)
	defer sc.release()
	word := func(w string) { out = v.mergeWord(w, out, sc) }
	if addSpecial && v.AddBOS && v.BOS >= 0 {
		out = append(out, v.BOS)
	}
	for _, run := range v.partitionSpecial(text, true) {
		if run.id >= 0 {
			out = append(out, run.id)
			continue
		}
		v.preSplitEach(run.text, sc, word)
	}
	return out
}

type specialRun struct {
	text string
	// id is the token when this run is a special token, and -1 for ordinary
	// text. Always set it: the zero value is token 0.
	id int32
}

// indexSpecial collects the tokens partitionSpecial matches literally: every
// control and user-defined token, which is llama.cpp's tokenizer_st_partition
// with parse_special on.
func (v *Vocab) indexSpecial() {
	v.special = v.special[:0]
	for id, k := range v.kind {
		if k == typeControl || k == typeUserDefined {
			v.special = append(v.special, int32(id))
		}
	}
	// Longest first, so a special token that is a prefix of another cannot win.
	for i := 1; i < len(v.special); i++ {
		for j := i; j > 0 && len(v.text[v.special[j]]) > len(v.text[v.special[j-1]]); j-- {
			v.special[j], v.special[j-1] = v.special[j-1], v.special[j]
		}
	}
	for _, id := range v.special {
		if s := v.text[id]; s != "" {
			v.specialAt[s[0]] = append(v.specialAt[s[0]], id)
		}
	}
}

// partitionSpecial cuts the text at every special token, longest match first.
// control says whether control tokens count (llama.cpp's parse_special);
// user-defined ones always do.
func (v *Vocab) partitionSpecial(text string, control bool) []specialRun {
	if len(v.special) == 0 {
		return []specialRun{{text: text, id: -1}}
	}
	var out []specialRun
	last := 0
	for i := 0; i < len(text); {
		hit := int32(-1)
		hlen := 0
		for _, id := range v.specialAt[text[i]] {
			if !control && v.kind[id] == typeControl {
				continue
			}
			if s := v.text[id]; strings.HasPrefix(text[i:], s) {
				hit, hlen = id, len(s)
				break // special is sorted longest first
			}
		}
		if hit < 0 {
			i++
			continue
		}
		if i > last {
			out = append(out, specialRun{text: text[last:i], id: -1})
		}
		out = append(out, specialRun{id: hit})
		i += hlen
		last = i
	}
	if last < len(text) {
		out = append(out, specialRun{text: text[last:], id: -1})
	}
	return out
}

// bpeSym is one symbol of a word being merged: its token (dead once merged
// into its left neighbour), in a doubly linked list so a merge is O(1) and the
// neighbours are cheap to re-queue.
type bpeSym struct {
	id         int32
	prev, next int32
}

// deadSym marks a symbol merged into its left neighbour. -1 is taken: it is a
// byte whose rune the vocabulary lacks, which is live and merges with nothing.
const deadSym = -2

// bpeMerge is one entry of the merge table: its rank and what it makes.
type bpeMerge struct{ rank, id int32 }

func pairKey(l, r int32) uint64 { return uint64(uint32(l))<<32 | uint64(uint32(r)) }

// bpeScratch is mergeWord's working memory, reused across the words of one
// Encode, so a symbol list and a queue are not allocated per word.
type bpeScratch struct {
	syms []bpeSym
	q    bpeQ
	buf  []byte
	pre  []runes // one per pre-tokenizer stage; see runes
}

// bpeScratches keeps scratch between Encodes. One that has grown past
// maxPooledRunes is dropped, so a single giant input does not pin its buffers.
var bpeScratches = sync.Pool{New: func() any { return new(bpeScratch) }}

const maxPooledRunes = 1 << 20

func (sc *bpeScratch) release() {
	for _, st := range sc.pre {
		if cap(st.rs) > maxPooledRunes {
			return
		}
	}
	bpeScratches.Put(sc)
}

// stages is n pre-tokenizer stage buffers, kept across calls.
func (sc *bpeScratch) stages(n int) []runes {
	for len(sc.pre) < n {
		sc.pre = append(sc.pre, runes{})
	}
	return sc.pre[:n]
}

// mergeWord runs BPE over one pre-tokenized word, in raw bytes. The word is
// never byte-encoded: the byte-level alphabet is one rune per byte, so the
// initial symbols are byteID and every merge is an id lookup. Only the
// ignoreMerges whole-word lookup builds the encoded text, in scratch.
func (v *Vocab) mergeWord(word string, out []int32, sc *bpeScratch) []int32 {
	if word == "" {
		return out
	}
	if v.bpe.ignoreMerges {
		sc.buf = sc.buf[:0]
		for i := 0; i < len(word); i++ {
			sc.buf = utf8.AppendRune(sc.buf, byteToRune[word[i]])
		}
		if id, ok := v.ids[string(sc.buf)]; ok {
			return append(out, id)
		}
	}
	syms := sc.syms[:0]
	for i := 0; i < len(word); i++ {
		syms = append(syms, bpeSym{id: v.bpe.byteID[word[i]], prev: int32(i - 1), next: int32(i + 1)})
	}
	syms[len(syms)-1].next = -1
	sc.syms = syms

	q := &sc.q
	*q = (*q)[:0]
	push := func(l, r int32) {
		if l < 0 || r < 0 {
			return
		}
		lid, rid := syms[l].id, syms[r].id
		if m, ok := v.bpe.pairs[pairKey(lid, rid)]; ok {
			q.push(bpeBigram{left: l, right: r, rank: m.rank, lid: lid, rid: rid, id: m.id})
		}
	}
	for i := int32(1); i < int32(len(syms)); i++ {
		push(i-1, i)
	}
	for len(*q) > 0 {
		bg := q.pop()
		l, r := &syms[bg.left], &syms[bg.right]
		// The queue holds stale entries by design: a merge invalidates its
		// neighbours' pairs but they are cheaper to skip here than to remove.
		// A symbol's text and its id determine each other, so an entry is
		// current iff both halves still carry the ids it was queued with --
		// which a dead symbol never does.
		if l.id != bg.lid || r.id != bg.rid {
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
	// A symbol that is still -1 is a byte whose rune the vocabulary lacks;
	// byte-level vocabularies carry all 256, and one that did not emits
	// nothing for it.
	for i := int32(0); i != -1; i = syms[i].next {
		if id := syms[i].id; id >= 0 {
			out = append(out, id)
		}
	}
	return out
}

type bpeBigram struct {
	left, right int32
	rank        int32
	lid, rid    int32 // the halves' tokens when queued
	id          int32 // what the merge makes
}

// bpeQ is a binary min-heap of bigrams, written out rather than driven through
// container/heap, whose interface calls and boxing are measurable cost.
type bpeQ []bpeBigram

// less is llama.cpp's comparator: lower rank wins, and on a tie the leftmost
// pair does. Ties are common, and breaking them differently changes the
// tokenization of ordinary text.
func (a bpeBigram) less(b bpeBigram) bool {
	return a.rank < b.rank || (a.rank == b.rank && a.left < b.left)
}

func (q *bpeQ) push(x bpeBigram) {
	h := append(*q, x)
	for i := len(h) - 1; i > 0; {
		p := (i - 1) / 2
		if !h[i].less(h[p]) {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
	*q = h
}

func (q *bpeQ) pop() bpeBigram {
	h := *q
	top := h[0]
	n := len(h) - 1
	h[0] = h[n]
	h = h[:n]
	for i := 0; ; {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && h[c+1].less(h[c]) {
			c++
		}
		if !h[c].less(h[i]) {
			break
		}
		h[i], h[c] = h[c], h[i]
		i = c
	}
	*q = h
	return top
}

// decodeBPE is Vocab.Decode for a byte-level BPE vocabulary.
func (v *Vocab) decodeBPE(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(v.text) {
			continue
		}
		if k := v.kind[id]; k == typeControl || k == typeUnused {
			continue
		}
		byteDecode(v.text[id], &b)
	}
	return b.String()
}

// llama3Family is the pre-tokenizer group llama.cpp gives the llama3 regex, and
// with it add_bos and ignore_merges. It is the legacy path, reached only by a
// container that stored no stages.
//
// "default" is here only to keep an older container's answer what it was; it
// is wrong for it (llama.cpp gives "default" its own pipeline, which the
// converter now stores as pretok.LlamaCppDefault). Reconvert to fix.
var llama3Family = map[string]bool{
	"default": true, "llama3": true, "llama-v3": true, "llama-bpe": true,
	"falcon3": true, "falcon-h1": true, "pixtral": true,
	"midm-2.0": true, "lfm2": true, "jina-v5-nano": true,
}

// digitGroupFor is the \p{N} quantifier the model's own tokenizer.json asks
// for. Most of the qwen2-shaped names want one digit at a time (Qwen2's own
// pattern is `\p{N}`); only dbrx and smaug-bpe want three.
func digitGroupFor(pre string) int {
	switch pre {
	case "dbrx", "smaug-bpe":
		return 3 // llama.cpp gives these the {1,3} pattern
	}
	return 1
}

// qwen2Family and dbrxFamily share llama3's SHAPE but not its digit rule; see
// digitGroupFor. They also do not take its ignore_merges or add_bos default.
var qwen2Family = map[string]bool{
	"qwen2": true, "qwen35": true, "hunyuan": true, "solar-open": true,
	"minicpm5":         true, // llama3-shaped but single-digit; see digitGroupFor
	"deepseek-r1-qwen": true, "kormo": true, "f2llmv2": true,
	"dbrx": true, "smaug-bpe": true, "stablelm2": true,
}

// gpt2Family gets the original GPT-2 pattern: no case-insensitive
// contractions, no digit grouping in threes, no newline runs. It is a
// different split from llama3's, not a simplification of it.
var gpt2Family = map[string]bool{
	"gpt-2": true, "mpt": true, "olmo": true, "jais": true, "trillion": true,
	"granite-docling": true, "phi-2": true, "gigachat": true, "mellum": true,
	"modern-bert": true, "a.x-4.0": true,
	"jina-es": true, "jina-de": true, "jina-v2-es": true, "jina-v2-de": true,
}

// starcoderFamily is gpt2's pattern with digits split one at a time first.
var starcoderFamily = map[string]bool{
	"starcoder": true, "refact": true, "command-r": true, "smollm": true,
	"codeshell": true, "exaone": true, "minerva-7b": true, "mellum2": true,
}

// preSplit applies the architecture's pre-tokenizer pattern and gathers the
// pieces; Encode streams them instead (preSplitEach).
func (v *Vocab) preSplit(text string) []string {
	var sc bpeScratch
	return collect(func(emit func(string)) { v.preSplitEach(text, &sc, emit) })
}

// preSplitEach applies the architecture's pre-tokenizer pattern, handing each
// word to emit the moment the last stage finds it. No stage's output is
// collected, which avoids most of Encode's allocation; order is preserved because
// every stage is an order-preserving split of the piece it is given.
func (v *Vocab) preSplitEach(text string, sc *bpeScratch, emit func(string)) {
	// The composed pipeline from the model's tokenizer.json, when there is one.
	if v.pipe != nil {
		v.pipe.each(text, sc.stages(len(v.pipe.ops)), emit)
		return
	}
	switch {
	case llama3Family[v.bpe.pre]:
		splitLlama3Each(text, llama3Params(3), &sc.stages(1)[0], emit)
	case qwen2Family[v.bpe.pre]:
		splitLlama3Each(text, llama3Params(digitGroupFor(v.bpe.pre)), &sc.stages(1)[0], emit)
	case gpt2Family[v.bpe.pre]:
		splitGPT2Each(text, &sc.stages(1)[0], emit)
	case starcoderFamily[v.bpe.pre]:
		st := sc.stages(2)
		splitDigitsEach(text, &st[0], func(run string) { splitGPT2Each(run, &st[1], emit) })
	}
	// An unknown pre-tokenizer is an error, not a default: the patterns are not
	// interchangeable. tok.New refuses such a vocabulary before reaching here.
}

// splitLlama3 is the parameterised Split stage at llama3's settings, the
// pattern llama3, qwen2 and qwen3 share:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//	| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Hand-rolled because `\s+(?!\S)` is a negative lookahead RE2 cannot run.
// It follows llama.cpp's hand-written version step for step so the two agree
// where the pattern alone does not settle a case.
func splitLlama3(text string, digitGroup int) []string {
	return splitLlama3Params(text, llama3Params(digitGroup))
}

func llama3Params(digitGroup int) pretok.SplitParams {
	return pretok.SplitParams{Contractions: true, CaseFold: true, DigitGroup: digitGroup}
}

// splitLlama3Params is the Split stage. Every pattern in the wild is this
// skeleton with different parameters; see pretok.SplitParams.
func splitLlama3Params(text string, sp pretok.SplitParams) []string {
	return collect(func(emit func(string)) { splitLlama3Each(text, sp, &runes{}, emit) })
}

// splitLlama3Each is the Split stage, handing each piece to emit as it is found.
func splitLlama3Each(text string, sp pretok.SplitParams, sc *runes, emit func(string)) {
	digitGroup := sp.DigitGroup
	if digitGroup < 1 {
		digitGroup = 1
	}
	sc.load(text)
	rs := sc.rs
	n := len(rs)
	prev := 0
	add := func(end int) {
		if end > prev {
			emit(sc.piece(text, prev, end))
		}
		prev = end
	}
	isLetter := func(i int) bool {
		if i < 0 || i >= n {
			return false
		}
		// qwen35's pattern is [\p{L}\p{M}]+ -- letters AND combining marks --
		// where llama3's is \p{L}+. A mark joining the letter run or starting a
		// new piece is a different tokenization of the same text.
		if sp.LetterMarks && ucFlags(rs[i])&ucFlagAccentMark != 0 {
			return true
		}
		return ucIsLetter(rs[i])
	}
	isNumber := func(i int) bool { return i >= 0 && i < n && ucIsNumber(rs[i]) }
	isSpace := func(i int) bool { return i >= 0 && i < n && ucIsSpace(rs[i]) }
	// isHan is \p{Han}, and under HanRuns it is subtracted from the letter
	// classes rather than merely taken first -- see SplitParams.HanRuns for why
	// the two halves are one flag.
	isHan := func(i int) bool { return sp.HanRuns && i >= 0 && i < n && ucIsHan(rs[i]) }
	// Han is \p{Lo}, a letter to every predicate below and to the upper
	// class, so the subtraction is wrapped into the classes once rather than
	// patched at each call site.
	isLetterX := func(i int) bool { return isLetter(i) && !isHan(i) }
	// plain is "in range and none of letter/number/whitespace", which is the
	// [^\s\p{L}\p{N}] class plus the bounds check llama.cpp writes as as_uint().
	// It uses the unsubtracted isLetter on purpose: those classes are plain
	// \p{L} in the pattern, so a Han codepoint is never punctuation.
	plain := func(i int) bool { return i >= 0 && i < n && !isLetter(i) && !isNumber(i) && !isSpace(i) }

	// contractAt matches (?i:'s|'t|'re|'ve|'m|'ll|'d) at p and returns where it
	// ends, or p. SuffixContract decides where it may appear.
	contractAt := func(p int) int {
		if !sp.Contractions || p >= n || rs[p] != '\'' || p+1 >= n {
			return p
		}
		c1 := ucFold(rs[p+1])
		if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
			return p + 2
		}
		if p+2 < n {
			c2 := ucFold(rs[p+2])
			if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
				return p + 3
			}
		}
		return p
	}
	// caseRun matches o200k's two letter alternatives at s, in order:
	//
	//	[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]* [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
	//	[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+ [\p{Ll}\p{Lm}\p{Lo}\p{M}]*
	//
	// The backtrack is required: Lm, Lo and M are in both classes, so a
	// greedy upper run can swallow what the first alternative's mandatory
	// lower run needs. Without it every CJK word takes the second alternative.
	upperAt := func(i int) bool { return i < n && ucUpperClass(rs[i]) && !isHan(i) }
	lowerAt := func(i int) bool { return i < n && ucLowerClass(rs[i]) && !isHan(i) }
	caseRun := func(s int) int {
		u := s
		for upperAt(u) {
			u++
		}
		for k := u; k >= s; k-- {
			if lowerAt(k) {
				e := k
				for lowerAt(e) {
					e++
				}
				return e
			}
		}
		if u > s {
			e := u
			for lowerAt(e) {
				e++
			}
			return e
		}
		return s
	}

	for pos := 0; pos < n; {
		// [\p{Han}]+ -- kimi-k2's first alternative, and first here for the same
		// reason: a greedy alternation tries it before the letter runs, and the
		// letter runs have had Han subtracted so they cannot take it back.
		if isHan(pos) {
			for isHan(pos) {
				pos++
			}
			add(pos)
			continue
		}
		// o200k: [^\r\n\p{L}\p{N}]? (case runs) (?i:'s|'t|...)?
		//
		// The optional leading character is tried first and given back if
		// no letters follow, as a greedy `?` does.
		if sp.CaseRuns {
			end := pos
			if rs[pos] != '\r' && rs[pos] != '\n' && !isLetter(pos) && !isNumber(pos) {
				if e := caseRun(pos + 1); e > pos+1 {
					end = e
				}
			}
			if end == pos {
				end = caseRun(pos)
			}
			if end > pos {
				if sp.SuffixContract {
					end = contractAt(end)
				}
				pos = end
				add(pos)
				continue
			}
		}
		// (?i:'s|'t|'re|'ve|'m|'ll|'d), as its own leading alternative. Under
		// SuffixContract it is not one: a lone "'t" with no letters before it is
		// punctuation there, not a contraction.
		if sp.Contractions && !sp.SuffixContract && rs[pos] == '\'' && pos+1 < n {
			c1 := ucFold(rs[pos+1])
			if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
				pos += 2
				add(pos)
				continue
			}
			if pos+2 < n {
				c2 := ucFold(rs[pos+2])
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					pos += 3
					add(pos)
					continue
				}
			}
		}
		// [^\r\n\p{L}\p{N}]?\p{L}+ -- an optional single non-letter, then letters.
		if !sp.CaseRuns && rs[pos] != '\r' && rs[pos] != '\n' && !isNumber(pos) {
			if isLetterX(pos) || isLetterX(pos+1) {
				pos++
				for isLetterX(pos) {
					pos++
				}
				add(pos)
				continue
			}
		}
		// llama3 is \p{N}{1,3} and qwen2 a bare \p{N}; see digitGroupFor.
		if isNumber(pos) {
			ini := pos
			for isNumber(pos) {
				pos++
				if pos-ini >= digitGroup {
					add(pos)
					ini = pos
				}
			}
			add(pos)
			continue
		}
		// ' ?[^\s\p{L}\p{N}]+[\r\n]*'
		probe := pos
		if rs[pos] == ' ' {
			probe = pos + 1
		}
		if plain(probe) {
			if rs[pos] == ' ' {
				pos++
			}
			for plain(pos) {
				pos++
			}
			for pos < n && !sp.PunctNoNewline && (rs[pos] == '\r' || rs[pos] == '\n' || (sp.PunctSlash && rs[pos] == '/')) {
				pos++
			}
			add(pos)
			continue
		}
		// The three whitespace alternatives, which differ only in where they stop.
		nws, lastNL := 0, 0
		for isSpace(pos + nws) {
			if rs[pos+nws] == '\r' || rs[pos+nws] == '\n' {
				lastNL = pos + nws + 1
			}
			nws++
		}
		switch {
		case lastNL > 0: // \s*[\r\n]+
			pos = lastNL
		case nws > 1 && pos+nws < n: // \s+(?!\S): give the last space back
			pos += nws - 1
		case nws > 0: // \s+
			pos += nws
		default:
			pos++
		}
		add(pos)
	}
}

// splitGPT2Each implements the original GPT-2 pattern, which llama.cpp gives to
// gpt-2, mpt, olmo and jais, handing each piece to emit as it is found:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Unlike splitLlama3 the contractions are case-sensitive, numbers run
// unbounded, and a newline is plain whitespace. Hand-rolled for the same
// negative lookahead.
func splitGPT2Each(text string, sc *runes, emit func(string)) {
	sc.load(text)
	rs := sc.rs
	n := len(rs)
	prev := 0
	add := func(end int) {
		if end > prev {
			emit(sc.piece(text, prev, end))
		}
		prev = end
	}
	isLetter := func(i int) bool { return i >= 0 && i < n && ucIsLetter(rs[i]) }
	isNumber := func(i int) bool { return i >= 0 && i < n && ucIsNumber(rs[i]) }
	isSpace := func(i int) bool { return i >= 0 && i < n && ucIsSpace(rs[i]) }
	plain := func(i int) bool { return i >= 0 && i < n && !isLetter(i) && !isNumber(i) && !isSpace(i) }

	for pos := 0; pos < n; {
		// 's|'t|'re|'ve|'m|'ll|'d -- no case folding, unlike llama3's.
		if rs[pos] == '\'' && pos+1 < n {
			c1 := rs[pos+1]
			if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
				pos += 2
				add(pos)
				continue
			}
			if pos+2 < n {
				c2 := rs[pos+2]
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					pos += 3
					add(pos)
					continue
				}
			}
		}
		probe := pos
		if rs[pos] == ' ' {
			probe = pos + 1
		}
		// ' ?\p{L}+', then ' ?\p{N}+', then ' ?[^\s\p{L}\p{N}]+'
		var cls func(int) bool
		switch {
		case isLetter(probe):
			cls = isLetter
		case isNumber(probe):
			cls = isNumber
		case plain(probe):
			cls = plain
		}
		if cls != nil {
			if rs[pos] == ' ' {
				pos++
			}
			for cls(pos) {
				pos++
			}
			add(pos)
			continue
		}
		nws := 0
		for isSpace(pos + nws) {
			nws++
		}
		switch {
		case nws > 1 && pos+nws < n: // \s+(?!\S): the last space starts the next word
			pos += nws - 1
		case nws > 0:
			pos += nws
		default:
			pos++
		}
		add(pos)
	}
}

// splitDigitsEach cuts every digit into its own piece, which is the "\p{N}"
// first pass the starcoder family runs before the gpt2 pattern, handing each
// piece to emit as it is found.
func splitDigitsEach(text string, sc *runes, emit func(string)) {
	sc.load(text)
	rs := sc.rs
	prev := 0
	for i, r := range rs {
		if !ucIsNumber(r) {
			continue
		}
		if i > prev {
			emit(sc.piece(text, prev, i))
		}
		emit(sc.piece(text, i, i+1))
		prev = i + 1
	}
	if prev < len(rs) {
		emit(sc.piece(text, prev, len(rs)))
	}
}

// runes is one pre-tokenizer stage's decoded view of its input: the code
// points, and the byte offset each starts at, so a piece is handed on as a
// slice of the input. Each stage owns one and reuses it.
//
// Invalid UTF-8 is rebuilt from the runes instead, turning each bad byte into
// U+FFFD as llama.cpp's splitters do; a slice would keep the raw byte.
type runes struct {
	rs    []rune
	off   []int32
	valid bool
}

func (r *runes) load(s string) {
	// Sized once from the input, since a byte is at most one rune.
	if cap(r.rs) < len(s) {
		r.rs, r.off = make([]rune, 0, len(s)), make([]int32, 0, len(s)+1)
	}
	r.rs, r.off = r.rs[:0], r.off[:0]
	for i, c := range s {
		r.rs = append(r.rs, c)
		r.off = append(r.off, int32(i))
	}
	r.off = append(r.off, int32(len(s)))
	r.valid = utf8.ValidString(s)
}

// piece is runes [a, b) of s, which load was given.
func (r *runes) piece(s string, a, b int) string {
	if r.valid {
		return s[r.off[a]:r.off[b]]
	}
	return string(r.rs[a:b])
}

// collect runs a streaming splitter and gathers what it emits: the slice
// signature the tests compare against.
func collect(each func(emit func(string))) []string {
	var out []string
	each(func(p string) { out = append(out, p) })
	return out
}

// opsFromContainer turns the container's stored pre-tokenizer stages into the
// pipeline's own op list. ok is false when the container carries none, which is
// how a caller tells "no stages" from "stages nobody recorded".
func opsFromContainer(ps []jlm.PreOp) ([]pretok.Op, bool) {
	if len(ps) == 0 {
		return nil, false
	}
	ops := make([]pretok.Op, 0, len(ps))
	for _, p := range ps {
		o := pretok.Op{
			UseRegex: p.UseRegex, AddPrefixSpace: p.AddPrefixSpace,
			Individual: p.Individual, Behavior: p.Behavior,
			Split: pretok.SplitParams{
				Contractions: p.Contractions, CaseFold: p.CaseFold,
				CaseRuns: p.CaseRuns, SuffixContract: p.SuffixContract,
				PunctSlash: p.PunctSlash, HanRuns: p.HanRuns, PunctNoNewline: p.PunctNoNewline,
				LetterMarks: p.LetterMarks, DigitGroup: int(p.DigitGroup),
			},
		}
		switch p.Kind {
		case jlm.PreSplit:
			o.Kind = pretok.OpSplit
		case jlm.PreByteLevel:
			o.Kind = pretok.OpByteLevel
		case jlm.PreDigits:
			o.Kind = pretok.OpDigits
		case jlm.PrePunctuation:
			o.Kind = pretok.OpPunctuation
		case jlm.PreDigitTriples:
			o.Kind = pretok.OpDigitTriples
		case jlm.PrePunctDefault:
			o.Kind = pretok.OpPunctDefault
		case jlm.PreDigitGroups:
			o.Kind = pretok.OpDigitGroups
		case jlm.PreKanaHanRuns:
			o.Kind = pretok.OpKanaHanRuns
		case jlm.PreHunyuanSplit:
			o.Kind = pretok.OpHunyuanSplit
		case jlm.PreSparkSplit:
			o.Kind = pretok.OpSparkSplit
		default:
			return nil, false
		}
		ops = append(ops, o)
	}
	return ops, true
}
