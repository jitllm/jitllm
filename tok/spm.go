// Package tok implements the SentencePiece tokenizer that llama-architecture
// GGUF models carry (tokenizer.ggml.model == "llama").
//
// It is a greedy highest-score bigram merge, not Viterbi and not byte-pair
// encoding:
//
//   - tokenizer.ggml.scores are merge priorities, not log probabilities: the
//     highest-scoring adjacent pair merges, repeatedly, until no adjacent pair
//     is in the vocabulary.
//   - tokenizer.ggml.merges is never read for this type (a sentencepiece model
//     has no merge table); BPE from that array gives almost-right output.
package tok

import (
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/pretok"
)

// spaceMark is U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's visible space.
const spaceMark = "▁"

// Token type codes from tokenizer.ggml.token_type.
const (
	typeNormal      = 1
	typeUnknown     = 2
	typeControl     = 3
	typeByte        = 6
	typeUserDefined = 4
	// typeUnused is a placeholder a converter invents to pad a vocabulary out
	// to the embedding's rows ("[PAD50954]", "<unused_256000>"). It decodes to
	// nothing: no model's text contains it.
	typeUnused = 5
)

// Vocab is a loaded SPM vocabulary.
type Vocab struct {
	// scratch holds the BPE encoders' working memory between calls.
	scratch sync.Pool
	ids     map[string]int32
	text    []string
	score   []float32
	kind    []int32

	BOS, EOS, Unk int32
	byteTok       [256]int32 // byte value -> <0xNN> token, or -1
	// spaceGlue is every character some token puts directly before a space
	// mark that is not its first character -- "▁" in the runs of spaces every
	// SPM vocabulary here carries, and ">" in gemma-3's ">▁</". Encode cuts the
	// text into words before a space mark whose preceding character is not in
	// it, since no merge can cross such a cut.
	spaceGlue map[string]bool

	// special is the ids of tokens that must be matched literally before
	// anything else, longest first so "<|im_start|>" beats "<". SPM and BPE
	// both partition on them; see indexSpecial.
	special []int32
	// specialAt is special bucketed by first byte, each bucket still longest
	// first, so a byte that starts no special costs one lookup.
	specialAt [256][]int32

	// eog is every id that ends a generation; see IsEOG.
	eog []int32
	// harmony is gpt-oss's message framing, looked up once; see DecodeChat.
	harmony harmony

	// pipe is the composed pre-tokenizer for this model's tokenizer.ggml.pre,
	// built once at load from the generated table. nil means no generated
	// pipeline covers this name, and the legacy family splitters run instead.
	pipe *pipeline

	// AddBOS and AddSpacePrefix default to true for SPM where the file does not
	// say: sentencepiece's add_dummy_prefix and a LlamaTokenizer's
	// add_bos_token.
	AddBOS         bool
	AddSpacePrefix bool

	// bpe is non-nil for a byte-level BPE vocabulary, in which case Encode and
	// Decode take that path instead. nil is SPM.
	bpe *bpeState
	// wpm is non-nil for a WordPiece vocabulary (the BERT family); see wpm.go.
	wpm *wpmState
	// spmBPE is non-nil for a SentencePiece-style BPE vocabulary (Gemma 4);
	// see bpespm.go. Decode is SPM's.
	spmBPE *spmBPE

	// AddEOS is the file's add_eos_token. Encode honours it for WordPiece only;
	// for SPM and BPE an EOS would end a generation prompt's turn, so an
	// embedding caller that wants it appends it (model.Model.EmbedText).
	AddEOS bool
}

// loadOpts is what the Options accumulate into, read once inside New.
type loadOpts struct {
	// tokenizerJSON is the caller's asserted tokenizer.json, or nil. It is
	// parsed inside New so a malformed file is New's error.
	tokenizerJSON io.Reader
	// namedStops is WithNamedStops.
	namedStops bool
}

// Option adjusts how a vocabulary is loaded.
type Option func(*loadOpts)

// WithTokenizer replaces the pre-tokenizer with the pipeline described by a
// HuggingFace tokenizer.json read from r. The GGUF still supplies the vocabulary
// and merges; only the pre-tokenizer comes from r.
//
// tokenizer.ggml.pre is a label a converter wrote, not proof of what the
// weights were trained against; the GGUF stays the default authority and this
// is the caller's explicit override (see AGENTS.md RULE 7m). The parsed
// pipeline wins over the stored stages and the legacy splitters; add_bos and
// ignore_merges still come from the container.
func WithTokenizer(r io.Reader) Option {
	return func(o *loadOpts) { o.tokenizerJSON = r }
}

// WithNamedStops adds to the stop set the end-of-turn pieces a vocabulary
// holds by name (namedStops), the stop set of a container written before it
// stated its own (jlm.Config.Generation nil): such a container stops where
// it did until it is converted again.
func WithNamedStops() Option {
	return func(o *loadOpts) { o.namedStops = true }
}

// New builds a vocabulary from a model's tokenizer metadata.
func New(vc *jlm.Vocab, options ...Option) (*Vocab, error) {
	var lo loadOpts
	for _, o := range options {
		o(&lo)
	}
	if vc == nil {
		return nil, fmt.Errorf("tok: the container has no tokenizer section")
	}
	switch vc.Kind {
	case jlm.VocabSPM, jlm.VocabBPE, jlm.VocabWPM, jlm.VocabBPESPM:
	default:
		return nil, fmt.Errorf("tok: vocabulary kind %v; only spm, bpe, bpe-spm and wpm are implemented", vc.Kind)
	}
	isBPE := vc.Kind == jlm.VocabBPE
	// A tokenizer.json on a non-BPE vocabulary is an error, not a no-op: SPM
	// has no pre-tokenizer stage, so the assertion would silently not apply.
	var override []pretok.Op
	if lo.tokenizerJSON != nil {
		if !isBPE {
			return nil, fmt.Errorf("tok: WithTokenizer: this vocabulary is %v; a "+
				"tokenizer.json pre-tokenizer applies to byte-level BPE only", vc.Kind)
		}
		var err error
		if override, err = pretok.ParsePreTokenizer(lo.tokenizerJSON); err != nil {
			return nil, fmt.Errorf("tok: WithTokenizer: %w", err)
		}
	}
	n := len(vc.Tokens)
	if n == 0 {
		return nil, fmt.Errorf("tok: the container's tokenizer has no tokens")
	}
	v := &Vocab{
		ids:            make(map[string]int32, n),
		text:           make([]string, n),
		AddBOS:         vc.AddBOS,
		AddEOS:         vc.AddEOS,
		AddSpacePrefix: vc.AddSpacePrefix,
	}
	for i, str := range vc.Tokens {
		v.text[i] = str
		// First id wins. Duplicate token strings exist in some vocabularies.
		if _, dup := v.ids[str]; !dup {
			v.ids[str] = int32(i)
		}
	}
	v.kind = make([]int32, n)
	for i, k := range vc.Kinds {
		v.kind[i] = int32(k)
	}
	if len(vc.Kinds) != 0 && len(vc.Kinds) != n {
		return nil, fmt.Errorf("tok: %d tokens but %d kinds", n, len(vc.Kinds))
	}
	// Scores are SPM's merge priorities. A BPE vocabulary either omits them or
	// ships zeros, and either way nothing reads them; nor does WordPiece.
	if vc.Kind == jlm.VocabSPM {
		v.score = vc.Scores
		if len(v.score) != n {
			return nil, fmt.Errorf("tok: %d tokens but %d scores", n, len(v.score))
		}
	}
	v.BOS, v.EOS, v.Unk = vc.BOS, vc.EOS, vc.Unk
	v.eog = eogOf(v, vc.Stop)
	if lo.namedStops {
		v.eog = namedStops(v, v.eog)
	}
	v.harmony = harmonyOf(v)
	if vc.Kind == jlm.VocabWPM {
		return v, newWPM(vc, v)
	}

	if isBPE {
		// BPE never inserts a leading space: the pre-tokenizer keeps the space
		// attached to the following word, which is what "Ġthe" is.
		v.AddSpacePrefix = false
		if err := newBPE(vc, v, override); err != nil {
			return nil, err
		}
		if v.preSplit("x") == nil {
			return nil, fmt.Errorf("tok: pre-tokenizer %q is not implemented", v.bpe.pre)
		}
		return v, nil
	}

	// Byte fallback: any byte that cannot be covered by a merge is emitted as
	// its own <0xNN> token (token_type 6), or Unk when the vocabulary lacks it.
	const hex = "0123456789ABCDEF"
	for b := 0; b < 256; b++ {
		id, ok := v.ids["<0x"+string(hex[b>>4])+string(hex[b&15])+">"]
		if !ok {
			v.byteTok[b] = -1
			continue
		}
		v.byteTok[b] = id
	}
	if vc.Kind == jlm.VocabBPESPM {
		v.indexSpecial()
		return v, newSPMBPE(vc, v)
	}
	// SPM partitions on its special tokens too, as the model's tokenizer matches
	// its added tokens, before the score merge: Gemma's turn markers are control tokens, and gemma3's
	// whitespace runs are user-defined tokens matched before any merge.
	v.indexSpecial()
	v.spaceGlue = map[string]bool{}
	for _, t := range v.text {
		// From byte 1, not len(spaceMark): a token need not open with a space
		// mark, and gemma-3's ">▁</" has its only one at byte 1.
		for k := 1; k < len(t); {
			i := strings.Index(t[k:], spaceMark)
			if i < 0 {
				break
			}
			at := k + i
			_, w := utf8.DecodeLastRuneInString(t[:at])
			v.spaceGlue[t[at-w:at]] = true
			k = at + len(spaceMark)
		}
	}
	return v, nil
}

// Size is the vocabulary size.
func (v *Vocab) Size() int { return len(v.text) }

// ID is the id whose token text is exactly this string, or false. It is not Encode: BPE-merging a marker's text can land on a
// different split than the vocabulary entry.
//
// Failing a raw lookup it compares against the decoded piece, which is what
// makes it work on a byte-level BPE vocabulary ("\n\n" is stored as "\u010a\u010a").
func (v *Vocab) ID(text string) (int32, bool) {
	if id, ok := v.ids[text]; ok {
		return id, true
	}
	for i := range v.text {
		if v.Decode([]int32{int32(i)}) == text {
			return int32(i), true
		}
	}
	return 0, false
}

// Text returns the raw token string for an id.
func (v *Vocab) Text(id int32) string {
	if id < 0 || int(id) >= len(v.text) {
		return ""
	}
	return v.text[id]
}

// utf8Len is the length a UTF-8 lead byte announces (its high nibble), which
// consumes a malformed sequence whole. utf8.DecodeRune is not a substitute: it
// resegments malformed input differently, which changes its tokens.
func utf8Len(c byte) int {
	return [16]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}[c>>4]
}

type symbol struct {
	off, n     int // slice of the working text
	prev, next int
}

type bigram struct {
	left, right int
	score       float32
	size        int
}

// bigramHeap pops the highest score first, breaking ties toward the leftmost
// pair. Both halves matter: sentencepiece's BPE agenda orders pairs
// (l.score < r.score) || (l.score == r.score && l.left > r.left).
type bigramHeap []bigram

// The heap is container/heap's sift-up and sift-down written out, avoiding its
// interface calls and boxing; the same operations in the same order, so the
// same pops.
func (a bigram) less(b bigram) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	return a.left < b.left
}

func (h *bigramHeap) push(x bigram) {
	q := append(*h, x)
	for i := len(q) - 1; i > 0; {
		p := (i - 1) / 2
		if !q[i].less(q[p]) {
			break
		}
		q[i], q[p] = q[p], q[i]
		i = p
	}
	*h = q
}

func (h *bigramHeap) pop() bigram {
	q := *h
	top := q[0]
	n := len(q) - 1
	q[0] = q[n]
	q = q[:n]
	for i := 0; ; {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && q[c+1].less(q[c]) {
			c++
		}
		if !q[c].less(q[i]) {
			break
		}
		q[i], q[c] = q[c], q[i]
		i = c
	}
	*h = q
	return top
}

// eogOf is the stop set: EOS and the ids the container states (jlm.Vocab.Stop:
// a safetensors model's generation_config.json eos_token_id list, a GGUF's
// eot and eom ids and the pieces its family's library statement lists).
func eogOf(v *Vocab, stated []int32) []int32 {
	var out []int32
	for _, id := range append([]int32{v.EOS}, stated...) {
		if id >= 0 && !slices.Contains(out, id) {
			out = append(out, id)
		}
	}
	return out
}

// namedStops is eog with the end-of-turn pieces v holds by name, and the
// fill-in-the-middle pad, repo and file-separator pieces. <|end|> closes a
// message in gpt-oss's harmony format, not the reply, so it is dropped where
// a vocabulary has both <|return|> and <|call|> (<|calls|> and <|flush|>
// likewise); </s> is not a stop on a vocabulary with gemma4's
// <|tool_response>.
func namedStops(v *Vocab, eog []int32) []int32 {
	has := func(n string) bool { _, ok := v.ids[n]; return ok }
	out := slices.Clone(eog)
	for _, n := range []string{"<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
		"<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>",
		"<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>",
		"<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[",
		"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
		"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
		"<|file_sep|>"} {
		if id, ok := v.ids[n]; ok && !slices.Contains(out, id) {
			out = append(out, id)
		}
	}
	drop := func(n string) {
		if id, ok := v.ids[n]; ok {
			out = slices.DeleteFunc(out, func(x int32) bool { return x == id })
		}
	}
	if (has("<|return|>") && has("<|call|>")) || (has("<|calls|>") && has("<|flush|>")) {
		drop("<|end|>")
	}
	if has("<|tool_response>") {
		drop("</s>")
	}
	return out
}

// IsEOG reports whether id ends a generation: EOS or a stop id the container
// states, or with WithNamedStops a piece namedStops names.
func (v *Vocab) IsEOG(id int32) bool { return slices.Contains(v.eog, id) }

// Encode tokenizes text. addSpecial controls the leading BOS (the tokenizers
// library's add_special_tokens).
//
// Specials are not parsed: a control token's text, "<s>" or "<|im_start|>",
// is characters, and a user-defined token is still matched. EncodeSpecial
// parses them.
func (v *Vocab) Encode(text string, addSpecial bool) []int32 {
	return v.encode(text, addSpecial, false)
}

// EncodeSpecial is Encode with the vocabulary's control tokens matched in the
// text as themselves, which a rendered chat template needs: the model's
// tokenizer matches its special added tokens so by default.
func (v *Vocab) EncodeSpecial(text string, addSpecial bool) []int32 {
	return v.encode(text, addSpecial, true)
}

func (v *Vocab) encode(text string, addSpecial, control bool) []int32 {
	if v.bpe != nil {
		return v.encodeBPE(text, addSpecial, control)
	}
	if v.spmBPE != nil {
		return v.encodeSPMBPE(text, addSpecial, control)
	}
	if v.wpm != nil {
		return v.encodeWPM(text, addSpecial, control)
	}
	out := make([]int32, 0, len(text)/3+2) // see encodeBPE
	if addSpecial && v.AddBOS {
		out = append(out, v.BOS)
	}
	// The space prefix is per text run after a special, exactly as at the
	// start of the prompt.
	prevSpecial := true
	for _, run := range v.partitionSpecial(text, control) {
		if run.id >= 0 {
			out = append(out, run.id)
			prevSpecial = true
			continue
		}
		out = v.encodeSPMRun(run.text, prevSpecial, out)
		prevSpecial = false
	}
	return out
}

// encodeSPMRun is the score merge over one run of text with no special token
// in it.
func (v *Vocab) encodeSPMRun(text string, prefix bool, out []int32) []int32 {
	if text == "" {
		return out
	}
	// A leading space is prepended before escaping, so the very first word
	// carries the same "▁" mark as every later one. This is why "Hello" encodes
	// as ' Hello' (15043) and not as 'Hello'.
	if v.AddSpacePrefix && prefix {
		text = " " + text
	}
	work := strings.ReplaceAll(text, " ", spaceMark)

	// One heap and one symbol buffer per word, not over the whole text, with
	// the same result as one over the whole text: no merge can cross a cut
	// placed before a space mark whose preceding character no token glues to a
	// space mark (spaceGlue), so each word's merge order is independent, and a
	// word-sized heap stays in L1.
	var syms []symbol
	h := &bigramHeap{}
	add := func(left, right int) {
		if left == -1 || right == -1 {
			return
		}
		s := work[syms[left].off : syms[right].off+syms[right].n]
		id, ok := v.ids[s]
		if !ok {
			return
		}
		h.push(bigram{left: left, right: right, score: v.score[id], size: len(s)})
	}
	for lo := 0; lo < len(work); {
		syms = syms[:0]
		prev, hi := lo, lo
		for hi < len(work) {
			n := min(utf8Len(work[hi]), len(work)-hi)
			if hi > lo && work[hi:hi+n] == spaceMark && !v.spaceGlue[work[prev:hi]] {
				break
			}
			syms = append(syms, symbol{off: hi, n: n, prev: len(syms) - 1, next: len(syms) + 1})
			prev, hi = hi, hi+n
		}
		syms[len(syms)-1].next = -1
		for i := 1; i < len(syms); i++ {
			add(i-1, i)
		}
		for len(*h) > 0 {
			b := h.pop()
			l, r := &syms[b.left], &syms[b.right]
			// Either side may already have been merged away, or grown, since
			// this pair was queued. Stale entries are skipped, not removed
			// eagerly.
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
		for i := 0; i != -1; i = syms[i].next {
			out = v.emitSymbol(work[syms[i].off:syms[i].off+syms[i].n], out)
		}
		lo = hi
	}
	return out
}

// emitSymbol emits one final symbol: its token, or its raw bytes.
//
// A merged symbol is always a token, so no merge has to be undone here; the byte fallback only sees a single character the
// vocabulary lacks.
func (v *Vocab) emitSymbol(s string, out []int32) []int32 {
	if id, ok := v.ids[s]; ok {
		return append(out, id)
	}
	for j := 0; j < len(s); j++ {
		if id := v.byteTok[s[j]]; id >= 0 {
			out = append(out, id)
		} else {
			out = append(out, v.Unk)
		}
	}
	return out
}

// Decode turns ids back into text, undoing the space escaping.
func (v *Vocab) Decode(ids []int32) string {
	if v.bpe != nil {
		return v.decodeBPE(ids)
	}
	if v.wpm != nil {
		return v.decodeWPM(ids)
	}
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(v.text) {
			continue
		}
		switch v.kind[id] {
		case typeControl, typeUnused:
			continue
		case typeByte:
			s := v.text[id]
			if len(s) == 6 && s[0] == '<' { // <0xNN>
				b.WriteByte(hexByte(s[3], s[4]))
				continue
			}
		}
		b.WriteString(strings.ReplaceAll(v.text[id], spaceMark, " "))
	}
	return b.String()
}

func hexByte(hi, lo byte) byte { return nibble(hi)<<4 | nibble(lo) }

func nibble(c byte) byte {
	switch {
	case c >= '0' && c <= '9':
		return c - '0'
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10
	}
	return 0
}

// Piece is the bytes token id adds to decoded text, false for a token that
// adds none: a control or unused token, or an id outside the vocabulary.
func (v *Vocab) Piece(id int32) (string, bool) {
	if id < 0 || int(id) >= len(v.text) {
		return "", false
	}
	if k := v.kind[id]; k == typeControl || k == typeUnused {
		return "", false
	}
	s := v.Decode([]int32{id})
	return s, s != ""
}

// Literal is token id's text as a chat template writes it: a control token's
// own text ([TOOL_CALLS], <|call|>), which Decode drops, and Decode's piece for
// every other. A tool call written in control tokens is read from it.
func (v *Vocab) Literal(id int32) string {
	if id < 0 || int(id) >= len(v.text) {
		return ""
	}
	if k := v.kind[id]; k == typeControl || k == typeUnused {
		return v.text[id]
	}
	one := [1]int32{id}
	return v.Decode(one[:])
}

// Harmony reports whether the vocabulary frames messages as gpt-oss's
// harmony does (DecodeChat).
func (v *Vocab) Harmony() bool { return v.harmony.ok }
