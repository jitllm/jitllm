package tok

import (
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/jitllm/jitllm/format/jlm"
)

// WordPiece: the BERT family's tokenizer, the one the bert and nomic-bert
// embedding models ship.
//
// The vocabulary marks the word-initial piece rather than the continuation:
// "ing" for BERT's "##ing" and "▁the" for "the", so a word is matched as
// "▁" + word, greedily, longest piece first. The segmentation is the same,
// since a piece at byte 0 can only be word-initial.
//
// It is the model's own WordPiece (its tokenizer.json): a word over
// maxWordChars characters is one [UNK].

// maxWordChars is WordPiece's max_input_chars_per_word: 100 in every census
// model's tokenizer.json (all-MiniLM-L6-v2, nomic-embed-text-v1.5,
// mxbai-embed-large-v1, snowflake-arctic-embed-xs) and the tokenizers
// default; a GGUF does not carry it.
const maxWordChars = 100

type wpmState struct {
	lower, strip bool
	maxLen       int // the longest token, in BYTES: the match window
	// special is every control token, longest first, so a literal "[SEP]" in
	// the text is that token and not five characters where the caller parses
	// specials, as the tokenizers library matches an added token in the text.
	special []int32
}

func newWPM(vc *jlm.Vocab, v *Vocab) error {
	w := &wpmState{lower: true, strip: true}
	for _, op := range vc.Pre {
		if op.Kind == jlm.PreBert {
			w.lower, w.strip = op.CaseFold, !op.LetterMarks
		}
	}
	for i, t := range v.text {
		if len(t) > w.maxLen {
			w.maxLen = len(t)
		}
		if v.kind[i] == typeControl && t != "" {
			w.special = append(w.special, int32(i))
		}
	}
	sort.SliceStable(w.special, func(a, b int) bool { return len(v.text[w.special[a]]) > len(v.text[w.special[b]]) })
	v.wpm = w
	return nil
}

// encodeWPM is [CLS] words [SEP], the post-processor a WordPiece model's
// tokenizer.json states (BertProcessing).
func (v *Vocab) encodeWPM(text string, addSpecial, control bool) []int32 {
	out := make([]int32, 0, len(text)/4+2)
	special := v.wpm.special
	if !control {
		special = nil
	}
	if addSpecial && v.AddBOS && v.BOS >= 0 {
		out = append(out, v.BOS)
	}
	last := 0
	for i := 0; i < len(text); {
		hit := int32(-1)
		for _, id := range special {
			if strings.HasPrefix(text[i:], v.text[id]) {
				hit = id
				break
			}
		}
		if hit < 0 {
			i++
			continue
		}
		out = v.wordPieces(text[last:i], out)
		out = append(out, hit)
		i += len(v.text[hit])
		last = i
	}
	out = v.wordPieces(text[last:], out)
	if addSpecial && v.AddEOS && v.EOS >= 0 {
		out = append(out, v.EOS)
	}
	return out
}

// wordPieces segments one run of ordinary text.
func (v *Vocab) wordPieces(text string, out []int32) []int32 {
	for _, word := range v.wpm.words(text) {
		if utf8.RuneCountInString(word) > maxWordChars {
			out = append(out, v.Unk)
			continue
		}
		w := spaceMark + word
		start := len(out)
		for i := 0; i < len(w); {
			hi := min(len(w), i+v.wpm.maxLen)
			j := hi
			for ; j > i; j-- {
				if id, ok := v.ids[w[i:j]]; ok {
					out = append(out, id)
					break
				}
			}
			if j == i { // no piece starts here: the whole word is unknown
				out = out[:start]
				break
			}
			i = j
		}
		if len(out) == start {
			out = append(out, v.Unk)
		}
	}
	return out
}

// words is BertNormalizer then BertPreTokenizer: control characters dropped,
// accents stripped and case folded as the vocabulary says, the text split on
// whitespace, and every punctuation mark, ASCII symbol and CJK ideograph made
// a word of its own.
func (w *wpmState) words(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		if w.strip {
			r = ucNFDBase(r)
		}
		fl := ucFlags(r)
		if ucIsSpace(r) {
			flush()
			continue
		}
		if r == 0 || r == utf8.RuneError || fl&ucFlagControl != 0 {
			continue
		}
		if w.strip && fl&ucFlagAccentMark != 0 {
			continue
		}
		if w.lower {
			r = ucToLower(r)
		}
		if fl&ucFlagPunctuation != 0 || (r < 0x7F && fl&ucFlagSymbol != 0) || isCJK(r) {
			flush()
			out = append(out, string(r))
			continue
		}
		cur.WriteRune(r)
	}
	flush()
	return out
}

// ucNFDBase is the first codepoint of r's canonical decomposition, or r.
func ucNFDBase(r rune) rune {
	u := uint32(r)
	if u < 0xC0 {
		return r
	}
	i := ucLast(ucNFD, 9, u)
	if i >= 0 && u <= ucWord(ucNFD, i*9+3) {
		return rune(ucWord(ucNFD, i*9+6))
	}
	return r
}

// isCJK is BERT's _is_chinese_char as the tokenizers library a tokenizer.json
// runs under writes it: its one range that differs from the Python
// BertTokenizer's starts at 0x2B920, where the Python's starts at 0x2B820.
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) || (r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B920 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}

// decodeWPM joins pieces back into text: a word-initial piece opens with a
// space, a continuation does not, and control tokens say nothing.
func (v *Vocab) decodeWPM(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || int(id) >= len(v.text) || v.kind[id] == typeControl {
			continue
		}
		b.WriteString(strings.ReplaceAll(v.text[id], spaceMark, " "))
	}
	return strings.TrimPrefix(b.String(), " ")
}
