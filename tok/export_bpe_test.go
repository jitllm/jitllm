package tok

import "strings"

// Representable is text with every character a byte-level vocabulary cannot
// spell removed. A byte with no token in the alphabet is dropped by the
// model's own tokenizer as well -- Hunyuan's vocabulary has no "\r", and
// HuggingFace tokenizers encodes "\r\n" as "\n" -- so no round trip keeps it.
// A vocabulary that is not byte-level returns text unchanged.
func Representable(v *Vocab, text string) string {
	if v.bpe == nil {
		return text
	}
	var b strings.Builder
	for _, r := range text {
		ok := true
		for _, c := range []byte(string(r)) {
			if _, has := v.ids.find(string(byteToRune[c])); !has {
				ok = false
			}
		}
		if ok {
			b.WriteRune(r)
		}
	}
	return b.String()
}
