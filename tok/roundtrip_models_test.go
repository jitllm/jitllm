package tok

import (
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestEveryVocabularyGivesTheBytesBack: Decode(Encode(s)) is s, byte for byte,
// on one container of every vocabulary in the model directory, for valid UTF-8
// a prompt can carry -- joined emoji, combining marks, scripts, NUL, a BOM,
// U+10FFFF. A vocabulary that loses or rewrites a byte changes the prompt the
// model sees. WordPiece normalises by design and is left out.
//
// Two inputs are deliberately not here, because the reference does the same:
//   - invalid UTF-8: byte-level BPE reads it as U+FFFD, as llama.cpp's
//     unicode_cpts_from_utf8 does, and SentencePiece cuts symbols by
//     llama.cpp's utf8_len, so a malformed lead byte takes the next bytes with
//     it (a following space comes back as its raw bytes). The models' own
//     tokenizers take only valid text; HTTP requests arrive as valid JSON.
//   - a carriage return: HunyuanOCR's tokenizer.json drops it, so a CR does
//     not survive there either.
func TestEveryVocabularyGivesTheBytesBack(t *testing.T) {
	inputs := []string{
		"plain ascii, punctuation!? 1234567890",
		"👩‍👩‍👧‍👦 🏳️‍🌈 👍🏽 é ñ ​ zero width",
		"日本語 中文 한국어 العربية עברית ไทย देवनागरी",
		"line\nunix\n\ttab\x00nul\x7f",
		"  leading   and trailing  ",
		"\uFEFFbom � replacement \U0010FFFF max",
	}
	seen := map[string]bool{}
	checked := 0
	filepath.WalkDir(testmodels.Dir(), func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".jlm") {
			return nil
		}
		f, err := jlm.Open(p)
		if err != nil {
			return nil
		}
		vc := f.Vocab()
		f.Close()
		if vc == nil || vc.Tokens.Len() == 0 {
			return nil
		}
		var head []string
		for i := range min(vc.Tokens.Len(), 300) {
			head = append(head, vc.Tokens.At(i))
		}
		key := strings.Join(head, "\x00")
		if seen[key] {
			return nil
		}
		seen[key] = true
		v, err := New(vc)
		if err != nil || v.wpm != nil {
			return nil
		}
		checked++
		for _, in := range inputs {
			got := v.Decode(v.Encode(in, false))
			// SentencePiece's AddSpacePrefix puts a real space in front.
			if got != in && strings.TrimPrefix(got, " ") != in {
				t.Errorf("%s: %q came back as %q", filepath.Base(p), in, got)
			}
		}
		return nil
	})
	if checked == 0 {
		testmodels.Missing(t, "%s", "no container with a vocabulary under "+testmodels.Dir())
	}
	t.Logf("%d vocabularies, %d inputs each", checked, len(inputs))
}
