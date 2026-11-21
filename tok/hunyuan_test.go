package tok

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok/pretok"
)

// TestHunyuanSplitsAsItsTokenizerDoes runs the hunyuan-dense pre-tokenizer
// against HuggingFace tokenizers running Hunyuan's own tokenizer.json
// (scripts/hunyuansplitgold.py). Its three Split stages are each a whole
// pattern, run as written; read as an alternation skeleton, the row was once
// three llama3 splits and "12345" came out as five digits where the model's
// tokenizer has "123" "45".
//
// The corpus carries what each stage is about: digit runs longer than three,
// CJK and kana beside Latin and digits, an ASCII mark glued to the letters
// after it, combining marks, and every kind of whitespace run.
func TestHunyuanSplitsAsItsTokenizerDoes(t *testing.T) {
	splitsAsItsTokenizerDoes(t, "hunyuan-dense", "hunyuan_dense_split.json",
		pretok.OpPattern, pretok.OpPattern, pretok.OpPattern, pretok.OpByteLevel)
}

// TestSparkSplitsAsItsTokenizerDoes is the same for Spark-X2.5, whose third
// Split is a variant (pretok.SparkWords: no newline tail on a punctuation
// run, a newline a piece of its own) followed by individual digits. Read as
// three llama3 splits it would be wrong; the corpus's newline cases are where
// the variant parts from Hunyuan's.
func TestSparkSplitsAsItsTokenizerDoes(t *testing.T) {
	splitsAsItsTokenizerDoes(t, "spark2_5", "spark2_5_split.json",
		pretok.OpPattern, pretok.OpPattern, pretok.OpPattern, pretok.OpDigits, pretok.OpByteLevel)
}

// splitsAsItsTokenizerDoes holds pre-tokenizer name's table row to the
// stage kinds want and to the pieces HuggingFace tokenizers made of the
// golden corpus (scripts/hunyuansplitgold.py).
func splitsAsItsTokenizerDoes(t *testing.T, name, goldenFile string, want ...pretok.OpKind) {
	t.Helper()
	ops, ok := pretok.PreOps(name)
	if !ok {
		t.Fatalf("pretok has no %s entry", name)
	}
	var got []pretok.OpKind
	for _, o := range ops {
		got = append(got, o.Kind)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s's stages are %v, want %v", name, got, want)
	}
	pipe, err := buildPipeline(ops)
	if err != nil {
		t.Fatalf("%s does not compose: %v", name, err)
	}
	b, err := os.ReadFile(filepath.Join("testdata", goldenFile))
	if err != nil {
		t.Fatalf("golden: %v (RULE 11: regenerate it, do not skip)", err)
	}
	var golden []struct {
		Text   string   `json:"text"`
		Pieces []string `json:"pieces"`
	}
	if err := json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden) < 20 {
		t.Fatalf("golden has %d case(s) -- too few to prove anything", len(golden))
	}
	groups, kana := 0, 0
	for _, c := range golden {
		g := pipe.apply(c.Text)
		if len(g) == 0 && len(c.Pieces) == 0 {
			continue
		}
		if !reflect.DeepEqual(g, c.Pieces) {
			t.Errorf("%q\n  got  %q\n  want %q", c.Text, g, c.Pieces)
		}
		if strings.Contains(c.Text, "12345") {
			groups++
		}
		for _, r := range c.Text {
			if isKanaHan(r) {
				kana++
				break
			}
		}
	}
	if groups == 0 || kana < 3 {
		t.Errorf("the corpus has %d digit run(s) over three and %d case(s) with CJK or kana", groups, kana)
	}
	t.Logf("%d case(s) against tokenizers", len(golden))
}
