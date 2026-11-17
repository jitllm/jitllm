package tok

import (
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/jitllm/jitllm/tok/pretok"
)

// TestSplitPatternSemantics holds the engine to what Oniguruma, under the
// tokenizers library's Split, does with each construct: the behaviors on the
// library's own documented example ("the-final--countdown" on "-"), greedy
// backtracking, counted and lazy repetition, leftmost-first alternation,
// lookahead, case folding (U+017F folds to s), classes with ranges, negation,
// nesting and intersection, properties and scripts, escapes, and an empty
// match.
func TestSplitPatternSemantics(t *testing.T) {
	cases := []struct {
		pat, behavior string
		invert        bool
		text          string
		want          []string
	}{
		{"-", "Removed", false, "the-final--countdown", []string{"the", "final", "countdown"}},
		{"-", "Isolated", false, "the-final--countdown", []string{"the", "-", "final", "-", "-", "countdown"}},
		{"-", "MergedWithPrevious", false, "the-final--countdown", []string{"the-", "final-", "-", "countdown"}},
		{"-", "MergedWithNext", false, "the-final--countdown", []string{"the", "-final", "-", "-countdown"}},
		{"-", "Contiguous", false, "the-final--countdown", []string{"the", "-", "final", "--", "countdown"}},
		{"-", "Removed", true, "the-final--countdown", []string{"-", "-", "-"}},
		{`\p{L}+`, "Isolated", false, "ab12cd", []string{"ab", "12", "cd"}},
		{`a+ab`, "Isolated", false, "aaab", []string{"aaab"}},
		{`\p{N}{1,3}`, "Isolated", false, "1234567", []string{"123", "456", "7"}},
		{`a{2}`, "Isolated", false, "aaaaa", []string{"aa", "aa", "a"}},
		{`a+?`, "Isolated", false, "aaa", []string{"a", "a", "a"}},
		{`a|ab`, "Isolated", false, "ab", []string{"a", "b"}},
		{`ab|a`, "Isolated", false, "ab", []string{"ab"}},
		{`\s+(?!\S)|\s+`, "Isolated", false, "a   b", []string{"a", "  ", " ", "b"}},
		{`\s+(?!\S)|\s+`, "Isolated", false, "a   ", []string{"a", "   "}},
		{`x(?=y)`, "Isolated", false, "xyxzxy", []string{"x", "yxz", "x", "y"}},
		{`(?i:'s|'t)`, "Isolated", false, "it'S a'ſ't", []string{"it", "'S", " a", "'ſ", "'t"}},
		{`'(?i:[sdmt]|ll|ve|re)`, "Isolated", false, "I'LL we'Re", []string{"I", "'LL", " we", "'Re"}},
		{`(?i)ab|cd`, "Isolated", false, "ABxCD", []string{"AB", "x", "CD"}},
		{`[^\s\p{L}\p{N}]+`, "Isolated", false, "a+-b ?", []string{"a", "+-", "b ", "?"}},
		{`[a-c\d]+`, "Isolated", false, "abcd123", []string{"abc", "d", "123"}},
		{`[\p{L}&&[^\p{Han}]]+`, "Isolated", false, "ab中文cd", []string{"ab", "中文", "cd"}},
		{`[[0-9][a-b]]+`, "Isolated", false, "1a2bz", []string{"1a2b", "z"}},
		{`\p{Han}+`, "Isolated", false, "a中文b", []string{"a", "中文", "b"}},
		{`\P{L}+`, "Isolated", false, "ab, cd", []string{"ab", ", ", "cd"}},
		{`[\p{^L}]+`, "Isolated", false, "ab, cd", []string{"ab", ", ", "cd"}},
		{`\x{4e2d}|\u6587|\x41`, "Isolated", false, "A中文", []string{"A", "中", "文"}},
		{`.`, "Contiguous", false, "ab\ncd", []string{"ab", "\n", "cd"}},
		{`[\r\n]`, "Isolated", false, "a\r\nb", []string{"a", "\r", "\n", "b"}},
		{`\w+`, "Isolated", false, "a_b1 c", []string{"a_b1", " ", "c"}},
		{`\w+`, "Isolated", false, "a①b٣", []string{"a", "①", "b٣"}},
		{`b*`, "Isolated", false, "abba", []string{"a", "bb", "a"}},
		{`(?:ab)+`, "Isolated", false, "ababa", []string{"abab", "a"}},
		{`(?:a|)+b`, "Isolated", false, "aab", []string{"aab"}},
		{`\{|}`, "Isolated", false, "a{b}", []string{"a", "{", "b", "}"}},
		{`x{,2}y`, "Isolated", false, "xxxy", []string{"x", "xxy"}},
		{`\s+$`, "Isolated", false, "a  \nb  ", []string{"a", "  ", "\nb", "  "}},
		{`^a|b$`, "Isolated", false, "ab\nab", []string{"a", "b", "\n", "a", "b"}},
		{`\Aa|a\z`, "Isolated", false, "aaa", []string{"a", "a", "a"}},
		{`a?+a`, "Isolated", false, "aa", []string{"aa"}},
		{`a*+a`, "Isolated", false, "aaa", []string{"aaa"}},
		{`(?>a+)b|a`, "Isolated", false, "aab", []string{"aab"}},
		{`(?:ab)?+ab`, "Isolated", false, "abab", []string{"abab"}},
		{`\d{1,3}(?=(?:\d{3})*\b)`, "Isolated", false, "1234567 89", []string{"1", "234", "567", " ", "89"}},
		{`\Bb`, "Isolated", false, "ab b", []string{"a", "b", " b"}},
		{`\Gx`, "Isolated", false, "xzx", []string{"x", "zx"}},
	}
	for _, c := range cases {
		sp, err := compileSplit(c.pat, c.behavior, c.invert)
		if err != nil {
			t.Errorf("%q: %v", c.pat, err)
			continue
		}
		if got := sp.split(c.text); !slices.Equal(got, c.want) {
			t.Errorf("%q %s invert=%v on %q:\n  got  %q\n  want %q", c.pat, c.behavior, c.invert, c.text, got, c.want)
		}
	}
	for _, bad := range []string{`(?<=a)b`, `(?<!a)b`, `\Kx`, `[[:alpha:]]`, `(a`, `a)`, `\p{NoSuchThing}`, `*a`, `\1`} {
		if _, err := compileSplit(bad, "Isolated", false); err == nil {
			t.Errorf("%q compiled; it is a construct the engine refuses", bad)
		}
	}
	if _, err := compileSplit("a", "Splitted", false); err == nil {
		t.Error("an unknown behavior compiled")
	}
	// Nested counted repetitions multiply: 200,000 and 20,000,000 instructions.
	for _, big := range []string{`(?:(?:ab){1000}){100}`, `(?:(?:(?:ab){1000}){1000}){10}`} {
		_, err := compileSplit(big, "Isolated", false)
		if err == nil || !strings.Contains(err.Error(), big) {
			t.Errorf("%q: %v; want a refusal naming the pattern, before it is written out", big, err)
		}
	}
}

// TestSplitIsLinearOnDigitRuns counts the engine's steps (instructions run,
// code points scanned, frames popped) over 40,000-digit runs through the
// patterns that look ahead along a run, as their tokenizer.json files write
// them: cohere2moe's digit triples counted to a word boundary, afmoe's groups
// of 510 and of three, superbpe's empty split before each triple. A lookahead
// re-run from every start of the run would take steps quadratic in its length.
func TestSplitIsLinearOnDigitRuns(t *testing.T) {
	const n, perRune = 40000, 128
	digits := strings.Repeat("0123456789", n/10)
	texts := []string{digits, digits + " x", "a " + digits[1:], digits[:n-1] + "\n" + digits}
	for _, pat := range []string{
		`\d{1,3}(?=(?:\d{3})*\b)`,
		`\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))|\G\p{Nd}{510}`,
		`\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)`,
		`(?=(\d{3})+(?!\d))`,
	} {
		sp, err := compileSplit(pat, "Isolated", false)
		if err != nil {
			t.Fatalf("%q: %v", pat, err)
		}
		for _, text := range texts {
			count := utf8.RuneCountInString(text)
			var sc runes
			var pieces []string
			sp.each(text, &sc, func(s string) { pieces = append(pieces, s) })
			if strings.Join(pieces, "") != text {
				t.Errorf("%q: the pieces of a %d-rune run do not make it up", pat, count)
			}
			t.Logf("%.40q, %d runes: %d steps, %.1f a rune", pat, count, sc.m.steps, float64(sc.m.steps)/float64(count))
			if sc.m.steps > perRune*count {
				t.Errorf("%q: %d steps over %d runes, more than %d a rune", pat, sc.m.steps, count, perRune)
			}
		}
	}
	// Each one's split of the bare run, as the regex reads it: the first
	// piece, then every other piece one length.
	for _, c := range []struct {
		pat         string
		first, rest int
	}{
		{`\d{1,3}(?=(?:\d{3})*\b)`, n % 3, 3},
		{`\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))|\G\p{Nd}{510}`, n % 510, 510},
		{`\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)`, n % 3, n - n%3},
		{`(?=(\d{3})+(?!\d))`, n % 3, 3},
	} {
		sp, err := compileSplit(c.pat, "Isolated", false)
		if err != nil {
			t.Fatal(err)
		}
		got := sp.split(digits)
		ok := len(got) > 1 && len(got[0]) == c.first
		for _, p := range got[1:] {
			ok = ok && len(p) == c.rest
		}
		if !ok {
			t.Errorf("%q cut a %d-digit run into %d pieces, the first %d long; want %d, then %d each", c.pat, n, len(got), len(got[0]), c.first, c.rest)
		}
	}
}

// patternCorpus is the text the differential gates split: multilingual prose,
// code, every case of the tokenizer's split goldens and TestGoldens' texts,
// the o200k and Unicode-pin cases, edge cases of each
// construct the patterns use, seeded random strings over the classes they
// distinguish, and invalid UTF-8.
func patternCorpus(t testing.TB) []string {
	t.Helper()
	corpus := []string{
		"", " ", "  ", "\n", "\r\n", "\t", "a", "1",
		"The quick brown fox jumps over the lazy dog. It's 12:45 -- isn't it?",
		"Hello world\n\nNew paragraph   with   spaces\t\ttabs \r\nand CRLF.\n",
		"L'été dernier, nous sommes allés à Montréal; c'était très agréable !",
		"Straße, Maß, Fußgänger: Öl über Ähren. ſ ß ẞ",
		"Ελληνικά: ὁ λόγος, σοφίας; ΣΊΣΥΦΟΣ ς",
		"Русский текст: Съешь же ещё этих мягких французских булок, да выпей чаю.",
		"العربية: مرحبا بالعالم ١٢٣٤٥٦٧ ٣",
		"עברית: שָׁלוֹם עוֹלָם",
		"हिन्दी: नमस्ते दुनिया क्षत्रिय",
		"ภาษาไทย สวัสดีชาวโลก",
		"中文：你好，世界！我们在2024年见面。",
		"日本語：こんにちは、世界。カタカナとひらがなと漢字。ｶﾀｶﾅ",
		"한국어: 안녕하세요 세계",
		"Emoji 👋🏽 family 👨‍👩‍👧 flags 🇫🇷 keycap 1️⃣ ✔️",
		"Numbers: 3.14159 1,000,000 ⅫⅣ ①② ¾ ²³ 0x1F 1e-9",
		"Contractions: don't I'M we'Re they'VE she'll HE'D it's 'tis 'ſ o'clock",
		"Mixed caseRuns: HTTPServer getUserName iPhone XMLHttpRequest ABC'S Isn't",
		"Marks: e\u0301 a\u0308 n\u0303 \u0301alone \u0300\u0301\u0302 क\u094D",
		"Spaces: a\u00a0b\u3000c\u2028d\u2029e\u0085f\u200bg \u2003 \u202f",
		"Punct run: ):\n ...!!! ?!/ \"quoted\" ‘single’ «guillemets» — dash",
		"func main() {\n\tfmt.Println(\"hi\") // comment\n\tx := a[1:3]\n}\n",
		"def getUserName(self):\n    return 12345\n\n\nclass A:\n    pass",
		"{\"key\": [1, 2, 3], \"nested\": {\"a\": null, \"b\": true}}",
		"for i in $(seq 1 10); do echo \"$i\" | grep -E '^[0-9]+$' >/dev/null; done",
		"<html><body class=\"x\">&amp; &lt;tag&gt;</body></html>",
		"SELECT * FROM users WHERE id = 42 AND name LIKE '%bob%';",
		"   leading and trailing   ", "\n\n\nnewlines\n\n", "a\r\rb\n\rc",
		"中文English混合mixed日本語にほんご한국어",
		"123456789012345 1234 12 1",
		"\xff\xfe invalid \x80 bytes \xe4\xb8 truncated \xc3", "ok\xc3\xa9\xe2\x82",
		"a\u017Fb 'S 'Ll", "x/y/z ab/ /",
		// TestO200KSplitsByCaseAndSuffixesContractions' and the Unicode pins'.
		"helloWorld", "HTTPServer", "Isn't", "'tis", ".\n/x",
		"a\u1C89b", "a\u088Fb", "a\uA7CBb", "ab",
	}
	files := []string{"hunyuan_dense_split.json", "kimi_k2_split.json", "spark2_5_split.json", "gemma4_encode.json"}
	golden, err := filepath.Glob(filepath.Join("..", "testdata", "golden", "tokens", "*.json"))
	if err != nil || len(golden) == 0 {
		t.Fatalf("corpus: TestGoldens' texts: %v, %d files", err, len(golden))
	}
	for _, f := range files {
		golden = append(golden, filepath.Join("testdata", f))
	}
	for _, f := range golden {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("corpus: %v", err)
		}
		var cases []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(b, &cases); err != nil {
			t.Fatalf("corpus %s: %v", f, err)
		}
		for _, c := range cases {
			corpus = append(corpus, c.Text)
		}
	}
	pools := [][]rune{
		[]rune("abcxyzABCXYZ"), []rune("0123456789"), []rune(" \t\n\r\u00a0\u3000\u2028\u0085"),
		[]rune("'sStTmMdDlLvVrReE\u017F"), []rune(".,;:!?\"()[]{}<>/\\-_+=*&^%$#@~`|"),
		[]rune("éüßøñçĀǅǈΣςσЖж"), []rune("\u0300\u0301\u0308\u094D\u0E31\u302A"),
		[]rune("中文字漢語一丁丂々〆"), []rune("あいうアイウーゝヽ"), []rune("٣١۵߁०๑①Ⅻ¾²"),
		[]rune("€£¥©®™°±×÷→⇒∑√∞≠≤"), []rune("👋🏽😀🇫🇷\u200d\ufe0f"), []rune("\u0378\U000E0001\uFFFD\uE000"),
		[]rune("가나다라ㄱㅏ"), []rune("ابتثעבר"),
	}
	rng := rand.New(rand.NewPCG(407, 1))
	for range 3000 {
		var b strings.Builder
		for range 1 + rng.IntN(40) {
			pool := pools[rng.IntN(len(pools))]
			b.WriteRune(pool[rng.IntN(len(pool))])
		}
		corpus = append(corpus, b.String())
	}
	for range 300 {
		n := 1 + rng.IntN(24)
		raw := make([]byte, n)
		for i := range raw {
			if rng.IntN(3) == 0 {
				raw[i] = byte(0x80 + rng.IntN(0x80))
			} else {
				raw[i] = byte(0x20 + rng.IntN(0x5F))
			}
		}
		corpus = append(corpus, string(raw))
	}
	return corpus
}

// stagesOf is every distinct stage an older container carries -- each stage
// the generated table held before every Split became its own pattern, and
// the stated defaults -- with the variants the stages take (Punctuation's two
// behaviors, both digit forms, ByteLevel's own regex, the legacy families').
func stagesOf() []pretok.Op {
	var ops []pretok.Op
	add := func(op pretok.Op) {
		if !slices.ContainsFunc(ops, func(o pretok.Op) bool { return o == op }) {
			ops = append(ops, op)
		}
	}
	for _, op := range []pretok.Op{
		{Kind: pretok.OpByteLevel, UseRegex: false, AddPrefixSpace: false},
		{Kind: pretok.OpByteLevel, UseRegex: true, AddPrefixSpace: false},
		{Kind: pretok.OpByteLevel, UseRegex: true, AddPrefixSpace: true},
		{Kind: pretok.OpDigitGroups, Split: pretok.SplitParams{DigitGroup: 3}},
		{Kind: pretok.OpDigits, Individual: false},
		{Kind: pretok.OpDigits, Individual: true},
		{Kind: pretok.OpDigitTriples},
		{Kind: pretok.OpHunyuanSplit},
		{Kind: pretok.OpKanaHanRuns},
		{Kind: pretok.OpPunctuation, Behavior: "Contiguous"},
		{Kind: pretok.OpSparkSplit},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: false, CaseFold: false, LetterMarks: false, DigitGroup: 1, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: false, CaseFold: false, LetterMarks: false, DigitGroup: 1, CaseRuns: true, SuffixContract: false, PunctSlash: true, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: false, CaseFold: false, LetterMarks: false, DigitGroup: 3, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: false, CaseFold: false, LetterMarks: true, DigitGroup: 1, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: false, DigitGroup: 1, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: false, DigitGroup: 1, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false, PunctNoNewline: true}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: false, DigitGroup: 1, CaseRuns: true, SuffixContract: true, PunctSlash: true, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: false, DigitGroup: 3, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: false, DigitGroup: 3, CaseRuns: true, SuffixContract: true, PunctSlash: false, HanRuns: true}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: false, DigitGroup: 3, CaseRuns: true, SuffixContract: true, PunctSlash: true, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, LetterMarks: true, DigitGroup: 1, CaseRuns: false, SuffixContract: false, PunctSlash: false, HanRuns: false}},
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, DigitGroup: 3, CaseRuns: true, SuffixContract: true, HanRuns: true}},
	} {
		add(op)
	}
	for _, op := range pretok.DefaultBPE() {
		add(op)
	}
	// The legacy families' splits, for a container that stored no stages.
	add(pretok.Op{Kind: pretok.OpSplit, Split: llama3Params(3)})
	add(pretok.Op{Kind: pretok.OpSplit, Split: llama3Params(1)})
	add(pretok.Op{Kind: pretok.OpPunctuation, Behavior: "Isolated"})
	return ops
}

// adjudicated is the tokenizers library's own split of every corpus text a
// stage and the engine split differently (testdata/pattern_split.json, from
// scripts/patternsplitgold.py), keyed by the stage -- its pattern and
// behavior, or the dumped pre_tokenizer's file -- and the text as the stages
// read it.
func adjudicated(t *testing.T) map[string][]string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "pattern_split.json"))
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Pattern, Behavior, File, Text string
		Library                       []string
	}
	if err := json.Unmarshal(b, &entries); err != nil {
		t.Fatal(err)
	}
	out := make(map[string][]string, len(entries))
	for _, e := range entries {
		stage := e.File
		if stage == "" {
			stage = e.Pattern + "\x00" + e.Behavior
		}
		out[stage+"\x00"+e.Text] = norm(e.Library)
	}
	return out
}

// norm is pieces as the next stage reads them: an invalid byte is U+FFFD
// whichever stage cut it.
func norm(ps []string) []string {
	out := make([]string, len(ps))
	for i, p := range ps {
		out[i] = string([]rune(p))
	}
	return out
}

// judge compares the engine's split of text with a stage's. Where the two
// differ the tokenizers library has the last word: the text must be in the
// adjudicated golden, and the engine must split it as the library does. It
// reports whether the library had to be asked.
func judge(t *testing.T, lib map[string][]string, stage, text string, engine, splitter []string, diffs *[]patternDiff, d patternDiff) bool {
	t.Helper()
	got, want := norm(engine), norm(splitter)
	if slices.Equal(got, want) {
		return false
	}
	ref, ok := lib[stage+"\x00"+string([]rune(text))]
	switch {
	case !ok:
		d.Text, d.Engine, d.Stage = text, engine, splitter
		*diffs = append(*diffs, d)
		t.Errorf("%q on %q: the engine and the stage differ and the library has not been asked "+
			"(scripts/patternsplitgold.py):\n  engine %q\n  stage  %q", stage, text, got, want)
	case !slices.Equal(got, ref):
		t.Errorf("%q on %q: the engine is not the library's split:\n  engine  %q\n  library %q\n  stage   %q",
			stage, text, got, ref, want)
	}
	return true
}

// TestSplitPatternsMatchTheStages is the differential gate: over the corpus,
// the engine running the pattern each stage kind stands for (pretok.Op.AsPattern)
// splits exactly as that stage's own splitter does, for every stage in the
// generated table and the stated defaults -- or, where the two differ, as the
// tokenizers library does, which on every such text is the engine's split and
// not the splitter's.
func TestSplitPatternsMatchTheStages(t *testing.T) {
	corpus := patternCorpus(t)
	lib := adjudicated(t)
	compared, agreed, asked := 0, 0, 0
	var diffs []patternDiff
	for _, op := range stagesOf() {
		pat, behavior, ok := op.AsPattern()
		if !ok {
			continue
		}
		sp, err := compileSplit(pat, behavior, false)
		if err != nil {
			t.Errorf("%+v: %q does not compile: %v", op, pat, err)
			continue
		}
		var sc runes
		here := 0
		for _, text := range corpus {
			want := collect(func(emit func(string)) { applyOp(op, text, &sc, emit) })
			if judge(t, lib, pat+"\x00"+behavior, text, sp.split(text), want, &diffs, patternDiff{Pattern: pat, Behavior: behavior}) {
				here++
			} else {
				agreed++
			}
		}
		if here > 0 {
			t.Logf("%s %.60q: %d of %d texts split as the library does and not as the stage", op.Kind, pat, here, len(corpus))
		}
		asked += here
		compared++
	}
	dumpDiffs(t, diffs)
	if want := 20; compared < want {
		t.Errorf("compared %d stages, want at least %d: the gate ran over fewer stages than the table holds", compared, want)
	}
	t.Logf("%d stages over %d texts: %d splits agree, %d settled by the library", compared, len(corpus), agreed, asked)
}

// splitsOf reads a dumped pre_tokenizer's Split stages as they are written,
// pattern text and behavior, and its byte-level stage's regex as its own
// pattern.
func splitsOf(t *testing.T, b []byte) []*splitPattern {
	t.Helper()
	obj, err := pretok.PreTokenizerObject(b)
	if err != nil {
		t.Fatal(err)
	}
	var walk func(n map[string]any) []*splitPattern
	walk = func(n map[string]any) []*splitPattern {
		var out []*splitPattern
		switch n["type"] {
		case "Sequence":
			for _, v := range n["pretokenizers"].([]any) {
				out = append(out, walk(v.(map[string]any))...)
			}
		case "Split":
			pat := n["pattern"].(map[string]any)["Regex"].(string)
			inv, _ := n["invert"].(bool)
			sp, err := compileSplit(pat, n["behavior"].(string), inv)
			if err != nil {
				t.Fatalf("%q: %v", pat, err)
			}
			out = append(out, sp)
		case "ByteLevel":
			if v, ok := n["use_regex"].(bool); !ok || v {
				sp, err := compileSplit(pretok.GPT2Pattern, "Isolated", false)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, sp)
			}
		case "Digits":
			pat := pretok.DigitRunsPattern
			if v, _ := n["individual_digits"].(bool); v {
				pat = pretok.DigitsPattern
			}
			sp, err := compileSplit(pat, "Isolated", false)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, sp)
		case "Punctuation":
			beh, _ := n["behavior"].(string)
			if beh == "" {
				beh = "Isolated"
			}
			sp, err := compileSplit(pretok.PunctuationPattern, beh, false)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, sp)
		default:
			t.Fatalf("stage %v", n["type"])
		}
		return out
	}
	var root map[string]any
	if err := json.Unmarshal(obj, &root); err != nil {
		t.Fatal(err)
	}
	return walk(root)
}

// patternDiff is one text the engine and a stage split differently, kept so
// the tokenizers library can be asked which is right
// (scripts/patternsplitgold.py reads JITLLM_PATTERN_DIFFS).
type patternDiff struct {
	Pattern, Behavior string `json:",omitempty"`
	PreTokenizer      string `json:",omitempty"`
	File              string `json:",omitempty"`
	Text              string
	Engine, Stage     []string
}

func dumpDiffs(t *testing.T, diffs []patternDiff) {
	path := os.Getenv("JITLLM_PATTERN_DIFFS")
	if path == "" || len(diffs) == 0 {
		return
	}
	b, err := json.Marshal(diffs)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
}

func chainSplit(stages []*splitPattern, text string) []string {
	pieces := []string{text}
	for _, sp := range stages {
		var next []string
		for _, p := range pieces {
			next = append(next, sp.split(p)...)
		}
		pieces = next
	}
	return pieces
}

// TestEngineSplitsAsTheLibrary holds the engine to every split in the
// adjudicated golden directly, with no splitter beside it: each stage pattern
// and each dumped tokenizer.json's whole pre_tokenizer, run through the engine,
// splits each text as the tokenizers library did.
func TestEngineSplitsAsTheLibrary(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "pattern_split.json"))
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Pattern, Behavior, File, Text string
		Library                       []string
	}
	if err := json.Unmarshal(b, &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) < 1000 {
		t.Fatalf("the golden holds %d splits", len(entries))
	}
	compiled := map[string]*splitPattern{}
	files := map[string][]*splitPattern{}
	patterns, wholes := 0, 0
	for _, e := range entries {
		var got []string
		if e.File != "" {
			stages, ok := files[e.File]
			if !ok {
				raw, err := os.ReadFile(filepath.Join("testdata", "pre", e.File))
				if err != nil {
					t.Fatal(err)
				}
				stages = splitsOf(t, raw)
				files[e.File] = stages
			}
			got = chainSplit(stages, e.Text)
			wholes++
		} else {
			key := e.Pattern + "\x00" + e.Behavior
			sp, ok := compiled[key]
			if !ok {
				if sp, err = compileSplit(e.Pattern, e.Behavior, false); err != nil {
					t.Fatal(err)
				}
				compiled[key] = sp
			}
			got = sp.split(e.Text)
			patterns++
		}
		if !slices.Equal(norm(got), norm(e.Library)) {
			t.Errorf("%s%.40q on %q:\n  engine  %q\n  library %q", e.File, e.Pattern, e.Text, got, e.Library)
		}
	}
	if patterns == 0 || wholes == 0 {
		t.Errorf("%d stage splits and %d whole pre_tokenizers: the golden lost a half", patterns, wholes)
	}
	t.Logf("%d stage splits, %d whole pre_tokenizers, as the library splits them", patterns, wholes)
}

// TestOwnPatternsMatchTheTokenizersGoldens holds the engine to the tokenizers
// library itself: the pieces it split each golden's text into, under the
// model's own pre_tokenizer.
func TestOwnPatternsMatchTheTokenizersGoldens(t *testing.T) {
	for golden, pre := range map[string]string{
		"hunyuan_dense_split.json": "hunyuan-dense.json",
		"spark2_5_split.json":      "spark2_5.json",
	} {
		b, err := os.ReadFile(filepath.Join("testdata", "pre", pre))
		if err != nil {
			t.Fatal(err)
		}
		stages := splitsOf(t, b)
		g, err := os.ReadFile(filepath.Join("testdata", golden))
		if err != nil {
			t.Fatal(err)
		}
		var cases []struct {
			Text   string   `json:"text"`
			Pieces []string `json:"pieces"`
		}
		if err := json.Unmarshal(g, &cases); err != nil {
			t.Fatal(err)
		}
		if len(cases) < 20 {
			t.Fatalf("%s: %d cases", golden, len(cases))
		}
		for _, c := range cases {
			got := chainSplit(stages, c.Text)
			if len(got) == 0 && len(c.Pieces) == 0 {
				continue
			}
			if !slices.Equal(got, c.Pieces) {
				t.Errorf("%s %q:\n  got  %q\n  want %q", golden, c.Text, got, c.Pieces)
			}
		}
		t.Logf("%s: %d cases", golden, len(cases))
	}
}
