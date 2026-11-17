// Package pretok is the pre-tokenizer model: the stages a tokenizer runs before
// merging, the parser that reads them out of a tokenizer.json, and the table
// generated from the artefacts themselves.
//
// It is its own package because the tokenizer runs these stages and the
// converter resolves them (so a container carries stages, not a name), and
// neither of those packages should import the other.
package pretok

type OpKind uint8

const (
	OpSplit OpKind = iota
	OpByteLevel
	OpDigits
	OpPunctuation
	// OpDigitTriples isolates every run of three ASCII digits, left to right:
	// HuggingFace's Split on "[0-9][0-9][0-9]" with behavior Isolated, which
	// falcon's pipeline ends with ("12345" -> "123", "45").
	//
	// Containers converted before every Split was stored as its own pattern
	// carry it; a conversion now stores that Split as an OpPattern.
	OpDigitTriples
	// OpPunctDefault isolates each contiguous run of \p{P} and the ASCII
	// symbols $+<=>^~| -- "[\p{P}\$\+<=>\^~\|]+", the first stage of
	// DefaultBPE. Unlike OpPunctuation (HuggingFace's
	// is_punc) it leaves the backtick alone. It is reached only through
	// DefaultBPE; no tokenizer.json produces it.
	OpPunctDefault
	// OpDigitGroups isolates \p{N} in groups of Split.DigitGroup, left to
	// right: HuggingFace's Split on "\p{N}{1,3}" with behavior Isolated, as a
	// stage of its own ("12345" -> "123", "45"; what lies between stays
	// whole). Hunyuan's pipeline opens with it. Read as the alternation
	// skeleton it became a whole llama3 split, which isolated every digit.
	OpDigitGroups
	// OpKanaHanRuns isolates runs of HuggingFace's
	// "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+": the CJK unified ideographs
	// to U+9FA5, hiragana and katakana. Hunyuan's second stage.
	OpKanaHanRuns
	// OpHunyuanSplit is Hunyuan's word split, a different alternation from
	// llama3's (no contractions and no digit alternative, an ASCII mark glued
	// to the ASCII letters after it, a prefix that may be a digit, and
	// [\p{P}\p{S}] where llama3 has [^\s\p{L}\p{N}]):
	//
	//	[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
	//	|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
	//	| ?[\p{P}\p{S}]+[\r\n]*
	//	|\s*[\r\n]+|\s+(?!\S)|\s+
	//
	// What no alternative matches is kept as a piece (Isolated).
	OpHunyuanSplit
	// OpSparkSplit is Spark-X2.5's variant of OpHunyuanSplit: the
	// punctuation run takes no newline tail, and a newline is a piece of one
	// character rather than the end of a whitespace run:
	//
	//	[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
	//	|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
	//	| ?[\p{P}\p{S}]+
	//	|[\r\n]|\s+(?!\S)|\s+
	OpSparkSplit
	// OpPattern is a Split as the model's tokenizer.json writes it: its
	// regex (Pattern), its behavior (Behavior, by the tokenizers library's
	// names) and whether it is inverted. Every Split a tokenizer.json holds
	// parses to one; the kinds above are what older containers carry, each
	// read as the pattern it stands for (Op.Pattern).
	OpPattern
)

type Op struct {
	Kind OpKind

	// Split
	Split SplitParams

	// ByteLevel: use_regex runs GPT-2's built-in pattern before byte encoding,
	// which is a different stage from an explicit Split even when the pattern
	// coincides.
	UseRegex       bool
	AddPrefixSpace bool

	// Digits. individual_digits=true isolates every digit; false groups runs.
	Individual bool

	// Punctuation behaviour, e.g. "Contiguous" or "Isolated"; a Pattern
	// stage's split behavior.
	Behavior string

	// Pattern: the regex, and whether the split is inverted.
	Pattern string
	Invert  bool
}

type SplitParams struct {
	// contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d). CaseFold is what the original
	// regex means; an expansion to '[sS]' cannot match U+017F.
	Contractions bool
	CaseFold     bool
	// LetterMarks is [\p{L}\p{M}]+ rather than \p{L}+ (qwen35).
	LetterMarks bool
	// DigitGroup is the \p{N} quantifier: 3 for \p{N}{1,3}, 1 for a bare \p{N}.
	DigitGroup int

	// The three below are o200k's shape, a different algorithm rather than
	// different parameters. Each is read off the pattern, not off a family
	// list (gpt-4o's pattern also contains `(?i:'s`).

	// CaseRuns splits a word by case instead of taking one \p{L}+ run:
	//
	//	[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+   then
	//	[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
	//
	// so "HTTPServer" is "HTTP" + "Server" where llama3 takes it whole.
	CaseRuns bool
	// SuffixContract puts the contraction at the end of a letter run as an
	// optional suffix, rather than as its own leading alternative: "Isn't" is
	// one piece under o200k and two under llama3, and a lone "'t" with no
	// letters before it is not a contraction at all.
	SuffixContract bool
	// PunctSlash is the punctuation run's trailing class: [\r\n/]* rather than
	// [\r\n]*, so a slash after punctuation joins the same piece.
	PunctSlash bool
	// PunctNoNewline is a punctuation run with no newline tail at all,
	// ` ?[^\s\p{L}\p{N}\r\n]+` (seed-coder, Seed-OSS): "):\n" is "):" and "\n",
	// where llama3's [\r\n]* keeps them one piece.
	PunctNoNewline bool

	// HanRuns is kimi-k2's one addition to o200k: two changes that always
	// travel together,
	//
	//	[\p{Han}]+                          a leading alternative of its own
	//	[\p{Lu}...\p{M}&&[^\p{Han}]]        Han subtracted from every letter class
	//
	// so CJK never merges with the letters beside it. One flag, because
	// neither half is useful alone.
	HanRuns bool
}

// DefaultBPE is the pipeline a BPE vocabulary takes when its GGUF names no
// pre-tokenizer, or names "default":
//
//	"[\p{P}\$\+<=>\^~\|]+"                                   -> OpPunctDefault
//	"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
//	                                                           -> ByteLevel's GPT-2 regex
//	"\p{N}+"                                                  -> Digits, grouped
//	"[0-9][0-9][0-9]"                                          -> OpDigitTriples
//
// with add_bos and ignore_merges false unless the file says otherwise. It is
// not llama3's split, which only the names that state it take.
//
// A file with no pre name carries no claim about its tokenizer; the model's own
// tokenizer.json (tok.WithTokenizer) outranks this.
func DefaultBPE() []Op {
	return []Op{
		{Kind: OpPunctDefault},
		{Kind: OpByteLevel, UseRegex: true},
		{Kind: OpDigits, Individual: false},
		{Kind: OpDigitTriples},
	}
}

// PreOps returns the ordered stages a pre-tokenizer name denotes, from the
// table generated out of each model's own tokenizer.json. It is for the
// converter: a container stores the stages, so a name the table lacks is a
// question at conversion rather than a refusal at load.
func PreOps(name string) ([]Op, bool) {
	ops, ok := Table()[name]
	return ops, ok
}

func (k OpKind) String() string {
	switch k {
	case OpSplit:
		return "Split"
	case OpByteLevel:
		return "ByteLevel"
	case OpDigits:
		return "Digits"
	case OpPunctuation:
		return "Punctuation"
	case OpDigitTriples:
		return "DigitTriples"
	case OpPunctDefault:
		return "PunctDefault"
	case OpDigitGroups:
		return "DigitGroups"
	case OpKanaHanRuns:
		return "KanaHanRuns"
	case OpHunyuanSplit:
		return "HunyuanSplit"
	case OpSparkSplit:
		return "SparkSplit"
	case OpPattern:
		return "Pattern"
	}
	return "?"
}
