package jlm

// The container's tokenizer section. It carries the parsed pre-tokenizer ops,
// not a name for one: the source stores a family label that every engine maps
// through a hand-maintained table. Resolving it once at conversion (from the
// name, or better the model's own tokenizer.json) means an unknown name is a
// conversion-time question, IgnoreMerges and ByteFallback are stored rather
// than inferred, and PreName (the label the source claimed) sits beside Pre
// (what runs) with nothing resolving one from the other at load.

// VocabKind is the tokenization algorithm.
type VocabKind uint8

const (
	VocabNone VocabKind = 0
	VocabSPM  VocabKind = 1 // score-driven longest-match, byte tokens for the rest
	VocabBPE  VocabKind = 2 // byte-level, merge-rank driven
	VocabWPM  VocabKind = 3 // WordPiece
	// VocabBPESPM is SentencePiece-style BPE (Gemma 4's "gemma4" tokenizer):
	// spaces become U+2581, then merges by rank run over the UTF-8 characters
	// of the whole run, with no pre-tokenizer split and no byte-level
	// alphabet, and a character the vocabulary lacks becomes its bytes'
	// <0xNN> tokens. A reader that predates it refuses the kind.
	VocabBPESPM VocabKind = 4
)

func (k VocabKind) String() string {
	switch k {
	case VocabSPM:
		return "spm"
	case VocabBPE:
		return "bpe"
	case VocabWPM:
		return "wpm"
	case VocabBPESPM:
		return "bpe-spm"
	}
	return "vocab(" + itoa(uint64(k)) + ")"
}

// TokenKind classifies one vocabulary entry.
type TokenKind uint8

const (
	TokenNormal      TokenKind = 1
	TokenUnknown     TokenKind = 2
	TokenControl     TokenKind = 3 // <s>, <|im_start|>: matched literally, never merged
	TokenUserDefined TokenKind = 4
	TokenUnused      TokenKind = 5
	TokenByte        TokenKind = 6 // <0xNN>
)

// 5 and 6 are the source's order, which convert/vocab.go copies verbatim
// from a GGUF's token_type and tok reads 6 as a byte token.

// PreOpKind is one stage of the pre-tokenizer pipeline.
type PreOpKind uint8

const (
	PreSplit       PreOpKind = 1 // the contraction/letter/digit/other regex family
	PreByteLevel   PreOpKind = 2 // GPT-2 byte encoding, optionally with its own regex
	PreDigits      PreOpKind = 3
	PrePunctuation PreOpKind = 4
	// PreBert is BERT's normaliser and pre-tokenizer as one stage, which is
	// what a WordPiece vocabulary runs before its longest-match: control
	// characters dropped, text split on whitespace, and every punctuation
	// mark, ASCII symbol and CJK ideograph made a word of its own.
	//
	// Two of PreOp's fields carry its options:
	//
	//	CaseFold     lowercase (BertNormalizer.lowercase)
	//	LetterMarks  keep combining marks; false strips accents, which is
	//	             BertNormalizer.strip_accents (NFD, then drop \p{Mn})
	//
	// LetterMarks reads as it does for PreSplit ("marks are part of the
	// letters"). Reusing the two fields keeps this a new code, not a new
	// record layout.
	PreBert PreOpKind = 5
	// PreDigitTriples isolates each run of three ASCII digits (HuggingFace's
	// Split on "[0-9][0-9][0-9]", Isolated), falcon's last stage. See
	// pretok.OpDigitTriples. No parameters, so a new code, not a new record
	// layout; an older reader falls back to its own table row for the name.
	PreDigitTriples PreOpKind = 6
	// PrePunctDefault isolates runs of llama.cpp's "[\p{P}\$\+<=>\^~\|]",
	// the first stage of the pipeline a vocabulary with no pre name gets
	// (pretok.LlamaCppDefault). No parameters; a new code, as above.
	PrePunctDefault PreOpKind = 7
	// PreDigitGroups isolates \p{N} in groups of DigitGroup (HuggingFace's
	// Split on "\p{N}{1,3}", Isolated); PreKanaHanRuns isolates runs of CJK
	// ideographs, hiragana and katakana; PreHunyuanSplit is Hunyuan's word
	// split. See the pretok ops of the same names. Hunyuan's pipeline is the
	// three then ByteLevel; a reader that predates them refuses the arch.
	PreDigitGroups  PreOpKind = 8
	PreKanaHanRuns  PreOpKind = 9
	PreHunyuanSplit PreOpKind = 10
	// PreSparkSplit is Spark-X2.5's variant of it (pretok.OpSparkSplit); a
	// reader that predates it refuses the stage.
	PreSparkSplit PreOpKind = 11
)

// PreOp is one stage, with every parameter the stage takes. A reader runs these
// in order; there is nothing else to know.
type PreOp struct {
	Kind PreOpKind

	// PreSplit.
	//
	// CaseFold is a real divergence and is stored: artefacts write the
	// contraction alternation (?i:'s|'t|...), and re-implementations expand it
	// to '[sS]', which misses U+017F LONG S under case folding. Which one a
	// container runs is decided at conversion.
	Contractions bool
	CaseFold     bool
	LetterMarks  bool  // [\p{L}\p{M}]+ rather than \p{L}+
	DigitGroup   uint8 // the \p{N} quantifier: 3 for {1,3}, 1 for a bare \p{N}

	// o200k's shape, a different algorithm rather than a parameter of the one
	// above; see pretok.SplitParams. Stored for the same reason as CaseFold.
	CaseRuns       bool // letter runs split by CASE, two alternatives
	SuffixContract bool // the contraction is a suffix of a letter run, not a lone alternative
	PunctSlash     bool // the punctuation run's tail is [\r\n/]* rather than [\r\n]*
	HanRuns        bool // Han is its own run and is subtracted from the letter classes
	// PunctNoNewline: the punctuation run has no [\r\n]* tail (seed-coder).
	// It travels in the vocab record's optional tail (encodeVocab), so it
	// cost no container version; a reader that predates it splits such a
	// vocabulary as llama3's, as every reader did before it existed.
	PunctNoNewline bool

	// PreByteLevel.
	UseRegex       bool
	AddPrefixSpace bool

	// PreDigits.
	Individual bool

	// PrePunctuation.
	Behavior string
}

// Merge is one BPE merge rule, as an explicit pair: the source's joined
// "left right" string is ambiguous once either side contains a space.
type Merge struct{ Left, Right string }

// Merges is a vocabulary's merge rules in order, as one string table.
type Merges struct{ t Strings }

// NewMerges is a table of m.
func NewMerges(m []Merge) Merges {
	s := make([]string, 0, 2*len(m))
	for _, x := range m {
		s = append(s, x.Left, x.Right)
	}
	return Merges{t: NewStrings(s)}
}

// Len is the number of merges.
func (m Merges) Len() int { return m.t.Len() / 2 }

// At is merge i.
func (m Merges) At(i int) Merge { return Merge{Left: m.t.At(2 * i), Right: m.t.At(2*i + 1)} }

// ChatTemplate is one named template. A model may ship several -- a default and
// a tool-calling variant, say -- where the source format has room for exactly
// one and unnamed.
type ChatTemplate struct{ Name, Body string }

// Vocab is the container's tokenizer.
type Vocab struct {
	Kind   VocabKind
	Tokens Strings
	Scores []float32    // SPM; empty for BPE
	Kinds  []TokenKind  // parallel to Tokens
	Merges Merges       // BPE
	Added  []AddedToken // tokens added after training, with their own handling

	// Special ids, -1 when the model has none. Pad, Sep and Mask are carried
	// because an embedding or classification caller needs them.
	BOS, EOS, Unk, Pad, Sep, Mask int32
	// Stop is every id besides EOS that the source states ends a generation:
	// a GGUF's tokenizer.ggml.eot/eom and FIM pad/repo/sep ids, a safetensors
	// model's generation_config eos_token_id list. The tokenizer adds the ids
	// llama.cpp recognises by name (tok.Vocab.IsEOG). Absent from a container
	// written before it, which then stops on EOS and the names alone.
	Stop []int32

	AddBOS, AddEOS bool
	AddSpacePrefix bool
	ByteFallback   bool // an unmatched byte becomes its <0xNN> token
	// IgnoreMerges emits a pre-token that is already a vocabulary entry whole,
	// instead of splitting it to runes and merging back up. Stored, not
	// inferred from a family name, since the two do not always agree.
	IgnoreMerges bool

	Pre []PreOp // the pipeline, in order
	// PreName is the label the source claimed, kept for provenance and
	// diagnostics. Nothing resolves a pipeline from it.
	PreName string

	Templates []ChatTemplate
}

// AddedToken is a token introduced after training, which the pre-tokenizer must
// match literally and whose surrounding whitespace handling is its own.
type AddedToken struct {
	ID         int32
	Content    string
	Special    bool
	LStrip     bool
	RStrip     bool
	Normalized bool
	SingleWord bool
}

// Template returns the named chat template. "" is the model's default: an
// unnamed one, else the one named "default" (transformers' choice from a named
// list), else the first.
func (v *Vocab) Template(name string) (string, bool) {
	for i := range v.Templates {
		if v.Templates[i].Name == name {
			return v.Templates[i].Body, true
		}
	}
	if name != "" {
		return "", false
	}
	for i := range v.Templates {
		if v.Templates[i].Name == "default" {
			return v.Templates[i].Body, true
		}
	}
	if len(v.Templates) > 0 {
		return v.Templates[0].Body, true
	}
	return "", false
}
