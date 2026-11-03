package pretok

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Reading a pre-tokenizer out of a model's own tokenizer.json, at runtime. It
// is the same parse scripts/gentok runs for the generated table, and it is
// what tok.WithTokenizer uses when the caller asserts the model's own file
// over the GGUF's tokenizer.ggml.pre label (AGENTS.md RULE 7m).
//
// Nothing here fetches: downloading is scripts/gentok's job.

// ParsePreTokenizer reads the ordered pre-tokenizer stages out of a
// HuggingFace tokenizer.json.
//
// r may hold only a prefix of the file: pre_tokenizer sits near the top and the
// megabytes below it are vocabulary the GGUF already carries, so a ranged fetch
// of the first few hundred KB is enough. A prefix that stops inside the object
// is an error rather than a partial answer.
//
// An op this package cannot build is refused, naming the op, rather than routed
// to the nearest-looking splitter.
func ParsePreTokenizer(r io.Reader) ([]Op, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("tok: reading tokenizer.json: %w", err)
	}
	obj, err := PreTokenizerObject(b)
	if err != nil {
		return nil, err
	}
	var node any
	if err := json.Unmarshal(obj, &node); err != nil {
		return nil, fmt.Errorf("tok: pre_tokenizer does not parse: %w", err)
	}
	ops := flattenPreTokenizer(node)
	if len(ops) == 0 {
		return nil, fmt.Errorf("tok: pre_tokenizer has no recognised stages")
	}
	// Validate here so the refusal lands at the parse, not at the first
	// Encode. It fires on a digit-triple Split in a form other than Isolated
	// (TestTheDigitTripleSplitIsReadOnlyIsolated).
	if err := Validate(ops); err != nil {
		return nil, err
	}
	return ops, nil
}

// PreTokenizerObject returns the complete pre_tokenizer object out of a
// tokenizer.json (or a prefix of one) by balancing braces, never by scanning
// for a field: on a model with no Split a scan runs into model.vocab and
// returns a vocabulary token as a pattern. It is exported for scripts/gentok.
func PreTokenizerObject(b []byte) ([]byte, error) {
	s := string(b)
	i := strings.Index(s, `"pre_tokenizer"`)
	if i < 0 {
		return nil, fmt.Errorf(`tok: no "pre_tokenizer" in the first %d bytes`, len(b))
	}
	// A null pre_tokenizer is a value (sentencepiece-derived files carry it);
	// scanning for the next "{" would parse some other object. The value is
	// inspected and an explicit null reported as itself.
	v := strings.TrimLeft(s[i+len(`"pre_tokenizer"`):], " \t\r\n")
	v = strings.TrimPrefix(v, ":")
	v = strings.TrimLeft(v, " \t\r\n")
	if strings.HasPrefix(v, "null") {
		return nil, fmt.Errorf("tok: pre_tokenizer is null -- this tokenizer.json " +
			"pre-tokenizes nothing, which is what a sentencepiece-derived BPE " +
			"looks like; it needs the SPM path, not a pre-tokenizer pipeline")
	}
	if !strings.HasPrefix(v, "{") {
		return nil, fmt.Errorf("tok: pre_tokenizer is not an object")
	}
	start := i + len(s[i:]) - len(v)
	depth, inStr, esc := 0, false, false
	for k := start; k < len(s); k++ {
		c := s[k]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			if depth--; depth == 0 {
				return []byte(s[start : k+1]), nil
			}
		}
	}
	return nil, fmt.Errorf("tok: pre_tokenizer object truncated at %d bytes", len(b))
}

// flattenPreTokenizer walks a Sequence into an ordered Op list. Order is
// semantic: Digits after ByteLevel operates on byte-encoded symbols, which is
// not the same op as Digits before it.
func flattenPreTokenizer(node any) []Op {
	n, ok := node.(map[string]any)
	if !ok {
		return nil
	}
	t, _ := n["type"].(string)
	switch t {
	case "Sequence":
		var out []Op
		if lst, ok := n["pretokenizers"].([]any); ok {
			for _, v := range lst {
				out = append(out, flattenPreTokenizer(v)...)
			}
		}
		return out
	case "Split":
		pat := ""
		if p, ok := n["pattern"].(map[string]any); ok {
			if r, ok := p["Regex"].(string); ok {
				pat = r
			}
		}
		if pat == "" {
			return nil // a Split we cannot read is not a Split we may guess at
		}
		// The digit-triple Split is its own stage, and only in the form the
		// artefacts carry it: isolated, not inverted.
		if pat == "[0-9][0-9][0-9]" {
			if n["behavior"] == "Isolated" && n["invert"] != true {
				return []Op{{Kind: OpDigitTriples}}
			}
			return []Op{{Kind: opUnreadable}}
		}
		// The whole-pattern stages, recognised by their exact text: each is
		// not the alternation skeleton parseSplit reads parameters off.
		if op, ok := isolatedSplit(pat); ok {
			if n["behavior"] == "Isolated" && n["invert"] != true {
				return []Op{op}
			}
			return []Op{{Kind: opUnreadable}}
		}
		return []Op{{Kind: OpSplit, Split: parseSplit(pat)}}
	case "ByteLevel":
		op := Op{Kind: OpByteLevel, UseRegex: true}
		if v, ok := n["use_regex"].(bool); ok {
			op.UseRegex = v
		}
		if v, ok := n["add_prefix_space"].(bool); ok {
			op.AddPrefixSpace = v
		}
		return []Op{op}
	case "Digits":
		op := Op{Kind: OpDigits}
		if v, ok := n["individual_digits"].(bool); ok {
			op.Individual = v
		}
		return []Op{op}
	case "Punctuation":
		op := Op{Kind: OpPunctuation, Behavior: "Isolated"}
		if v, ok := n["behavior"].(string); ok {
			op.Behavior = v
		}
		return []Op{op}
	}
	return nil
}

// HunyuanWords is the exact pattern of Hunyuan's third Split stage, as its
// tokenizer.json decodes (the \r and \n are the characters).
const HunyuanWords = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+" +
	"|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+" +
	"| ?[\\p{P}\\p{S}]+[\r\n]*" +
	"|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"

// SparkWords is Spark-X2.5's third Split stage (OpSparkSplit), as its
// tokenizer.json decodes.
const SparkWords = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+" +
	"|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+" +
	"| ?[\\p{P}\\p{S}]+" +
	"|[\r\n]|\\s+(?!\\S)|\\s+"

// isolatedSplit is the stage of a Split pattern that is a stage of its own
// rather than the alternation skeleton, by its exact text.
func isolatedSplit(pat string) (Op, bool) {
	switch pat {
	case `\p{N}{1,3}`:
		return Op{Kind: OpDigitGroups, Split: SplitParams{DigitGroup: 3}}, true
	case "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+":
		return Op{Kind: OpKanaHanRuns}, true
	case HunyuanWords:
		return Op{Kind: OpHunyuanSplit}, true
	case SparkWords:
		return Op{Kind: OpSparkSplit}, true
	}
	return Op{}, false
}

// parseSplit reads the parameters out of a Split pattern.
//
// It recognises, it does not interpret: the patterns in the wild are one
// alternation skeleton with different quantifiers and classes. An unrecognised
// shape must end up as a refusal, never as a default.
func parseSplit(pat string) SplitParams {
	sp := SplitParams{DigitGroup: 1}
	if strings.Contains(pat, `(?i:'s`) || strings.Contains(pat, `'(?i:[sdmt]`) {
		sp.Contractions, sp.CaseFold = true, true
	} else if strings.Contains(pat, `'[sS]`) {
		sp.Contractions = true // an already-adapted alternation: case-exact
	}
	if strings.Contains(pat, `\p{N}{1,3}`) {
		sp.DigitGroup = 3
	}
	if strings.Contains(pat, `[\p{L}\p{M}]`) {
		sp.LetterMarks = true
	}
	// o200k's shape, read off the pattern rather than the name. These select a
	// different algorithm, so each keys on a construct only it has (gpt-4o
	// also contains `(?i:'s`).
	if strings.Contains(pat, `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`) {
		sp.CaseRuns = true
	}
	// The contraction group followed by `?` is the suffix form; llama3's stands
	// alone and is followed by `|`.
	if strings.Contains(pat, `'d)?`) || strings.Contains(pat, `'D)?`) {
		sp.SuffixContract = true
	}
	if strings.Contains(pat, `[\r\n/]*`) {
		sp.PunctSlash = true
	}
	// The punctuation alternative with no tail. The artefact spells the
	// newlines either escaped or, after JSON decoding, as the characters
	// themselves; both are read, and the tail's absence is what is keyed on.
	for _, nl := range []string{`\r\n`, "\r\n"} {
		if strings.Contains(pat, ` ?[^\s\p{L}\p{N}`+nl+`]+|`) {
			sp.PunctNoNewline = true
		}
	}
	return sp
}

// GoSource renders the op as a Go composite literal for the generated table.
//
// The generator parses with ParsePreTokenizer and prints with this, so the
// table and the loader share one mapping.
//
// An unknown op kind prints as an empty literal, which buildPipeline refuses
// at load, so a new kind cannot become a silently valid table entry.
func (op Op) GoSource() string {
	switch op.Kind {
	case OpSplit:
		// PunctNoNewline prints only when set, so other rows are unchanged.
		tail := ""
		if op.Split.PunctNoNewline {
			tail = ", PunctNoNewline: true"
		}
		return fmt.Sprintf("{Kind: OpSplit, Split: SplitParams{Contractions: %v, CaseFold: %v, LetterMarks: %v, DigitGroup: %d, CaseRuns: %v, SuffixContract: %v, PunctSlash: %v, HanRuns: %v%s}}",
			op.Split.Contractions, op.Split.CaseFold, op.Split.LetterMarks, op.Split.DigitGroup,
			op.Split.CaseRuns, op.Split.SuffixContract, op.Split.PunctSlash, op.Split.HanRuns, tail)
	case OpByteLevel:
		return fmt.Sprintf("{Kind: OpByteLevel, UseRegex: %v, AddPrefixSpace: %v}", op.UseRegex, op.AddPrefixSpace)
	case OpDigits:
		return fmt.Sprintf("{Kind: OpDigits, Individual: %v}", op.Individual)
	case OpPunctuation:
		return fmt.Sprintf("{Kind: OpPunctuation, Behavior: %q}", op.Behavior)
	case OpDigitTriples:
		return "{Kind: OpDigitTriples}"
	case OpPunctDefault:
		return "{Kind: OpPunctDefault}"
	case OpDigitGroups:
		return fmt.Sprintf("{Kind: OpDigitGroups, Split: SplitParams{DigitGroup: %d}}", op.Split.DigitGroup)
	case OpKanaHanRuns:
		return "{Kind: OpKanaHanRuns}"
	case OpHunyuanSplit:
		return "{Kind: OpHunyuanSplit}"
	case OpSparkSplit:
		return "{Kind: OpSparkSplit}"
	}
	return "{}"
}

// Validate reports whether every stage is one a pipeline can be built from.
//
// It is the half of buildPipeline that belongs to the model: whether a stage
// is well formed, as opposed to whether it can be compiled into a splitter. The
// parser asks it so a refusal lands at the parse. Only opUnreadable
// reaches an error; the other checks guard kinds a future flattener may emit.
func Validate(ops []Op) error {
	for _, o := range ops {
		switch o.Kind {
		case OpSplit, OpByteLevel, OpDigits, OpPunctuation, OpDigitTriples, OpPunctDefault,
			OpDigitGroups, OpKanaHanRuns, OpHunyuanSplit, OpSparkSplit:
		case opUnreadable:
			return fmt.Errorf("pretok: a digit-triple Split that is not Isolated and uninverted")
		default:
			return fmt.Errorf("pretok: stage kind %d is not one this model defines", o.Kind)
		}
		if (o.Kind == OpSplit || o.Kind == OpDigitGroups) && o.Split.DigitGroup < 1 {
			return fmt.Errorf("pretok: a split stage with digit group %d", o.Split.DigitGroup)
		}
	}
	return nil
}
