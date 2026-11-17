package pretok

import "strconv"

// GPT2Pattern is the pattern the byte-level pre-tokenizer runs when its
// use_regex is set, the tokenizers library's own string for it.
const GPT2Pattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

// The patterns the other fixed stages stand for, each as the tokenizers
// library would split it: Digits is a split on \p{N} (Rust's char::is_numeric),
// Punctuation one on its is_punc, ASCII punctuation or \p{P}.
const (
	DigitsPattern        = `\p{N}`
	DigitRunsPattern     = `\p{N}+`
	PunctuationPattern   = `[\p{P}\x21-\x2F\x3A-\x40\x5B-\x60\x7B-\x7E]`
	DigitTriplesPattern  = `[0-9][0-9][0-9]`
	PunctDefaultPattern  = `[\p{P}\$\+<=>\^~\|]+`
	KanaHanRunsPattern   = "[一-龥぀-ゟ゠-ヿ]+"
	contractionAlternate = `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
)

// AsPattern is the Split a stage stands for: its pattern and its
// behavior, by the tokenizers library's names. ok is false for a stage that
// splits nothing (ByteLevel without use_regex) and for a kind with no
// pattern.
//
// An OpSplit's pattern is rebuilt from its parameters as the alternation
// skeleton they were read off, so a container that stored parameters splits
// as it did when it was written.
func (op Op) AsPattern() (pattern, behavior string, ok bool) {
	switch op.Kind {
	case OpSplit:
		return op.Split.Pattern(), "Isolated", true
	case OpByteLevel:
		return GPT2Pattern, "Isolated", op.UseRegex
	case OpDigits:
		if op.Individual {
			return DigitsPattern, "Isolated", true
		}
		return DigitRunsPattern, "Isolated", true
	case OpPunctuation:
		// Only Contiguous kept a run; every other behavior was read as
		// Isolated by the stage this stands for.
		if op.Behavior == "Contiguous" {
			return PunctuationPattern, "Contiguous", true
		}
		return PunctuationPattern, "Isolated", true
	case OpDigitTriples:
		return DigitTriplesPattern, "Isolated", true
	case OpPunctDefault:
		return PunctDefaultPattern, "Isolated", true
	case OpDigitGroups:
		return `\p{N}{1,` + strconv.Itoa(max(op.Split.DigitGroup, 1)) + `}`, "Isolated", true
	case OpKanaHanRuns:
		return KanaHanRunsPattern, "Isolated", true
	case OpHunyuanSplit:
		return HunyuanWords, "Isolated", true
	case OpSparkSplit:
		return SparkWords, "Isolated", true
	}
	return "", "", false
}

// Pattern is the alternation the parameters were read off.
func (sp SplitParams) Pattern() string {
	letter, upper, lower := `\p{L}`, `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`, `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`
	plain := `[^\s\p{L}\p{N}]`
	if sp.LetterMarks {
		letter, plain = `[\p{L}\p{M}]`, `[^\s\p{L}\p{M}\p{N}]`
	}
	if sp.HanRuns {
		letter = `[` + letter + `&&[^\p{Han}]]`
		upper = `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`
		lower = `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`
	}
	var alts []string
	if sp.HanRuns {
		alts = append(alts, `[\p{Han}]+`)
	}
	if sp.CaseRuns {
		suffix := ""
		if sp.SuffixContract && sp.Contractions {
			suffix = contractionAlternate + "?"
		}
		alts = append(alts,
			`[^\r\n\p{L}\p{N}]?`+upper+`*`+lower+`+`+suffix,
			`[^\r\n\p{L}\p{N}]?`+upper+`+`+lower+`*`+suffix)
	}
	if sp.Contractions && !sp.SuffixContract {
		alts = append(alts, contractionAlternate)
	}
	if !sp.CaseRuns {
		alts = append(alts, `[^\r\n\p{L}\p{N}]?`+letter+`+`)
	}
	if g := max(sp.DigitGroup, 1); g == 1 {
		alts = append(alts, `\p{N}`)
	} else {
		alts = append(alts, `\p{N}{1,`+strconv.Itoa(g)+`}`)
	}
	switch {
	case sp.PunctNoNewline:
		alts = append(alts, ` ?`+plain+`+`)
	case sp.PunctSlash:
		alts = append(alts, ` ?`+plain+`+[\r\n/]*`)
	default:
		alts = append(alts, ` ?`+plain+`+[\r\n]*`)
	}
	alts = append(alts, `\s*[\r\n]+`, `\s+(?!\S)`, `\s+`)
	out := alts[0]
	for _, a := range alts[1:] {
		out += "|" + a
	}
	return out
}
