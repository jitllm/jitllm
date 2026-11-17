package tok

import (
	"fmt"

	"github.com/jitllm/jitllm/tok/pretok"
)

// A pre-tokenizer is a pipeline of typed stages, composed at load time rather
// than hand-written per family. The pipelines that occur in the models' own
// tokenizer.json files include:
//
//	Split -> ByteLevel(use_regex=false)                       qwen2, pixtral, lfm2,
//	                                                          hunyuan, stablelm2,
//	                                                          smaug-bpe, gpt-4o, tekken
//	ByteLevel(use_regex=true)                                 gpt-2
//	Digits(individual) -> ByteLevel(use_regex=true)           starcoder, refact, smollm
//	Punctuation -> ByteLevel(use_regex=true) -> Digits(indiv) falcon3
//	Punctuation -> Split -> ByteLevel(=false) -> Split        falcon-h1
//	DigitGroups -> KanaHanRuns -> HunyuanSplit -> ByteLevel(=false)
//	                                                          hunyuan-dense
//	DigitGroups -> KanaHanRuns -> SparkSplit -> Digits(indiv) -> ByteLevel(=false)
//	                                                          spark2_5
//
// A name-based family table cannot express these (falcon3 and falcon-h1 are
// three and four stages, not one Split). An op this package cannot build is an
// error at load, naming the op.

// pipeline is a composed pre-tokenizer, built once per model at load.
type pipeline struct {
	ops []pretok.Op
	// raw keeps the patterns the stages came from, so a wrong split can be
	// traced to the artefact rather than guessed at.
	raw []string
}

// buildPipeline composes the stages, refusing anything it cannot build.
func buildPipeline(ops []pretok.Op) (*pipeline, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("tok: empty pre-tokenizer pipeline")
	}
	for i, op := range ops {
		switch op.Kind {
		case pretok.OpSplit:
			if op.Split.DigitGroup < 1 {
				return nil, fmt.Errorf("tok: stage %d Split: digit group %d is not a quantifier", i, op.Split.DigitGroup)
			}
		case pretok.OpDigitGroups:
			if op.Split.DigitGroup < 1 {
				return nil, fmt.Errorf("tok: stage %d DigitGroups: group %d", i, op.Split.DigitGroup)
			}
		case pretok.OpByteLevel, pretok.OpDigits, pretok.OpPunctuation, pretok.OpDigitTriples,
			pretok.OpPunctDefault, pretok.OpKanaHanRuns, pretok.OpHunyuanSplit, pretok.OpSparkSplit:
		default:
			return nil, fmt.Errorf("tok: stage %d: unsupported pre-tokenizer op %q -- "+
				"refusing rather than routing it to the nearest-looking splitter", i, op.Kind)
		}
	}
	return &pipeline{ops: ops}, nil
}

// apply runs the stages in order. Each stage takes the pieces the previous one
// produced, which is HuggingFace's own composition rule and the reason Digits
// after ByteLevel is not the same as Digits before it.
func (p *pipeline) apply(text string) []string {
	return collect(func(emit func(string)) { p.each(text, make([]runes, len(p.ops)), emit) })
}

// each runs the stages over text, handing every final piece to emit the moment
// it is found: stage k's piece goes straight into stage k+1, so no stage's
// output is ever gathered. sc is one buffer per stage.
//
// The per-stage emitters are built once per call, not per piece, so the
// recursion allocates no closures.
func (p *pipeline) each(text string, sc []runes, emit func(string)) {
	next := emit
	for k := len(p.ops) - 1; k >= 0; k-- {
		op, st, to := p.ops[k], &sc[k], next
		next = func(s string) { applyOp(op, s, st, to) }
	}
	next(text)
}

// applyOp runs one stage over s, handing its pieces to emit. It is a function
// because the stage type belongs to pretok, which describes stages but does not
// run them.
func applyOp(op pretok.Op, s string, sc *runes, emit func(string)) {
	switch op.Kind {
	case pretok.OpSplit:
		splitLlama3Each(s, op.Split, sc, emit)
	case pretok.OpDigits:
		splitDigitsOpEach(s, op.Individual, sc, emit)
	case pretok.OpPunctuation:
		splitPunctuationEach(s, op.Behavior, sc, emit)
	case pretok.OpDigitTriples:
		splitDigitTriplesEach(s, emit)
	case pretok.OpPunctDefault:
		splitRunsEach(s, isPunctDefault, sc, emit)
	case pretok.OpDigitGroups:
		splitDigitGroupsEach(s, op.Split.DigitGroup, sc, emit)
	case pretok.OpKanaHanRuns:
		splitRunsEach(s, isKanaHan, sc, emit)
	case pretok.OpHunyuanSplit:
		splitHunyuanEach(s, false, sc, emit)
	case pretok.OpSparkSplit:
		splitHunyuanEach(s, true, sc, emit)
	case pretok.OpByteLevel:
		if op.UseRegex {
			splitGPT2Each(s, sc, emit)
			return
		}
		emit(s)
	default:
		emit(s)
	}
}

// splitDigitsOpEach isolates digits: individual_digits=true gives one piece per
// digit, false groups each run.
func splitDigitsOpEach(s string, individual bool, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	for i := 0; i < len(rs); {
		if !ucIsNumber(rs[i]) {
			j := i
			for j < len(rs) && !ucIsNumber(rs[j]) {
				j++
			}
			emit(sc.piece(s, i, j))
			i = j
			continue
		}
		if individual {
			emit(sc.piece(s, i, i+1))
			i++
			continue
		}
		j := i
		for j < len(rs) && ucIsNumber(rs[j]) {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// splitDigitTriplesEach isolates each run of three ASCII digits, scanning left
// to right without overlap -- the regex "[0-9][0-9][0-9]" under HuggingFace's
// Split with behavior Isolated. What lies between matches is emitted whole.
// ASCII only, as the class says: a byte test needs no rune decode.
func splitDigitTriplesEach(s string, emit func(string)) {
	isD := func(b byte) bool { return b >= '0' && b <= '9' }
	start := 0
	for i := 0; i+3 <= len(s); {
		if isD(s[i]) && isD(s[i+1]) && isD(s[i+2]) {
			if i > start {
				emit(s[start:i])
			}
			emit(s[i : i+3])
			i += 3
			start = i
			continue
		}
		i++
	}
	if start < len(s) {
		emit(s[start:])
	}
}

// isPunctDefault is "[\p{P}\$\+<=>\^~\|]" -- \p{P} and the eight named
// ASCII symbols, and not the backtick HuggingFace's is_punc includes.
func isPunctDefault(r rune) bool {
	switch r {
	case '$', '+', '<', '=', '>', '^', '~', '|':
		return true
	}
	return ucFlags(r)&ucFlagPunctuation != 0
}

// splitRunsEach emits maximal runs of in(r) and of !in(r), in order.
func splitRunsEach(s string, in func(rune) bool, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	for i := 0; i < len(rs); {
		j, want := i, in(rs[i])
		for j < len(rs) && in(rs[j]) == want {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// splitPunctuationEach isolates punctuation. "Contiguous" keeps a run together;
// anything else isolates each mark.
//
// Punctuation is HuggingFace's is_punc: is_ascii_punctuation || \p{P}. The
// ASCII symbols $+<=>^`|~ are \p{S}, not \p{P}, and HF still isolates them.
// BERT's wordPiece stage reads the same class (wpm.go).
func splitPunctuationEach(s string, behavior string, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	isPunct := func(r rune) bool {
		fl := ucFlags(r)
		return fl&ucFlagPunctuation != 0 || (r < 0x7F && fl&ucFlagSymbol != 0)
	}
	for i := 0; i < len(rs); {
		if !isPunct(rs[i]) {
			j := i
			for j < len(rs) && !isPunct(rs[j]) {
				j++
			}
			emit(sc.piece(s, i, j))
			i = j
			continue
		}
		if behavior != "Contiguous" {
			emit(sc.piece(s, i, i+1))
			i++
			continue
		}
		j := i
		for j < len(rs) && isPunct(rs[j]) {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}
