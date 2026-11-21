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
//	Split -> Split -> Split -> ByteLevel(=false)              hunyuan-dense
//	Split -> Split -> Split -> Digits(indiv) -> ByteLevel(=false)
//	                                                          spark2_5
//
// A name-based family table cannot express these (falcon3 and falcon-h1 are
// three and four stages, not one Split). Every stage runs through the pattern
// engine (pattern.go): a Split as its own regex, the other kinds as the
// pattern each stands for (pretok.Op.AsPattern). An op this package cannot
// build -- an unknown kind, a pattern that does not compile -- is an error at
// load, naming the op.

// pipeline is a composed pre-tokenizer, built once per model at load: each
// stage's pattern compiled for the engine (pattern.go), nil for a stage that
// splits nothing.
type pipeline struct {
	ops    []pretok.Op
	splits []*splitPattern
}

// buildPipeline composes the stages, refusing anything it cannot build: a
// stage kind it does not know, and a pattern that does not compile.
func buildPipeline(ops []pretok.Op) (*pipeline, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("tok: empty pre-tokenizer pipeline")
	}
	p := &pipeline{ops: ops, splits: make([]*splitPattern, len(ops))}
	for i, op := range ops {
		switch op.Kind {
		case pretok.OpPattern:
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
		pat, behavior, ok := op.AsPattern()
		if !ok {
			continue
		}
		sp, err := compileSplit(pat, behavior, op.Invert)
		if err != nil {
			return nil, fmt.Errorf("tok: stage %d %s: %w", i, op.Kind, err)
		}
		p.splits[i] = sp
	}
	return p, nil
}

// CheckPreOps refuses a pre-tokenizer this package could not build: a stage
// kind it does not know, or a pattern that does not compile. It is what a
// vocabulary carrying these stages would be refused with at load.
func CheckPreOps(ops []pretok.Op) error {
	_, err := buildPipeline(ops)
	return err
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
		sp, st, to := p.splits[k], &sc[k], next
		if sp == nil {
			continue
		}
		next = func(s string) { sp.each(s, st, to) }
	}
	next(text)
}
