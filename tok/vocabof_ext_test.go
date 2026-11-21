package tok_test

import (
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// vocabOfMaybe is the container's tokenizer section, built from a parsed
// source, for a sweep: it returns nil where the model carries none (the
// internal package's vocabOf skips there instead).
//
// The tests go through the converter: tok.New reads a jlm.Vocab as conversion
// resolved it, so a hand-built Vocab would test a vocabulary no converter
// produces. t.Skip in a loop is runtime.Goexit and ends the whole test, silently dropping
// every later model; a tokenizer-less file (a projector GGUF) must be stepped
// over instead.
func vocabOfMaybe(t testing.TB, f *meta.File) *jlm.Vocab {
	t.Helper()
	v, err := convert.VocabOf(f)
	if err != nil {
		t.Fatalf("convert.VocabOf: %v", err)
	}
	return v
}

// vocabOf is the container's tokenizer section, built from a parsed source.
//
// The tests go through the converter: tok.New reads a jlm.Vocab as conversion
// resolved it, so a hand-built Vocab would test a vocabulary no converter
// produces.
func vocabOf(t testing.TB, f *meta.File) *jlm.Vocab {
	t.Helper()
	v, err := convert.VocabOf(f)
	if err != nil {
		t.Fatalf("convert.VocabOf: %v", err)
	}
	if v == nil {
		t.Skip("this model carries no tokenizer section -- the gate would prove nothing")
	}
	return v
}
