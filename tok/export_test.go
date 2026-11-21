package tok

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/pretok"
)

// What the external tests read of the package's insides: the tests that read
// a vocabulary through the converter are package tok_test, since the
// converter imports tok. Compiled into the package for its tests only.

// Piped reports whether the vocabulary splits by a composed pipeline, empty
// or not.
func (v *Vocab) Piped() bool { return v.pipe != nil }

// PipeOps is the pipeline's stages, nil where there is no pipeline.
func (v *Vocab) PipeOps() []pretok.Op {
	if v.pipe == nil {
		return nil
	}
	return v.pipe.ops
}

// PreSplit is the vocabulary's pre-tokenizer split of text.
func (v *Vocab) PreSplit(text string) []string { return v.preSplit(text) }

// IDs is the vocabulary's token text to id.
func (v *Vocab) IDs() map[string]int32 { return v.ids }

// Kind is token id's type.
func (v *Vocab) Kind(id int32) int32 { return v.kind[id] }

// EOG is the end-of-generation ids.
func (v *Vocab) EOG() []int32 { return v.eog }

// ByteToken is byte b's <0xNN> token, -1 where there is none.
func (v *Vocab) ByteToken(b int) int32 { return v.byteTok[b] }

// TypeUnused is a converter's padding token type.
const TypeUnused = typeUnused

// UcIsNumber reports the pinned tables' Number property.
func UcIsNumber(r rune) bool { return ucIsNumber(r) }

// Llama3Family, Qwen2Family, GPT2Family and StarcoderFamily report the
// pre-tokenizer names each split serves.
func Llama3Family(pre string) bool    { return llama3Family(pre) }
func Qwen2Family(pre string) bool     { return qwen2Family(pre) }
func GPT2Family(pre string) bool      { return gpt2Family(pre) }
func StarcoderFamily(pre string) bool { return starcoderFamily(pre) }

// Pipeline is a composed pre-tokenizer, built from its stages.
type Pipeline struct{ p *pipeline }

// BuildPipeline composes ops.
func BuildPipeline(ops []pretok.Op) (Pipeline, error) {
	p, err := buildPipeline(ops)
	return Pipeline{p}, err
}

// Apply is the pipeline's split of text.
func (p Pipeline) Apply(text string) []string { return p.p.apply(text) }

// WPMVocab is a WordPiece vocabulary of the tests' (wpm_test.go).
func WPMVocab(lower, keepMarks bool) *jlm.Vocab { return wpmVocab(lower, keepMarks) }

// RealTokenizerJSON returns a tokenizer.json body carrying obj as its
// pre_tokenizer, laid out the way HuggingFace writes one: the key sits between
// other top-level keys and a vocabulary follows it.
func RealTokenizerJSON(obj string) string {
	return `{"version":"1.0","truncation":null,"padding":null,` +
		`"added_tokens":[{"id":0,"content":"<|endoftext|>","special":true}],` +
		`"normalizer":null,"pre_tokenizer":` + obj +
		`,"post_processor":null,"decoder":{"type":"ByteLevel"},` +
		`"model":{"type":"BPE","vocab":{"Richard":42,"Regex":43},"merges":["a b"]}}`
}
