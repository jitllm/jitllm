package convert

import (
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The cross-format gate: one model, two input formats, one container.
//
// It is the only gate that can see a wrong name mapping: a q projection wired
// to k, a norm applied in the wrong place, or an unpermuted q all load and run
// fluently, and no oracle inside one format sees them. SmolLM2-360M-Instruct
// is used as a bf16 HuggingFace directory and as llama.cpp's Q8_0 GGUF;
// converting both and comparing tensor by tensor turns each into a number.
//
// The bound is Q8_0's own quantisation error, not a tolerance: an unpermuted q
// scores ~2 against a 1e-4 pass.

const (
	hfDirName = "SmolLM2-360M-Instruct"
	hfGGUF    = "SmolLM2-360M-Instruct-Q8_0.gguf"
	// Measured: the worst of this model's 290 tensors is 5.843e-05, one order
	// under the bound and four under a mis-wired tensor.
	q8NMSE = 1e-3
)

func hfModelDir(t testing.TB) (dir, ggufPath string) {
	t.Helper()
	dir = testmodels.Path(hfDirName)
	ggufPath = testmodels.Path(hfGGUF)
	for _, p := range []string{dir, ggufPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing. "+
				"Fetch it: huggingface.co/HuggingFaceTB/%s", p, err, hfDirName)
		}
	}
	return dir, ggufPath
}

func hfSource(t testing.TB, dir string) *jlm.Source {
	t.Helper()
	_, shards, err := hfShards(dir)
	if err != nil {
		t.Fatal(err)
	}
	var open []*safetensors.File
	for _, p := range shards {
		f, err := safetensors.Open(p)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		open = append(open, f)
	}
	s, err := hfSourceLoaded(dir, open)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func ggufSource(t testing.TB, path string) *jlm.Source {
	t.Helper()
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	s, err := sourceOf(f)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestSafetensorsConfigMatchesItsGGUF compares the two converters' answers to
// every hyperparameter question.
//
// config.json and a GGUF header are two independent witnesses (llama.cpp's
// converter read the same config.json), so a disagreement is one of the two
// readers being wrong about what a key means.
func TestSafetensorsConfigMatchesItsGGUF(t *testing.T) {
	dir, gp := hfModelDir(t)
	a, b := hfSource(t, dir).Config, ggufSource(t, gp).Config
	fields := []struct {
		name string
		x, y any
	}{
		{"Arch", a.Arch, b.Arch},
		{"NLayer", a.NLayer, b.NLayer},
		{"NEmbd", a.NEmbd, b.NEmbd},
		{"NHead", a.NHead, b.NHead},
		{"NKVHead", a.NKVHead, b.NKVHead},
		{"HeadDim", a.HeadDim, b.HeadDim},
		{"NRot", a.NRot, b.NRot},
		{"NFFN", a.NFFN, b.NFFN},
		{"NVocab", a.NVocab, b.NVocab},
		{"NCtx", a.NCtx, b.NCtx},
		{"RMSEps", a.RMSEps, b.RMSEps},
		{"RopeBase", a.RopeBase, b.RopeBase},
		{"EmbdScale", a.EmbdScale, b.EmbdScale},
		{"AttnFactor", a.AttnFactor, b.AttnFactor},
		{"NExpert", a.NExpert, b.NExpert},
		{"Flags", a.Flags, b.Flags},
	}
	n := 0
	for _, f := range fields {
		if fmt.Sprint(f.x) != fmt.Sprint(f.y) {
			t.Errorf("%s: safetensors %v, gguf %v", f.name, f.x, f.y)
			continue
		}
		n++
	}
	if n != len(fields) {
		t.Fatalf("%d of %d config fields agree", n, len(fields))
	}
	t.Logf("%d config fields agree; arch=%v L=%d d=%d heads=%d/%d hd=%d ffn=%d vocab=%d flags=%v",
		n, a.Arch, a.NLayer, a.NEmbd, a.NHead, a.NKVHead, a.HeadDim, a.NFFN, a.NVocab, a.Flags)
}

// TestSafetensorsTensorsMatchTheirGGUF is the name-mapping and rotary-layout
// gate. See the file comment for why nothing else can see either.
func TestSafetensorsTensorsMatchTheirGGUF(t *testing.T) {
	dir, gp := hfModelDir(t)
	a, b := hfSource(t, dir), ggufSource(t, gp)

	type key struct {
		role  jlm.Role
		block int32
		index int32
	}
	index := func(s *jlm.Source) map[key]*jlm.Tensor {
		m := map[key]*jlm.Tensor{}
		for i := range s.Tensors {
			m[key{s.Tensors[i].Role, s.Tensors[i].Block, s.Tensors[i].Index}] = &s.Tensors[i]
		}
		return m
	}
	ma, mb := index(a), index(b)
	if len(ma) != len(a.Tensors) || len(mb) != len(b.Tensors) {
		t.Fatalf("a (role, block, index) is claimed twice: %d/%d and %d/%d",
			len(ma), len(a.Tensors), len(mb), len(b.Tensors))
	}
	for k := range ma {
		if _, ok := mb[k]; !ok {
			t.Errorf("safetensors has %v block %d index %d and the gguf does not", k.role, k.block, k.index)
		}
	}
	for k := range mb {
		if _, ok := ma[k]; !ok {
			t.Errorf("the gguf has %v block %d index %d and safetensors does not", k.role, k.block, k.index)
		}
	}
	if t.Failed() {
		t.Fatalf("the two inputs do not describe the same set of tensors")
	}

	keys := make([]key, 0, len(ma))
	for k := range ma {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].block != keys[j].block {
			return keys[i].block < keys[j].block
		}
		return keys[i].role < keys[j].role
	})

	worst, worstAt := 0.0, ""
	checked := 0
	xa := []float32{}
	xb := []float32{}
	for _, k := range keys {
		ta, tb := ma[k], mb[k]
		if ta.NDim != tb.NDim || ta.Dims != tb.Dims {
			t.Errorf("%v block %d: safetensors %v, gguf %v",
				k.role, k.block, ta.Dims[:ta.NDim], tb.Dims[:tb.NDim])
			continue
		}
		n := 1
		for i := 0; i < int(ta.NDim); i++ {
			n *= int(ta.Dims[i])
		}
		qa, _ := jlm.SourceType(ta.Type)
		qb, _ := jlm.SourceType(tb.Type)
		if cap(xa) < n {
			xa, xb = make([]float32, n), make([]float32, n)
		}
		xa, xb = xa[:n], xb[:n]
		if err := quant.Dequant32(qa, ta.Data, xa); err != nil {
			t.Fatalf("%v block %d (%v): %v", k.role, k.block, qa, err)
		}
		if err := quant.Dequant32(qb, tb.Data, xb); err != nil {
			t.Fatalf("%v block %d (%v): %v", k.role, k.block, qb, err)
		}
		var num, den float64
		for i := range xa {
			d := float64(xa[i]) - float64(xb[i])
			num += d * d
			den += float64(xb[i]) * float64(xb[i])
		}
		if den == 0 {
			t.Fatalf("%v block %d: the gguf arm is all zeros, so its NMSE is degenerate",
				k.role, k.block)
		}
		nmse := num / den
		if nmse > worst {
			worst, worstAt = nmse, fmt.Sprintf("%v block %d (%v vs %v)", k.role, k.block, qa, qb)
		}
		if nmse > q8NMSE {
			t.Errorf("%v block %d: NMSE %.3e against %.0e (%v vs %v)",
				k.role, k.block, nmse, q8NMSE, qa, qb)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no tensor was compared -- this gate proved nothing")
	}
	t.Logf("%d tensors agree; worst NMSE %.3e at %s (bound %.0e)", checked, worst, worstAt, q8NMSE)
}

// TestSafetensorsTokenizerMatchesItsGGUF is RULE 7m's comparison, run rather
// than asserted.
//
// The safetensors arm parses the model's own tokenizer.json; the GGUF arm
// resolves its label through pretok.Table. Disagreement is a finding, so this
// reports every field rather than asserting one.
func TestSafetensorsTokenizerMatchesItsGGUF(t *testing.T) {
	dir, gp := hfModelDir(t)
	a, b := hfSource(t, dir).Vocab, ggufSource(t, gp).Vocab
	if a == nil || b == nil {
		t.Fatalf("a container with no tokenizer: %v %v", a == nil, b == nil)
	}
	if a.Kind != b.Kind {
		t.Fatalf("kind %v against %v", a.Kind, b.Kind)
	}
	if a.Tokens.Len() != b.Tokens.Len() {
		t.Fatalf("%d tokens against %d", a.Tokens.Len(), b.Tokens.Len())
	}
	diff := 0
	for i := range a.Tokens.Len() {
		if a.Tokens.At(i) != b.Tokens.At(i) {
			if diff < 5 {
				t.Errorf("token %d: %q against %q", i, a.Tokens.At(i), b.Tokens.At(i))
			}
			diff++
		}
	}
	if diff != 0 {
		t.Fatalf("%d of %d tokens differ", diff, a.Tokens.Len())
	}
	if a.Merges.Len() != b.Merges.Len() {
		t.Fatalf("%d merges against %d", a.Merges.Len(), b.Merges.Len())
	}
	for i := range a.Merges.Len() {
		if a.Merges.At(i) != b.Merges.At(i) {
			t.Fatalf("merge %d: %v against %v", i, a.Merges.At(i), b.Merges.At(i))
		}
	}
	// The pipeline must denote the same stages in the same order: Digits
	// before ByteLevel is not the same op as Digits after it.
	if len(a.Pre) != len(b.Pre) {
		t.Fatalf("pre-tokenizer: %d stages (%+v) against %d (%+v)",
			len(a.Pre), a.Pre, len(b.Pre), b.Pre)
	}
	for i := range a.Pre {
		if a.Pre[i] != b.Pre[i] {
			t.Errorf("stage %d: %+v against %+v", i, a.Pre[i], b.Pre[i])
		}
	}
	if len(a.Pre) == 0 {
		t.Fatal("no pre-tokenizer stages on either side -- this gate proved nothing")
	}
	for _, f := range []struct {
		name string
		x, y any
	}{
		{"BOS", a.BOS, b.BOS}, {"EOS", a.EOS, b.EOS},
		{"AddBOS", a.AddBOS, b.AddBOS}, {"AddEOS", a.AddEOS, b.AddEOS},
		{"IgnoreMerges", a.IgnoreMerges, b.IgnoreMerges},
		{"ByteFallback", a.ByteFallback, b.ByteFallback},
		{"templates", len(a.Templates), len(b.Templates)},
	} {
		if fmt.Sprint(f.x) != fmt.Sprint(f.y) {
			t.Errorf("%s: safetensors %v, gguf %v", f.name, f.x, f.y)
		}
	}
	// What the GGUF has no room for at all.
	t.Logf("%d tokens, %d merges, %d stages agree; PreName %q against %q; "+
		"safetensors also carries %d added tokens the GGUF has no field for",
		a.Tokens.Len(), a.Merges.Len(), len(a.Pre), a.PreName, b.PreName, len(a.Added))
	if len(a.Added) == 0 {
		t.Error("no added tokens came across, and this model ships some")
	}
}
