package engine

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"strings"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

// sideInputsSuffix names a towered fixture restated with per-layer inputs and
// an embedding norm (withSideInputs): Qwen3-VL's deepstack taps and Gemma 3n's
// per-layer inputs in one side input, which front.json states as the taps'
// zero rows joined to the per-layer rows, and an RMS norm over the token
// embeddings, which front.json states for a token_embd_norm with no layer
// norm. No published model and no family's fixture takes either.
const sideInputsSuffix = "@sideinputs"

// statedSuffixes are the restatements a fixture's name can carry:
// <fixture>@<suffix> is the fixture converted with its source restated
// (convert.WithStatedSource), its graphs stated over the restatement as any
// conversion states them, so the restated model runs what a converted one
// does.
func statedSuffixes() map[string]func(*jlm.Source) {
	return map[string]func(*jlm.Source){sideInputsSuffix: withSideInputs, loopSuffix: loopedTwice}
}

// loopSuffix names a fixture whose layers run twice over (loopedTwice):
// blocks n..2n-1 a second pass of blocks 0..n-1 reading the first pass's
// pages (jlm.Config.WeightsOf), each block with its own history. A
// mixture's second pass reads its first pass's expert bank.
const loopSuffix = "@loop"

// loopedTwice states s's layers run twice, as a library file's unroll
// states a loop (convert.UnrollSource). A loop the converter refuses
// leaves a statement of the pages that the writer refuses in turn.
func loopedTwice(s *jlm.Source) {
	n := int(s.Config.NLayer)
	order := make([]int, 2*n)
	for i := range order {
		order[i] = i % n
	}
	if err := convert.UnrollSource(s, order); err != nil {
		s.Config.WeightsOf = []uint32{math.MaxUint32}
	}
}

// statedSource is the fixture a name with a statedSuffixes suffix restates
// and the suffix, false for any other name.
func statedSource(path string) (src, suffix string, ok bool) {
	for sf := range statedSuffixes() {
		if s, ok := strings.CutSuffix(path, sf); ok {
			return s, sf, true
		}
	}
	return "", "", false
}

// censusRestated are the restated fixtures, relative to the model directory,
// the census converts and the sweeps run, each with its tower where the
// fixture has one: the statements no other model takes.
func censusRestated() []string {
	return []string{"synth-qwen3vl.gguf" + sideInputsSuffix}
}

// sideInputsDim is the per-layer width withSideInputs states.
const sideInputsDim = 16

// withSideInputs gives s per-layer inputs of sideInputsDim a layer -- a
// per-layer token table, a projection of the embeddings and its norm, as
// Gemma 3n's -- and a norm over the token embeddings, every value drawn at
// random from a fixed seed: rows of the token table near 0, the norms' gains
// near 1.
func withSideInputs(s *jlm.Source) {
	c := s.Config
	vocab := uint64(0)
	for _, t := range s.Tensors {
		if t.Role == jlm.RoleTokenEmbd && t.Block == jlm.DenseBlock {
			vocab = t.Dims[1]
		}
	}
	c.PLEDim = sideInputsDim
	width := uint64(c.NLayer) * sideInputsDim
	r := rand.New(rand.NewPCG(20261120, 1))
	add := func(role jlm.Role, name string, dims []uint64, at func() float32) {
		n := uint64(1)
		t := jlm.Tensor{Role: role, Block: jlm.DenseBlock, Index: -1, Type: jlm.TypeF32, NDim: uint8(len(dims)), Name: name}
		for i, d := range dims {
			t.Dims[i] = d
			n *= d
		}
		t.Data = make([]byte, 4*n)
		for i := range n {
			binary.LittleEndian.PutUint32(t.Data[4*i:], math.Float32bits(at()))
		}
		s.Tensors = append(s.Tensors, t)
	}
	small := func() float32 { return float32(0.05 * r.NormFloat64()) }
	gain := func() float32 { return float32(1 + 0.1*r.NormFloat64()) }
	add(jlm.RolePLETokEmbd, "per_layer_token_embd.weight", []uint64{width, vocab}, small)
	add(jlm.RolePLEModelProj, "per_layer_model_proj.weight", []uint64{uint64(c.NEmbd), width}, small)
	add(jlm.RolePLEProjNorm, "per_layer_proj_norm.weight", []uint64{sideInputsDim}, gain)
	add(jlm.RoleTokenEmbdNorm, "token_embd_norm.weight", []uint64{uint64(c.NEmbd)}, gain)
}

// restatedSize is the bytes of the fixture a restated name reads, false
// where the box does not hold it.
func restatedSize(path string) (int64, bool) {
	src, _, _ := statedSource(path)
	fi, err := os.Stat(src)
	if err != nil {
		return 0, false
	}
	return fi.Size(), true
}
