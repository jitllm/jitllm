package convert

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestMixtureFamiliesConvertAlikeFromEitherInput holds each mixture family's
// two inputs to one container. scripts/moegold.py saves one fixture with the
// family's own class (safetensors, linked as <name>-hf) and writes the same
// weights to GGUF with llama.cpp's converter, every tensor F32 -- so the two
// converters must agree on every config field this gate names and on every
// tensor BYTE for byte: a role, a squeeze, a fold or a rotary base that one
// path gets and the other does not is a difference here, where the transformers
// gate (model.TestSafetensorsMatchTransformers) holds only the safetensors arm.
func TestMixtureFamiliesConvertAlikeFromEitherInput(t *testing.T) {
	ran := 0
	for _, name := range []string{"synth-glm4moe", "synth-qwen2moe", "synth-ernie45moe",
		"synth-ernie45", "synth-hunyuanmoe", "synth-hunyuan", "synth-dots1", "synth-phimoe"} {
		t.Run(name, func(t *testing.T) {
			dir, gp := testmodels.Path(name+"-hf"), testmodels.Path(name+".gguf")
			for _, p := range []string{dir, gp} {
				if _, err := os.Stat(p); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %v -- run scripts/moegold.py %s and link its "+
						"safetensors as %s-hf (RULE 11)", err, name, name)
				}
			}
			a, b := hfSource(t, dir), ggufSource(t, gp)
			ca, cb := a.Config, b.Config
			for _, f := range []struct {
				name string
				x, y any
			}{
				{"NLayer", ca.NLayer, cb.NLayer}, {"NMTP", ca.NMTP, cb.NMTP},
				{"NEmbd", ca.NEmbd, cb.NEmbd}, {"NHead", ca.NHead, cb.NHead}, {"NKVHead", ca.NKVHead, cb.NKVHead},
				{"HeadDim", ca.HeadDim, cb.HeadDim}, {"NRot", ca.NRot, cb.NRot}, {"NFFN", ca.NFFN, cb.NFFN},
				{"NVocab", ca.NVocab, cb.NVocab}, {"RMSEps", ca.RMSEps, cb.RMSEps},
				{"RopeBase", ca.RopeBase, cb.RopeBase}, {"NExpert", ca.NExpert, cb.NExpert},
				{"NExpertUsed", ca.NExpertUsed, cb.NExpertUsed}, {"NFFNExp", ca.NFFNExp, cb.NFFNExp},
				{"NFFNShExp", ca.NFFNShExp, cb.NFFNShExp}, {"NDenseLead", ca.NDenseLead, cb.NDenseLead},
				{"MoEStep", ca.MoEStep, cb.MoEStep}, {"ExpertScale", ca.ExpertScale, cb.ExpertScale},
				{"NExpertGroup", ca.NExpertGroup, cb.NExpertGroup}, {"Flags", ca.Flags, cb.Flags},
				{"AttnFactor", ca.AttnFactor, cb.AttnFactor}, {"SWAWindow", ca.SWAWindow, cb.SWAWindow},
				{"SWAPeriod", ca.SWAPeriod, cb.SWAPeriod},
			} {
				if fmt.Sprint(f.x) != fmt.Sprint(f.y) {
					t.Errorf("%s: safetensors %v, gguf %v", f.name, f.x, f.y)
				}
			}
			type key struct {
				r    jlm.Role
				b, i int32
			}
			ib := map[key]*jlm.Tensor{}
			for i := range b.Tensors {
				x := &b.Tensors[i]
				ib[key{x.Role, x.Block, x.Index}] = x
			}
			same := 0
			for i := range a.Tensors {
				x := &a.Tensors[i]
				y, ok := ib[key{x.Role, x.Block, x.Index}]
				if !ok {
					t.Errorf("safetensors has %v block %d and the gguf does not", x.Role, x.Block)
					continue
				}
				delete(ib, key{x.Role, x.Block, x.Index})
				if x.Type != y.Type || x.NDim != y.NDim || x.Dims != y.Dims {
					t.Errorf("%v block %d: safetensors %v %v, gguf %v %v", x.Role, x.Block,
						x.Type, x.Dims[:x.NDim], y.Type, y.Dims[:y.NDim])
					continue
				}
				yb, err := y.Bytes()
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(x.Data, yb) {
					t.Errorf("%v block %d: the bytes differ", x.Role, x.Block)
					continue
				}
				same++
			}
			for k := range ib {
				t.Errorf("the gguf has %v block %d and safetensors does not", k.r, k.b)
			}
			if same == 0 {
				t.Fatal("no tensor compared -- this gate proved nothing")
			}
			ran++
			t.Logf("%d tensors byte-identical, config agrees (flags %v)", same, ca.Flags)
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "%s", "no family compared")
	}
}
