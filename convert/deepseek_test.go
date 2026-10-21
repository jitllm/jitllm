package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The DeepSeek (MLA) half of the safetensors reader.
//
// Its one hard gate is arithmetic, not dimensional. kv_b_proj holds two
// up-projections per head, one stored transposed and the other verbatim:
//
//	score = q_nope . (W_k c) = (W_k^T q_nope) . c   -> attn_k_b is W_k^T
//	out   = sum_j a_j (W_v c_j) = W_v (sum a_j c_j) -> attn_v_b is W_v
//
// synth-deepseek sets qk_nope_head_dim == v_head_dim == kv_lora_rank == 16, so
// every wrong split has an identical table; only reconstructing the source
// numbers separates them.

const dsDir = "synth-deepseek"

// ---------------------------------------------------------------------------
// the real fixture

func deepseekSource(t testing.TB) (*jlm.Source, *safetensors.File) {
	t.Helper()
	dir := testmodels.Path(dsDir)
	p := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing (RULE 10). "+
			"Make it: $JITLLM_HF_PY scripts/hfgold.py %s", p, err, dsDir)
	}
	f, err := safetensors.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	s, err := hfSourceLoaded(dir, []*safetensors.File{f})
	if err != nil {
		t.Fatalf("synth-deepseek does not convert: %v", err)
	}
	return s, f
}

// TestDeepseekConverts pins every Config field the container carries for
// this architecture, against the numbers in the fixture's own config.json.
//
// Each is a number nothing else checks (HeadDim is not head_dim, the expert
// count has a third spelling, the router knobs are unique to this arch), and
// a wrong default loads.
func TestDeepseekConverts(t *testing.T) {
	s, _ := deepseekSource(t)
	c := s.Config
	// config.json: hidden 64, 4 heads, 3 layers, qk_nope 16 + qk_rope 8,
	// v_head 16, kv_lora 16, q_lora 32, first_k_dense_replace 1, 8 routed
	// experts of 32 with 2 used, 1 shared, n_group 2 / topk_group 1, scale
	// 2.5, sigmoid, norm_topk_prob true.
	for _, k := range []struct {
		name     string
		got, exp uint32
	}{
		{"NLayer", c.NLayer, 3},
		{"NEmbd", c.NEmbd, 64},
		{"NHead", c.NHead, 4},
		{"NKVHead", c.NKVHead, 4},
		{"NFFN", c.NFFN, 96},
		{"NVocab", c.NVocab, 151665},
		{"NCtx", c.NCtx, 512},
		// 24, not the file's head_dim of 8. See deepseekHFConfig.
		{"HeadDim", c.HeadDim, 24},
		{"HeadDimV", c.HeadDimV, 16},
		{"NRot", c.NRot, 8},
		{"QLoraRank", c.QLoraRank, 32},
		{"KVLoraRank", c.KVLoraRank, 16},
		{"NDenseLead", c.NDenseLead, 1},
		{"NExpert", c.NExpert, 8},
		{"NExpertUsed", c.NExpertUsed, 2},
		{"NFFNExp", c.NFFNExp, 32},
		{"NFFNShExp", c.NFFNShExp, 32},
		{"NExpertGroup", c.NExpertGroup, 2},
		{"NExpertGroupUsed", c.NExpertGroupUsed, 1},
	} {
		if k.got != k.exp {
			t.Errorf("%s = %d, want %d", k.name, k.got, k.exp)
		}
	}
	if c.Arch != jlm.ArchDeepseek2 {
		t.Errorf("Arch = %v, want deepseek2", c.Arch)
	}
	if c.ExpertScale != 2.5 {
		t.Errorf("ExpertScale = %v, want 2.5", c.ExpertScale)
	}
	if c.RMSEps != 1e-6 || c.RopeBase != 10000 {
		t.Errorf("RMSEps %v RopeBase %v", c.RMSEps, c.RopeBase)
	}
	// rope_parameters says rope_type "default", so no YaRN and no correction.
	if c.YarnFactor != 0 || c.YarnLogMul != 0 || c.AttnFactor != 1 {
		t.Errorf("YarnFactor %v YarnLogMul %v AttnFactor %v, want 0/0/1",
			c.YarnFactor, c.YarnLogMul, c.AttnFactor)
	}
	if !c.Flags.Has(jlm.FlagExpertSigmoid) {
		t.Error("scoring_func is sigmoid and FlagExpertSigmoid is not set")
	}
	// norm_topk_prob is true here and in DeepseekV3Config's default, so
	// FlagNoExpertNorm must not be set.
	if c.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Error("norm_topk_prob is true and FlagNoExpertNorm is set")
	}
	// Not NEOX: llama.cpp gives deepseek2 LLAMA_ROPE_TYPE_NORM, transformers
	// rotates adjacent pairs, and no row permutation happens on the way in.
	if c.Flags.Has(jlm.FlagRopeNeox) {
		t.Error("FlagRopeNeox is set: deepseek2 rotates ADJACENT pairs")
	}
	if c.Flags.Has(jlm.FlagTiedEmbd) {
		t.Error("lm_head.weight is present and FlagTiedEmbd is set")
	}
	if s.Vocab == nil || s.Vocab.Tokens.Len() != int(c.NVocab) {
		t.Fatalf("vocab %v", s.Vocab)
	}
}

// TestDeepseekLayerSets is the per-layer half: block 0 is the leading dense
// block and blocks 1-2 are mixtures, and the container must say so tensor by
// tensor, since NDenseLead is one integer and the roles come from a flat
// name table.
func TestDeepseekLayerSets(t *testing.T) {
	s, _ := deepseekSource(t)
	byBlock := map[int32]map[jlm.Role]int{}
	for i := range s.Tensors {
		e := &s.Tensors[i]
		if byBlock[e.Block] == nil {
			byBlock[e.Block] = map[jlm.Role]int{}
		}
		byBlock[e.Block][e.Role]++
	}
	// Every block, dense or mixture, carries the same attention.
	mla := []jlm.Role{jlm.RoleAttnNorm, jlm.RoleFFNNorm, jlm.RoleAttnQA, jlm.RoleAttnQANorm,
		jlm.RoleAttnQB, jlm.RoleAttnKVA, jlm.RoleAttnKVANorm, jlm.RoleAttnKB,
		jlm.RoleAttnVB, jlm.RoleAttnOut}
	dense := []jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown}
	// RoleExpProbsB, not RoleRouterBias: DeepSeek's e_score_correction_bias
	// biases the selection only, where gpt-oss's ffn_gate_inp.bias biases the
	// router logits and reaches the weights. llama.cpp keeps two tensors too.
	moe := []jlm.Role{jlm.RoleRouter, jlm.RoleExpProbsB,
		jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank,
		jlm.RoleShExpGate, jlm.RoleShExpUp, jlm.RoleShExpDown}
	for b := int32(0); b < 3; b++ {
		got := byBlock[b]
		if got == nil {
			t.Fatalf("block %d has no tensor at all", b)
		}
		want, absent := append(append([]jlm.Role{}, mla...), dense...), moe
		if b > 0 {
			want, absent = append(append([]jlm.Role{}, mla...), moe...), dense
		}
		for _, r := range want {
			if got[r] != 1 {
				t.Errorf("block %d: %v appears %d time(s), want 1", b, r, got[r])
			}
		}
		for _, r := range absent {
			if got[r] != 0 {
				t.Errorf("block %d: %v appears %d time(s), want 0", b, r, got[r])
			}
		}
		if len(got) != len(want) {
			t.Errorf("block %d carries %d role(s), want %d: %v", b, len(got), len(want), got)
		}
	}
	// And the roles this architecture must not have: MLA has no key or
	// value projection.
	for i := range s.Tensors {
		switch s.Tensors[i].Role {
		case jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnQKV:
			t.Errorf("%s landed as %v, and latent attention has no key projection",
				s.Tensors[i].Name, s.Tensors[i].Role)
		}
	}
}

// TestDeepseekKVBSplitIsAbsorbed reconstructs W_k and W_v from the container
// and compares them, element by element, against the safetensors bytes.
//
// It carries its own violation arms (RULE 10): readings a swap or a missing
// transpose would produce must not match, so a fixture change that makes the
// transpose a no-op fails here instead of passing vacuously.
func TestDeepseekKVBSplitIsAbsorbed(t *testing.T) {
	s, f := deepseekSource(t)
	c := s.Config
	src := map[int32][]float32{}
	for i := range f.Tensors {
		tt := &f.Tensors[i]
		if !strings.HasSuffix(tt.Name, ".self_attn.kv_b_proj.weight") {
			continue
		}
		b, err := f.ReadTensor(tt)
		if err != nil {
			t.Fatal(err)
		}
		n := strings.TrimPrefix(tt.Name, "model.layers.")
		var li int32
		if _, err := fmt.Sscanf(n, "%d.", &li); err != nil {
			t.Fatal(err)
		}
		src[li] = f32s(b)
	}
	if len(src) != int(c.NLayer) {
		t.Fatalf("%d kv_b_proj in the checkpoint, want %d", len(src), c.NLayer)
	}
	deepseekCheckKVB(t, s, c, src)
}

// deepseekCheckKVB is the arithmetic, shared by the real fixture (the three
// widths equal, so only arithmetic separates the readings) and the hermetic
// one (all different, so the indexing is exercised).
func deepseekCheckKVB(t *testing.T, s *jlm.Source, c *jlm.Config, src map[int32][]float32) {
	t.Helper()
	nope := int(c.HeadDim - c.NRot)
	vdim := int(hfHeadDimV(c))
	lat, heads := int(c.KVLoraRank), int(c.NHead)
	find := func(role jlm.Role, block int32) *jlm.Tensor {
		for i := range s.Tensors {
			if s.Tensors[i].Role == role && s.Tensors[i].Block == block {
				return &s.Tensors[i]
			}
		}
		t.Fatalf("no %v in block %d", role, block)
		return nil
	}
	ran := 0
	for block, w := range src {
		if len(w) != heads*(nope+vdim)*lat {
			t.Fatalf("block %d: kv_b_proj has %d elements, want %d",
				block, len(w), heads*(nope+vdim)*lat)
		}
		// The source, addressed the way HuggingFace stores it: row-major
		// [heads*(nope+vdim), lat], head h's W_k first and its W_v after.
		wk := func(h, i, j int) float32 { return w[(h*(nope+vdim)+i)*lat+j] }
		wv := func(h, r, j int) float32 { return w[(h*(nope+vdim)+nope+r)*lat+j] }

		kb, vb := find(jlm.RoleAttnKB, block), find(jlm.RoleAttnVB, block)
		if kb.NDim != 3 || kb.Dims != [4]uint64{uint64(nope), uint64(lat), uint64(heads)} {
			t.Fatalf("attn_k_b is %dD %v, want [%d %d %d]", kb.NDim, kb.Dims[:kb.NDim], nope, lat, heads)
		}
		if vb.NDim != 3 || vb.Dims != [4]uint64{uint64(lat), uint64(vdim), uint64(heads)} {
			t.Fatalf("attn_v_b is %dD %v, want [%d %d %d]", vb.NDim, vb.Dims[:vb.NDim], lat, vdim, heads)
		}
		kd, vd := f32s(kb.Data), f32s(vb.Data)
		if len(kd) != heads*lat*nope || len(vd) != heads*vdim*lat {
			t.Fatalf("block %d: k_b %d elements, v_b %d", block, len(kd), len(vd))
		}
		// jlm.sheetsOf: sheet h is rows*k elements, row-major, k fastest.
		kbAt := func(h, row, col int) float32 { return kd[h*lat*nope+row*nope+col] }
		vbAt := func(h, row, col int) float32 { return vd[h*vdim*lat+row*lat+col] }

		// The claim: attn_k_b is W_k transposed, attn_v_b is W_v as it is.
		for h := 0; h < heads; h++ {
			for i := 0; i < nope; i++ {
				for j := 0; j < lat; j++ {
					if got, exp := kbAt(h, j, i), wk(h, i, j); got != exp {
						t.Fatalf("block %d head %d: attn_k_b[%d][%d] = %v, and W_k^T wants "+
							"W_k[%d][%d] = %v -- k_b is the k half TRANSPOSED",
							block, h, j, i, got, i, j, exp)
					}
				}
			}
			for r := 0; r < vdim; r++ {
				for j := 0; j < lat; j++ {
					if got, exp := vbAt(h, r, j), wv(h, r, j); got != exp {
						t.Fatalf("block %d head %d: attn_v_b[%d][%d] = %v, want W_v[%d][%d] = %v "+
							"-- v_b is the v half UNCHANGED", block, h, r, j, got, r, j, exp)
					}
				}
			}
		}

		// The violation arms: each is a plausible wrong reading, and each must
		// be distinguishable from the one above.
		differs := func(name string, f func() bool) {
			if !f() {
				t.Errorf("block %d: %s is indistinguishable from the shipped reading, so "+
					"the assertions above hold for a container that is wrong", block, name)
			}
		}
		differs("k_b WITHOUT the transpose", func() bool {
			for h := 0; h < heads; h++ {
				for i := 0; i < nope; i++ {
					for j := 0; j < lat; j++ {
						if kd[h*lat*nope+i*lat+j] != wk(h, i, j) {
							return true
						}
					}
				}
			}
			return false
		})
		differs("v_b WITH a transpose", func() bool {
			for h := 0; h < heads; h++ {
				for r := 0; r < vdim; r++ {
					for j := 0; j < lat; j++ {
						if vd[h*vdim*lat+j*vdim+r] != wv(h, r, j) {
							return true
						}
					}
				}
			}
			return false
		})
		differs("k_b and v_b swapped", func() bool {
			if len(kd) != len(vd) {
				return true
			}
			for i := range kd {
				if kd[i] != vd[i] {
					return true
				}
			}
			return false
		})
		ran++
	}
	if ran == 0 {
		t.Fatal("no block was checked -- this gate proved nothing (RULE 10)")
	}
	t.Logf("kv_b_proj absorbed over %d block(s): %d head(s), nope %d, latent %d, value %d",
		ran, heads, nope, lat, vdim)
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// ---------------------------------------------------------------------------
// a hermetic fixture, for the refusals and for a second geometry
//
// Every width here differs (qk_nope 4, v_head_dim 3, kv_lora_rank 5), unlike
// the real fixture, so the indexing is exercised and a shape check has
// something to say.
const (
	dsEmbd   = 8
	dsHead   = 2
	dsNope   = 4
	dsRope   = 2
	dsHeadD  = dsNope + dsRope // 6
	dsVDim   = 3
	dsLat    = 5
	dsQLora  = 9
	dsFFN    = 16
	dsExp    = 4
	dsUsed   = 2
	dsExpFFN = 6
	dsLayer  = 3
	dsLead   = 1
	dsVocab  = sNVocab
)

func newSynthDeepseek() *synthModel {
	base := newSynth()
	m := &synthModel{
		cfg: map[string]any{
			"architectures":           []any{"DeepseekV3ForCausalLM"},
			"model_type":              "deepseek_v3",
			"hidden_size":             dsEmbd,
			"intermediate_size":       dsFFN,
			"num_hidden_layers":       dsLayer,
			"num_attention_heads":     dsHead,
			"num_key_value_heads":     dsHead,
			"head_dim":                dsRope, // what DeepseekV3Config derives
			"max_position_embeddings": 64,
			"rms_norm_eps":            1e-6,
			"rope_theta":              10000,
			"vocab_size":              dsVocab,
			"hidden_act":              "silu",
			"tie_word_embeddings":     false,
			"q_lora_rank":             dsQLora,
			"kv_lora_rank":            dsLat,
			"qk_nope_head_dim":        dsNope,
			"qk_rope_head_dim":        dsRope,
			"v_head_dim":              dsVDim,
			"first_k_dense_replace":   dsLead,
			"n_routed_experts":        dsExp,
			"num_experts_per_tok":     dsUsed,
			"n_shared_experts":        1,
			"moe_intermediate_size":   dsExpFFN,
			"n_group":                 2,
			"topk_group":              1,
			"routed_scaling_factor":   2.5,
			"norm_topk_prob":          true,
			"scoring_func":            "sigmoid",
		},
		tensors: map[string]*synthTensor{},
		tokJSON: base.tokJSON,
		tokCfg:  base.tokCfg,
	}
	add := func(name string, shape ...uint64) {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		m.tensors[name] = &synthTensor{safetensors.F32, shape, f32Bytes(n, uint32(len(name)*13+n))}
	}
	add("model.embed_tokens.weight", dsVocab, dsEmbd)
	add("lm_head.weight", dsVocab, dsEmbd)
	add("model.norm.weight", dsEmbd)
	for l := 0; l < dsLayer; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"input_layernorm.weight", dsEmbd)
		add(p+"post_attention_layernorm.weight", dsEmbd)
		add(p+"self_attn.q_a_proj.weight", dsQLora, dsEmbd)
		add(p+"self_attn.q_a_layernorm.weight", dsQLora)
		add(p+"self_attn.q_b_proj.weight", dsHead*dsHeadD, dsQLora)
		add(p+"self_attn.kv_a_proj_with_mqa.weight", dsLat+dsRope, dsEmbd)
		add(p+"self_attn.kv_a_layernorm.weight", dsLat)
		add(p+"self_attn.kv_b_proj.weight", dsHead*(dsNope+dsVDim), dsLat)
		add(p+"self_attn.o_proj.weight", dsEmbd, dsHead*dsVDim)
		if l < dsLead {
			add(p+"mlp.gate_proj.weight", dsFFN, dsEmbd)
			add(p+"mlp.up_proj.weight", dsFFN, dsEmbd)
			add(p+"mlp.down_proj.weight", dsEmbd, dsFFN)
			continue
		}
		add(p+"mlp.gate.weight", dsExp, dsEmbd)
		add(p+"mlp.gate.e_score_correction_bias", dsExp)
		add(p+"mlp.shared_experts.gate_proj.weight", dsExpFFN, dsEmbd)
		add(p+"mlp.shared_experts.up_proj.weight", dsExpFFN, dsEmbd)
		add(p+"mlp.shared_experts.down_proj.weight", dsEmbd, dsExpFFN)
		for e := 0; e < dsExp; e++ {
			q := fmt.Sprintf("%smlp.experts.%d.", p, e)
			add(q+"gate_proj.weight", dsExpFFN, dsEmbd)
			add(q+"up_proj.weight", dsExpFFN, dsEmbd)
			add(q+"down_proj.weight", dsEmbd, dsExpFFN)
		}
	}
	return m
}

// TestSynthDeepseekConverts is the control for the refusal table below, and the
// second geometry for the kv_b arithmetic.
func TestSynthDeepseekConverts(t *testing.T) {
	m := newSynthDeepseek()
	s, err := m.source(t)
	if err != nil {
		t.Fatalf("the unmutated fixture does not convert, so no refusal below proves anything: %v", err)
	}
	c := s.Config
	if c.Arch != jlm.ArchDeepseek2 || c.HeadDim != dsHeadD || c.HeadDimV != dsVDim ||
		c.NRot != dsRope || c.QLoraRank != dsQLora || c.KVLoraRank != dsLat ||
		c.NDenseLead != dsLead || c.NExpert != dsExp || c.NExpertUsed != dsUsed ||
		c.NFFNExp != dsExpFFN || c.NFFNShExp != dsExpFFN ||
		c.NExpertGroup != 2 || c.NExpertGroupUsed != 1 || c.ExpertScale != 2.5 {
		t.Fatalf("config %+v", c)
	}
	if !c.Flags.Has(jlm.FlagExpertSigmoid) || c.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Errorf("flags %v", c.Flags)
	}
	// The same arithmetic as the real fixture, over widths that are all
	// different.
	src := map[int32][]float32{}
	for l := 0; l < dsLayer; l++ {
		src[int32(l)] = f32s(m.tensors[fmt.Sprintf("model.layers.%d.self_attn.kv_b_proj.weight", l)].data)
	}
	deepseekCheckKVB(t, s, c, src)
}

// TestSynthDeepseekAlreadySplitIsNotReSplit covers the checkpoint that ships
// the absorbed halves separately: they are carried through, not cut again.
func TestSynthDeepseekAlreadySplitIsNotReSplit(t *testing.T) {
	m := newSynthDeepseek()
	for l := 0; l < dsLayer; l++ {
		p := fmt.Sprintf("model.layers.%d.self_attn.", l)
		fused := m.tensors[p+"kv_b_proj.weight"]
		delete(m.tensors, p+"kv_b_proj.weight")
		// The shapes llama.cpp already writes: wk_b is {qk_nope, kv_lora,
		// n_head} and wv_b is {kv_lora, v_head, n_head}, which safetensors
		// lists slowest-first.
		m.tensors[p+"k_b_proj.weight"] = &synthTensor{safetensors.F32,
			[]uint64{dsHead, dsLat, dsNope}, fused.data[:dsHead*dsLat*dsNope*4]}
		m.tensors[p+"v_b_proj.weight"] = &synthTensor{safetensors.F32,
			[]uint64{dsHead, dsVDim, dsLat}, f32Bytes(dsHead*dsVDim*dsLat, uint32(l+7))}
	}
	s, err := m.source(t)
	if err != nil {
		t.Fatalf("a checkpoint with the halves already absorbed was refused: %v", err)
	}
	n := 0
	for i := range s.Tensors {
		switch s.Tensors[i].Role {
		case jlm.RoleAttnKB, jlm.RoleAttnVB:
			if !strings.HasSuffix(s.Tensors[i].Name, "_b_proj.weight") {
				t.Errorf("%s was re-split: a pre-absorbed half must be carried through",
					s.Tensors[i].Name)
			}
			n++
		}
	}
	if n != 2*dsLayer {
		t.Fatalf("%d absorbed tensor(s), want %d", n, 2*dsLayer)
	}
}

// TestDeepseekRefusals is the violation table.
//
// Every row is a container that would load, and several would be fluent.
func TestDeepseekRefusals(t *testing.T) {
	L := func(l int, s string) string { return fmt.Sprintf("model.layers.%d.%s", l, s) }
	cases := []struct {
		name string
		mut  func(*synthModel)
		want string
	}{
		{"a mixture block with a dense feed-forward", func(m *synthModel) {
			m.tensors[L(2, "mlp.gate_proj.weight")] = &synthTensor{safetensors.F32,
				[]uint64{dsFFN, dsEmbd}, f32Bytes(dsFFN*dsEmbd, 1)}
		}, "block 2 carries mlp.gate_proj.weight, which is dense"},
		{"a dense block with a router", func(m *synthModel) {
			m.tensors[L(0, "mlp.gate.weight")] = &synthTensor{safetensors.F32,
				[]uint64{dsExp, dsEmbd}, f32Bytes(dsExp*dsEmbd, 2)}
		}, "block 0 carries mlp.gate.weight, which is a mixture"},
		{"a mixture block with no router", func(m *synthModel) {
			delete(m.tensors, L(2, "mlp.gate.weight"))
		}, "block 2 is a mixture block and has no mlp.gate.weight"},
		{"a dense block with no feed-forward", func(m *synthModel) {
			delete(m.tensors, L(0, "mlp.gate_proj.weight"))
			delete(m.tensors, L(0, "mlp.up_proj.weight"))
			delete(m.tensors, L(0, "mlp.down_proj.weight"))
		}, "block 0 is inside the leading dense run"},
		{"first_k_dense_replace past the model", func(m *synthModel) {
			m.cfg["first_k_dense_replace"] = dsLayer + 1
		}, "first_k_dense_replace 4 in a 3-block model"},
		// The shape a real DeepSeek-V3 has, a multi-token-prediction block at
		// index num_hidden_layers, without the num_nextn_predict_layers that
		// declares it: most of its tensors are named in the table, so the
		// refusal must come from the index.
		{"an undeclared multi-token-prediction block", func(m *synthModel) {
			p := fmt.Sprintf("model.layers.%d.", dsLayer)
			m.tensors[p+"input_layernorm.weight"] = &synthTensor{safetensors.F32,
				[]uint64{dsEmbd}, f32Bytes(dsEmbd, 6)}
			m.tensors[p+"eh_proj.weight"] = &synthTensor{safetensors.F32,
				[]uint64{dsEmbd, 2 * dsEmbd}, f32Bytes(2*dsEmbd*dsEmbd, 7)}
		}, "with 0 prediction block(s) (num_nextn_predict_layers)"},
		// The two query forms, each with the other's weights.
		{"two-step weights and a null rank", func(m *synthModel) {
			m.cfg["q_lora_rank"] = nil
		}, "the query is projected in two steps and nothing would read them"},
		{"one-step weights and a rank", func(m *synthModel) {
			for l := 0; l < dsLayer; l++ {
				p := fmt.Sprintf("model.layers.%d.self_attn.", l)
				delete(m.tensors, p+"q_a_proj.weight")
				delete(m.tensors, p+"q_a_layernorm.weight")
				delete(m.tensors, p+"q_b_proj.weight")
				m.tensors[p+"q_proj.weight"] = &synthTensor{safetensors.F32,
					[]uint64{dsHead * dsHeadD, dsEmbd}, f32Bytes(dsHead*dsHeadD*dsEmbd, 3)}
			}
		}, "the query is projected in one step and the rank is unread"},
		// MLA shares one latent across every head.
		{"grouped-query latent attention", func(m *synthModel) {
			m.cfg["num_key_value_heads"] = 1
		}, "latent attention shares one key-value latent across every head"},
		// The MLA geometry itself.
		{"no kv_lora_rank", func(m *synthModel) { delete(m.cfg, "kv_lora_rank") },
			"this architecture is multi-head latent attention"},
		{"kv_b_proj of the wrong height", func(m *synthModel) {
			tt := m.tensors[L(0, "self_attn.kv_b_proj.weight")]
			tt.shape = []uint64{dsHead * (dsNope + dsVDim + 1), dsLat}
			tt.data = f32Bytes(dsHead*(dsNope+dsVDim+1)*dsLat, 4)
		}, "head(s) of (4 nope + 3 value) rows over a 5-wide latent"},
		// The output projection reads the value head: here 3 against 6, and
		// reading the wrong one would refuse every real DeepSeek.
		{"o_proj sized by the key head", func(m *synthModel) {
			tt := m.tensors[L(0, "self_attn.o_proj.weight")]
			tt.shape = []uint64{dsEmbd, dsHead * dsHeadD}
			tt.data = f32Bytes(dsEmbd*dsHead*dsHeadD, 5)
		}, "attn_out wants [6 8]"},
		// The router's three knobs.
		{"an unimplemented gate", func(m *synthModel) { m.cfg["scoring_func"] = "softmax_topk" },
			`scoring_func "softmax_topk" is not implemented`},
		{"groups that do not divide the experts", func(m *synthModel) { m.cfg["n_group"] = 3 },
			"4 experts do not divide into n_group 3"},
		{"more groups used than exist", func(m *synthModel) { m.cfg["topk_group"] = 3 },
			"topk_group 3 of n_group 2"},
		{"no routed experts", func(m *synthModel) { delete(m.cfg, "n_routed_experts") },
			"n_routed_experts is 0"},
		// A scaling this architecture does not implement is still refused by
		// name: consuming yarn must not open the door to the rest.
		{"llama3 rope scaling", func(m *synthModel) {
			m.cfg["rope_scaling"] = map[string]any{"rope_type": "llama3", "factor": 32.0}
		}, `rope_scaling "llama3" is not implemented`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newSynthDeepseek()
			c.mut(m)
			_, err := m.source(t)
			if err == nil {
				t.Fatalf("accepted -- this gate proved nothing")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused for the wrong reason:\n got %v\nwant ...%s...", err, c.want)
			}
			t.Logf("refused: %v", err)
		})
	}
}

// TestDeepseekYarnIsRead is the positive control for the scaling the table
// above only refuses, case by case against transformers' arithmetic.
//
// cos and sin carry _compute_yarn_parameters' attention_factor: the stated one,
// else mscale(f, mscale)/mscale(f, mscale_all_dim) when both are set, else
// mscale(f, 1). The score carries yarn_apply_mscale's mscale(f,
// mscale_all_dim) squared, as YarnLogMul = 0.1*mscale_all_dim -- the GGUF key
// rope.scaling.yarn_log_multiplier's convention, so both inputs write one
// container.
func TestDeepseekYarnIsRead(t *testing.T) {
	ms := func(k float64) float32 { return float32(0.1*k*math.Log(40) + 1) }
	for _, c := range []struct {
		name         string
		extra        map[string]any
		attn, logMul float32
	}{
		// Every shipped DeepSeek: the rotary is untouched.
		{"mscale equals mscale_all_dim", map[string]any{"mscale": 0.707, "mscale_all_dim": 0.707}, 1, 0.1 * 0.707},
		{"mscale against mscale_all_dim", map[string]any{"mscale": 1.0, "mscale_all_dim": 0.707},
			float32(float64(ms(1)) / float64(ms(0.707))), 0.1 * 0.707},
		{"neither", map[string]any{}, ms(1), 0},
		{"mscale_all_dim alone", map[string]any{"mscale_all_dim": 0.707}, ms(1), 0.1 * 0.707},
		{"a stated attention_factor", map[string]any{"mscale": 0.707, "mscale_all_dim": 0.707,
			"attention_factor": 1.5}, 1.5, 0.1 * 0.707},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := newSynthDeepseek()
			rs := map[string]any{
				"rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 32,
				"beta_fast": 32.0, "beta_slow": 1.0,
			}
			for k, v := range c.extra {
				rs[k] = v
			}
			m.cfg["rope_scaling"] = rs
			s, err := m.source(t)
			if err != nil {
				t.Fatalf("a yarn deepseek was refused: %v", err)
			}
			cf := s.Config
			if cf.YarnFactor != 40 || cf.YarnOrigCtx != 32 || cf.YarnBetaFast != 32 || cf.YarnBetaSlow != 1 {
				t.Errorf("yarn %v/%v/%v/%v", cf.YarnFactor, cf.YarnOrigCtx, cf.YarnBetaFast, cf.YarnBetaSlow)
			}
			if cf.YarnLogMul != c.logMul {
				t.Errorf("YarnLogMul = %v, want %v", cf.YarnLogMul, c.logMul)
			}
			if math.Abs(float64(cf.AttnFactor-c.attn)) > 1e-6 {
				t.Errorf("AttnFactor = %v, want %v", cf.AttnFactor, c.attn)
			}
		})
	}
}

// TestDeepseekContainerRoundTrips writes the fixture as a real .jlm and reads
// it back, so the v23 config encoder and the 3-D absorbed entries are exercised
// by the file rather than only by the in-memory Source.
func TestDeepseekContainerRoundTrips(t *testing.T) {
	s, _ := deepseekSource(t)
	dst := filepath.Join(t.TempDir(), "deepseek.jlm")
	if _, err := jlm.Write(dst, s, jlm.Fingerprint{}); err != nil {
		t.Fatalf("jlm.Write: %v", err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatalf("jlm.Open: %v", err)
	}
	defer f.Close()
	c := f.Config()
	if c.Arch != jlm.ArchDeepseek2 || c.KVLoraRank != 16 || c.QLoraRank != 32 ||
		c.HeadDim != 24 || c.HeadDimV != 16 || c.NRot != 8 || c.NDenseLead != 1 ||
		c.NExpertGroup != 2 || c.NExpertGroupUsed != 1 || c.ExpertScale != 2.5 ||
		!c.Flags.Has(jlm.FlagExpertSigmoid) {
		t.Fatalf("config did not survive the round trip: %+v", c)
	}
	kb, vb := 0, 0
	for _, e := range f.Entries() {
		switch e.Role {
		case jlm.RoleAttnKB:
			kb++
			if e.NDim != 3 || e.Dims[2] != uint64(c.NHead) {
				t.Errorf("attn_k_b block %d is %dD %v, want one sheet per head",
					e.Block, e.NDim, e.Dims[:e.NDim])
			}
		case jlm.RoleAttnVB:
			vb++
		}
	}
	if kb != int(c.NLayer) || vb != int(c.NLayer) {
		t.Fatalf("%d attn_k_b and %d attn_v_b in a %d-block container", kb, vb, c.NLayer)
	}
}
