package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// roleOf is the source format's tensor-name vocabulary, and the only place it
// appears: a container identifies a tensor by (Role, Block, Index), so this
// table is read once per conversion. An unrecognised name is a refusal, not a
// skip, because a tensor dropped at conversion is a model that loads and is
// wrong.
func roleOf(name string) (jlm.Role, bool) {
	if r, ok := coreRoleOf(name); ok {
		return r, true
	}
	for _, group := range roleGroups() {
		if r, ok := group(name); ok {
			return r, true
		}
	}
	return 0, false
}

// roleGroups are the tables beside the core one, one per family of
// architectures. A name in two of them would be read as the first's role;
// TestRoleNamesAreDisjoint holds them apart.
func roleGroups() []func(string) (jlm.Role, bool) {
	return []func(string) (jlm.Role, bool){
		clefRoleOf, deepseek4RoleOf, flagshipRoleOf, gemma3nRoleOf, gemma3nvRoleOf,
		kimiK3RoleOf, qwen4expRoleOf, visionFamRoleOf,
	}
}

func coreRoleOf(name string) (jlm.Role, bool) {
	switch name {
	case "token_embd.weight":
		return jlm.RoleTokenEmbd, true
	case "output.weight":
		return jlm.RoleOutput, true
	case "output_norm.weight":
		return jlm.RoleOutputNorm, true
	case "output_norm.bias":
		return jlm.RoleOutputNormBias, true
	case "output.bias":
		return jlm.RoleOutputBias, true
	case "rope_factors_long.weight":
		return jlm.RoleRopeFactorsLong, true
	case "rope_factors_short.weight":
		return jlm.RoleRopeFactorsShort, true
	case "rope_freqs.weight":
		return jlm.RoleRopeFreqs, true
	case "token_types.weight":
		return jlm.RoleTokenTypes, true
	case "position_embd.weight":
		return jlm.RolePosEmbd, true
	case "token_embd_norm.weight":
		return jlm.RoleTokenEmbdNorm, true
	case "token_embd_norm.bias":
		return jlm.RoleTokenEmbdNormBias, true
	case "cls.norm.weight":
		return jlm.RoleScorerNorm, true
	case "cls.norm.bias":
		return jlm.RoleScorerNormBias, true
	case "cls.weight":
		return jlm.RoleScorer, true
	case "cls.bias":
		return jlm.RoleScorerBias, true
	case "cls.output.weight":
		return jlm.RoleScorerOut, true
	case "cls.output.bias":
		return jlm.RoleScorerOutBias, true
	case "decision.scorer.weight":
		return jlm.RoleScorer, true
	case "decision.scorer.bias":
		return jlm.RoleScorerBias, true
	case "decision.scorer_out.weight":
		return jlm.RoleScorerOut, true
	case "decision.scorer_out.bias":
		return jlm.RoleScorerOutBias, true
	case "dense_2.weight":
		return jlm.RoleEmbdDense1, true
	case "dense_3.weight":
		return jlm.RoleEmbdDense2, true
	case "dense.0.weight":
		return jlm.RoleEmbdDense1, true
	case "dense.1.weight":
		return jlm.RoleEmbdDense2, true
	case "attn_norm.weight":
		return jlm.RoleAttnNorm, true
	case "attn_q.weight":
		return jlm.RoleAttnQ, true
	case "attn_k.weight":
		return jlm.RoleAttnK, true
	case "attn_v.weight":
		return jlm.RoleAttnV, true
	case "attn_output.weight":
		return jlm.RoleAttnOut, true
	case "attn_qkv.weight":
		return jlm.RoleAttnQKV, true
	case "attn_qkv.bias":
		return jlm.RoleAttnQKVBias, true
	case "attn_norm.bias":
		return jlm.RoleAttnNormBias, true
	case "attn_norm_2.weight":
		return jlm.RoleAttnNorm2, true
	case "attn_norm_2.bias":
		return jlm.RoleAttnNorm2Bias, true
	case "ffn_norm.bias":
		return jlm.RoleFFNNormBias, true
	case "attn_q.bias":
		return jlm.RoleAttnQBias, true
	case "attn_k.bias":
		return jlm.RoleAttnKBias, true
	case "attn_v.bias":
		return jlm.RoleAttnVBias, true
	case "attn_output.bias":
		return jlm.RoleAttnOutBias, true
	case "attn_q_norm.weight":
		return jlm.RoleAttnQNorm, true
	case "attn_k_norm.weight":
		return jlm.RoleAttnKNorm, true
	case "attn_gate.weight":
		return jlm.RoleAttnGate, true
	case "attn_sinks.weight":
		return jlm.RoleAttnSinks, true
	case "post_attention_norm.weight":
		return jlm.RolePostAttnNorm, true
	case "post_ffw_norm.weight":
		return jlm.RolePostFFNNorm, true
	case "ffn_norm.weight":
		return jlm.RoleFFNNorm, true
	case "ffn_gate.weight":
		return jlm.RoleFFNGate, true
	case "ffn_up.weight":
		return jlm.RoleFFNUp, true
	case "ffn_down.weight":
		return jlm.RoleFFNDown, true
	case "attn_sub_norm.weight":
		return jlm.RoleAttnSubNorm, true
	case "ffn_sub_norm.weight":
		return jlm.RoleFFNSubNorm, true
	case "attn_output_norm.weight":
		return jlm.RoleAttnOutNorm, true
	case "attn_output_norm.bias":
		return jlm.RoleAttnOutNormBias, true
	case "layer_output_norm.weight":
		return jlm.RoleLayerOutNorm, true
	case "layer_output_norm.bias":
		return jlm.RoleLayerOutNormBias, true
	case "ffn_up.bias":
		return jlm.RoleFFNUpBias, true
	case "ffn_down.bias":
		return jlm.RoleFFNDownBias, true
	case "ffn_gate_inp.weight":
		return jlm.RoleRouter, true
	case "ffn_gate_inp.bias":
		return jlm.RoleRouterBias, true
	case "exp_probs_b.bias":
		return jlm.RoleExpProbsB, true
	case "ffn_gate_exps.bias":
		return jlm.RoleExpGateBias, true
	case "ffn_up_exps.bias":
		return jlm.RoleExpUpBias, true
	case "ffn_down_exps.bias":
		return jlm.RoleExpDownBias, true
	case "ffn_gate_exps.weight":
		return jlm.RoleExpGateBank, true
	case "ffn_up_exps.weight":
		return jlm.RoleExpUpBank, true
	case "ffn_down_exps.weight":
		return jlm.RoleExpDownBank, true
	case "attn_q_a.weight":
		return jlm.RoleAttnQA, true
	case "attn_q_a_norm.weight":
		return jlm.RoleAttnQANorm, true
	case "attn_q_b.weight":
		return jlm.RoleAttnQB, true
	case "attn_kv_a_mqa.weight":
		return jlm.RoleAttnKVA, true
	case "attn_kv_a_norm.weight":
		return jlm.RoleAttnKVANorm, true
	case "attn_kv_b.weight":
		return jlm.RoleAttnKVB, true
	case "attn_k_b.weight":
		return jlm.RoleAttnKB, true
	case "attn_v_b.weight":
		return jlm.RoleAttnVB, true
	case "nextn.eh_proj.weight":
		return jlm.RoleNextnEHProj, true
	case "nextn.enorm.weight":
		return jlm.RoleNextnENorm, true
	case "nextn.hnorm.weight":
		return jlm.RoleNextnHNorm, true
	case "nextn.shared_head_norm.weight":
		return jlm.RoleNextnHeadNorm, true
	case "nextn.shared_head_head.weight":
		return jlm.RoleNextnHead, true
	case "nextn.embed_tokens.weight":
		return jlm.RoleNextnEmbd, true
	case "ffn_gate_inp_shexp.weight":
		return jlm.RoleShRouter, true
	case "ffn_gate_shexp.weight":
		return jlm.RoleShExpGate, true
	case "ffn_up_shexp.weight":
		return jlm.RoleShExpUp, true
	case "ffn_down_shexp.weight":
		return jlm.RoleShExpDown, true
	case "ssm_ba.weight":
		return jlm.RoleSSMBA, true
	case "ssm_conv1d.weight":
		return jlm.RoleSSMConv1d, true
	case "ssm_a":
		return jlm.RoleSSMA, true
	case "ssm_dt.bias":
		return jlm.RoleSSMDtBias, true
	case "ssm_norm.weight":
		return jlm.RoleSSMNorm, true
	case "ssm_out.weight":
		return jlm.RoleSSMOut, true
	case "ssm_in.weight":
		return jlm.RoleSSMInProj, true
	case "ssm_d":
		return jlm.RoleSSMD, true
	case "ssm_conv1d.bias":
		return jlm.RoleSSMConvBias, true
	case "ssm_x.weight":
		return jlm.RoleSSMXDt, true
	case "ssm_dt.weight":
		return jlm.RoleSSMDtProj, true
	case "ssm_dt_norm.weight":
		return jlm.RoleSSMDtNorm, true
	case "ssm_b_norm.weight":
		return jlm.RoleSSMBNorm, true
	case "ssm_c_norm.weight":
		return jlm.RoleSSMCNorm, true
	case "ffn_norm":
		return jlm.RoleFFNNorm, true
	case "shortconv.in_proj.weight":
		return jlm.RoleSSMInProj, true
	case "shortconv.conv.weight":
		return jlm.RoleSSMConv1d, true
	case "shortconv.out_proj.weight":
		return jlm.RoleSSMOut, true
	case "v.patch_embd.weight":
		return jlm.RoleVPatchEmbd, true
	case "v.patch_embd.bias":
		return jlm.RoleVPatchBias, true
	case "v.position_embd.weight":
		return jlm.RoleVPosEmbd, true
	case "v.post_ln.weight":
		return jlm.RoleVPostNorm, true
	case "v.post_ln.bias":
		return jlm.RoleVPostNormBias, true
	case "mm.model.fc.weight":
		return jlm.RoleVProj, true
	case "mm.model.fc.bias":
		return jlm.RoleVProjBias, true
	case "mm.input_projection.weight":
		return jlm.RoleVProj, true
	case "mm.soft_emb_norm.weight":
		return jlm.RoleVProjNorm, true
	case "v.class_embd":
		return jlm.RoleVClassEmbd, true
	case "v.pre_ln.weight":
		return jlm.RoleVPreNorm, true
	case "v.pre_ln.bias":
		return jlm.RoleVPreNormBias, true
	case "mm.0.weight":
		return jlm.RoleVProj, true
	case "mm.0.bias":
		return jlm.RoleVProjBias, true
	case "mm.2.weight":
		return jlm.RoleVProj2, true
	case "mm.2.bias":
		return jlm.RoleVProj2Bias, true
	case "v.ln1.weight":
		return jlm.RoleVAttnNorm, true
	case "v.ln1.bias":
		return jlm.RoleVAttnNormBias, true
	case "v.ln2.weight":
		return jlm.RoleVFFNNorm, true
	case "v.ln2.bias":
		return jlm.RoleVFFNNormBias, true
	case "v.attn_q.weight":
		return jlm.RoleVAttnQ, true
	case "v.attn_k.weight":
		return jlm.RoleVAttnK, true
	case "v.attn_v.weight":
		return jlm.RoleVAttnV, true
	case "v.attn_out.weight":
		return jlm.RoleVAttnOut, true
	case "v.attn_q.bias":
		return jlm.RoleVAttnQBias, true
	case "v.attn_k.bias":
		return jlm.RoleVAttnKBias, true
	case "v.attn_v.bias":
		return jlm.RoleVAttnVBias, true
	case "v.attn_out.bias":
		return jlm.RoleVAttnOutBias, true
	case "v.ffn_up.weight":
		return jlm.RoleVFC1, true
	case "v.ffn_up.bias":
		return jlm.RoleVFC1Bias, true
	case "v.ffn_down.weight":
		return jlm.RoleVFC2, true
	case "v.ffn_down.bias":
		return jlm.RoleVFC2Bias, true
	case "v.ffn_gate.weight":
		return jlm.RoleVFFNGate, true
	case "v.ffn_gate.bias":
		return jlm.RoleVFFNGateBias, true
	case "resampler.query":
		return jlm.RoleVRsQuery, true
	case "resampler.kv.weight":
		return jlm.RoleVRsKV, true
	case "resampler.ln_q.weight":
		return jlm.RoleVRsLnQ, true
	case "resampler.ln_q.bias":
		return jlm.RoleVRsLnQBias, true
	case "resampler.ln_kv.weight":
		return jlm.RoleVRsLnKV, true
	case "resampler.ln_kv.bias":
		return jlm.RoleVRsLnKVBias, true
	case "resampler.attn.q.weight":
		return jlm.RoleVRsQ, true
	case "resampler.attn.q.bias":
		return jlm.RoleVRsQBias, true
	case "resampler.attn.k.weight":
		return jlm.RoleVRsK, true
	case "resampler.attn.k.bias":
		return jlm.RoleVRsKBias, true
	case "resampler.attn.v.weight":
		return jlm.RoleVRsV, true
	case "resampler.attn.v.bias":
		return jlm.RoleVRsVBias, true
	case "resampler.attn.out.weight":
		return jlm.RoleVRsOut, true
	case "resampler.attn.out.bias":
		return jlm.RoleVRsOutBias, true
	case "resampler.ln_post.weight":
		return jlm.RoleVRsLnPost, true
	case "resampler.ln_post.bias":
		return jlm.RoleVRsLnPostBias, true
	case "resampler.proj.weight":
		return jlm.RoleVRsProj, true
	}
	return 0, false
}

// towerName is a tower tensor's name in the vocabulary identify reads, where
// a projector spells one differently: Janus-Pro's second aligner matrix is
// mm.1, the graph position llava's mm.2 holds. v is nil for a text file.
func towerName(v *jlm.Vision, name string) string {
	if v != nil && v.Projector == jlm.ProjJanus && strings.HasPrefix(name, "mm.1.") {
		return "mm.2." + strings.TrimPrefix(name, "mm.1.")
	}
	// Llama 4's post_attention_layernorm is its pre-MLP norm (ln2), and
	// llama.cpp's tensor map names it attn_post_norm for a transformers 5
	// checkpoint (vision_model.model.layers.N.post_attention_layernorm,
	// which it lists for Gemma 4's genuine post-attention norm).
	if v != nil && v.Projector == jlm.ProjLlama4 && strings.Contains(name, ".attn_post_norm.") {
		return strings.Replace(name, ".attn_post_norm.", ".ln2.", 1)
	}
	// Kimi-VL's and LFM2-VL's are mm.1 and mm.2: llava's mm.0 and mm.2
	// positions.
	if v != nil && (v.Projector == jlm.ProjKimiVL || v.Projector == jlm.ProjLFM2) && strings.HasPrefix(name, "mm.1.") {
		return "mm.0." + strings.TrimPrefix(name, "mm.1.")
	}
	return name
}

// unreadTower names the tower tensors a file carries and no graph reads, each
// with the reason. Every other unknown name is refused (identify): a tensor
// dropped at conversion is a model that loads and is wrong, so the exceptions
// are a list, never a pattern.
func unreadTower(name string) (string, bool) {
	switch name {
	case "resampler.pos_embed_k":
		return "the resampler's position is computed per grid", true
	}
	return "", false
}

// indexedRole is the per-expert form: "ffn_gate.3.weight" is expert 3's gate.
func indexedRole(name string) (jlm.Role, bool) {
	switch name {
	case "ffn_gate":
		return jlm.RoleExpGate, true
	case "ffn_up":
		return jlm.RoleExpUp, true
	case "ffn_down":
		return jlm.RoleExpDown, true
	}
	return 0, false
}

// retarget maps a role whose name means different things in different
// architectures onto what the tensor does. gemma2/gemma3's post_attention_norm
// is applied to the attention output; qwen3next's, gpt-oss's and glm4moe's tensor of that
// name is the FFN's input norm (ffn_norm by position, which they do not ship
// under that name). Resolving it here keeps the source's ambiguity out of the
// container.
func retarget(arch jlm.Arch, role jlm.Role) jlm.Role {
	if (arch == jlm.ArchQwen3Next || arch == jlm.ArchGPTOSS || arch == jlm.ArchGLM4MoE || arch == jlm.ArchGLM4VMoE ||
		arch == jlm.ArchSeedOSS) &&
		role == jlm.RolePostAttnNorm {
		return jlm.RoleFFNNorm
	}
	// DBRX's attn_output_norm is its FFN pre-norm (llama.cpp's dbrx.cpp),
	// where BERT's is applied after the residual add.
	if arch == jlm.ArchDBRX && role == jlm.RoleAttnOutNorm {
		return jlm.RoleFFNNorm
	}
	// LFM2's final norm is named token_embd_norm (llama.cpp's
	// LLM_TENSOR_OUTPUT_NORM_LFM2, "fix for wrong tensor name"); it is applied
	// after the last block, not to the embeddings.
	if (arch == jlm.ArchLFM2 || arch == jlm.ArchLFM2Enc) && role == jlm.RoleTokenEmbdNorm {
		return jlm.RoleOutputNorm
	}
	// Qwen3.8-Flash-Next's n-gram table carries Gemma's per-layer table name
	// and is another object: a row table of hashed n-grams.
	if arch == jlm.ArchQwen4Exp && role == jlm.RolePLETokEmbd {
		return jlm.RoleNGramTable
	}
	return role
}

// identify turns a source tensor name into the triple a container stores.
// Index is -1 unless the name carries one.
func identify(name string) (role jlm.Role, block int32, index int32, err error) {
	block, index = jlm.DenseBlock, -1
	rest := name
	// "v." marks the vision tower; its blocks are "v.blk.N.".
	vision := strings.HasPrefix(rest, "v.")
	// A decision head's blocks are "dec.blk.N." (Clef's): model-level tensors
	// with the head block in Index, so they never meet a backbone block.
	if r, ok := strings.CutPrefix(rest, "dec.blk."); ok {
		i := strings.IndexByte(r, '.')
		n, e := strconv.ParseInt(r[:max(i, 0)], 10, 32)
		if i <= 0 || e != nil || n < 0 {
			return 0, 0, 0, fmt.Errorf("convert: %q: no head block index", name)
		}
		if role, ok := roleOf("dec." + r[i+1:]); ok {
			return role, jlm.DenseBlock, int32(n), nil
		}
		return 0, 0, 0, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary", name)
	}
	for _, p := range []string{"v.blk.", "blk."} {
		if strings.HasPrefix(rest, p) {
			r := rest[len(p):]
			i := strings.IndexByte(r, '.')
			if i <= 0 {
				return 0, 0, 0, fmt.Errorf("convert: %q: no block index", name)
			}
			n, e := strconv.Atoi(r[:i])
			if e != nil || n < 0 {
				return 0, 0, 0, fmt.Errorf("convert: %q: block index %q", name, r[:i])
			}
			block, rest = int32(n), r[i+1:]
			if vision {
				rest = "v." + rest
			}
			break
		}
	}
	if r, ok := roleOf(rest); ok {
		return r, block, index, nil
	}
	// The second temporal plane of Qwen2-VL's conv3d patch embedding, which the
	// GGUF spells as a second 4-D tensor. It is carried as index 1, folded into
	// index -1 by prepareTower (the sum a still image reads), and kept as
	// itself for a video's second frame (foldPatchPlanes).
	if rest == "v.patch_embd.weight.1" {
		return jlm.RoleVPatchEmbd, block, 1, nil
	}
	// The indexed per-expert form.
	if f := strings.Split(rest, "."); len(f) == 3 && f[2] == "weight" {
		if r, ok := indexedRole(f[0]); ok {
			n, e := strconv.Atoi(f[1])
			if e == nil && n >= 0 {
				return r, block, int32(n), nil
			}
		}
	}
	return 0, 0, 0, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
		"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
}

// typeOf maps a source weight type onto the container's own code.
//
// A type the container does not define is a refusal rather than a stored
// source number.
func typeOf(t quant.Type) (jlm.Type, error) {
	ty, ok := jlm.TypeOf(t)
	if !ok {
		return 0, fmt.Errorf("convert: weight type %s has no code in this container's vocabulary", t)
	}
	return ty, nil
}

// sourceType is typeOf's inverse, for the gate below.
func sourceType(t jlm.Type) (quant.Type, bool) { return jlm.SourceType(t) }
