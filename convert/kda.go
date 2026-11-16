package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// k3LatentNormEps is the epsilon of the MLA latent norms (q_a_layernorm,
// kv_a_layernorm): Moonshot's KimiMLAAttention builds both as
// KimiRMSNorm(rank) at its default 1e-6, and transformers' KimiLinearAttention
// does the same; llama.cpp norms them at rms_norm_eps (RULE 7m, chosen: the
// model's own code).
const k3LatentNormEps = 1e-6

// Kimi K3 is llama.cpp's kimi-k3.cpp and Moonshot's modeling_kimi_linear.py
// (the text model of KimiK3ForConditionalGeneration), text only:
//
//	head_count_kv per layer, 0 on a KDA layer    -> LayerKinds (LayerLinearAttn
//	                                               there, LayerFullAttn else)
//	ssm.conv_kernel, kda.head_dim, head_count      -> SSM: KDA's heads are the
//	                                               attention's (llama.cpp's
//	                                               n_head_kda is n_head)
//	kda.gate_lower_bound                           -> DeltaDecayBounded: the decay
//	                                               lb*sigmoid(exp(A_log)*(f+dt))
//	                                               in place of -exp(A_log)*
//	                                               softplus(f+dt)
//	MLA with no rotary (mla_use_nope)              -> FlagNoPosEnc
//	the MLA latent norms' eps                      -> LatentNormEps (1e-6)
//
// The ranks, the residual-attention block, the experts' latent and the situ
// bounds, like the rest of this table, are latent_delta.json's container statement. The router is a sigmoid
// with the selection bias (llama.cpp's converter asserts it; KimiMoEGate would
// honour softmax, which no K3 states).

// kindsByKVHeads is each layer's kind off head_count_kv, which llama.cpp's
// converter writes per layer with zero on a KDA layer (its loader's own rule,
// is_recr = n_head_kv == 0): the layers_by_kv_heads statement. A file with no
// such layer, or only such layers, is no hybrid and refused.
func kindsByKVHeads(f *meta.File, n uint32) ([]jlm.LayerKind, error) {
	kv, ok := f.Key("attention.head_count_kv")
	if !ok {
		return nil, fmt.Errorf("no attention.head_count_kv: the layer kinds are read off it")
	}
	vs, err := kv.Int32s()
	if err != nil {
		return nil, fmt.Errorf("attention.head_count_kv: %w (one count per layer, zero on a KDA layer)", err)
	}
	if uint32(len(vs)) != n {
		return nil, fmt.Errorf("attention.head_count_kv has %d entries for %d layers", len(vs), n)
	}
	kinds := make([]jlm.LayerKind, n)
	linear := 0
	for i, v := range vs {
		kinds[i] = jlm.LayerFullAttn
		if v == 0 {
			kinds[i], linear = jlm.LayerLinearAttn, linear+1
		}
	}
	if linear == 0 || linear == len(vs) {
		return nil, fmt.Errorf("%d of %d layers are linear: the statement states a hybrid", linear, len(vs))
	}
	return kinds, nil
}

// kdaTensors finishes a source whose linear blocks store KDA's projections
// and convolutions apart (split_kda, llama.cpp's Kimi-K3) into the
// container's KDA layout: q, k and v joined into one mixed projection
// (RoleAttnQKV, q|k|v by rows) and their three convolutions into one filter
// bank ([taps, channels] with the channels in the same order), the output
// projection as RoleSSMOut, and the per-head decay rate -exp(A_log) (llama.cpp's
// converter folds it) repeated per channel, which is the rate KDA's gate reads.
func kdaTensors(c *Config, s *jlm.Source) error {
	if !c.Stated.splitKDA {
		return nil
	}
	type parts struct{ q, k, v, cq, ck, cv *jlm.Tensor }
	blocks := map[int32]*parts{}
	out := s.Tensors[:0:0]
	for i := range s.Tensors {
		e := s.Tensors[i]
		if e.Block < 0 || uint32(e.Block) >= c.NLayer || !c.LayerKinds[e.Block].Recurrent() {
			out = append(out, e)
			continue
		}
		p := blocks[e.Block]
		if p == nil {
			p = &parts{}
			blocks[e.Block] = p
		}
		x := e
		switch {
		case e.Role == jlm.RoleAttnQ:
			p.q = &x
		case e.Role == jlm.RoleAttnK:
			p.k = &x
		case e.Role == jlm.RoleAttnV:
			p.v = &x
		case e.Role == jlm.RoleSSMConv1d && strings.HasSuffix(e.Name, ".ssm_conv1d_q.weight"):
			p.cq = &x
		case e.Role == jlm.RoleSSMConv1d && strings.HasSuffix(e.Name, ".ssm_conv1d_k.weight"):
			p.ck = &x
		case e.Role == jlm.RoleSSMConv1d && strings.HasSuffix(e.Name, ".ssm_conv1d_v.weight"):
			p.cv = &x
		case e.Role == jlm.RoleAttnOut:
			x.Role = jlm.RoleSSMOut
			out = append(out, x)
		case e.Role == jlm.RoleSSMA:
			a, err := kdaExpandA(x, c)
			if err != nil {
				return err
			}
			out = append(out, a)
		default:
			out = append(out, e)
		}
	}
	for b := int32(0); b < int32(c.NLayer); b++ {
		if !c.LayerKinds[b].Recurrent() {
			continue
		}
		p := blocks[b]
		if p == nil || p.q == nil || p.k == nil || p.v == nil || p.cq == nil || p.ck == nil || p.cv == nil {
			return fmt.Errorf("convert: KDA block %d lacks one of attn_q/k/v or ssm_conv1d_q/k/v", b)
		}
		qkv, err := kdaJoinRows([]*jlm.Tensor{p.q, p.k, p.v}, jlm.RoleAttnQKV, uint64(c.NEmbd),
			uint64(c.SSM.Inner))
		if err != nil {
			return err
		}
		cv, err := kdaJoinConv([]*jlm.Tensor{p.cq, p.ck, p.cv}, uint64(c.SSM.ConvKernel), uint64(c.SSM.Inner))
		if err != nil {
			return err
		}
		out = append(out, qkv, cv)
	}
	s.Tensors = out
	return nil
}

// kdaJoinRows stacks matrices of one k by rows, each `rows` rows of k. Of one
// type, the bytes go back to back (GGUF's row-major blocks). Of several -- a
// mixed quantization such as Q4_K_M stores attn_q at Q4_K beside attn_k and
// attn_v at Q8_0 -- every part is dequantized and the whole re-stored as Q8_0
// (F32 where k is not whole Q8_0 blocks), as fuseBetaAlpha does for
// Qwen3.5's gates: a Q8_0 part comes back bit for bit (its block's largest
// code is 127, so the requantization finds the stored scale again), and a
// narrower part pays Q8_0's bytes and its rounding.
func kdaJoinRows(ts []*jlm.Tensor, role jlm.Role, k, rows uint64) (jlm.Tensor, error) {
	e := jlm.Tensor{Role: role, Block: ts[0].Block, Index: -1, Type: ts[0].Type, NDim: 2,
		Name: ts[0].Name + "(joined q|k|v)"}
	e.Dims[0], e.Dims[1] = k, rows*uint64(len(ts))
	mixed := false
	for _, t := range ts {
		if t.NDim != 2 || t.Dims[0] != k || t.Dims[1] != rows {
			return jlm.Tensor{}, fmt.Errorf("convert: %s is %v, want {%d, %d} beside %s",
				t.Name, t.Dims[:t.NDim], k, rows, ts[0].Name)
		}
		mixed = mixed || t.Type != e.Type
	}
	if !mixed {
		for _, t := range ts {
			b, err := t.Bytes()
			if err != nil {
				return jlm.Tensor{}, err
			}
			e.Data = append(e.Data, b...)
		}
		return e, nil
	}
	n := int(k * rows)
	vals := make([]float32, len(ts)*n)
	for i, t := range ts {
		src, ok := sourceType(t.Type)
		if !ok {
			return jlm.Tensor{}, fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
		}
		b, err := t.Bytes()
		if err != nil {
			return jlm.Tensor{}, err
		}
		if err := quant.Dequant32(src, b, vals[i*n:(i+1)*n]); err != nil {
			return jlm.Tensor{}, fmt.Errorf("convert: %s: %w", t.Name, err)
		}
	}
	if k%q8Elems == 0 {
		e.Type, e.Data = jlm.TypeQ8, quantizeQ8Rows(vals, int(k), len(ts)*int(rows))
		return e, nil
	}
	e.Type, e.Data = jlm.TypeF32, f32AsBytes(vals)
	return e, nil
}

// kdaJoinConv joins three causal convolutions, each {taps, 1, channels} (or
// {taps, 1, channels, 1}) of F32 with every channel's taps contiguous, into
// RoleSSMConv1d's {taps, 3*channels}: the same bytes back to back.
func kdaJoinConv(ts []*jlm.Tensor, taps, ch uint64) (jlm.Tensor, error) {
	e := jlm.Tensor{Role: jlm.RoleSSMConv1d, Block: ts[0].Block, Index: -1, Type: jlm.TypeF32, NDim: 2,
		Name: ts[0].Name + "(joined q|k|v)"}
	e.Dims[0], e.Dims[1] = taps, 3*ch
	for _, t := range ts {
		n := uint64(1)
		for _, d := range t.Dims[:t.NDim] {
			n *= d
		}
		if t.Type != jlm.TypeF32 || t.Dims[0] != taps || n != taps*ch {
			return jlm.Tensor{}, fmt.Errorf("convert: %s is %v of %v, want %d taps of %d channels in F32",
				t.Name, t.Dims[:t.NDim], t.Type, taps, ch)
		}
		b, err := t.Bytes()
		if err != nil {
			return jlm.Tensor{}, err
		}
		e.Data = append(e.Data, b...)
	}
	return e, nil
}

// kdaExpandA repeats each head's decay rate over its head_dim channels.
func kdaExpandA(e jlm.Tensor, c *Config) (jlm.Tensor, error) {
	heads, kd := int(c.SSM.NHeadV), int(c.SSM.StateSize)
	b, err := e.Bytes()
	if err != nil {
		return jlm.Tensor{}, err
	}
	if e.Type != jlm.TypeF32 || len(b) != 4*heads {
		return jlm.Tensor{}, fmt.Errorf("convert: %s is %d bytes of %v, want %d F32 rates, one per head",
			e.Name, len(b), e.Type, heads)
	}
	out := make([]byte, 0, 4*heads*kd)
	for h := 0; h < heads; h++ {
		w := b[4*h : 4*h+4]
		if a := math.Float32frombits(binary.LittleEndian.Uint32(w)); !(a < 0) {
			return jlm.Tensor{}, fmt.Errorf("convert: %s head %d has rate %g; it is -exp(A_log), negative",
				e.Name, h, a)
		}
		for i := 0; i < kd; i++ {
			out = append(out, w...)
		}
	}
	e.NDim, e.Dims, e.Data, e.Load = 1, [4]uint64{uint64(heads * kd)}, out, nil
	return e, nil
}
