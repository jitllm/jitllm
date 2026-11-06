package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

func gemma3nRoleOf(name string) (jlm.Role, bool) {
	switch name {
	case "altup_proj.weight":
		return jlm.RoleStreamExpand, true
	// altup_unembd_proj is one {n, n, AltUp-1} tensor in the file; gemma3nTensors
	// cuts it into one matrix per stream.
	case "altup_unembd_proj.weight":
		return jlm.RoleStreamCollapse1, true
	case "altup_router.weight":
		return jlm.RoleStreamRouter, true
	case "altup_router_norm.weight":
		return jlm.RoleStreamRouterNorm, true
	case "altup_predict_coef.weight":
		return jlm.RoleStreamPredictCoef, true
	case "altup_correct_coef.weight":
		return jlm.RoleStreamCorrectCoef, true
	case "altup_correct_scale.weight":
		return jlm.RoleStreamCorrectScale, true
	case "laurel_l.weight":
		return jlm.RoleLowRankIn, true
	case "laurel_r.weight":
		return jlm.RoleLowRankOut, true
	case "laurel_post_norm.weight":
		return jlm.RoleLowRankNorm, true
	}
	return 0, false
}

// gemma3nConfig is llama.cpp's gemma3n.cpp and transformers'
// Gemma3nForCausalLM, text only:
//
//	gemma3's attention: sqrt(n_embd) embedding,          -> EmbdScale, FlagGELU,
//	  GELU-tanh, per-head q/k RMSNorm, NEOX on the          FlagQKNorm, FlagRopeNeox
//	  whole head, post-attention and post-FFN norms,
//	  a final softcap                                    -> FinalSoftcap
//	sliding layers by attention.sliding_window_pattern  -> SWAWindow, SWAPeriod
//	  at their own base                                  -> RopeBaseSWA
//	Gemma 4's E-model pieces: a weightless v norm, an   -> Flag2VNorm, AttnScale 1,
//	  attention scale of one, per-layer embeddings, the    PLEDim, NKVShared
//	  last shared_kv_layers reading the KV of the last
//	  earlier layer of their own kind
//	AltUp's parallel residual streams                   -> AltUp (altup.num_inputs)
//	the gaussian top-k on the first FFNs' gates         -> NSparse, SparseStd
//	  (activation_sparsity_scale: the standard normal's
//	  quantile per layer, -inf where none)
//
// LAuReL and AltUp's per-block matrices are tensors (gemma3nTensors).
func gemma3nConfig(f *meta.File, c *jlm.Config) error {
	c.EmbdScale = float32(math.Sqrt(float64(c.NEmbd)))
	c.Flags |= jlm.FlagGELU | jlm.FlagRopeNeox | jlm.FlagQKNorm
	c.Flags2 |= jlm.Flag2VNorm
	c.AttnScale = 1
	c.FinalSoftcap = float32(f.FloatKey("final_logit_softcapping", 0))
	c.NKVShared = uint32(f.UintKey("attention.shared_kv_layers", 0))
	if c.NKVShared >= c.NLayer {
		return fmt.Errorf("%d KV-sharing layers of %d", c.NKVShared, c.NLayer)
	}
	c.PLEDim = uint32(f.UintKey("embedding_length_per_layer_input", 0))
	if c.PLEDim == 0 {
		return fmt.Errorf("no embedding_length_per_layer_input: gemma3n gates a per-layer input into every block")
	}
	if _, ok := f.Get("per_layer_token_embd.weight"); !ok {
		return fmt.Errorf("per-layer embeddings %d wide and no per_layer_token_embd", c.PLEDim)
	}
	if c.NFFN == 0 {
		kv, ok := f.Key("feed_forward_length")
		if !ok {
			return fmt.Errorf("no feed_forward_length")
		}
		ws, err := kv.Int32s()
		if err != nil || len(ws) != int(c.NLayer) || ws[0] <= 0 {
			return fmt.Errorf("feed_forward_length: %d values for %d layers (%v)", len(ws), c.NLayer, err)
		}
		for i, w := range ws {
			if w != ws[0] {
				return fmt.Errorf("feed_forward_length %v: layer %d differs from layer 0: %w", ws, i, ErrNotImplemented)
			}
		}
		c.NFFN = uint32(ws[0])
	}
	c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0))
	if c.SWAWindow == 0 {
		return fmt.Errorf("no attention.sliding_window")
	}
	if err := swaPatternOf(f, c, 0); err != nil {
		return err
	}
	if c.SWAPeriod == 0 || c.SWAPeriod > c.NLayer {
		return fmt.Errorf("no sliding pattern with a global layer in it (period %d of %d layers)",
			c.SWAPeriod, c.NLayer)
	}
	c.RopeBaseSWA = float32(f.FloatKey("rope.freq_base_swa", 10000))
	if c.NRot != c.HeadDim || c.HeadDimV != 0 {
		return fmt.Errorf("rotary %d and value width %d on a %d-wide head", c.NRot, c.HeadDimV, c.HeadDim)
	}
	c.PredStreams = uint32(f.UintKey("altup.num_inputs", 0))
	if c.PredStreams < 2 {
		return fmt.Errorf("altup.num_inputs %d: gemma3n runs two or more streams", c.PredStreams)
	}
	// The active stream is the first in every published checkpoint and in
	// both references' defaults; another index would only relabel streams, and
	// is refused rather than relabelled.
	if a := f.UintKey("altup.active_idx", 0); a != 0 {
		return fmt.Errorf("altup.active_idx %d: %w", a, ErrNotImplemented)
	}
	return sparsityOf(f, c)
}

// sparsityOf reads activation_sparsity_scale: the standard normal's quantile
// at each layer's target sparsity, which transformers computes as
// Normal(0,1).icdf(p) and llama.cpp's converter writes per layer, -inf where a
// layer has none (icdf(0)). The published pattern is one value on a leading
// run of layers, which is what the container holds; any other shape is
// refused rather than approximated.
func sparsityOf(f *meta.File, c *jlm.Config) error {
	kv, ok := f.Key("activation_sparsity_scale")
	if !ok {
		return nil
	}
	vs, err := kv.Float32s()
	if err != nil || len(vs) != int(c.NLayer) {
		return fmt.Errorf("activation_sparsity_scale: %d values for %d layers (%v)", len(vs), c.NLayer, err)
	}
	n := 0
	for n < len(vs) && !math.IsInf(float64(vs[n]), -1) {
		if vs[n] != vs[0] || math.IsNaN(float64(vs[n])) {
			return fmt.Errorf("activation_sparsity_scale %v: layer %d differs from layer 0: %w", vs, n, ErrNotImplemented)
		}
		n++
	}
	for i := n; i < len(vs); i++ {
		if !math.IsInf(float64(vs[i]), -1) {
			return fmt.Errorf("activation_sparsity_scale %v: layer %d is sparse after a dense one: %w",
				vs, i, ErrNotImplemented)
		}
	}
	c.NSparse = uint32(n)
	if n > 0 {
		c.SparseStd = vs[0]
	}
	return nil
}

// gemma3nTensors finishes Gemma 3n's tensors. The unembedding's AltUp-1
// matrices, stacked in one tensor in the file, become one tensor per stream;
// each block's prediction and correction coefficients are read as float
// matrices the router's weights multiply (stateAltUp).
func gemma3nTensors(s *jlm.Source) error {
	c := s.Config
	if c == nil || c.Arch != jlm.ArchGemma3n {
		return nil
	}
	n, a := uint64(c.NEmbd), uint64(c.PredStreams)
	out := s.Tensors[:0:0]
	for _, t := range s.Tensors {
		switch {
		case t.Block >= 0 && c.KVShared(int(t.Block)) &&
			(t.Role == jlm.RoleAttnK || t.Role == jlm.RoleAttnV || t.Role == jlm.RoleAttnKNorm):
			// A KV-sharing block computes no k or v (Gemma 4's rule).
			continue
		case t.Role == jlm.RoleStreamCollapse1 && t.Block < 0:
			if t.NDim != 3 || t.Dims[0] != n || t.Dims[1] != n || t.Dims[2] != a-1 {
				return fmt.Errorf("convert: %s is %v, want {%d, %d, %d}", t.Name, t.Dims[:t.NDim], n, n, a-1)
			}
			if a-1 > 3 {
				return fmt.Errorf("convert: %d AltUp streams, and the container names three unembeddings: %w",
					a, ErrNotImplemented)
			}
			b, err := t.Bytes()
			if err != nil {
				return err
			}
			sheet := uint64(len(b)) / (a - 1)
			for k := uint64(0); k < a-1; k++ {
				u := t
				u.Role = jlm.RoleStreamCollapse1 + jlm.Role(k)
				u.NDim, u.Dims = 2, [4]uint64{n, n}
				u.Data, u.Load = b[k*sheet:(k+1)*sheet], nil
				u.Name = fmt.Sprintf("%s/%d", t.Name, k+1)
				out = append(out, u)
			}
			continue
		case t.Role == jlm.RoleStreamExpand:
			if t.NDim != 3 || t.Dims[0] != n || t.Dims[1] != n || t.Dims[2] != a-1 {
				return fmt.Errorf("convert: %s is %v, want {%d, %d, %d}", t.Name, t.Dims[:t.NDim], n, n, a-1)
			}
			// The AltUp-1 projections stacked by rows: one matrix of
			// (AltUp-1)*n rows, so the expansion is one matvec.
			t.NDim, t.Dims = 2, [4]uint64{n, n * (a - 1)}
		case t.Role == jlm.RoleStreamPredictCoef, t.Role == jlm.RoleStreamCorrectCoef:
			cols := a // the correction's out width: one coefficient per stream
			if t.Role == jlm.RoleStreamPredictCoef {
				cols = a * a
			}
			if err := widenF32(&t); err != nil {
				return err
			}
			if t.NDim != 2 || t.Dims[0] != a || t.Dims[1] != cols {
				return fmt.Errorf("convert: %s is %v, want {%d, %d}", t.Name, t.Dims[:t.NDim], a, cols)
			}
		case t.Role == jlm.RolePLETokEmbd && t.NDim == 2:
			// llama.cpp pads the per-layer table from its own vocabulary to
			// the model's with zero rows (its "vision/audio token slots");
			// transformers reads row 0 for every id past the table
			// (Gemma3nModel's per_layer_inputs_mask), so the padding is cut
			// and the reader maps such an id to row 0.
			if err := trimZeroRows(&t); err != nil {
				return err
			}
		case t.Role == jlm.RoleStreamRouter:
			// Read as plain floats on every tier: a published GGUF stores it
			// F16, and a matrix of four rows gains nothing from staying narrow.
			if err := widenF32(&t); err != nil {
				return err
			}
			if t.Dims[0] != n || t.Dims[1] != a {
				return fmt.Errorf("convert: %s is %v, want {%d, %d}", t.Name, t.Dims[:t.NDim], n, a)
			}
		}
		out = append(out, t)
	}
	s.Tensors = out
	return nil
}

// widenF32 rewrites a small float tensor (F16, BF16) as F32.
func widenF32(t *jlm.Tensor) error {
	if t.Type == jlm.TypeF32 {
		return nil
	}
	src, ok := sourceType(t.Type)
	if !ok || (src != quant.F16 && src != quant.BF16) {
		return fmt.Errorf("convert: %s is %v, want a float tensor", t.Name, t.Type)
	}
	b, err := t.Bytes()
	if err != nil {
		return err
	}
	f := make([]float32, len(b)/2)
	if err := quant.Dequant32(src, b, f); err != nil {
		return fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	d := make([]byte, 4*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint32(d[4*i:], math.Float32bits(v))
	}
	t.Type, t.Data, t.Load = jlm.TypeF32, d, nil
	return nil
}

// trimZeroRows cuts a 2-D table's trailing rows whose bytes are all zero, the
// padding a converter appended. A row of real weights is never all zero; a
// zero row quantizes to zero bytes in every format.
func trimZeroRows(t *jlm.Tensor) error {
	b, err := t.Bytes()
	if err != nil {
		return err
	}
	rows := int(t.Dims[1])
	if rows == 0 || len(b)%rows != 0 {
		return fmt.Errorf("convert: %s is %d bytes over %d rows", t.Name, len(b), rows)
	}
	rb := len(b) / rows
	keep := rows
	for keep > 1 {
		zero := true
		for _, c := range b[(keep-1)*rb : keep*rb] {
			if c != 0 {
				zero = false
				break
			}
		}
		if !zero {
			break
		}
		keep--
	}
	if keep == rows {
		return nil
	}
	t.Data, t.Load = b[:keep*rb], nil
	t.Dims[1] = uint64(keep)
	return nil
}
