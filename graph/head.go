package graph

import "github.com/jitllm/jitllm/format/jlm"

// Head is the model's head as ops: the row's norm, its projection to the
// vocabulary with the projection's stored bias, then the final softcap and
// the logit scale, in that order over the logits.
type Head struct {
	NEmbd, NVocab int
	// Eps is the norm's epsilon.
	Eps float64
	// LayerNorm makes the norm a LayerNorm, with its stored bias where
	// NormBias is set; otherwise it is an RMSNorm.
	LayerNorm, NormBias bool
	// Output is the projection's role (the embedding's where the model ties
	// them), and Bias says its bias is stored.
	Output jlm.Role
	Bias   bool
	// Softcap is c of c*tanh(x/c), 0 for none. Scale is the divisor of the
	// logits, 0 or 1 for none.
	Softcap, Scale float64
	// Hidden makes the norm's rows the graph's second output, after the
	// logits (the input's rows, with no norm): what a step reading the
	// trunk's normed rows back reads.
	Hidden bool
	// NoNorm projects the row as it comes, where the model has no output
	// norm (a mix of its streams stands in for it).
	NoNorm bool
	// Pick, where set, projects only the rows of Pick ids, the graph's
	// second input: the logits at those ids, a decision's labels, in place of
	// the vocabulary's.
	Pick int
}

// Build is the head's graph.
func (h *Head) Build() *Graph {
	b := NewBlock(-1)
	x := b.In("x", h.NEmbd)
	width := h.NVocab
	var ids ValueID
	if h.Pick > 0 {
		ids, width = b.In("pick.ids", h.Pick), h.Pick
		b.G.Values[ids].Elem = ElemU32
	}
	n := x
	switch {
	case h.NoNorm:
	case h.LayerNorm:
		w := []Binding{{Role: jlm.RoleOutputNorm, K: h.NEmbd}}
		if h.NormBias {
			w = append(w, Binding{Role: jlm.RoleOutputNormBias, K: h.NEmbd})
		}
		n = b.Emit(OpLayerNorm, []ValueID{x}, []Result{{Name: jlm.RoleOutputNorm.String(), Width: h.NEmbd}},
			w, Attrs{Eps: h.Eps}, nil, nil)[0]
	default:
		n = b.Norm(x, jlm.RoleOutputNorm, h.Eps)
	}
	var bias []jlm.Role
	if h.Bias {
		bias = []jlm.Role{jlm.RoleOutputBias}
	}
	var l ValueID
	if h.Pick > 0 {
		w := []Binding{{Role: h.Output, K: h.NEmbd, Rows: h.NVocab}}
		for _, r := range bias {
			w = append(w, Binding{Role: r, K: h.NVocab})
		}
		l = b.Emit(OpMatVec, []ValueID{n, ids}, []Result{{Name: "picked", Width: h.Pick}}, w, Attrs{}, nil, nil)[0]
	} else {
		l = b.MatVec(n, h.Output, h.NVocab, bias...)
	}
	op := func(name string, fn MapFn, k float64, in ...ValueID) ValueID {
		return b.Emit(OpMap, in, []Result{{Name: name, Width: width}}, nil, Attrs{Fn: fn, Const: k}, nil, nil)[0]
	}
	if s := h.Softcap; s != 0 {
		// s*tanh(l/s), tanh(y) as 2*sigmoid(2y) - 1: finite at both ends,
		// where an exponential of 2y overflows past |y| of 44.
		sg := op("softcap.sigmoid", MapSigmoid, 0, op("softcap.arg", MapMul, 2/s, l))
		t := op("softcap.tanh", MapSub, 1, op("softcap.twice", MapMul, 2, sg))
		l = op("capped", MapMul, s, t)
	}
	if s := h.Scale; s != 0 && s != 1 {
		l = op("scaled", MapMul, 1/s, l)
	}
	b.Out(l)
	if h.Hidden {
		// With no norm the rows are the input's, copied: an output the
		// graph writes (Copy).
		if h.NoNorm {
			n = b.Emit(OpSlice, []ValueID{x}, []Result{{Name: "hidden", Width: h.NEmbd}}, nil, Attrs{Width: h.NEmbd}, nil, nil)[0]
		}
		b.Out(n)
	}
	return b.G
}
