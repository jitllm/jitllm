package graph

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// Decompose is g with every composite op it can state as primitives replaced
// by them (docs/design/composer.md, "Decomposition"), written once from the
// op's attributes and bindings and never from a family's name. An op it
// cannot yet state stays as it is; the program counts those
// (TestCompositeKindsOnlyShrink) and each falls as its decomposition lands.
//
// A route becomes its selection's indices (the route's own output value,
// now an index row) and its weights, a value of their own, which the
// selection's combine (and a gather weighting its pairs' inputs) reads as one
// more input.
func Decompose(g *Graph) *Graph {
	d := &Graph{Version: g.Version, Block: g.Block, Values: append([]Value(nil), g.Values...),
		In: append([]ValueID(nil), g.In...), Out: append([]ValueID(nil), g.Out...)}
	weights := map[ValueID]ValueID{}
	for _, o := range g.Ops {
		if o.Kind == OpRoute {
			if w, ok := d.route(o); ok {
				weights[o.Out[0]] = w
				continue
			}
		}
		switch o.Kind {
		case OpStreamMix:
			if d.streamMix(o) {
				continue
			}
		case OpStreamPlace:
			if d.streamPlace(o) {
				continue
			}
		}
		o.In = append([]ValueID(nil), o.In...)
		if w, ok := weights[selOf(o)]; ok && (o.Kind == OpCombine || o.Kind == OpGather && o.Attr.WeightIn) {
			o.In = append(o.In, w)
		}
		d.op(o)
	}
	return d
}

// selOf is the selection op o reads, or -1.
func selOf(o Op) ValueID {
	switch o.Kind {
	case OpGather, OpExpertMatVec, OpCombine:
		if len(o.In) > 1 {
			return o.In[1]
		}
	}
	return -1
}

// MaskedOut is the value a masked key takes: below every score, finite (an
// infinity does not fold identically in every lowering), and equal for every
// masked element, so ties among them go to the lower index.
const MaskedOut = -math.MaxFloat32

// RouteSumFloor is added to a renormalising sum, DeepSeek's sum + 1e-20: an
// all-underflowed selection divides to 0 rather than NaN.
const RouteSumFloor = 1e-20

// route writes route op o as primitives into d and returns its weights'
// value, or false where o states a router the primitives here do not yet
// express (sparsemixer, a hash-routed block's table).
func (d *Graph) route(o Op) (ValueID, bool) {
	a := o.Attr
	E, K := a.Experts, a.TopK
	var router, bias, selBias *Binding
	for i := range o.Weight {
		switch w := &o.Weight[i]; w.Role {
		case jlm.RoleRouter:
			router = w
		case jlm.RoleRouterBias:
			bias = w
		case jlm.RoleExpProbsB:
			selBias = w
		default:
			return 0, false
		}
	}
	gated := a.Router == RouterSigmoid || a.Router == RouterSqrtSoftplus
	if router == nil || E < 1 || K < 1 || K > E || !gated && a.Router != RouterSoftmax ||
		a.Groups > 1 && (E%a.Groups != 0 || a.GroupsUsed < 1 || a.GroupsUsed > a.Groups) {
		return 0, false
	}
	p := prims{d: d}
	logits := p.out("route.logits", E, Op{Kind: OpMatVec, In: []ValueID{o.In[0]}, Weight: []Binding{*router}})
	if bias != nil {
		logits = p.mapw("route.logits_b", E, MapAdd, *bias, logits)
	}
	// The key the selection ranks: the logit, the gate's score, or the
	// softmax's probability where a selection bias is added to it.
	key, score := logits, logits
	if gated {
		score = p.gate(a.Router, logits, E)
		key = score
	}
	if selBias != nil {
		if !gated {
			key = p.softmax("route.probs", logits, E)
		}
		key = p.mapw("route.key", E, MapAdd, *selBias, key)
	}
	if a.Groups > 1 {
		// Each group scored by the sum of its two largest keys, the groups
		// ranked, and the keys of the groups not kept masked out.
		span := E / a.Groups
		take := min(2, span)
		_, top := p.topk("route.group_top", key, take, span, E)
		gs := p.out("route.group_scores", a.Groups, Op{Kind: OpReduce, In: []ValueID{top},
			Attr: Attrs{Fn: MapAdd, Order: OrderSeq, Span: take}})
		rank := p.outElem("route.group_rank", a.Groups, ElemU32, Op{Kind: OpRank, In: []ValueID{gs}})
		keep := p.out("route.group_keep", a.Groups, Op{Kind: OpMap, In: []ValueID{rank},
			Attr: Attrs{Fn: MapLess, Const: float64(a.GroupsUsed)}})
		key = p.out("route.masked", E, Op{Kind: OpMap, In: []ValueID{keep, key},
			Attr: Attrs{Fn: MapWhere, Const: MaskedOut}})
	}
	// The selection: the route's output value, now an index row.
	sel := o.Out[0]
	d.Values[sel].Width, d.Values[sel].Elem = K, ElemU32
	top := d.value("route.top", K)
	d.op(Op{Kind: OpTopK, In: []ValueID{key}, Out: []ValueID{sel, top}, Attr: Attrs{TopK: K}})
	var w ValueID
	switch {
	case gated:
		// The selected unbiased scores, renormalised by their sum in selection
		// order plus the floor.
		w = p.out("route.scores", K, Op{Kind: OpTake, In: []ValueID{score, sel}})
		if a.RouteNorm {
			s := p.out("route.sum", 1, Op{Kind: OpReduce, In: []ValueID{w}, Attr: Attrs{Fn: MapAdd, Order: OrderSeq}})
			s = p.out("route.divisor", 1, Op{Kind: OpMap, In: []ValueID{s}, Attr: Attrs{Fn: MapAdd, Const: RouteSumFloor}})
			w = p.out("route.weights_n", K, Op{Kind: OpMap, In: []ValueID{w, s}, Attr: Attrs{Fn: MapDiv}})
		}
	case a.RouteNorm:
		// A softmax over the selected logits: the full softmax renormalised
		// over the selection, its normaliser cancelling.
		w = p.softmax("route.weights_n", p.out("route.top_logits", K, Op{Kind: OpTake, In: []ValueID{logits, sel}}), K)
	default:
		// The full softmax's probabilities of the selected experts.
		w = p.out("route.weights_p", K, Op{Kind: OpTake, In: []ValueID{p.softmax("route.probs_all", logits, E), sel}})
	}
	if a.Scale != 0 && a.Scale != 1 {
		w = p.out("route.weights_s", K, Op{Kind: OpMap, In: []ValueID{w}, Attr: Attrs{Fn: MapMul, Const: float64(float32(a.Scale))}})
	}
	return w, true
}

// prims writes primitives into a graph, each into a new value.
type prims struct{ d *Graph }

// out appends o with a new f32 value of width w as its output.
func (p prims) out(name string, w int, o Op) ValueID { return p.outElem(name, w, ElemF32, o) }

func (p prims) outElem(name string, w int, e Elem, o Op) ValueID {
	v := p.d.value(name, w)
	p.d.Values[v].Elem = e
	o.Out = []ValueID{v}
	p.d.op(o)
	return v
}

// mapw is f of x and a bound vector b.
func (p prims) mapw(name string, w int, f MapFn, b Binding, x ValueID) ValueID {
	return p.out(name, w, Op{Kind: OpMap, In: []ValueID{x}, Weight: []Binding{b}, Attr: Attrs{Fn: f}})
}

// gate is router r's score of each logit: its sigmoid, or the square root of
// its softplus.
func (p prims) gate(r Router, x ValueID, w int) ValueID {
	if r == RouterSqrtSoftplus {
		sp := p.out("route.softplus", w, Op{Kind: OpMap, In: []ValueID{x}, Attr: Attrs{Fn: MapSoftplus}})
		return p.out("route.scores_all", w, Op{Kind: OpMap, In: []ValueID{sp}, Attr: Attrs{Fn: MapSqrt}})
	}
	return p.out("route.scores_all", w, Op{Kind: OpMap, In: []ValueID{x}, Attr: Attrs{Fn: MapSigmoid}})
}

// softmax is x's softmax over its row of w: the maximum from the first
// element, the exponentials' sum from zero, each in order, and each
// exponential over the sum.
func (p prims) softmax(name string, x ValueID, w int) ValueID {
	m := p.out(name+".max", 1, Op{Kind: OpReduce, In: []ValueID{x}, Attr: Attrs{Fn: MapMax, Order: OrderSeq}})
	c := p.out(name+".centred", w, Op{Kind: OpMap, In: []ValueID{x, m}, Attr: Attrs{Fn: MapSub}})
	e := p.out(name+".exp", w, Op{Kind: OpMap, In: []ValueID{c}, Attr: Attrs{Fn: MapExp}})
	z := p.out(name+".sum", 1, Op{Kind: OpReduce, In: []ValueID{e}, Attr: Attrs{Fn: MapAdd, Order: OrderSeq}})
	return p.out(name, w, Op{Kind: OpMap, In: []ValueID{e, z}, Attr: Attrs{Fn: MapDiv}})
}

// topk is the k largest of each group of span of x (a row of w): their
// indices and their values.
func (p prims) topk(name string, x ValueID, k, span, w int) (ix, v ValueID) {
	n := k * (w / max(span, 1))
	if span == 0 {
		n = k
	}
	ix = p.d.value(name+".ix", n)
	p.d.Values[ix].Elem = ElemU32
	v = p.d.value(name, n)
	p.d.op(Op{Kind: OpTopK, In: []ValueID{x}, Out: []ValueID{ix, v}, Attr: Attrs{TopK: k, Span: span}})
	return ix, v
}
