package graph

import "github.com/jitllm/jitllm/format/jlm"

// Embed is a step's input rows: each row's token's row of the embedding,
// times Scale where it is neither 0 nor 1 (Gemma's sqrt(n_embd)), plus its
// position's row of a learned position table of Positions rows where there
// is one (StarCoder's).
//
// Streams, where above one, makes the row a residual of that many streams,
// stream-major, the input row stream 0 and the others zero (Kimi-K3's
// residual attention) or, where AltUp, its projections matched to its
// magnitude (Gemma 3n's). PLE, where set, adds each layer's input as an
// output after the row (Out[1+layer]), from the row before its streams.
type Embed struct {
	NEmbd, NVocab int
	Scale         float64
	Positions     int
	Streams       int
	AltUp         bool
	PLE           *PLE
	Eps           float64
}

func (e *Embed) Build() *Graph {
	g := &Graph{Block: -1}
	scaled := e.Scale != 0 && e.Scale != 1
	name := "x"
	if scaled {
		name = "embd"
	}
	x := g.value(name, e.NEmbd)
	g.op(Op{Kind: OpEmbed, Out: []ValueID{x}, Weight: []Binding{{Role: jlm.RoleTokenEmbd, K: e.NEmbd, Rows: e.NVocab}}})
	if scaled {
		y := g.value("x", e.NEmbd)
		g.op(Op{Kind: OpMap, In: []ValueID{x}, Out: []ValueID{y}, Attr: Attrs{Fn: MapMul, Const: e.Scale}})
		x = y
	}
	if e.Positions > 0 {
		p := g.value("pos_embd", e.NEmbd)
		g.op(Op{Kind: OpEmbed, Out: []ValueID{p}, Weight: []Binding{{Role: jlm.RolePosEmbd, K: e.NEmbd, Rows: e.Positions}},
			Attr: Attrs{Positional: true}})
		y := g.value("x_pos", e.NEmbd)
		g.op(Op{Kind: OpAdd, In: []ValueID{x, p}, Out: []ValueID{y}})
		x = y
	}
	var ple []ValueID
	if e.PLE != nil {
		ple = e.PLE.inputs(g, x, e.NEmbd, e.Eps)
	}
	if e.Streams > 1 && e.AltUp {
		x = g.altExpand(x, e.Streams, e.NEmbd)
	} else if e.Streams > 1 {
		rest := (e.Streams - 1) * e.NEmbd
		t := g.value("streams.tiled", rest)
		g.op(Op{Kind: OpTile, In: []ValueID{x}, Out: []ValueID{t}, Attr: Attrs{Span: e.NEmbd}})
		z := g.value("streams.zero", rest)
		g.op(Op{Kind: OpMap, In: []ValueID{t}, Out: []ValueID{z}, Attr: Attrs{Fn: MapMul, Const: 0}})
		y := g.value("x_streams", e.Streams*e.NEmbd)
		g.op(Op{Kind: OpConcat, In: []ValueID{x, z}, Out: []ValueID{y}})
		x = y
	}
	g.Out = append([]ValueID{x}, ple...)
	return g
}
