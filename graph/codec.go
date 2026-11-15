package graph

import (
	"bytes"
	"encoding/gob"
	"fmt"
)

// A stored graph is the graph's own values, its op kinds by name: a kind
// added or moved in the list reads back as itself, and a name this build
// does not know is refused. The block's index is not stored, so blocks of
// one structure share one encoding.
type storedGraph struct {
	Version int
	Values  []Value
	Ops     []storedOp
	In, Out []ValueID
	Routed  Region
}

type storedOp struct {
	Kind   string
	In     []ValueID
	Out    []ValueID
	Weight []Binding
	Reads  []StateRef
	Writes []StateRef
	Attr   Attrs
}

// MaxStored is the most bytes a stored graph may hold: a decoder reads no
// more, whatever a corrupt container claims (RULE 9).
const MaxStored = 1 << 24

// Encode is g as a container stores a block's graph, read back by Decode.
func Encode(g *Graph) ([]byte, error) {
	s := storedGraph{Version: g.Version, Values: g.Values, In: g.In, Out: g.Out, Routed: g.Routed}
	for _, o := range g.Ops {
		if o.Kind <= OpInvalid || o.Kind >= numOps {
			return nil, fmt.Errorf("graph: block %d holds op kind %d", g.Block, o.Kind)
		}
		s.Ops = append(s.Ops, storedOp{Kind: o.Kind.String(), In: o.In, Out: o.Out, Weight: o.Weight,
			Reads: o.Reads, Writes: o.Writes, Attr: o.Attr})
	}
	var b bytes.Buffer
	if err := gob.NewEncoder(&b).Encode(&s); err != nil {
		return nil, fmt.Errorf("graph: encoding block %d: %w", g.Block, err)
	}
	if b.Len() > MaxStored {
		return nil, fmt.Errorf("graph: block %d encodes to %d bytes, past %d", g.Block, b.Len(), MaxStored)
	}
	return b.Bytes(), nil
}

// Decode is the graph Encode stored, as block li's, held to what the
// container stores for the block (Verify).
func Decode(b []byte, li int, f Facts) (*Graph, error) {
	if len(b) > MaxStored {
		return nil, fmt.Errorf("graph: block %d's stored graph is %d bytes, past %d", li, len(b), MaxStored)
	}
	var s storedGraph
	if err := gob.NewDecoder(bytes.NewReader(b)).Decode(&s); err != nil {
		return nil, fmt.Errorf("graph: block %d's stored graph: %w", li, err)
	}
	g := &Graph{Version: s.Version, Block: li, Values: s.Values, In: s.In, Out: s.Out, Routed: s.Routed}
	for _, o := range s.Ops {
		k := kindNamed(o.Kind)
		if k == OpInvalid {
			return nil, fmt.Errorf("graph: block %d's stored graph holds op %q, which this build does not know", li, o.Kind)
		}
		g.Ops = append(g.Ops, Op{Kind: k, In: o.In, Out: o.Out, Weight: o.Weight, Reads: o.Reads, Writes: o.Writes, Attr: o.Attr})
	}
	if r := g.Routed; r != (Region{}) && (r.Lo < 0 || r.Hi > len(g.Ops) || r.Lo >= r.Hi) {
		return nil, fmt.Errorf("graph: block %d's routed ops [%d, %d) are not its ops", li, r.Lo, r.Hi)
	}
	if err := Verify(g, f); err != nil {
		return nil, err
	}
	return g, nil
}

// kindNamed is the op kind named name, OpInvalid for none.
func kindNamed(name string) OpKind {
	for k := OpInvalid + 1; k < numOps; k++ {
		if k.String() == name {
			return k
		}
	}
	return OpInvalid
}
