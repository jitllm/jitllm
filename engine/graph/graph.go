// Package graph is the block graph: a versioned, typed SSA description of
// one block's values and persistent state effects, built from what the
// container states (docs/design/composer.md, "Above the recipes"). It holds
// data and a verifier and computes nothing: a graph is lowered to the
// generated operations the engine already runs, never interpreted here.
package graph

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// Version is the op language this package writes and the verifier reads. A
// graph of another version is refused, never guessed at.
const Version = 1

// OpKind is an operation. Each kind's semantics are fixed per Version: a
// change of meaning is a new kind or a new Version, not a reinterpretation.
type OpKind uint8

const (
	OpInvalid OpKind = iota
	// OpNorm is RMSNorm over a row: x * rsqrt(mean(x^2) + Eps) * w, the
	// epsilon inside the root, accumulated in f32. Weight[0] is the scale.
	OpNorm
	// OpHeadNorm is OpNorm over each head of HeadDim within the row, one
	// scale of HeadDim shared by every head (qwen3's q/k norm).
	OpHeadNorm
	// OpMatVec is out = W x, W's rows the output; Weight[0] is W, an
	// optional Weight[1] its bias added after.
	OpMatVec
	// OpRope turns each head's first NRot elements by the position, in the
	// pair Layout, at Base.
	OpRope
	// OpKVAppend writes k and v (In[0], In[1]) as this position's row of the
	// block's KV: it reads state version n and writes n+1.
	OpKVAppend
	// OpAttend is causal attention of q (In[0]) over the KV version it reads,
	// GQA by the head counts, scaled by Scale, over a Window (0: the whole
	// history), scores capped by Softcap (0: none).
	OpAttend
	// OpAdd is In[0] + In[1].
	OpAdd
	// OpActMul is Act(In[0]) * In[1], elementwise.
	OpActMul
	// OpGatedDelta is a gated delta rule over its input row (In[0], the
	// sublayer's normed input): the projections, causal convolution, decay
	// and scan that turn it into one output row per value head (Out[0],
	// VHeads*VDim), reading the recurrence's version n and writing n+1. It is
	// opaque in this version: the rule's own internals are one op, its
	// weights all bound by it, and only what follows it is matched.
	OpGatedDelta
	numOps
)

var opNames = [...]string{"invalid", "norm", "head_norm", "matvec", "rope", "kv_append", "attend", "add", "act_mul", "gated_delta"}

func (k OpKind) String() string {
	if int(k) < len(opNames) {
		return opNames[k]
	}
	return fmt.Sprintf("op%d", k)
}

// RopeLayout is how a rotary pairs a head's elements.
type RopeLayout uint8

const (
	// RopeNorm pairs (2i, 2i+1).
	RopeNorm RopeLayout = iota + 1
	// RopeNeox pairs (i, i+NRot/2).
	RopeNeox
)

func (l RopeLayout) String() string {
	switch l {
	case RopeNorm:
		return "norm"
	case RopeNeox:
		return "neox"
	}
	return fmt.Sprintf("layout%d", l)
}

// Act is an activation an OpActMul applies to its first input.
type Act uint8

const (
	ActSiLU Act = iota + 1
	ActGELUTanh
	// ActSigmoid is sigma(In[0]) * In[1]: a gated norm's sigmoid gate.
	ActSigmoid
)

// ValueID names a value in a Graph's Values.
type ValueID int32

// Value is one SSA value: a row of Width f32 elements per token.
type Value struct {
	Name  string
	Width int
}

// Binding is a weight an op reads: its container role in the graph's block,
// with the shape the op requires (K the input width, Rows the output; Rows 0
// for a vector of K).
type Binding struct {
	Role    jlm.Role
	K, Rows int
}

// StateRef is a version of the block's persistent state.
type StateRef struct {
	Name    string // "kv"
	Version int
}

// Attrs are an op's parameters. Only those its kind defines are read; the
// verifier refuses one set on a kind that does not define it.
type Attrs struct {
	Eps     float64
	HeadDim int
	NRot    int
	Base    float64
	Layout  RopeLayout
	Freqs   bool // the rotary divides each pair's theta by stored factors (rope_freqs, YaRN)
	NHead   int
	NKVHead int
	Scale   float64
	Window  int
	Softcap float64
	Act     Act
}

// Op is one operation.
type Op struct {
	Kind   OpKind
	In     []ValueID
	Out    []ValueID
	Weight []Binding
	Reads  []StateRef
	Writes []StateRef
	Attr   Attrs
}

// Graph is one block.
type Graph struct {
	Version int
	Block   int
	Values  []Value
	Ops     []Op
	In, Out []ValueID // the residual at the block's boundary
}

func (g *Graph) value(name string, width int) ValueID {
	g.Values = append(g.Values, Value{Name: name, Width: width})
	return ValueID(len(g.Values) - 1)
}

func (g *Graph) op(o Op) { g.Ops = append(g.Ops, o) }

// String is the graph in the illustrative form docs/design/composer.md uses.
func (g *Graph) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "block %d (graph v%d)\n", g.Block, g.Version)
	name := func(ids []ValueID) string {
		s := make([]string, len(ids))
		for i, id := range ids {
			if id >= 0 && int(id) < len(g.Values) {
				s[i] = "%" + g.Values[id].Name
			} else {
				s[i] = fmt.Sprintf("%%?%d", id)
			}
		}
		return strings.Join(s, ", ")
	}
	for _, o := range g.Ops {
		fmt.Fprintf(&b, "  %s = %s(%s", name(o.Out), o.Kind, name(o.In))
		for _, w := range o.Weight {
			fmt.Fprintf(&b, ", %s", w.Role)
		}
		for _, r := range o.Reads {
			fmt.Fprintf(&b, ", reads %s@%d", r.Name, r.Version)
		}
		for _, w := range o.Writes {
			fmt.Fprintf(&b, ", writes %s@%d", w.Name, w.Version)
		}
		b.WriteString(")\n")
	}
	return b.String()
}
