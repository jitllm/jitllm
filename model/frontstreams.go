package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// A block over several residual streams the converter states
// (jlm.Streams): the graph reads each stream as an input of its own, a row of
// n_embd, and writes each as an output, stream k's output over stream k's
// input. The call site binds them at their offsets in main's stream-major
// residual.

// streams is a block's residual streams as its graph reads them: the
// inputs, and the stream the block banks its input into, written before any
// stream is read (0 for none).
type streams struct {
	in   []graph.ValueID
	bank graph.ValueID
	// cur is the streams after a sublayer that writes every one, and ex
	// the values the sublayer's read exports to its writes.
	cur []graph.ValueID
	ex  []graph.ValueID
}

// statedBinds is every role block li's stated reads bind.
func (c *Config) statedBinds(li int) []jlm.Role {
	if c.stated == nil || li < 0 || li >= len(c.stated.Blocks) {
		return nil
	}
	var out []jlm.Role
	bl := &c.stated.Blocks[li]
	for _, u := range []*jlm.SubStreams{&bl.Pre, &bl.Attn, &bl.FFN} {
		if u.Read == nil {
			continue
		}
		for _, op := range u.Read.Ops {
			out = append(out, op.Bind...)
		}
	}
	return out
}

// ResidStreams says the residual is every stream, which the input rows
// expand into (jlm.Streams.Expand): a row's hidden state is all of them, as
// the head's mix reads them, not the output norm's row.
func (c *Config) ResidStreams() bool { return c.stated != nil && len(c.stated.Expand) != 0 }

// A block over a bank of checkpoints (jlm.BlockStreams.Bank): stream 0 is the
// running residual and stream 1+j checkpoint j. Each sublayer reads a
// softmax mix of the checkpoints banked so far and the running residual,
// each scored by its stored vector's dot with the stream RMS-normed with no
// weight; a checkpoint block banks its input first and restarts the running
// residual from its attention's output.
//
//	in_a = mix(x_1..x_nb, x_0)                     the attention's input
//	x_0' = attn(in_a), or x_0 + attn(in_a)          a checkpoint block, or not
//	in_f = mix(x_1..x_nb', x_0')                    with the input banked
//	x_0'' = x_0' + ffn(in_f)

// streamRead is graph.StreamRead over the block's per-layer inputs alone
// where the side row carries the taps' rows before them.
func (m *Model) streamRead(b *graph.Builder, f graph.Facts, u *jlm.SubStreams, xs []graph.ValueID, in, run, side graph.ValueID,
	ops func([]graph.ValueID, *jlm.TowerOp) (graph.ValueID, error)) (graph.ValueID, []graph.ValueID, error) {
	if c := m.Cfg; side != 0 && c.TapsWidth() > 0 && c.PLEDim != 0 {
		side = b.Emit(graph.OpSlice, []graph.ValueID{side}, []graph.Result{{Name: "side.ple", Width: c.PleWidth()}}, nil,
			graph.Attrs{Offset: c.TapsWidth(), Width: c.PleWidth()}, nil, nil)[0]
	}
	return graph.StreamRead(b, f, u, xs, in, run, side, ops)
}

// StatesStreams says the model's several residual streams are the ones the
// converter states, and nothing else carries a stream.
func (c *Config) StatesStreams() bool {
	return c.statedStreams() > 1 && c.Streams() == c.statedStreams()
}

// Streams is how many residual streams a token carries: the streams the
// converter states, DeepSeek V4's hyper-connections, or one.
func (c *Config) Streams() int { return max(c.HCMult, c.statedStreams(), 1) }

// statedStreams is how many streams the converter states, 0 for none.
func (c *Config) statedStreams() int {
	if c.stated == nil {
		return 0
	}
	return int(c.stated.N)
}

// ResidW is one row's residual width, every stream of it.
func (c *Config) ResidW() int { return c.NEmbd * c.Streams() }

// Stream is stream k of row r of a stream-major residual of `rows` rows.
func (c *Config) Stream(x []float32, k, rows, r int) []float32 {
	d := c.NEmbd
	o := (k*rows + r) * d
	return x[o : o+d]
}

// banked writes the block's streams: the running residual y, the banked
// stream its input, every other stream its input itself, which no kernel
// writes.
func (m *Model) banked(b *graph.Builder, f graph.Facts, li int, st *streams, y graph.ValueID, has func(jlm.Role) bool) (*graph.Graph, Uncovered, error) {
	out := slices.Clone(st.in)
	out[0] = outScaled(b, has, y)
	if k := m.Cfg.stated.Blocks[li].Bank; k != 0 {
		out[k] = st.bank
	}
	b.Out(out...)
	return verified(b, f)
}

// streamCopy is a row of v as a value of its own.
func streamCopy(b *graph.Builder, v graph.ValueID) graph.ValueID {
	w := b.G.Values[v].Width
	return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: "stream", Width: w}}, nil, graph.Attrs{Width: w}, nil, nil)[0]
}

// ExpandsStreams says the input rows' streams past the first are made from
// the first: the converter states their expansion.
func (m *Model) ExpandsStreams() bool {
	c := m.Cfg
	return c.stated != nil && len(c.stated.Expand) != 0
}

// streamExpand is the input rows' streams 1 and up from stream 0, x, as the
// converter states them: each its program over x, or a copy of x.
func (m *Model) streamExpand(b *graph.Builder, f graph.Facts, x graph.ValueID) ([]graph.ValueID, error) {
	var out []graph.ValueID
	for k, p := range m.Cfg.stated.Expand {
		if p == nil {
			out = append(out, streamCopy(b, x))
			continue
		}
		v, err := graph.LowerStated(b, p, []graph.ValueID{x}, f)
		if err != nil {
			return nil, fmt.Errorf("stream %d's expansion: %w", k+1, err)
		}
		out = append(out, v)
	}
	return out, nil
}

// streamMix is the head's read of every stream in b, an input of b a
// stream: the converter's stated head read over facts f.
func (m *Model) streamMix(b *graph.Builder, f graph.Facts) (graph.ValueID, error) {
	c := m.Cfg
	s, d := c.Streams(), c.NEmbd
	xs := make([]graph.ValueID, s)
	xs[0] = b.In("x", d)
	for k := 1; k < s; k++ {
		xs[k] = b.In(fmt.Sprintf("x.%d", k), d)
	}
	y, _, err := m.streamRead(b, f, &c.stated.Head, xs, xs[0], xs[0], 0, nil)
	if err != nil {
		return 0, fmt.Errorf("the streams' stated head read: %w", err)
	}
	return y, nil
}

// plainActAt is block li's stated activation without the sparsity the
// container states on its FFN's gate (unsparsified): the activation a gate
// that is not an FFN's (the per-layer input's) reads.
func (m *Model) plainActAt(li int) *jlm.Expr {
	e := m.Container.Config().ActAt(li)
	if e == nil {
		return nil
	}
	return Unsparsified(e)
}

// Unsparsified is e with each term keeping max(g - t, 0) of its gate g
// (operand 0), t a value reduced over the gate -- a gaussian top-k, t the
// gate's mean and a stated number of deviations -- read as the gate itself,
// and the terms nothing reads then dropped: e itself where it has no such
// term.
func Unsparsified(e *jlm.Expr) *jlm.Expr {
	gate := jlm.Operand(0)
	ts := slices.Clone(e.Terms)
	// overGate reports whether a reads a reduction over the gate, gated
	// whether it reads the gate at all.
	var overGate, gated func(a jlm.Arg) bool
	gated = func(a jlm.Arg) bool {
		if a == gate {
			return true
		}
		if a.Kind != jlm.ArgTerm {
			return false
		}
		return slices.ContainsFunc(ts[a.Index].Args[:], gated)
	}
	overGate = func(a jlm.Arg) bool {
		if a.Kind != jlm.ArgTerm {
			return false
		}
		t := ts[a.Index]
		if t.Reduce && gated(t.Args[0]) {
			return true
		}
		return slices.ContainsFunc(t.Args[:], overGate)
	}
	cut := false
	for k, t := range ts {
		if t.Fn != jlm.FnMax || t.Reduce || t.Args[1] != jlm.Num(0) || t.Args[0].Kind != jlm.ArgTerm {
			continue
		}
		s := ts[t.Args[0].Index]
		if s.Fn != jlm.FnSub || s.Reduce || s.Args[0] != gate || !overGate(s.Args[1]) {
			continue
		}
		at := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(k)}
		for j := k + 1; j < len(ts); j++ {
			for i, a := range ts[j].Args {
				if a == at {
					ts[j].Args[i] = gate
				}
			}
		}
		cut = true
	}
	if !cut {
		return e
	}
	live := make([]bool, len(ts))
	live[len(ts)-1] = true
	for k := len(ts) - 1; k >= 0; k-- {
		if !live[k] {
			continue
		}
		for _, a := range ts[k].Args {
			if a.Kind == jlm.ArgTerm {
				live[a.Index] = true
			}
		}
	}
	at := make([]uint32, len(ts))
	out := &jlm.Expr{}
	for k, t := range ts {
		if !live[k] {
			continue
		}
		for i, a := range t.Args {
			if a.Kind == jlm.ArgTerm {
				t.Args[i].Index = at[a.Index]
			}
		}
		at[k] = uint32(len(out.Terms))
		out.Terms = append(out.Terms, t)
	}
	return out
}
