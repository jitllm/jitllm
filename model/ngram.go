package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// An n-gram table (jlm.TOpNGramRows): each position's token and its
// predecessors are hashed per head (the op's jlm.NGramHash; a predecessor at
// or before an end-of-sequence token, or before the sequence, reads as that
// token) into a row of a shared table, read a row at
// a time from the container (jlm.RowTable): it is hundreds of millions of
// rows, of which a position reads one a head, each head's from its own
// range. The CPU device reads the rows a call's hash names into the table's
// view between its gathers' halves (cpuForm). The predecessors are the
// block's recurrent tail (jlm.Program.Tail), so they page, relocate and are
// prefix-cached with it.

// ngramTable is the container's n-gram table.
type ngramTable struct {
	E        *jlm.Entry
	typ      quant.Type
	RowBytes int
}

// loadNGramTable reads the n-gram block's table.
func loadNGramTable(m *Model, cfg *Config, f *jlm.File) error {
	op := cfg.ngramOp()
	if op == nil {
		return nil
	}
	h := op.Hash
	rowDim := int(op.Width) / (int(op.Span-1) * int(h.HeadsPer))
	e, ok := f.Find(op.Bind[0], jlm.DenseBlock, -1)
	if !ok || !jlm.RowTable(e) || e.NDim != 2 || int(e.Dims[0]) != rowDim {
		return fmt.Errorf("model: %s has an n-gram block and no n-gram table %d wide", cfg.Arch, rowDim)
	}
	t, ok := jlm.SourceType(e.Type)
	if !ok {
		return fmt.Errorf("model: the n-gram table is %v", e.Type)
	}
	switch t {
	case quant.F32, quant.F16, quant.BF16, quant.IQ4_NL:
	default:
		return fmt.Errorf("model: an n-gram table of %v has no row reader", t)
	}
	rb, err := jlm.RowBytes(e)
	if err != nil || rb > jlm.Align {
		return fmt.Errorf("model: n-gram rows of %d bytes (%v)", rb, err)
	}
	for i := range h.Voc {
		if h.Off[i]+h.Voc[i] > e.Dims[1] {
			return fmt.Errorf("model: n-gram head %d reads past the table's %d rows", i, e.Dims[1])
		}
	}
	m.Ngram = &ngramTable{E: e, typ: t, RowBytes: int(rb)}
	return nil
}

// ngramOp is the blocks' stated n-gram op (jlm.TOpNGramRows), nil where they
// state none.
func (c *Config) ngramOp() *jlm.TowerOp {
	if c.stated == nil {
		return nil
	}
	for i := range c.stated.Blocks {
		b := &c.stated.Blocks[i]
		for _, sub := range []*jlm.SubStreams{&b.Pre, &b.Attn, &b.FFN} {
			for _, p := range append([]*jlm.Program{sub.Read, sub.Join, sub.Post}, sub.Writes...) {
				if p == nil {
					continue
				}
				for j := range p.Ops {
					if p.Ops[j].Code == jlm.TOpNGramRows {
						return &p.Ops[j]
					}
				}
			}
		}
	}
	return nil
}
