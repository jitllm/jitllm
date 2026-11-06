package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// Qwen3.8-Flash-Next's one linear block adds the n-gram embedding before its
// mix (ngram.go); its hyper-connections and its attention's selection are
// the converter's statements (convert.StateStreams, convert.StateSelection).

// NGramAt reports that block li adds the n-gram embedding.
func (c *Config) NGramAt(li int) bool { return c.NGramBlock != 0 && li == c.NGramBlock-1 }

// ngramHeads is the n-gram embedding's head count.
func (c *Config) ngramHeads() int { return (c.NGram - 1) * c.NGramHeadsPer }

// ngramHist is how many earlier positions the n-gram convolution reads.
func (c *Config) ngramHist() int { return (c.NGramConv - 1) * c.NGram }

// ngramStateLen is one sequence's n-gram state: the NGram-1 tokens before the
// next position (stored as id+1, zero for none, which reads as the
// end-of-sequence id), then the convolution's history, ngramHist positions of
// every stream.
func (c *Config) ngramStateLen() int {
	return c.NGram - 1 + c.ngramHist()*c.HCMult*c.NEmbd
}

// NGramWidth is the gathered n-gram embedding's width: every head's row.
func (c *Config) NGramWidth() int { return c.ngramHeads() * c.NGramRowDim }

// loadNGramTable reads the n-gram block's table.
func loadNGramTable(m *Model, cfg *Config, f *jlm.File) error {
	if cfg.NGramBlock == 0 {
		return nil
	}
	e, ok := f.Find(jlm.RoleNGramTable, jlm.DenseBlock, -1)
	if !ok || !jlm.RowTable(e) || e.NDim != 2 || int(e.Dims[0]) != cfg.NGramRowDim {
		return fmt.Errorf("model: %s has an n-gram block and no n-gram table %d wide", cfg.Arch, cfg.NGramRowDim)
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
	for h := range cfg.NGramVoc {
		if cfg.NGramOff[h]+cfg.NGramVoc[h] > e.Dims[1] {
			return fmt.Errorf("model: n-gram head %d reads past the table's %d rows", h, e.Dims[1])
		}
	}
	m.Ngram = &ngramTable{E: e, typ: t, RowBytes: int(rb)}
	return nil
}
