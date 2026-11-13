package model

import "github.com/jitllm/jitllm/nn"

// infillTables is the vocabulary's infill tables (tok.Infill) over the
// logits' row: a row past the vocabulary, or an id a table names past the
// row, is a token with no piece.
func (m *Model) infillTables() nn.InfillTables {
	in := m.Vocab.Infill()
	n := m.Cfg.NVocab
	t := nn.InfillTables{Class: make([]uint32, n), Parent: make([]uint32, n), Depth: in.Depth, EOT: uint32(max(in.EOT, 0))}
	for i := range n {
		t.Class[i], t.Parent[i] = uint32(i), uint32(i)
		if i < len(in.Class) && int(in.Class[i]) < n && int(in.Parent[i]) < n {
			t.Class[i], t.Parent[i] = in.Class[i], in.Parent[i]
		}
	}
	for _, id := range in.EOG {
		t.EOG = append(t.EOG, float32(id))
	}
	if len(t.EOG) == 0 {
		t.EOG = []float32{-1}
	}
	return t
}
