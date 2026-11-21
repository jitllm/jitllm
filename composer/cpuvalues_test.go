package composer

import "github.com/jitllm/jitllm/graph"

// valueWords is a buffer for each of g's values past its first (the input
// the harness binds), rows rows where a value states none, as the harness's
// bind asks for them, and its output's buffer.
func valueWords(g *graph.Graph, rows int) (func(Operand) []uint32, []float32) {
	bufs := map[graph.ValueID][]float32{}
	for i, v := range g.Values {
		if i == 0 {
			continue
		}
		r := v.Rows
		if r == 0 {
			r = rows
		}
		bufs[graph.ValueID(i)] = make([]float32, r*v.Width+16)
	}
	return func(o Operand) []uint32 {
		if b, ok := bufs[o.Value]; ok && o.Kind == OperandValue {
			return f32Words(b)
		}
		return nil
	}, bufs[g.Out[0]]
}
