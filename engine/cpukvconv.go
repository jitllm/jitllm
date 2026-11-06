package engine

import (
	"slices"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// kvHalfRow is the floats a row of the binary16 conversions' forms holds.
const kvHalfRow = 256

// cpuPlaceKV composes the conversions where the instance's history is
// binary16 or q8, for the CPU device (cpudev.KVConv): a binary16 history's
// halves (graph.OpHalves, graph.OpWidenHalves) in rows of kvHalfRow floats,
// a q8 history's rows (graph.OpQ8Rows, graph.OpWidenQ8Rows) one a head width
// its layers hold.
func (s *Instance) cpuPlaceKV() {
	s.kvConv = nil
	if s.cpu.dev == nil {
		return
	}
	narrow, widen := graph.OpHalves, graph.OpWidenHalves
	var widths []int
	switch s.kvl.fmt {
	case cpu.KVF16:
		widths = []int{kvHalfRow}
	case cpu.KVQ8:
		narrow, widen = graph.OpQ8Rows, graph.OpWidenQ8Rows
		for li := s.lo; li < s.hi; li++ {
			if hd := s.kvlAt(li).headDim; !slices.Contains(widths, hd) {
				widths = append(widths, hd)
			}
		}
	default:
		return
	}
	var conv []cpudev.KVConvForms
	for _, fw := range widths {
		c := cpudev.KVConvForms{FW: fw, HW: fw / 2}
		if s.kvl.fmt == cpu.KVQ8 {
			c.HW = graph.Q8RowWords(fw)
		}
		nb := graph.New()
		nb.Out(nb.Emit(narrow, []graph.ValueID{nb.In("kv.f", c.FW)},
			[]graph.Result{{Name: "kv.h", Width: c.HW, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)...)
		var hf [2]cpuHeadForm
		s.cpuComposeHead(&hf[0], nb.G, nn.DeclineNoGraph, cpuRowsCap)
		wb := graph.New()
		h := wb.In("kv.h", c.HW)
		wb.G.Values[h].Elem = graph.ElemU32
		wb.Out(wb.Emit(widen, []graph.ValueID{h}, []graph.Result{{Name: "kv.f", Width: c.FW}}, nil, graph.Attrs{}, nil, nil)...)
		s.cpuComposeHead(&hf[1], wb.G, nn.DeclineNoGraph, cpuRowsCap)
		c.Narrow = cpudev.Part{Composed: hf[0].composed, F: hf[0].f, Why: hf[0].why, Err: hf[0].err}
		c.Widen = cpudev.Part{Composed: hf[1].composed, F: hf[1].f, Why: hf[1].why, Err: hf[1].err}
		conv = append(conv, c)
	}
	s.kvConv = cpudev.NewKVConv(s.cpu.dev, conv, s.kvl.fmt == cpu.KVF16)
}

// convertKV narrows the floats f of head rows hd wide into the history's
// bytes h, or widens h into f, on the device the instance places the
// conversions on (nn.KVConvDevice); false where they do not serve.
func (s *Instance) convertKV(narrow bool, hd int, f []float32, h []byte) bool {
	return s.kvConv != nil && s.kvConv.Convert(narrow, hd, f, h)
}
