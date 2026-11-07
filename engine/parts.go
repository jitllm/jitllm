package engine

import (
	"slices"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// kvHalfRow is the floats a row of the binary16 conversions' forms holds.
const kvHalfRow = 256

// partsPlan is what the instance's parts are placed from (nn.PartsPlan),
// each as the model states it: the head's and the samplers' graphs (graph.Head
// over the row the streams collapse to, the norm, the projection and what
// follows it; a step's rows' at cpuRowsCap; a label pick's over the picked rows
// alone; the greedy token's and a top-k draw's), the streams' expansion and
// collapse (a prediction block's own head mix where the instance runs one), an
// embedding model's readout at the segment's rows, the entry of the prediction
// block a draft instance starts at, the input rows' (the token rows' and an
// embedding span's given rows'), each rotary's table (a tower's grid table
// turns a picture's rows in one call) and a binary16 or q8 history's
// conversions (a binary16 history's halves in rows of kvHalfRow floats, a q8
// history's rows one a head width its layers hold).
func (s *Instance) partsPlan() *nn.PartsPlan {
	fg := s.m.FrontGraphs()
	p := &nn.PartsPlan{D: s.c.NEmbd, NVocab: s.c.NVocab, TopLog: model.MaxTopLogprobs,
		HeadNorm: fg.OutNorm != nil, HeadW: nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}}
	part := func(g *graph.Graph, none nn.Decline, rows int) nn.PartPlan { return s.partPlan(g, none, rows, nil) }
	p.Head[0] = part(fg.CpuHead, nn.DeclineNoGraph, 1)
	p.Head[1] = part(fg.CpuHidden, nn.DeclineNoGraph, 1)
	p.Head[2] = part(fg.CpuHidden, nn.DeclineNoGraph, cpuRowsCap)
	none := nn.DeclineNoGraph
	if fg.CpuHead != nil {
		none = nn.DeclineHeadBias
	}
	p.Head[3] = part(fg.PickHead, none, 1)
	for i, g := range []*graph.Graph{fg.CpuSample, fg.CpuWalk, fg.CpuRow, fg.CpuCut, fg.CpuWalkProb, fg.CpuCutProb,
		fg.CpuCutK, fg.CpuCutKProb} {
		p.Head[4+i] = part(g, nn.DeclineNoSample, 1)
	}
	p.Head[12] = nn.PartPlan{Skip: true}
	if fg.OutNorm != nil {
		p.Head[12] = part(fg.OutNorm, nn.DeclineNoGraph, cpuRowsCap)
	}
	p.Head[13] = part(fg.CpuLogprobs, nn.DeclineNoSample, 1)
	p.Head[14] = part(fg.CpuMask, nn.DeclineNoSample, 1)
	p.Head[15] = nn.PartPlan{Skip: true}
	if fg.CpuDecide != nil {
		p.Head[15] = part(fg.CpuDecide, nn.DeclineNoSample, 1)
	}

	if s.c.StatesStreams() {
		p.Streams, p.NStreams = true, s.c.Streams()
		p.StreamParts[0] = part(fg.Expand, nn.DeclineNoGraph, 1)
		p.StreamParts[1] = part(fg.Expand, nn.DeclineNoGraph, cpuRowsCap)
		collapse := fg.Collapse
		cw := nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}
		if i := s.lo - s.c.NLayer; i >= 0 && i < len(fg.MtpCollapse) {
			collapse, cw = fg.MtpCollapse[i], nn.PartWeights{Block: s.lo, Of: s.ws.tensors(s.lo, s.layer(s.lo))}
		}
		p.StreamParts[2] = part(collapse, nn.DeclineNoGraph, 1)
		p.StreamParts[3] = part(collapse, nn.DeclineNoGraph, cpuRowsCap)
		p.StreamW, p.CollapseW = nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}, cw
	}

	if g := fg.Pool; g != nil {
		p.Pool, p.PoolPart, p.PoolW = true, part(g, nn.DeclineNoGraph, s.maxSeq), nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}
		p.PoolRows, p.PoolWidth, p.PoolRaw = s.maxSeq, s.poolWidth(), len(g.Out) > 1
	}

	if i := s.lo - s.c.NLayer; i >= 0 && i < len(fg.MtpEntry) {
		g := fg.MtpEntry[i]
		p.Entry, p.EntryWidth = true, s.c.ResidW()
		p.EntryParts = [2]nn.PartPlan{part(g, nn.DeclineNoGraph, 1), part(g, nn.DeclineNoGraph, cpuRowsCap)}
		p.EntryW = nn.PartWeights{Block: s.lo, Of: s.ws.tensors(s.lo, s.layer(s.lo))}
	}

	p.Inputs[0] = part(fg.CpuEmbed, nn.DeclineNoEmbed, 1)
	p.Inputs[1] = part(fg.CpuEmbed, nn.DeclineNoEmbed, cpuRowsCap)
	p.Inputs[2], p.Inputs[3] = nn.PartPlan{Skip: true}, nn.PartPlan{Skip: true}
	if fg.Given != nil {
		p.Inputs[2] = part(fg.Given, nn.DeclineNoEmbed, 1)
		p.Inputs[3] = part(fg.Given, nn.DeclineNoEmbed, cpuRowsCap)
	}
	p.PleW = s.c.InputSideWidth()
	p.Given, p.WritesX = fg.Given != nil, fg.Given != nil && (len(fg.Given.Out) > 1 || p.PleW == 0)
	p.InputsW = nn.PartWeights{Block: cpuInputRows, Of: s.ws.tensors(cpuInputRows, nil)}

	tabs := s.tableGraphs()
	p.TableWidths = [2]int{s.c.RopeW(), s.c.LocalNRot()}
	p.Tables = make([]nn.TablePlan, len(tabs))
	for i, g := range tabs {
		// The tables are the model's rotaries; an instance whose config has
		// none (a tower's) composes none.
		if i < len(p.TableWidths) && p.TableWidths[i] == 0 {
			p.Tables[i] = nn.TablePlan{One: nn.PartPlan{Skip: true}, Many: nn.PartPlan{Skip: true}}
			continue
		}
		rows := cpuRowsCap
		if s.towered() {
			rows = s.maxSeq
		}
		p.Tables[i] = nn.TablePlan{One: s.tablePlan(g, 1), Many: s.tablePlan(g, rows)}
	}
	p.TablePlanes = [2][]float32{s.tablePlanes(0), s.tablePlanes(1)}
	p.TablesW = nn.PartWeights{Block: cpuTablesBlock, Of: s.ws.tensors(cpuTablesBlock, nil)}

	narrow, widen := graph.OpHalves, graph.OpWidenHalves
	var widths []int
	switch s.kvl.fmt {
	case nn.KVF16:
		widths = []int{kvHalfRow}
	case nn.KVQ8:
		narrow, widen = graph.OpQ8Rows, graph.OpWidenQ8Rows
		for li := s.lo; li < s.hi; li++ {
			if hd := s.kvlAt(li).headDim; !slices.Contains(widths, hd) {
				widths = append(widths, hd)
			}
		}
	}
	for _, fw := range widths {
		c := nn.KVConvPlan{FW: fw, HW: fw / 2}
		if s.kvl.fmt == nn.KVQ8 {
			c.HW = graph.Q8RowWords(fw)
		}
		nb := graph.New()
		nb.Out(nb.Emit(narrow, []graph.ValueID{nb.In("kv.f", c.FW)},
			[]graph.Result{{Name: "kv.h", Width: c.HW, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)...)
		c.Narrow = part(nb.G, nn.DeclineNoGraph, cpuRowsCap)
		wb := graph.New()
		h := wb.In("kv.h", c.HW)
		wb.G.Values[h].Elem = graph.ElemU32
		wb.Out(wb.Emit(widen, []graph.ValueID{h}, []graph.Result{{Name: "kv.f", Width: c.FW}}, nil, graph.Attrs{}, nil, nil)...)
		c.Widen = part(wb.G, nn.DeclineNoGraph, cpuRowsCap)
		p.KV = append(p.KV, c)
	}
	p.KVHalf = s.kvl.fmt == nn.KVF16
	return p
}

// tablePlan is rotary table graph g as a part at rows rows, the most marks a
// sequence holds the instance's.
func (s *Instance) tablePlan(g *graph.Graph, rows int) nn.PartPlan {
	return nn.PartPlan{Graph: g, Rows: rows, Seqs: 1, Marks: s.marksCap(), ActWindow: s.m.ActWindow,
		None: nn.DeclineNoGraph}
}

// placeParts makes the instance's parts on the CPU device from partsPlan.
func (s *Instance) placeParts() {
	pp := s.places[routeLanes].(nn.PartsPlacer).PlaceParts(s.partsPlan())
	s.kvConv = nil
	if pp.Head == nil {
		return
	}
	s.heads, s.inputs, s.tables = pp.Head, pp.Inputs, pp.Tables
	if pp.Streams != nil {
		s.streams = pp.Streams
	}
	if pp.Pool != nil {
		s.readout = pp.Pool
	}
	if pp.Entry != nil {
		s.entry = pp.Entry
	}
	if pp.KVConv != nil {
		s.kvConv = pp.KVConv
	}
}
