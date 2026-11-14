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
// turns a picture's rows in one call) and a binary16 or quantized history's
// conversions (a binary16 history's halves in rows of kvHalfRow floats, a q8
// or block history's rows one a head width its layers hold).
func (s *Instance) partsPlan() *nn.PartsPlan {
	fg := s.m.FrontGraphs()
	p := &nn.PartsPlan{D: s.c.NEmbd, NVocab: s.c.NVocab, TopLog: model.MaxTopLogprobs,
		HeadNorm: fg.OutNorm != nil, HeadW: nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}}
	part := func(g *graph.Graph, rows int) nn.PartPlan { return s.partPlan(g, rows, nil) }
	p.Head[0] = part(fg.CpuHead, 1)
	p.Head[1] = part(fg.CpuHidden, 1)
	p.Head[2] = part(fg.CpuHidden, cpuRowsCap)
	p.Head[3] = part(fg.PickHead, 1)
	for i, g := range []*graph.Graph{fg.CpuSample, fg.CpuWalk, fg.CpuRow, fg.CpuCut, fg.CpuWalkProb, fg.CpuCutProb,
		fg.CpuCutK, fg.CpuCutKProb} {
		p.Head[4+i] = part(g, 1)
	}
	p.Head[12] = nn.PartPlan{Skip: true}
	if fg.OutNorm != nil {
		p.Head[12] = part(fg.OutNorm, cpuRowsCap)
	}
	p.Head[13] = part(fg.CpuLogprobs, 1)
	p.Head[14] = part(fg.CpuMask, 1)
	p.Head[15] = nn.PartPlan{Skip: true}
	if fg.CpuDecide != nil {
		p.Head[15] = part(fg.CpuDecide, 1)
	}
	for i, g := range []*graph.Graph{fg.CpuCutTyp, fg.CpuCutTypProb, fg.CpuCutKTyp, fg.CpuCutKTypProb} {
		p.Head[16+i] = part(g, 1)
	}
	p.Head[20], p.Head[21] = part(fg.CpuMiro, 1), part(fg.CpuMiroV1, 1)
	p.Head[22], p.Infill = part(fg.CpuWalkInfill, 1), fg.Infill
	p.Head[23] = part(fg.CpuAdaptive, 1)
	p.Head[24] = part(fg.CpuStage, 1)
	p.Head[25] = part(fg.CpuLogprobsRow, 1)

	if s.c.StatesStreams() {
		p.Streams, p.NStreams = true, s.c.Streams()
		p.StreamParts[0] = part(fg.Expand, 1)
		p.StreamParts[1] = part(fg.Expand, cpuRowsCap)
		collapse := fg.Collapse
		cw := nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}
		if i := s.lo - s.c.NLayer; i >= 0 && i < len(fg.MtpCollapse) {
			collapse, cw = fg.MtpCollapse[i], nn.PartWeights{Block: s.lo, Of: s.ws.tensors(s.lo, s.layer(s.lo))}
		}
		p.StreamParts[2] = part(collapse, 1)
		p.StreamParts[3] = part(collapse, cpuRowsCap)
		p.StreamW, p.CollapseW = nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}, cw
	}

	if g := fg.Pool; g != nil {
		p.Pool, p.PoolPart, p.PoolW = true, part(g, s.maxSeq), nn.PartWeights{Block: -1, Of: s.ws.tensors(-1, nil)}
		p.PoolRows, p.PoolWidth, p.PoolRaw = s.maxSeq, s.poolWidth(), len(g.Out) > 1
		p.PoolNorm = slices.ContainsFunc(g.In, func(v graph.ValueID) bool { return g.Values[v].Name == "embed.norm" })
	}

	if i := s.lo - s.c.NLayer; i >= 0 && i < len(fg.MtpEntry) {
		g := fg.MtpEntry[i]
		p.Entry, p.EntryWidth = true, s.c.ResidW()
		p.EntryParts = [2]nn.PartPlan{part(g, 1), part(g, cpuRowsCap)}
		p.EntryW = nn.PartWeights{Block: s.lo, Of: s.ws.tensors(s.lo, s.layer(s.lo))}
	}

	p.Inputs[0] = part(fg.CpuEmbed, 1)
	p.Inputs[1] = part(fg.CpuEmbed, cpuRowsCap)
	p.Inputs[2], p.Inputs[3] = nn.PartPlan{Skip: true}, nn.PartPlan{Skip: true}
	// A tower's rows are its entry's: the text model's given rows' graph
	// (a position row, per-layer inputs) is not theirs, and run over them it
	// wrote the text model's rows in their place.
	given := fg.Given
	if s.towered() {
		given = nil
	}
	if given != nil {
		p.Inputs[2] = part(given, 1)
		p.Inputs[3] = part(given, cpuRowsCap)
	}
	p.PleW = s.c.InputSideWidth()
	p.Given, p.WritesX = given != nil, given != nil && (len(given.Out) > 1 || p.PleW == 0)
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

	for f := nn.KVF16; int(f) < nn.KVFormats; f++ {
		narrow, widen := graph.OpHalves, graph.OpWidenHalves
		widths := []int{kvHalfRow}
		at := graph.Attrs{Block: f.Block(), BF16: f == nn.KVBF16}
		if f.Quantized() {
			narrow, widen, widths = graph.OpQ8Rows, graph.OpWidenQ8Rows, nil
			if at.Block != 0 {
				narrow, widen = graph.OpBlockRows, graph.OpWidenBlockRows
			}
			for li := s.lo; li < s.hi; li++ {
				if hd := s.kvlAt(li).headDim; !slices.Contains(widths, hd) {
					widths = append(widths, hd)
				}
			}
		}
		for _, fw := range widths {
			c := nn.KVConvPlan{FW: fw, HW: fw / 2}
			switch {
			case at.Block != 0:
				c.HW = graph.BlockRowWords(at.Block, fw)
			case f == nn.KVQ8:
				c.HW = graph.Q8RowWords(fw)
			}
			nb := graph.New()
			nb.Out(nb.Emit(narrow, []graph.ValueID{nb.In("kv.f", c.FW)},
				[]graph.Result{{Name: "kv.h", Width: c.HW, Elem: graph.ElemU32}}, nil, at, nil, nil)...)
			c.Narrow = part(nb.G, cpuRowsCap)
			wb := graph.New()
			h := wb.In("kv.h", c.HW)
			wb.G.Values[h].Elem = graph.ElemU32
			wb.Out(wb.Emit(widen, []graph.ValueID{h}, []graph.Result{{Name: "kv.f", Width: c.FW}}, nil, at, nil, nil)...)
			c.Widen = part(wb.G, cpuRowsCap)
			p.KV[f] = append(p.KV[f], c)
		}
	}
	p.KVFmt = s.kvl.fmt
	return p
}

// tablePlan is rotary table graph g as a part at rows rows, the most marks a
// sequence holds the instance's.
func (s *Instance) tablePlan(g *graph.Graph, rows int) nn.PartPlan {
	if g == nil {
		return nn.PartPlan{Skip: true}
	}
	return nn.PartPlan{Graph: g, Rows: rows, Seqs: 1, Marks: s.marksCap(), ActWindow: s.m.ActWindow}
}

// placeParts makes the instance's parts on the CPU device from partsPlan.
func (s *Instance) placeParts() {
	pp := s.places[routeLanes].(nn.PartsPlacer).PlaceParts(s.partsPlan())
	s.kvConv, s.kvConvs = nil, [nn.KVFormats]nn.KVConvDevice{}
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
	s.kvConvs = pp.KVConv
	s.kvConv = pp.KVConv[s.kvl.fmt]
}
