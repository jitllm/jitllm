package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// PartsPlan is what the CPU device places an instance's parts from: each
// part's graphs as its plans, the weights each binds and the widths it runs
// at. Head is the head's and the samplers' plans in HeadParts' order;
// Streams, Pool and Entry say the instance has the residual streams' mix, a
// readout and a prediction block's entry; Tables is each rotary's table, a
// row's and a call's; KV is the history's conversions, nil for a history of
// floats.
type PartsPlan struct {
	Head     [15]PartPlan
	HeadNorm bool
	HeadW    Weights
	// D is a row's width, NVocab the vocabulary, TopLog the most log
	// probabilities a draw returns.
	D, NVocab, TopLog int

	Streams             bool
	StreamParts         [4]PartPlan
	StreamW, CollapseW  Weights
	NStreams            int
	StreamFault         bool
	Pool                bool
	PoolPart            PartPlan
	PoolW               Weights
	PoolRows, PoolWidth int
	PoolRaw             bool
	Entry               bool
	EntryParts          [2]PartPlan
	EntryW              Weights
	EntryWidth          int
	Inputs              [4]PartPlan
	Given, WritesX      bool
	InputsW             Weights
	PleW                int
	Tables              []TablePlan
	TablePlanes         [2][]float32
	TableWidths         [2]int
	TablesW             Weights
	KV                  []KVConvPlan
	KVHalf              bool
}

// TablePlan is a rotary's table: a row's and a call's rows'.
type TablePlan struct{ One, Many PartPlan }

// KVConvPlan is a history's conversions at one row width: the floats FW
// wide into HW words and back.
type KVConvPlan struct {
	FW, HW        int
	Narrow, Widen PartPlan
}

// Parts is an instance's parts on the CPU device; a part the plan states
// none of is nil.
type Parts struct {
	Head    *Head
	Streams *Streams
	Pool    *Pool
	Entry   *Entry
	Inputs  *Inputs
	Tables  *Tables
	KVConv  *KVConv
}

// PlaceParts composes every part of p and makes it the part the instance
// runs, each counted served or declined in the execution's steps. Nothing
// before the execution's first placement.
func (e *Exec) PlaceParts(p *PartsPlan) Parts {
	var out Parts
	if e.dev == nil {
		return out
	}
	var h [15]Part
	for i := range p.Head {
		h[i] = e.FormPart(p.Head[i])
	}
	out.Head = NewHead(e.dev, &e.steps, HeadParts{Head: h[0], Hidden: h[1], Rows: h[2], Pick: h[3], Sample: h[4], Walk: h[5],
		Row: h[6], Cut: h[7], WalkProb: h[8], CutProb: h[9], CutK: h[10], CutKProb: h[11], OutNorm: h[12], Logprobs: h[13],
		Mask: h[14]}, p.HeadNorm, p.HeadW, p.D, p.NVocab, p.TopLog)
	if p.Streams {
		var sp [4]Part
		for i := range sp {
			sp[i] = e.FormPart(p.StreamParts[i])
		}
		out.Streams = NewStreams(e.dev, &e.steps, sp[0], sp[1], sp[2], sp[3], p.StreamW, p.CollapseW, p.NStreams, p.D)
		out.Streams.Fault = p.StreamFault
	}
	if p.Pool {
		part := e.FormPart(p.PoolPart)
		var weights []unsafe.Pointer
		if f := part.F; f != nil {
			ws, why := p.PoolW.Bind(f)
			if why != nn.DeclineServes {
				part.F, part.Why = nil, why
			}
			weights = ws
		}
		out.Pool = NewPool(e.dev, &e.steps, part, weights, p.PoolRows, p.D, p.PoolWidth, p.PoolRaw)
	}
	if p.Entry {
		out.Entry = NewEntry(e.dev, &e.steps, e.FormPart(p.EntryParts[0]), e.FormPart(p.EntryParts[1]), p.EntryW, p.EntryWidth)
	}
	var in [4]Part
	for i := range in {
		in[i] = e.FormPart(p.Inputs[i])
	}
	out.Inputs = NewInputs(e.dev, &e.steps, in[0], in[1], in[2], in[3], p.Given, p.WritesX, p.InputsW, p.D, p.PleW)
	one, many := make([]Part, len(p.Tables)), make([]Part, len(p.Tables))
	for i, t := range p.Tables {
		one[i], many[i] = e.FormPart(t.One), e.FormPart(t.Many)
	}
	out.Tables = NewTables(e.dev, one, many, p.TablePlanes, p.TableWidths, p.TablesW)
	if p.KV != nil {
		conv := make([]KVConvForms, len(p.KV))
		for i, k := range p.KV {
			conv[i] = KVConvForms{FW: k.FW, HW: k.HW, Narrow: e.FormPart(k.Narrow), Widen: e.FormPart(k.Widen)}
		}
		out.KVConv = NewKVConv(e.dev, conv, p.KVHalf)
	}
	return out
}
