package cpudev

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Parts is an instance's parts on the CPU device; a part the plan states
// none of is nil.
type Parts struct {
	Head    *Head
	Streams *Streams
	Pool    *Pool
	Entry   *Entry
	Inputs  *Inputs
	Tables  *Tables
	KVConv  [nn.KVFormats]*KVConv
}

// PlaceParts composes every part of p and makes it the part the instance
// runs, each counted served or declined in the execution's steps. Nothing
// before the execution's first placement.
func (e *Exec) placeParts(p *PartsPlan) Parts {
	var out Parts
	if e.dev == nil {
		return out
	}
	var h [20]Part
	for i := range p.Head {
		h[i] = e.FormPart(p.Head[i])
	}
	out.Head = NewHead(e.dev, &e.steps, HeadParts{Head: h[0], Hidden: h[1], Rows: h[2], Pick: h[3], Sample: h[4], Walk: h[5],
		Row: h[6], Cut: h[7], WalkProb: h[8], CutProb: h[9], CutK: h[10], CutKProb: h[11], OutNorm: h[12], Logprobs: h[13],
		Mask: h[14], Decide: h[15], CutTyp: h[16], CutTypProb: h[17], CutKTyp: h[18], CutKTypProb: h[19]}, p.HeadNorm, boundWeights{e.b, p.HeadW}, p.D, p.NVocab, p.TopLog)
	if p.Streams {
		var sp [4]Part
		for i := range sp {
			sp[i] = e.FormPart(p.StreamParts[i])
		}
		out.Streams = NewStreams(e.dev, &e.steps, sp[0], sp[1], sp[2], sp[3], boundWeights{e.b, p.StreamW}, boundWeights{e.b, p.CollapseW}, p.NStreams, p.D)
		out.Streams.Fault = e.StreamFault
	}
	if p.Pool {
		part := e.FormPart(p.PoolPart)
		var weights []unsafe.Pointer
		if f := part.F; f != nil {
			ws, why := boundWeights{e.b, p.PoolW}.Bind(f)
			if why != nn.DeclineServes {
				part.F, part.Why = nil, why
				part.Err = fmt.Errorf("the readout's weights: %s", nn.DeclineName(int(why)))
			}
			weights = ws
		}
		out.Pool = NewPool(e.dev, &e.steps, part, weights, p.PoolRows, p.D, p.PoolWidth, p.PoolRaw)
	}
	if p.Entry {
		out.Entry = NewEntry(e.dev, &e.steps, e.FormPart(p.EntryParts[0]), e.FormPart(p.EntryParts[1]), boundWeights{e.b, p.EntryW}, p.EntryWidth)
	}
	var in [4]Part
	for i := range in {
		in[i] = e.FormPart(p.Inputs[i])
	}
	out.Inputs = NewInputs(e.dev, &e.steps, in[0], in[1], in[2], in[3], p.Given, p.WritesX, boundWeights{e.b, p.InputsW}, p.D, p.PleW)
	one, many := make([]Part, len(p.Tables)), make([]Part, len(p.Tables))
	for i, t := range p.Tables {
		one[i], many[i] = e.FormPart(t.One), e.FormPart(t.Many)
	}
	out.Tables = NewTables(e.dev, one, many, p.TablePlanes, p.TableWidths, boundWeights{e.b, p.TablesW})
	e.kvconv = nil
	for f, plans := range p.KV {
		if len(plans) == 0 {
			continue
		}
		conv := make([]KVConvForms, len(plans))
		for i, k := range plans {
			conv[i] = KVConvForms{FW: k.FW, HW: k.HW, Narrow: e.FormPart(k.Narrow), Widen: e.FormPart(k.Widen)}
		}
		kc := NewKVConv(e.dev, conv, nn.KVFmt(f) == nn.KVF16)
		out.KVConv[f] = kc
		if nn.KVFmt(f) == p.KVFmt {
			e.kvconv = kc
		}
	}
	return out
}
