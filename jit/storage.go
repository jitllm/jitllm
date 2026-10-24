package jit

import (
	"errors"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// Source is where a storage's contents come from before a step runs: what the
// engine puts there, or that the step writes it first.
type Source uint8

const (
	// SourceStep is a value or scratch a kernel of the step writes before
	// any reads it.
	SourceStep Source = iota
	// SourceWeight is a weight or norm of the model, as the container
	// stores it: Storage.Role of Storage.Block.
	SourceWeight
	// SourceInput is the step's input rows: each row's token embedding.
	SourceInput
	// SourceRows is the rows' history descriptors, RowWords a row (RowTab,
	// RowStart, RowEnd, RowWrite).
	SourceRows
	// SourcePages is the history's page table, the engine's.
	SourcePages
	// SourceQueryOffset is where the step's query region starts.
	SourceQueryOffset
	// SourceRopePositions is the step's live row count, then a position a
	// row.
	SourceRopePositions
	// SourceRopePlanes is the model's rotary planes (nn.Rope.TabPlanes).
	SourceRopePlanes
	// SourceKeys and SourceValues are a part's history pools: pages of
	// Step.Page positions of the width its kernels store, kept from step to
	// step.
	SourceKeys
	SourceValues
	// SourceWords is a block of constants the plan carries.
	SourceWords
	// SourceTransfer is a value the part before wrote on another device,
	// carried between them (Plan.Transfers).
	SourceTransfer
	// SourceTokens is the step's tokens, a word a row.
	SourceTokens
	// SourceState is a part's recurrent state, a sequence's own: zero at
	// its start, kept from step to step.
	SourceState
	// SourceRequest is a step's request, the caller's words: a decision's
	// question type, then its marker count, then a marker row a step row.
	SourceRequest
	// SourceWindows is the rows' descriptors as SourceRows lays them, each
	// row's attended range its window.
	SourceWindows
	// SourceRopeAxisPositions is a step's positions on a rotary's several
	// axes: the live row count, then RopeAxes words a row, axis a of row r
	// at 1+r*RopeAxes+a.
	SourceRopeAxisPositions
	numSources
)

var sourceNames = [numSources]string{"step", "weight", "input", "rows", "pages", "query offset",
	"rotary positions", "rotary planes", "keys", "values", "words", "transfer", "tokens", "state", "request", "windows",
	"rotary axis positions"}

func (s Source) String() string {
	if s < numSources {
		return sourceNames[s]
	}
	return fmt.Sprintf("source %d", s)
}

// Engine reports whether the engine sizes the storage from what it owns --
// its page table, the model's rotary -- rather than from the plan's Bytes.
func (s Source) Engine() bool {
	return s == SourcePages || s == SourceRopePlanes
}

// A row's history descriptor: the words the engine writes for each row of a
// step.
const (
	RowTab   = rowTab
	RowStart = rowStart
	RowEnd   = rowEnd
	RowWrite = rowWrite
	RowWords = rowWords
)

// RopeAxes is a row's positions in SourceRopeAxisPositions.
const RopeAxes = ropeAxes

// Storage is one buffer a step binds: the key every binding of it shares,
// where its contents come from, and the bytes the step reaches in it (0 for
// a storage the engine sizes).
type Storage struct {
	Key    Operand
	Source Source
	// Role and Block are a weight's tensor, Block -1 for one outside the
	// blocks, Index the index it is found at (-1 any: Part.Index), and
	// Region which of its stored regions (Operand.Region).
	Role   jlm.Role
	Block  int
	Index  int32
	Region int
	// Segment is the segment of the part that binds it first: a weight's
	// is the view its Role and Block are found in (Model.Segment).
	Segment Segment
	Bytes   int64
	Words   []uint32
}

// StorageKeys names the storage an operand lands in: a graph value by its
// part and id, a part's boundary value as the value it is the storage of --
// a block reads the residual the one before it wrote.
type StorageKeys struct {
	alias map[[2]int][2]int
}

// NewStorageKeys is the storage keys of the model's parts.
func NewStorageKeys(parts []Part) StorageKeys {
	out := map[[2]int][2]int{}
	for i := 1; i < len(parts); i++ {
		a, b := parts[i-1].Graph, parts[i].Graph
		if len(a.Out) == 0 || len(b.In) == 0 || fed(parts[i], 0) {
			continue
		}
		from := [2]int{i - 1, int(a.Out[0])}
		if x, ok := out[from]; ok {
			from = x
		}
		out[[2]int{i, int(b.In[0])}] = from
	}
	for i, p := range parts {
		for _, f := range p.Feeds {
			out[[2]int{i, int(p.Graph.In[f.In])}] = [2]int{f.From, int(parts[f.From].Graph.Out[f.Out])}
		}
	}
	return StorageKeys{alias: out}
}

// Key is the storage operand o names.
func (s StorageKeys) Key(o Operand) Operand {
	if o.Kind != OperandValue {
		return o
	}
	at := [2]int{o.Of, int(o.Value)}
	if x, ok := s.alias[at]; ok {
		at = x
	}
	return Operand{Kind: OperandValue, Of: at[0], Value: graph.ValueID(at[1])}
}

// tableSources is the source of each table the decompositions declare.
var tableSources = map[string]Source{
	tableRows.Name: SourceRows, tablePages.Name: SourcePages, tableQOff.Name: SourceQueryOffset,
	tablePos.Name: SourceRopePositions, tableAxisPos.Name: SourceRopeAxisPositions, tablePlanes.Name: SourceRopePlanes,
	tableKeys.Name: SourceKeys, tableValues.Name: SourceValues, tableRequest.Name: SourceRequest, tableWindows.Name: SourceWindows, tableAngles.Name: SourceStep,
	tableTokens.Name: SourceTokens,
}

// sourceOf is where storage key's contents come from: written says a kernel
// of the step writes it.
func sourceOf(key Operand, written bool, words []uint32) (Source, error) {
	switch {
	case words != nil:
		return SourceWords, nil
	case key.Kind == OperandWeight || key.Kind == OperandVector:
		return SourceWeight, nil
	case key.Kind == OperandTable:
		name := key.Name
		if strings.HasPrefix(name, "rec.") {
			return SourceState, nil
		}
		// A second rotary's table and planes are the first's sources.
		for _, t := range []Operand{tableAngles, tablePlanes} {
			if strings.HasPrefix(name, t.Name+".") {
				name = t.Name
			}
		}
		s, ok := tableSources[name]
		if !ok {
			return 0, fmt.Errorf("jit: table %q has no source", key.Name)
		}
		return s, nil
	case written:
		return SourceStep, nil
	case key.Kind == OperandValue:
		return SourceInput, nil
	}
	return 0, fmt.Errorf("jit: %v storage %q is read and nothing writes it", key.Kind, key.Name)
}

// bytes is what a buffer's accesses reach: its shape's elements, or its
// extent's, or its words; 0 where it states none.
func (b Buf) bytes() int64 {
	n := int64(ElemBytes(b))
	switch {
	case b.Shape != nil:
		for _, x := range b.Shape {
			n *= int64(x)
		}
		return n
	case b.Extent > 0:
		return n * int64(b.Extent)
	case b.Words != nil:
		return int64(4 * len(b.Words))
	}
	return 0
}

// storages gathers the plan's bindings into its storages, each binding
// naming its own.
func (p *Plan) storages(parts []Part) error {
	keys := NewStorageKeys(parts)
	at := map[Operand]int{}
	written, read := map[int]bool{}, map[int]bool{}
	for pi := range p.Parts {
		for ci := range p.Parts[pi].Codes {
			c := &p.Parts[pi].Codes[ci]
			for bi := range c.Binds {
				b := &c.Binds[bi]
				key := keys.Key(b.Operand)
				words := b.Words
				if words != nil {
					key.Name = fmt.Sprintf("%s#%d.%d", key.Name, pi, ci)
				}
				i, ok := at[key]
				if !ok {
					i = len(p.Storages)
					at[key] = i
					st := Storage{Key: key, Role: key.Role, Block: -1, Index: -1, Region: key.Region, Words: words}
					// A weight of a part that is not a block's is the model's.
					if key.Of >= 0 && key.Of < len(parts) && parts[key.Of].Kind == PartBlock {
						st.Block = parts[key.Of].Block
					}
					if key.Of >= 0 && key.Of < len(parts) {
						st.Index = parts[key.Of].Index()
					}
					// The binding part's: a table no part owns (its key's
					// Of is 0) is not part 0's segment's.
					st.Segment = p.Parts[pi].Part.Segment
					p.Storages = append(p.Storages, st)
				}
				b.Storage = i
				p.Storages[i].Bytes = max(p.Storages[i].Bytes, b.Bytes)
				written[i] = written[i] || b.Writes
				read[i] = read[i] || b.Read
			}
		}
	}
	carried := map[Operand]bool{}
	for _, tr := range p.Transfers {
		carried[keys.Key(tr.Value)] = true
	}
	var errs []error
	for i := range p.Storages {
		s := &p.Storages[i]
		if carried[s.Key] {
			s.Source = SourceTransfer
			continue
		}
		// A storage no kernel reads is the step's own whatever it held.
		src, err := sourceOf(s.Key, written[i] || !read[i] && s.Words == nil, s.Words)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		s.Source = src
		if src == SourceInput && s.Key != keys.input(parts) {
			errs = append(errs, fmt.Errorf("jit: value %d of part %d is read and nothing writes it", s.Key.Value, s.Key.Of))
		}
	}
	return errors.Join(errs...)
}

// input is the storage of the step's input rows: the input of the first
// part that takes one.
func (s StorageKeys) input(parts []Part) Operand {
	for i, p := range parts {
		if len(p.Graph.In) > 0 {
			return s.Key(Operand{Kind: OperandValue, Of: i, Value: p.Graph.In[0]})
		}
	}
	return Operand{}
}
