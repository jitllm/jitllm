package cpudev

import (
	"fmt"
	"slices"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// Src is where a call word's value comes from.
type Src uint8

// The argument sources of a composed CPU kernel's call words.
const (
	SrcValue Src = iota
	SrcX
	SrcOwn
	SrcWeight
	SrcVector
	SrcRope
	SrcKeys
	SrcRopeRows
	SrcRopeMarks
	SrcValues
	SrcPages
	SrcRows
	SrcZero
	SrcCoord
	SrcWord
	SrcCall
	// SrcRecState and SrcRecConv are a linear block's recurrent states
	// and its convolution's windows, a slot a sequence, read and written in
	// place; SrcRecRows the call's recurrent descriptor, which names the
	// call's sequence's slot. SrcRecConvOut is where a call's windows are
	// written: in place for a row, apart for a step of several (its runs
	// past the step's write a run's window again from the window read).
	SrcRecState
	SrcRecConv
	SrcRecConvOut
	// SrcShare is the rows a call of a Share code takes (composer.Code.Share).
	SrcShare
	SrcRecRows
	// SrcPoolPos is the positions table a pooled readout reads its live
	// count from (cpupool.go).
	SrcPoolPos
	// SrcIO is an input or output of a graph whose output is not its
	// input (the head's, the sampler's): the bound buffer of that place,
	// the inputs' then the outputs' (Call.io).
	SrcIO
	// SrcSide is a block graph's input past its residual streams, the
	// rows' per-layer inputs (Call.side).
	SrcSide
	// SrcRequest is the step's request words (Call.req), SrcPos
	// its positions (Call.pos).
	SrcRequest
	SrcPos
	// SrcPlanes is the block's rotary planes (nn.Rope.TabPlanes), which
	// a kernel turning at positions of its own reads (Call.Planes).
	SrcPlanes
	// SrcTokens is the step's tokens, a word a row (Call.tok).
	SrcTokens
	// NumSrc is the sources' count.
	NumSrc
)

// Arg is one call word: its byte field in cpu.Args and where its value
// comes from, with the elements it is offset by.
type Arg struct {
	field  uintptr
	src    Src
	region uint8
	coord  uint8
	role   jlm.Role
	i      int
	off    int
	elem   int
	val    uint64
	// stream is the residual stream a SrcX address is in.
	stream int
	// per is the elements the address moves a unit of each call
	// coordinate (composer.CallArg.Per), moves whether any does.
	per   [2]int
	moves bool
	// inst is a rotary table's instance (SrcRope, SrcPlanes).
	inst int
	// from is the first stored row a weight binds (graph.Binding.From).
	from int
}

// Code is one kernel of a form, mapped, and its call words; reads and
// writes are the residual streams whose input it reads and whose output it
// writes, a bit a stream.
type Code struct {
	reads, writes uint64
	code          *cpu.Code
	args          []Arg
	calls         []int
	n             int
	// serial is a kernel over calls that write a buffer every call shares.
	serial bool
	// live is the code's hold on a call's rows (composer.Code's Live and
	// LiveUnits): a call of rows rows runs the units of the leading axis
	// they reach, each of inner calls (composer.RowCalls).
	live  composer.Code
	inner int
	// tiles is a kernel whose inner calls tile a weight (composer.Code.TileMajor):
	// its calls run a tile for every live row before the next tile.
	tiles bool
	// name is the composer's name for the kernel, and label its region's
	// pprof label: the name, or the names of the kernels its phase runs;
	// nanos and runs its time and launches under WithProfile; regs and size
	// its register facts and its code's bytes.
	name, label string
	regs        composer.RegUse
	size        int
	desc        string
	nanos, runs int64
	// pcalls, pcallNs, pstartNs and pmaxNs are its calls under WithProfile:
	// their count, time, start after the launch's and each launch's last start.
	pcalls, pcallNs, pstartNs, pmaxNs int64
	// rd and wr are the buffers the kernel's calls read and write, as the
	// call site binds them; join is how many kernels after it run in its
	// pool phase, none reading or writing what another of the phase writes,
	// and ends their calls' running totals (group).
	rd, wr []bufRef
	join   int
	// chain is how many kernels after it, some reading what one before
	// writes, run in its phase unit by unit (chainPhases), of units units;
	// sliced is the rows of each buffer a call reaches alone (Code.slice).
	chain, units int
	sliced       []bufSlice
	// share is the code's hold on a weight's rows sized at the launch
	// (composer.Code.Share).
	share composer.Code
}

type bufSlice struct {
	buf bufRef
	per int
}

// bufRef is a buffer a call word binds, by where the call site finds it.
type bufRef struct {
	src    Src
	region uint8
	role   jlm.Role
	i      int
	stream int
}

// Form is a block's kernels for the CPU at one composer Key, mapped
// once for the model, and the words of each of its own buffers.
type Form struct {
	Codes []Code
	// Weights is the weights and vectors the codes bind, each once: a
	// call's addresses of them are its Call.Weights, in this order.
	Weights []Weight
	// Rec says the graph steps a recurrent state: a recurrent kind's block
	// whose convolution is centred over the call keeps none.
	Rec bool
	// Rows is the rows the form holds: 1 for decode's, else its capacity;
	// Keys the positions of a sequence's history it attends at most.
	Rows, Keys int
	// Unit is the most rows a unit of a kernel's leading axis holds
	// (composer.RowJam): a call's rows past a whole number of them are run
	// from a copy of the residual, so a unit's rows past the call's write
	// no other rows.
	Unit int
	// Own is each own buffer's words, Consts an own buffer's constant
	// words, by its index.
	Own    []int
	Consts map[int][]uint32
	// KeysEnd is the codes up to the last that writes the history: a
	// bidirectional run of more rows than a call runs them over every row
	// first. KeysX says one of them writes the residual.
	KeysEnd int
	KeysX   bool
	// Cross is the own buffers Codes[:KeysEnd] write and the rest read:
	// what a bidirectional run of more rows than a call holds for each
	// call between its two passes.
	Cross []int
	// A form over banks in the host's expert pages runs in two halves: its
	// codes before Split, up to the selection, then the rest, which read
	// the banks once the selected experts' pages are resident and held.
	// Sel is the own buffer holding the selection, of SelN words, and WSel
	// the picks' weights' (weightsOf), -1 where none holds them, read by
	// the codes WReads.
	Split, Sel, SelN, WSel int
	WReads                 []int
	// Toks says a code reads the rows' tokens (SrcTokens).
	Toks bool
	// A form gathering the n-gram table's rows likewise runs to RowSplit,
	// its first code that reads the table, then reads in the rows the own
	// buffer RowSel names, RowN a call's row, before the rest.
	RowSplit, RowSel, RowN int
}

// ownUse is a code's use of an own buffer, by their indices.
type ownUse struct{ code, buf int }

// Close unmaps f's kernels.
func (f *Form) Close() {
	for i := range f.Codes {
		f.Codes[i].code.Close()
	}
}

// IO is the most inputs and outputs a graph apart from its residual
// binds: the sampler's candidates read three and write two.
const IO = 6

// DeclineError is a form the CPU device refuses by a named reason.
type DeclineError struct {
	Why composer.Decline
	Err error
}

func (e DeclineError) Error() string {
	return fmt.Sprintf("%s: %v", e.Why.String(), e.Err)
}

// stepsState reports whether gr reads or writes a recurrent state.
func stepsState(gr *graph.Graph) bool {
	for _, o := range gr.Ops {
		for _, r := range slices.Concat(o.Reads, o.Writes) {
			if r.State == graph.StateRecState || r.State == graph.StateRecConv {
				return true
			}
		}
	}
	return false
}

// BuildForm maps codes onto h and plans each call word's source: the
// graph's input and output the residual, or with apart each an io place of
// its own. A whole form is run at its rows and no fewer (a convolutional
// block's, over its grid), so its kernels need not be held to a call's rows.
func BuildForm(h *cpu.Target, gr *graph.Graph, codes []composer.Code, rows int, apart, whole bool) (*Form, error) {
	if apart && len(gr.In)+len(gr.Out) > IO {
		return nil, fmt.Errorf("a graph of %d inputs and %d outputs, past the %d places the CPU's call site binds", len(gr.In), len(gr.Out), IO)
	}
	f := &Form{Consts: map[int][]uint32{}, Rows: rows, Sel: -1, WSel: -1, RowSel: -1, Rec: stepsState(gr)}
	selV, wsV, rowV := selOf(gr), weightsOf(gr), TableRowsOf(gr)
	wOwn := map[graph.ValueID]int{}
	var wrote, read []ownUse
	// A kernel's constant block is its own: two kernels' blocks may share a
	// name and hold different words.
	type ownKey struct {
		o    composer.Operand
		code int
	}
	own := map[ownKey]int{}
	ownOf := func(o composer.Operand, code, words int, consts []uint32) int {
		key := ownKey{o, -1}
		if consts != nil {
			key.code = code
		}
		if i, ok := own[key]; ok {
			f.Own[i] = max(f.Own[i], words)
			return i
		}
		f.Own = append(f.Own, words)
		i := len(f.Own) - 1
		own[key] = i
		if consts != nil {
			f.Consts[i] = consts
		}
		return i
	}
	for ci, c := range codes {
		if c.CPU == nil {
			f.Close()
			return nil, fmt.Errorf("%s has no CPU code", c.Name)
		}
		cc := Code{calls: c.Calls, n: 1, name: c.Name, regs: c.Regs, size: len(c.CPU), share: composer.Code{Share: c.Share, ShareUnit: c.ShareUnit}}
		if c.Share > 0 && len(c.Calls) != 1 {
			f.Close()
			return nil, fmt.Errorf("%s: a share of rows over calls %v", c.Name, c.Calls)
		}
		for _, n := range c.Calls {
			cc.n *= n
		}
		if rows > 1 && !c.Scan && !whole && (c.Live > 0 || len(c.Reach) > 0) {
			// A form of several rows runs the units of its calls' leading
			// axis the call's rows reach: a kernel not held to them would
			// read and write rows past the call's. A scan's calls are its
			// state's rows, each reading the call's rows its descriptor
			// names; a kernel that holds no rows of a step (a table over the
			// history) runs whole (composer.Code.LiveUnitsOf).
			if c.Live == 0 || len(c.Calls) == 0 || c.Calls[0] != c.LiveUnits {
				f.Close()
				var binds []string
				for _, b := range c.Binds {
					n := fmt.Sprintf("%s/v%d/r%d", b.Operand.Name, b.Operand.Value, b.Operand.Role)
					if b.Writes {
						n += "(w)"
					}
					binds = append(binds, n)
				}
				return nil, fmt.Errorf("%s at %d rows: its calls %v hold no rows of a call (live %d of %d units; passes %v; binds %v)",
					c.Name, rows, c.Calls, c.Live, c.LiveUnits, c.Passes, binds)
			}
			cc.live, cc.inner = composer.Code{Live: c.Live, LiveUnits: c.LiveUnits}, cc.n/c.Calls[0]
			cc.tiles = c.TileMajor && cc.inner > 1
			f.Unit = max(f.Unit, c.Live/c.LiveUnits)
		}
		bytes := map[composer.Operand]int64{}
		reach := map[composer.Operand][]int{}
		for _, r := range c.Reach {
			reach[r.Operand] = r.Per
		}
		// rw is each operand's use: 1 read, 2 written.
		rw := map[composer.Operand]uint8{}
		for _, bd := range c.Binds {
			if o := bd.Operand; bd.Writes && o.Kind == composer.OperandTable && (o.Table == composer.TableKeys || o.Table == composer.TableValues) {
				f.KeysEnd = ci + 1
			}
			if o := bd.Operand; o.Kind == composer.OperandValue {
				if k := streamIn(gr, o.Value); k >= 0 && bd.Read {
					cc.reads |= 1 << k
				}
				if k := slices.Index(gr.Out, o.Value); k >= 0 && bd.Writes {
					cc.writes |= 1 << k
				}
			}
			o := bd.Operand
			o.Copy = 0
			bytes[o] = max(bytes[o], bd.Bytes)
			if bd.Read {
				rw[o] |= 1
			}
			if bd.Writes {
				rw[o] |= 2
			}
			if bd.Writes && bd.Operand.Kind == composer.OperandScratch && !bd.Own && !callsOwnScratch(c, bd.Operand, bd.Bytes) {
				cc.serial = true
			}
			if bd.Writes {
				n := o.Name
				if o.Kind == composer.OperandValue && int(o.Value) < len(gr.Values) {
					n = gr.Values[o.Value].Name
				}
				cc.desc += " " + n
			}
		}
		for _, a := range c.Args {
			// A pair's second head's operand is the first's storage at its
			// offset.
			o := a.Operand
			o.Copy = 0
			ar := Arg{field: uintptr(a.Field), off: a.Offset, elem: 4}
			if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(cpu.Args{}) {
				f.Close()
				return nil, fmt.Errorf("%s: a call word at %d past the argument block", c.Name, a.Field)
			}
			if a.Elem > 0 {
				ar.elem = a.Elem
			}
			if len(a.Per) > len(ar.per) || len(a.Per) > len(c.Calls) {
				f.Close()
				return nil, fmt.Errorf("%s: an address moving along %d axes of a calls' space of %d", c.Name, len(a.Per), len(c.Calls))
			}
			for ax, n := range a.Per {
				ar.per[ax] = n
				ar.moves = ar.moves || n != 0
			}
			switch {
			case a.Coord > 0:
				ar.src, ar.coord = SrcCoord, uint8(a.Coord-1)
			case a.Share:
				ar.src = SrcShare
			case a.Length:
				ar.src, ar.val = SrcValue, uint64(a.Extent)
			case a.Value:
				f.Close()
				return nil, fmt.Errorf("%s: a count the caller computes", c.Name)
			case a.Moving:
				f.Close()
				return nil, fmt.Errorf("%s: an offset of %v the caller computes", c.Name, o)
			case a.Words != nil:
				ar.src, ar.i = SrcOwn, ownOf(o, ci, len(a.Words), a.Words)
			case apart && o.Kind == composer.OperandValue && slices.Contains(gr.In, o.Value):
				ar.src, ar.i = SrcIO, slices.Index(gr.In, o.Value)
			case apart && o.Kind == composer.OperandValue && slices.Contains(gr.Out, o.Value):
				ar.src, ar.i = SrcIO, len(gr.In)+slices.Index(gr.Out, o.Value)
			case o.Kind == composer.OperandValue && streamIn(gr, o.Value) >= 0:
				ar.src, ar.stream = SrcX, streamIn(gr, o.Value)
			case o.Kind == composer.OperandValue && slices.Contains(gr.Out, o.Value):
				ar.src, ar.stream = SrcX, slices.Index(gr.Out, o.Value)
			case o.Kind == composer.OperandValue && len(gr.In) == len(gr.Out)+1 && o.Value == gr.In[len(gr.Out)]:
				ar.src = SrcSide
			case o.Kind == composer.OperandValue:
				ar.src, ar.i = SrcOwn, ownOf(o, ci, max(gr.Values[o.Value].Width, int(bytes[o]/4))+a.Offset, nil)
				switch o.Value {
				case selV:
					f.Sel = ar.i
				case rowV:
					f.RowSel = ar.i
				default:
					if slices.Contains(wsV, o.Value) {
						wOwn[o.Value] = ar.i
					}
				}
			case o.Kind == composer.OperandScratch:
				ar.src, ar.i = SrcOwn, ownOf(o, ci, int(bytes[o]/4)+a.Offset+1, nil)
			case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3:
				ar.src, ar.role, ar.region, ar.from = SrcWeight, o.Role, uint8(o.Region), o.Instance
				ar.i = f.weight(Weight{Role: o.Role, Region: uint8(o.Region), From: o.Instance})
			case o.Kind == composer.OperandVector:
				ar.src, ar.role = SrcVector, o.Role
				ar.i = f.weight(Weight{Vector: true, Role: o.Role})
			case o.Kind == composer.OperandTable:
				switch t := o.Table; {
				case t == composer.TableAngles && o.Instance < 2:
					ar.src, ar.inst = SrcRope, o.Instance
				case t == composer.TablePlanes && o.Instance < 2:
					ar.src, ar.inst = SrcPlanes, o.Instance
				case t == composer.TableRopeRows:
					ar.src = SrcRopeRows
				case t == composer.TableRopeMarks:
					ar.src = SrcRopeMarks
				case t == composer.TableKeys:
					ar.src = SrcKeys
				case t == composer.TableValues:
					ar.src = SrcValues
				case t == composer.TablePages:
					ar.src = SrcPages
				case t == composer.TableRows, t == composer.TableWindows:
					// A row's descriptor is its own layer's: its range the
					// layer's window where it has one (kvPages.keyStart).
					ar.src = SrcRows
				case t == composer.TableCall:
					ar.src = SrcCall
				case t == composer.TableRecState, t == composer.TableRecStateOut:
					ar.src = SrcRecState
				case t == composer.TableRecConv:
					ar.src = SrcRecConv
				case t == composer.TableRecConvOut:
					ar.src = SrcRecConvOut
				case t == composer.TableRecRows:
					ar.src = SrcRecRows
				case t == composer.TableQOff:
					ar.src = SrcZero
				case t == composer.TableRequest:
					ar.src = SrcRequest
				case t == composer.TableTokens:
					ar.src, f.Toks = SrcTokens, true
				case t == composer.TablePos:
					// A pooled readout reads its live count (SrcPoolPos), any
					// other graph its rows' positions (Call.pos).
					ar.src = SrcPos
					if slices.ContainsFunc(gr.Ops, func(o graph.Op) bool { return o.Kind == graph.OpPool }) {
						ar.src = SrcPoolPos
					}
				default:
					f.Close()
					return nil, DeclineError{composer.DeclineTable, fmt.Errorf("%s reads the table %s", c.Name, o.Name)}
				}
			default:
				f.Close()
				return nil, fmt.Errorf("%s binds %+v, which the CPU's call site does not provide", c.Name, o)
			}
			if ar.src == SrcOwn && a.Words == nil {
				if rw[o]&2 != 0 {
					wrote = append(wrote, ownUse{ci, ar.i})
				}
				if rw[o] != 2 {
					read = append(read, ownUse{ci, ar.i})
				}
			}
			if a.Word {
				if ar.src != SrcOwn {
					f.Close()
					return nil, fmt.Errorf("%s passes a word of %+v by value, which the CPU's call site holds no words of", c.Name, o)
				}
				ar.src = SrcWord
			}
			cc.args = append(cc.args, ar)
			if ar.src != SrcValue && ar.src != SrcCoord && ar.src != SrcShare {
				buf := bufRef{src: ar.src, region: ar.region, role: ar.role, i: ar.i, stream: ar.stream}
				if buf.src == SrcWord {
					buf.src = SrcOwn
				}
				use := rw[o]
				if use == 0 || a.Word {
					// Unbound, or a word read when its call is set up: a
					// read, and, unbound, a write too.
					use |= 1
					if rw[o] == 0 {
						use |= 2
					}
				}
				if use&1 != 0 {
					cc.rd = append(cc.rd, buf)
				}
				if use&2 != 0 {
					cc.wr = append(cc.wr, buf)
				}
				cc.slice(buf, reach[o])
			}
		}
		code, err := h.MapNamed(c.CPU, c.Name)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		cc.code = code
		f.Codes = append(f.Codes, cc)
	}
	// The weights as the first of their values a kernel reads.
	for _, v := range wsV {
		own, ok := wOwn[v]
		for _, r := range read {
			if ok && r.buf == own && !slices.Contains(f.WReads, r.code) {
				f.WSel, f.WReads = own, append(f.WReads, r.code)
			}
		}
		if f.WSel >= 0 {
			break
		}
	}
	for _, cc := range f.Codes[:f.KeysEnd] {
		f.KeysX = f.KeysX || cc.writes != 0
	}
	for _, w := range wrote {
		if w.code >= f.KeysEnd || slices.Contains(f.Cross, w.buf) {
			continue
		}
		for _, r := range read {
			if r.buf == w.buf && r.code >= f.KeysEnd {
				f.Cross = append(f.Cross, w.buf)
				break
			}
		}
	}
	if !apart && !f.inPlace() {
		f.Close()
		return nil, fmt.Errorf("the block reads its residual after writing its output over it")
	}
	if err := f.bankHalves(gr, rows); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.rowHalves(gr); err != nil {
		f.Close()
		return nil, err
	}
	if rows == 1 {
		f.chainPhases()
		f.joinPhases()
	}
	f.labelPhases()
	f.shareOwn(wrote, read)
	return f, nil
}

// shareOwn lays the form's own buffers out on fewer, as the card's forms
// share theirs (the tier's partForm.share): a buffer whose first code writes
// it without reading it, live from that code's pool phase to the last
// phase that binds it, takes one whose last phase ran before its first. The
// phases run in order -- a joined or chained phase's kernels run together,
// so its buffers are live across all of it (joinPhases, chainPhases, which
// ran over the buffers unshared) -- so the earlier holder is done with it.
// A constant, a buffer first read (it
// carries what an earlier call left), the selection and the picks' weights
// (the host reads and writes them between the halves) and what a
// bidirectional run holds between its passes keep their own. Every own
// index the form holds is renumbered.
func (f *Form) shareOwn(wrote, read []ownUse) {
	n := len(f.Own)
	// Each code's phase: a joined or chained run's kernels share theirs.
	phase := make([]int, len(f.Codes))
	for ci, p := 0, 0; ci < len(f.Codes); p++ {
		end := ci + 1 + max(f.Codes[ci].join, f.Codes[ci].chain)
		for ; ci < end && ci < len(f.Codes); ci++ {
			phase[ci] = p
		}
	}
	first, last := make([]int, n), make([]int, n)
	for i := range first {
		first[i] = -1
	}
	for ci := range f.Codes {
		for _, ar := range f.Codes[ci].args {
			if ar.src != SrcOwn && ar.src != SrcWord {
				continue
			}
			if first[ar.i] < 0 {
				first[ar.i] = ci
			}
			last[ar.i] = ci
		}
	}
	fresh := make([]bool, n)
	for _, w := range wrote {
		if w.code == first[w.buf] {
			fresh[w.buf] = true
		}
	}
	for _, r := range read {
		if r.code == first[r.buf] {
			fresh[r.buf] = false
		}
	}
	pinned := func(i int) bool {
		_, k := f.Consts[i]
		return k || !fresh[i] || first[i] < 0 || i == f.Sel || i == f.WSel || slices.Contains(f.Cross, i)
	}
	slot := make([]int, n)
	var phys []int
	var free []int // each buffer's last code, len(f.Codes) where it is never given up
	var order []int
	for i := range n {
		if pinned(i) {
			slot[i] = len(phys)
			phys, free = append(phys, f.Own[i]), append(free, len(f.Codes))
			continue
		}
		order = append(order, i)
	}
	slices.SortStableFunc(order, func(a, b int) int { return first[a] - first[b] })
	for _, i := range order {
		need, at := f.Own[i], -1
		for p, end := range free {
			if end >= phase[first[i]] {
				continue
			}
			// The narrowest that holds it, else the widest, grown.
			switch {
			case at < 0:
				at = p
			case phys[p] >= need && (phys[at] < need || phys[p] < phys[at]):
				at = p
			case phys[at] < need && phys[p] > phys[at]:
				at = p
			}
		}
		if at < 0 {
			at = len(phys)
			phys, free = append(phys, 0), append(free, 0)
		}
		phys[at] = max(phys[at], need)
		free[at] = phase[last[i]]
		slot[i] = at
	}
	consts := map[int][]uint32{}
	for i, c := range f.Consts {
		consts[slot[i]] = c
	}
	f.Own, f.Consts = phys, consts
	re := func(b *bufRef) {
		if b.src == SrcOwn {
			b.i = slot[b.i]
		}
	}
	for ci := range f.Codes {
		c := &f.Codes[ci]
		for k := range c.args {
			if c.args[k].src == SrcOwn || c.args[k].src == SrcWord {
				c.args[k].i = slot[c.args[k].i]
			}
		}
		for k := range c.rd {
			re(&c.rd[k])
		}
		for k := range c.wr {
			re(&c.wr[k])
		}
		for k := range c.sliced {
			re(&c.sliced[k].buf)
		}
	}
	for k, i := range f.Cross {
		f.Cross[k] = slot[i]
	}
	if f.Sel >= 0 {
		f.Sel = slot[f.Sel]
	}
	if f.RowSel >= 0 {
		f.RowSel = slot[f.RowSel]
	}
	if f.WSel >= 0 {
		f.WSel = slot[f.WSel]
	}
}

// callsOwnScratch reports whether code c's calls each write a slice of
// scratch o no other call writes: every address of o the code is passed
// moves along every axis of its calls, each axis's step past the whole of
// the axes inside it, as an operand laid out by its primitive's leading axes
// is, and the code reaching no more of o than its calls' slices; an axis of
// one call moves nothing. A scratch some call's address does not move along
// is every call's.
func callsOwnScratch(c composer.Code, o composer.Operand, bytes int64) bool {
	if len(c.Calls) == 0 {
		return false
	}
	o.Copy = 0
	seen := false
	for _, a := range c.Args {
		x := a.Operand
		x.Copy = 0
		if x != o {
			continue
		}
		if a.Word || a.Coord > 0 || a.Length || a.Value || a.Moving || len(a.Per) != len(c.Calls) {
			return false
		}
		span := 0
		for ax := range c.Calls {
			if c.Calls[ax] == 1 {
				continue
			}
			inner := 1
			if ax+1 < len(c.Calls) {
				inner = a.Per[ax+1] * c.Calls[ax+1]
			}
			if a.Per[ax] <= 0 || a.Per[ax] < inner {
				return false
			}
			span = max(span, a.Per[ax]*c.Calls[ax])
		}
		elem := int64(4)
		if a.Elem > 0 {
			elem = int64(a.Elem)
		}
		if bytes <= 0 || bytes > int64(span)*elem {
			return false
		}
		seen = true
	}
	return seen
}

// joinPhases runs each run of consecutive kernels that share no buffer one
// of them writes in one pool phase: their calls one space, claimed as one
// kernel's. A run does not cross the selection's split.
func (f *Form) joinPhases() {
	for g := 0; g < len(f.Codes); {
		if f.Codes[g].chain > 0 {
			g += f.Codes[g].chain + 1
			continue
		}
		j := g + 1
		if f.Codes[g].joinable() {
			for j < len(f.Codes) && j != f.Split && j != f.RowSplit && f.Codes[j].joinable() && f.Codes[j].chain == 0 && apart(f.Codes[g:j], &f.Codes[j]) {
				j++
			}
		}
		f.Codes[g].join = j - g - 1
		g = j
	}
}

// labelPhases names each kernel's region for a profile: its own name, or
// the names of every kernel the phase it heads runs.
func (f *Form) labelPhases() {
	for g := 0; g < len(f.Codes); g++ {
		c := &f.Codes[g]
		k := c.join + c.chain
		if k == 0 {
			c.label = c.name
			continue
		}
		names := make([]string, k+1)
		for i := range names {
			names[i] = f.Codes[g+i].name
		}
		c.label = strings.Join(names, "+")
	}
}

// slice records the rows of buffer b a call reaches alone along the one
// axis of its calls past a single call (composer.Code.Reach), -1 for none.
func (c *Code) slice(b bufRef, reach []int) {
	per := -1
	if ax := oneAxis(c.calls); ax >= 0 && ax < len(reach) && reach[ax] > 0 {
		per = reach[ax]
	}
	for i := range c.sliced {
		if c.sliced[i].buf == b {
			if c.sliced[i].per != per {
				c.sliced[i].per = -1
			}
			return
		}
	}
	c.sliced = append(c.sliced, bufSlice{b, per})
}

// oneAxis is the one axis of calls past a single call, -1 for none or more.
func oneAxis(calls []int) int {
	ax := -1
	for i, n := range calls {
		if n == 1 {
			continue
		}
		if ax >= 0 {
			return -1
		}
		ax = i
	}
	return ax
}

// chainPhases runs each run of kernels that chained admits, one reading
// what another writes, in one pool phase: unit u runs its calls of each
// kernel in turn. A run does not cross the selection's split.
func (f *Form) chainPhases() {
	for g := 0; g < len(f.Codes); {
		j := g + 1
		for j < len(f.Codes) && j != f.Split && j != f.RowSplit && f.Codes[j].joinable() && chained(f.Codes[g:j+1]) {
			j++
		}
		dep := false
		for k := g + 1; k < j; k++ {
			dep = dep || !apart(f.Codes[g:k], &f.Codes[k])
		}
		if dep {
			f.Codes[g].chain, f.Codes[g].units = j-g-1, chainUnits(f.Codes[g:j])
		}
		g = j
	}
}

// chainUnits is the units of a phase of kernels cs: the fewest calls of
// one, every kernel's calls along one axis and a whole number a unit; 0
// where they are not.
func chainUnits(cs []Code) int {
	u := cs[0].n
	for i := range cs {
		u = min(u, cs[i].n)
	}
	ax := oneAxis(cs[0].calls)
	for i := range cs {
		if !cs[i].joinable() || cs[i].share.Share > 0 || ax < 0 || oneAxis(cs[i].calls) != ax || len(cs[i].calls) != len(cs[0].calls) || cs[i].n%u != 0 {
			return 0
		}
	}
	return u
}

// chained says kernels cs may run in one phase unit by unit: every buffer
// one writes and another reads or writes, every call of both reaches only
// its own slice of, a unit's calls' slices one span in each.
func chained(cs []Code) bool {
	u := chainUnits(cs)
	if u <= 1 {
		return false
	}
	span := func(c *Code, b bufRef) int {
		for _, s := range c.sliced {
			if s.buf == b && s.per > 0 {
				return s.per * (c.n / u)
			}
		}
		return -1
	}
	for i := range cs {
		for _, b := range cs[i].wr {
			for j := range cs {
				if j == i || !slices.Contains(cs[j].rd, b) && !slices.Contains(cs[j].wr, b) {
					continue
				}
				if s := span(&cs[i], b); s < 0 || s != span(&cs[j], b) {
					return false
				}
			}
		}
	}
	return true
}

// joinable says a kernel's calls may share a phase with another's: calls
// on the pool, each free of the others and of the step's rows. A kernel of
// one call runs on the caller, in no phase.
func (c *Code) joinable() bool {
	return c.live.Live == 0 && !c.tiles && c.n > 1 && !c.serial
}

// apart says kernel c reads nothing the kernels gs write and writes nothing
// they read or write.
func apart(gs []Code, c *Code) bool {
	for i := range gs {
		g := &gs[i]
		for _, w := range g.wr {
			if slices.Contains(c.rd, w) || slices.Contains(c.wr, w) {
				return false
			}
		}
		for _, w := range c.wr {
			if slices.Contains(g.rd, w) {
				return false
			}
		}
	}
	return true
}

// selOf is the selection gr's banks are read at, -1 for a graph that reads
// none.
func selOf(gr *graph.Graph) graph.ValueID {
	for _, o := range gr.Ops {
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			return o.In[1]
		}
	}
	return -1
}

// weightsOf is the picks' weights gr's banks' outputs are multiplied by,
// then the values of their width each is an elementwise map of, back to the
// first that is not: nil for a graph that weights none.
func weightsOf(gr *graph.Graph) []graph.ValueID {
	bank := map[graph.ValueID]bool{}
	made := map[graph.ValueID]int{}
	w := graph.ValueID(-1)
	for i, o := range gr.Ops {
		for _, v := range o.Out {
			made[v] = i
		}
		if o.Kind == graph.OpExpertMatVec && len(o.In) == 2 && len(o.Weight) > 0 && jlm.ExpertBank(o.Weight[0].Role) {
			bank[o.Out[0]] = true
		}
		if w >= 0 || o.Kind != graph.OpMap || o.Attr.Fn != graph.MapMul || len(o.In) != 2 {
			continue
		}
		switch {
		case bank[o.In[0]]:
			w = o.In[1]
		case bank[o.In[1]]:
			w = o.In[0]
		}
	}
	var ws []graph.ValueID
	for w >= 0 && len(ws) < len(gr.Ops) {
		ws = append(ws, w)
		i, ok := made[w]
		if o := gr.Ops[i]; !ok || o.Kind != graph.OpMap || len(o.In) == 0 || gr.Values[o.In[0]].Width != gr.Values[w].Width {
			break
		}
		w = gr.Ops[i].In[0]
	}
	return ws
}

// bankHalves finds where a form over banks is cut (Form.Split): at
// its first code that binds a bank, with the selection written by a code
// before it.
func (f *Form) bankHalves(gr *graph.Graph, rows int) error {
	sel := selOf(gr)
	if sel < 0 {
		return nil
	}
	f.Split = -1
	for ci, c := range f.Codes {
		for _, a := range c.args {
			if a.src == SrcWeight && jlm.ExpertBank(a.role) && f.Split < 0 {
				f.Split = ci
			}
		}
	}
	if f.Sel < 0 || f.Split <= 0 {
		return fmt.Errorf("a form over banks with no selection of its own (%d) or no code before its banks (%d)", f.Sel, f.Split)
	}
	f.SelN = rows * gr.Values[sel].Width
	return nil
}

// TableRowsOf is the rows the n-gram table's gathers read, the hash's, -1
// for a graph that gathers none.
func TableRowsOf(gr *graph.Graph) graph.ValueID {
	if !slices.ContainsFunc(gr.Ops, func(o graph.Op) bool {
		return o.Kind == graph.OpEmbed && len(o.Weight) == 1 && o.Weight[0].Role == jlm.RoleNGramTable
	}) {
		return -1
	}
	for _, o := range gr.Ops {
		if o.Kind == graph.OpNGramHash {
			return o.Out[0]
		}
	}
	return -1
}

// rowHalves finds where a form gathering the n-gram table's rows is cut:
// at its first code that reads the table, the rows written by a code
// before it.
func (f *Form) rowHalves(gr *graph.Graph) error {
	rv := TableRowsOf(gr)
	if rv < 0 {
		return nil
	}
	f.RowSplit = -1
	for ci, c := range f.Codes {
		for _, a := range c.args {
			if a.src == SrcWeight && a.role == jlm.RoleNGramTable && f.RowSplit < 0 {
				f.RowSplit = ci
			}
		}
	}
	if f.RowSel < 0 || f.RowSplit <= 0 || f.Split > 0 && f.Split <= f.RowSplit {
		return fmt.Errorf("a form gathering table rows with no rows of its own (%d), no code before its gathers (%d), or banks before them (%d)", f.RowSel, f.RowSplit, f.Split)
	}
	f.RowN = gr.Values[rv].Width
	return nil
}

// inPlace says the form may write each residual stream's output over its
// input: every kernel that reads a stream's input runs before the first
// that writes its output.
func (f *Form) inPlace() bool {
	for k := range 64 {
		lastRead, firstWrite := -1, len(f.Codes)
		for i, c := range f.Codes {
			if c.reads&(1<<k) != 0 {
				lastRead = i
			}
			if c.writes&(1<<k) != 0 {
				firstWrite = min(firstWrite, i)
			}
		}
		if lastRead >= firstWrite {
			return false
		}
	}
	return true
}

// streamIn is the residual stream whose input v is in block graph gr, -1
// for a value that is none: the graph's first inputs, one a stream it
// writes.
func streamIn(gr *graph.Graph, v graph.ValueID) int {
	if k := slices.Index(gr.In, v); k < len(gr.Out) {
		return k
	}
	return -1
}

// Weight is a weight or a vector one of a form's kernels binds, by role:
// a weight's plane (its packed quants, scales or sub-scales), and for a
// table read by rows, the first row the binding names.
type Weight struct {
	Vector bool
	Role   jlm.Role
	Region uint8
	From   int
}

// weight is w's index in the form's weights, added where new.
func (f *Form) weight(w Weight) int {
	if i := slices.Index(f.Weights, w); i >= 0 {
		return i
	}
	f.Weights = append(f.Weights, w)
	return len(f.Weights) - 1
}

// Name is the composer's name for kernel c, Regs its register facts and
// Size its code's bytes.
func (c *Code) Name() string          { return c.name }
func (c *Code) Regs() composer.RegUse { return c.regs }
func (c *Code) Size() int             { return c.size }
