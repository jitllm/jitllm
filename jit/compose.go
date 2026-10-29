// Package jit is the composer: a model and its placement in, for every
// device the kernels the model runs there and the plan that launches them.
// The model's graph is decomposed into primitives, each generated as naive
// IR, the IR rewritten by registered passes, and after the last pass lowered
// by the device's backend.
package jit

import (
	"errors"
	"fmt"
	"slices"
)

// Step is the shape of one step the engine runs: rows a step, the history's
// positions a page, the longest history a row attends, and, where not 0,
// how many activations share one int8 scale -- the engine's decision
// (Model.ActWindow), in place of the profile's.
type Step struct {
	Rows, Page, Keys int
	ActWindow        int
	// Seqs is how many sequences the rows are of, each its own history
	// (Keys positions): one for a prompt's chunk or an encoder's sequence,
	// 0 for a sequence a row.
	Seqs int
	// Picture is the rows a picture's step of the model's vision segment
	// runs over (its patches and the rows before them), 0 for a plan that
	// composes no vision segment.
	Picture int
}

// stepOf is the step a part of segment seg is composed for: a vision
// segment's a picture's, one bidirectional sequence of Picture rows.
func (s Step) stepOf(seg Segment) Step {
	if seg != SegmentVision {
		return s
	}
	return Step{Rows: s.Picture, Page: s.Page, Keys: s.Picture, ActWindow: s.ActWindow, Seqs: 1, Picture: s.Picture}
}

// Histories is how many histories a step's rows read: Seqs, or a row's
// own each.
func (s Step) Histories() int {
	if s.Seqs > 0 {
		return s.Seqs
	}
	return s.Rows
}

// Placement is the device each part of the model runs on.
type Placement func(Part) Device

// Code is one kernel as its backend runs it: the passes that formed it, the
// buffers it binds in its parameters' order, and on a GPU the workgroups a
// launch runs.
type Code struct {
	Name   string
	Passes []string
	Lowered
	Binds  []Binding
	Groups int
	// Live, where set, is the rows of a step the code holds at its form's
	// capacity: LiveUnits coordinates of its leading axis, each of
	// LiveInner elements (workgroups where Work), at LiveGroup threads a
	// workgroup. A call of fewer rows launches LiveGroups.
	Live, LiveUnits, LiveInner, LiveGroup int
	Work                                  bool
	// Few, where set, is a code launched in place of the code before it at
	// a call of at most Few rows; Many, where set, one launched only at a
	// call of more than Many rows (Kernel.Few, Kernel.Many).
	Few, Many int
	// Args is a host call's argument words.
	Args []CallArg
	// Calls is a host kernel's calls' space, the outermost first: the
	// kernel runs once at each of its coordinates, any order, each call's
	// own passed in the words its Args say. Empty is one call.
	Calls []int
	// TileMajor is a kernel whose calls' second axis tiles a weight
	// (Workers): its calls are best run every live row of a tile, then the
	// next tile, so a tile is read from memory once for the step's rows.
	TileMajor bool
	// Scan is a scan's code: its calls are its state's rows, each stepping
	// the rows of the call its run descriptor names, whatever the call's
	// rows.
	Scan bool
	// Matrix is Kernel.Matrix.
	Matrix [2]int
	// Share, where set, is the rows of a host kernel whose single calls'
	// axis is a call's first row in ShareUnit rows: the caller cuts them into
	// a call a worker, of a multiple of ShareUnit rows, and passes each its
	// rows in the Share word.
	Share, ShareUnit int
}

// Writes reports whether the kernel writes operand o.
func (c *Code) Writes(o Operand) bool {
	for _, b := range c.Binds {
		if b.Operand == o && b.Writes {
			return true
		}
	}
	return false
}

// CallArg is one argument word of a host call, at byte offset Field: the
// address of Operand's storage Offset elements in, with Words the contents
// of a constant block; or a value the caller computes, a run-time length
// where Length is set, of static extent Extent. Moving is set where the
// offset is the caller's to compute. Word is set where the word at Offset
// is passed itself, by value, rather than its address.
type CallArg struct {
	Field   int
	Operand Operand
	Offset  int
	Moving  bool
	Word    bool
	Words   []uint32
	Value   bool
	Length  bool
	Extent  int
	// Share is the length of a call's rows of Code.Share.
	Share bool
	// Coord, where set, is 1 + the axis of Calls whose coordinate the word
	// is, set for each call.
	Coord int
	// Per, where set, is the elements the address moves a unit of each
	// axis of Calls' coordinate: a call's rows (Workers). Elem, where set,
	// is the bytes of the elements Offset and Per count (a binary16
	// history's two); four where it is not.
	Per  []int
	Elem int
}

// Binding is a buffer a kernel binds: its operand and whether the kernel
// writes it.
type Binding struct {
	Operand Operand
	Writes  bool
	// Bytes is what the kernel reaches of it, 0 where its buffer states no
	// extent; Storage is its storage in the plan's Storages.
	Bytes   int64
	Storage int
	// Words is a block of constants' contents, the caller's to supply.
	Words []uint32
	// Read is whether a live load of the kernel reads it: a binding a
	// rewrite left unread is still bound, and is no reader of its storage.
	Read bool
	// Own is a buffer each call writes its own elements of, no other call
	// writing them.
	Own bool
}

// PartPlan is one part of the model on its device: its kernels in launch
// order.
type PartPlan struct {
	Part   Part
	Device Device
	Codes  []Code
}

// Transfer is a value carried between consecutive parts on different
// devices: the boundary value of part From read by part To.
type Transfer struct {
	From, To int
	Value    Operand
	Words    int
}

// Plan is the model on its placement: each part's launches, and the
// transfers at the seams between devices.
type Plan struct {
	Parts     []PartPlan
	Transfers []Transfer
	// Storages is every buffer the step binds: what each holds and where
	// its contents come from.
	Storages []Storage
}

// Compose forms the model in f at step s on placement p. A part it cannot
// form is an error, beside the parts it formed.
func (r *Registry) Compose(f Model, s Step, p Placement) (*Plan, error) {
	parts, err := Front(f, s.Picture > 0)
	if err != nil {
		return nil, err
	}
	plan := &Plan{}
	for _, part := range parts {
		plan.Parts = append(plan.Parts, PartPlan{Part: part, Device: p(part)})
	}
	for i := 1; i < len(parts); i++ {
		a, b := plan.Parts[i-1], plan.Parts[i]
		if a.Device == b.Device || len(a.Part.Graph.Out) == 0 {
			continue
		}
		v := a.Part.Graph.Out[0]
		plan.Transfers = append(plan.Transfers, Transfer{From: i - 1, To: i,
			Value: Operand{Kind: OperandValue, Value: v, Name: a.Part.Graph.Values[v].Name, Of: i - 1},
			Words: s.Rows * a.Part.Graph.Values[v].Width})
	}
	var errs []error
	// A run is consecutive parts on one device and of one segment: a
	// picture's step and a token's are launched apart.
	for i := 0; i < len(parts); {
		j := i + 1
		for j < len(parts) && plan.Parts[j].Device == plan.Parts[i].Device && parts[j].Segment == parts[i].Segment {
			j++
		}
		if err := r.formRun(f, plan.Parts[i:j], i, s); err != nil {
			errs = append(errs, err)
		}
		i = j
	}
	errs = append(errs, plan.storages(parts))
	return plan, errors.Join(errs...)
}

// formRun is a run of consecutive parts on one device: each decomposed and
// generated, the run passed as one, each part's kernels lowered.
func (r *Registry) formRun(f Model, run []PartPlan, first int, s Step) error {
	ks, err := r.Kernels(f, run, first, s)
	codes, lerr := r.Lower(ks, run[0].Device)
	for _, c := range codes {
		run[c.Part-first].Codes = append(run[c.Part-first].Codes, c.Code)
	}
	return errors.Join(err, lerr)
}

// Kernels is a run's final IR: its parts decomposed and generated, the run
// passed as one. Where the registers refuse a kernel, the largest expression
// its part formed as one map is formed a map a term and the run passed again;
// a part with none left is formed with its kernels joined to none.
func (r *Registry) Kernels(f Model, run []PartPlan, first int, s Step) ([]*Kernel, error) {
	prof, ok := r.Profiles[run[0].Device.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no profile for %s", run[0].Device.Kind)
	}
	prof = prof.on(run[0].Device)
	sp := newExprSplit()
	for {
		ks, errs, err := r.generated(f, run, first, s, sp)
		if err != nil {
			return nil, err
		}
		ks, err = prof.run(ks, run[0].Device)
		if sp.splitRefused(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		return ks, errors.Join(errs...)
	}
}

// generated is a run's parts decomposed and generated, and the parts that
// did not decompose.
func (r *Registry) generated(f Model, run []PartPlan, first int, s Step, sp *exprSplit) ([]*Kernel, []error, error) {
	d := run[0].Device
	var ks []*Kernel
	var errs []error
	for i, pp := range run {
		prims, err := r.decompose(f, pp.Part, first+i, s, d, sp)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", pp.Part, err))
			continue
		}
		for _, p := range prims {
			k, err := generate(p)
			if err != nil {
				return nil, nil, fmt.Errorf("%s: %w", pp.Part, err)
			}
			k.Part, k.Of, k.Unfused, k.Unjam, k.Unpaired, k.Unwide, k.Uncursored, k.Unstreamed = pp.Part.Kind, first+i, sp.unfused[first+i], sp.unjammed[first+i], sp.unpaired[first+i], sp.unwide[first+i], sp.uncursored[first+i], sp.unstreamed[first+i]
			ks = append(ks, k)
		}
	}
	boundaries(ks, run, first)
	return ks, errs, nil
}

// boundaries names each part's input in ks as the operand the run's part
// before it wrote it as (the storage both are, StorageKeys): a pass that
// joins kernels across the boundary then sees one buffer, its writes and
// its reads, where two names would hide a race between them.
func boundaries(ks []*Kernel, run []PartPlan, first int) {
	alias := map[Operand]Operand{}
	for i := 1; i < len(run); i++ {
		a, b := run[i-1].Part.Graph, run[i].Part.Graph
		// An input a feed names is that feed's, wherever its part runs
		// (a picture's rows, from the vision segment's run): never the
		// part before's output.
		if len(a.Out) == 0 || len(b.In) == 0 || fed(run[i].Part, 0) {
			continue
		}
		from := Operand{Kind: OperandValue, Value: a.Out[0], Of: first + i - 1}
		if x, ok := alias[from]; ok {
			from = x
		}
		alias[Operand{Kind: OperandValue, Value: b.In[0], Of: first + i}] = from
	}
	for i, pp := range run {
		for _, fd := range pp.Part.Feeds {
			if fd.From < first || fd.From >= first+len(run) {
				continue
			}
			src := run[fd.From-first].Part.Graph
			alias[Operand{Kind: OperandValue, Value: pp.Part.Graph.In[fd.In], Of: first + i}] =
				Operand{Kind: OperandValue, Value: src.Out[fd.Out], Of: fd.From}
		}
	}
	for _, k := range ks {
		for bi := range k.Bufs {
			o := &k.Bufs[bi].Operand
			if x, ok := alias[Operand{Kind: OperandValue, Value: o.Value, Of: o.Of}]; ok && o.Kind == OperandValue {
				o.Value, o.Of = x.Value, x.Of
				o.Name = run[x.Of-first].Part.Graph.Values[x.Value].Name
			}
		}
	}
}

// fed reports whether part p's input in is named by one of its feeds.
func fed(p Part, in int) bool {
	for _, f := range p.Feeds {
		if f.In == in {
			return true
		}
	}
	return false
}

// PartCode is a kernel's code and the part of the model it belongs to.
type PartCode struct {
	Code
	Part int
}

// Lower is a run's final IR on device d as its backend's code.
func (r *Registry) Lower(ks []*Kernel, d Device) ([]PartCode, error) {
	low, ok := r.Lowerings[d.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no lowering for %s", d.Kind)
	}
	var codes []PartCode
	var errs []error
	for _, k := range ks {
		l, err := low.Lower(k, d)
		if err != nil {
			errs = append(errs, fmt.Errorf("jit: lowering %s: %w", k.Name, err))
			continue
		}
		c := Code{Name: k.Name, Passes: k.Passes, Lowered: l, Groups: k.groups(), Work: k.Work, Few: k.Few, Many: k.Many, Scan: k.Kind == PrimScan, Matrix: k.Matrix}
		if l.CPU != nil && !k.Work && k.Group <= 0 {
			c.Calls = append([]int(nil), k.Space...)
			c.Share, c.ShareUnit = k.Share, k.ShareUnit
		}
		if k.Capacity > 0 && k.RowsHeld && len(k.Space) > 0 && k.Space[0] > 0 {
			c.Live, c.LiveUnits, c.LiveInner, c.LiveGroup = k.Capacity, k.Space[0], k.Elements()/k.Space[0], k.Group
			c.TileMajor = k.TileMajor
		}
		for _, a := range k.Args {
			if a.Buf >= 0 {
				b := k.Bufs[a.Buf]
				ca := CallArg{Field: a.Field, Operand: b.Operand, Words: b.Words, Word: b.Word > 0}
				if b.Offset > 0 {
					if nd := k.Nodes[b.Offset-1]; nd.Op == OpConst {
						ca.Offset, ca.Elem = int(nd.Imm), ElemBytes(b)
					} else if c, per, ok := k.callOffset(b.Offset - 1); ok && l.CPU != nil {
						ca.Offset, ca.Per, ca.Elem = c, per, ElemBytes(b)
					} else {
						ca.Moving = true
					}
				}
				c.Args = append(c.Args, ca)
				continue
			}
			if a.Coord > 0 {
				c.Args = append(c.Args, CallArg{Field: a.Field, Value: true, Coord: a.Coord})
				continue
			}
			nd := k.Nodes[a.Node-1]
			c.Args = append(c.Args, CallArg{Field: a.Field, Value: true, Length: nd.Op == OpLength, Extent: int(nd.Imm), Share: nd.Op == OpLength && nd.A < 0})
		}
		// What the code reads: the loads its stores, loop counts and launch
		// clamp reach, and the buffers its cursors walk.
		read, written := map[int]bool{}, map[int]bool{}
		live := k.Live()
		for at := []int{k.Clamp - 1}; len(at) > 0; {
			i := at[len(at)-1]
			at = at[:len(at)-1]
			if i < 0 || live[i] {
				continue
			}
			live[i] = true
			for _, a := range k.Operands(&k.Nodes[i]) {
				at = append(at, *a)
			}
		}
		for i := range live {
			if nd := k.Nodes[i]; nd.Op == OpLoad {
				read[nd.Buf] = true
			}
		}
		for _, l := range k.Loops {
			for _, c := range l.Cursors {
				read[c.Buf] = true
			}
		}
		for _, s := range k.Stores {
			if !s.Barrier && !s.Run {
				written[s.Buf] = true
			}
		}
		for bi, b := range k.Bufs {
			if b.Kind != BufShared {
				o := b.Operand
				o.Copy = 0
				own := !k.Work && k.Group == 0 && slices.Contains(k.Own, o)
				c.Binds = append(c.Binds, Binding{Operand: b.Operand, Writes: b.Kind == BufOut && written[bi], Bytes: b.bytes(), Words: b.Words, Read: read[bi], Own: own})
			}
		}
		codes = append(codes, PartCode{Code: c, Part: k.Of})
	}
	return codes, errors.Join(errs...)
}

// ShareRows is a call's rows of a Share code whose calls run on workers
// workers, a multiple of the code's unit, and the calls they take: one a
// worker, whatever workers the code was formed for.
func (c *Code) ShareRows(workers int) (rows, calls int) {
	if c.Share <= 0 {
		return 0, 0
	}
	w, u := max(workers, 1), max(c.ShareUnit, 1)
	rows = ((c.Share+w-1)/w + u - 1) / u * u
	return rows, (c.Share + rows - 1) / rows
}
