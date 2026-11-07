package composer

import (
	"fmt"
	"maps"
	"runtime"
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// DeviceKind is a backend.
type DeviceKind uint8

const (
	DeviceGPU DeviceKind = iota
	DeviceCPU
)

func (k DeviceKind) String() string {
	if k == DeviceCPU {
		return "cpu"
	}
	return "gpu"
}

// Device is what the composer knows of a device: its backend, the threads
// it holds resident and the width of a subgroup or a vector. What a device
// costs is engine/cost's.
type Device struct {
	Kind         DeviceKind
	Slots, Lanes int
	// Units, where known, is the multiprocessors (compute units) the Slots
	// are spread over.
	Units int
	// Regs, where known, is the 32-bit registers a thread holds when a
	// multiprocessor holds the most threads it can.
	Regs int
	// Shared, where known, is the bytes of workgroup memory a unit holds for
	// the workgroups resident on it.
	Shared int
	// ISA, on a CPU, is the instruction sets it executes, as engine/device/cpu names
	// them (cpu.ISA's bits): an x86's, or on arm64 FEAT_DotProd.
	ISA uint64
	// ExactDiv is a device whose float division is correctly rounded, as
	// IEEE 754 states it: a CPU's, PTX's div.rn. The int8 form's scale and
	// inverse are divisions there, and made exact in integers where not.
	ExactDiv bool
	// ExactArith is a device whose lowering rounds every float operation
	// alone (PTX's .rn), so two kernels computing a value by the same
	// operations in the same order write its same bits; a driver free to
	// contract them (SPIR-V's, MSL's) is not.
	ExactArith bool
	// Matrix, where set, is the collective tile (rows, columns, depth) the
	// device multiplies binary16 operands into a float32 accumulator in,
	// staged in workgroup memory and carried across a loop (ir's tile ops):
	// a reduction over several rows runs there (Matrix, the pass).
	Matrix [3]int
	// Tile, where set, is the matrix tile every projection takes in place
	// of the one the model chooses (Matrix.chosen).
	Tile MatrixTile
	// Fragment, where set, is the device's own matrix instruction, which
	// Matrix issues on fragments it lays out (OpMMA).
	Fragment ir.MMAShape
	// ScaleRows is a device whose lowering scales an accumulator tile's
	// rows by words of workgroup memory (OpTileScale).
	ScaleRows bool
	// HalfPairs is a device whose lowering runs packed binary16 arithmetic
	// (OpSubF16x2, OpFmaF16x2).
	HalfPairs bool
	// VectorLoads is a device whose lowering reads four consecutive words
	// of a buffer in one access (ir.OpLoadV).
	VectorLoads bool
	// Launch is the words its memory serves while one launch holds it, as
	// its probe measured; 0 where nothing did.
	Launch int
	// Buffers, where known, is the buffers one launch binds at most
	// (backend.Bindings): Phases merges no launch past it.
	Buffers int
}

// warp is the threads a combine by lane shuffles takes whole on d: its
// lanes, which on a GPU hold the IR's subgroup -- the IR's shuffles name
// every lane of one (ir.SubgroupLanes), and a device guaranteeing fewer
// runs none. Zero where d combines by no shuffle.
func (d Device) warp() int {
	if d.Lanes <= 1 || d.Kind == DeviceGPU && d.Lanes < ir.SubgroupLanes {
		return 0
	}
	return d.Lanes
}

// Lowered is a kernel in its backend's form.
type Lowered struct {
	GPU *ir.Kernel
	CPU []byte
	// Regs is a CPU kernel's register facts.
	Regs RegUse
}

// RegUse is what a CPU kernel asks of its host: Vec of the File vector
// registers its ISA has (highest index named + 1, -1 where the lowering does
// not count them), and Stack memory operands off the stack pointer, its
// spills. Zero Stack is the bar for a hot kernel.
type RegUse struct {
	Vec, File, Stack int
}

// Lowering turns a kernel's final IR into its backend's code.
type Lowering interface {
	Lower(k *Kernel, d Device) (Lowered, error)
}

// Context is what a decomposition sees beside its op: the part's graph and
// kind, its weights' storage formats, the step, and the int8 form the
// part's device keeps.
type Context struct {
	Graph   *graph.Graph
	Part    PartKind
	Formats map[jlm.Role]jlm.Type
	Step    Step
	Int8    *Int8Form
	// unsignedDot is whether this device's byte dot takes an unsigned
	// operand (Int8Form.UnsignedDot), exactDiv whether its division is
	// correctly rounded (Device.ExactDiv).
	unsignedDot, exactDiv bool
	// scanLoops is Profile.ScanLoops.
	scanLoops bool
	// split is the run's expressions formed a map a term.
	split *exprSplit
	// matrix is a step of rows the device's matrix unit runs a projection
	// over (Device.Matrix): its weight decoded, its input float.
	matrix bool
	// Rotaries is the part's segment's rotaries, each a table of the step,
	// the first named the model's RotaryBase'th (tableName).
	Rotaries   []graph.Rotary
	RotaryBase int
	// Of is the part's index in the model's parts, the blocks consecutive.
	Of int
	// History is how the part's history pools lay out a page.
	History History
	// More is Request.More.
	More []History
	// hist is the history the op being decomposed reads or writes
	// (onHistory), whose tables are at instance hist; base is history 0's
	// layout while another's is History.
	hist uint8
	base History
	// Paged is Request.Paged: the sheets the call site holds of each bank
	// it holds in pages.
	Paged map[jlm.Role]int
	// Strides is Request.Strides: the words from one sheet of a bank to the
	// next where they lie further apart than they pack.
	Strides map[jlm.Role]int
	formed  map[graph.ValueID]bool
	// grouped holds the groupings of a selection's pairs by expert formed
	// (expertGrouped), by their scratch's name: the projections a
	// selection routes share one.
	grouped map[string]bool
}

// blockPart is the part of block li, for a block part's decomposition: the
// blocks are consecutive parts.
func (c *Context) blockPart(li int) int { return c.Of - c.Graph.Block + li }

// Decomposition is one op kind's primitives.
type Decomposition func(o graph.Op, c *Context) ([]Prim, error)

// Registry is the composer's plugin surface: a decomposition for each op
// kind, a profile of passes and a lowering for each backend. Default is
// main's; a caller adds or replaces entries.
type Registry struct {
	Decompositions map[graph.OpKind]Decomposition
	Profiles       map[DeviceKind]Profile
	Lowerings      map[DeviceKind]Lowering
	// cache is what Form formed, by Key; nil forms every request afresh. A
	// copy of the registry shares it: the profile and the lowering are in
	// the Key.
	cache *formCache
}

// WithoutInt8 is r with no int8 form on device kind: a matvec there decodes
// its weights and keeps its activation float. It shares r's cache; the
// profile is in the Key.
func (r *Registry) WithoutInt8(kind DeviceKind) *Registry {
	fr := *r
	fr.Profiles = maps.Clone(r.Profiles)
	p := fr.Profiles[kind]
	p.Int8 = nil
	fr.Profiles[kind] = p
	return &fr
}

// WithoutHalves is r with device kind's histories kept at f32, as a State
// holding an f32 cache reads them: its profile without the Halves pass. It
// shares r's cache; the profile is in the Key.
func (r *Registry) WithoutHalves(kind DeviceKind) *Registry {
	fr := *r
	fr.Profiles = maps.Clone(r.Profiles)
	p := fr.Profiles[kind]
	halves := func(x Pass) bool {
		_, ok := x.(Halves)
		return ok
	}
	p.Passes = slices.DeleteFunc(slices.Clone(p.Passes), halves)
	if p.Narrow != nil {
		p.Narrow = slices.DeleteFunc(slices.Clone(p.Narrow), halves)
	}
	fr.Profiles[kind] = p
	return &fr
}

// ForPasses is r for forms over a history read in passes (History.Passes) on
// device kind: no split, whose merge normalizes the whole history, and no
// Flash, whose fused normalization leaves no pass maximum, sum or weighted
// values for the fold. It shares r's cache.
func (r *Registry) ForPasses(kind DeviceKind) *Registry {
	fr := r.WithoutSplits(kind)
	p := fr.Profiles[kind]
	flash := func(x Pass) bool {
		_, ok := x.(Flash)
		return ok
	}
	p.Passes = slices.DeleteFunc(p.Passes, flash)
	if p.Narrow != nil {
		p.Narrow = slices.DeleteFunc(p.Narrow, flash)
	}
	fr.Profiles[kind] = p
	return fr
}

// WithoutSplits is r with no sum, range or extreme cut into segments on
// device kind: every sum in its loop's order. It shares r's cache.
func (r *Registry) WithoutSplits(kind DeviceKind) *Registry {
	fr := *r
	fr.Profiles = maps.Clone(r.Profiles)
	p := fr.Profiles[kind]
	cut := func(x Pass) bool {
		switch x.(type) {
		case Split, SplitRange, SplitExtreme, GroupExtreme:
			return true
		}
		return false
	}
	p.Passes = slices.DeleteFunc(slices.Clone(p.Passes), cut)
	if p.Narrow != nil {
		p.Narrow = slices.DeleteFunc(slices.Clone(p.Narrow), cut)
	}
	fr.Profiles[kind] = p
	return &fr
}

// DropSegments has every split on device kind sum its segments but the
// last (Split.Drop): a gate's violation, never a shipping one.
func (r *Registry) DropSegments(kind DeviceKind) {
	p := r.Profiles[kind]
	p.Passes = slices.Clone(p.Passes)
	for i, x := range p.Passes {
		if s, ok := x.(Split); ok {
			s.Drop = true
			p.Passes[i] = s
		}
	}
	r.Profiles[kind] = p
}

// PoisonWorkgroupMemory has the GPU lowering poison every workgroup array
// at kernel start: a gate's choice, never a shipping one.
func (r *Registry) PoisonWorkgroupMemory() {
	r.Lowerings[DeviceGPU] = gpuLowering{Poison: true}
}

// Default is the registry whose profiles are set to main's choices.
func Default() *Registry {
	return &Registry{
		Decompositions: decompositions(),
		Profiles:       map[DeviceKind]Profile{DeviceGPU: gpuProfile(), DeviceCPU: cpuProfile()},
		Lowerings:      map[DeviceKind]Lowering{DeviceGPU: gpuLowering{}, DeviceCPU: cpuLowering{Shuffles: true, Arm64: runtime.GOARCH == "arm64"}},
		cache:          newFormCache(),
	}
}

func (r *Registry) decompose(f Model, part Part, of int, s Step, d Device, sp *exprSplit) ([]Prim, error) {
	view, rots, base := f, f.Rotaries(), 0
	if part.Segment == SegmentVision {
		view, base = f.Segment(), len(textRotaries(f))
		rots = view.Rotaries()
	}
	s = s.stepOf(part.Segment)
	c := r.context(part.Graph, part.Kind, formats(view, part), rots, s, d)
	c.RotaryBase, c.Of, c.split = base, of, sp
	return r.decomposeOps(c, of)
}

// context is a decomposition's context for graph g of a part of kind k on
// device d, the device's int8 form at the step's window.
func (r *Registry) context(g *graph.Graph, k PartKind, f map[jlm.Role]jlm.Type, rots []graph.Rotary, s Step, d Device) *Context {
	c := &Context{Graph: g, Part: k, Formats: f, Step: s, Int8: r.Profiles[d.Kind].Int8, Rotaries: rots, exactDiv: d.ExactDiv,
		formed: map[graph.ValueID]bool{}, grouped: map[string]bool{}, matrix: d.Matrix[0] > 0 && s.Rows >= d.Matrix[0],
		scanLoops: r.Profiles[d.Kind].ScanLoops}
	if q := c.Int8; q != nil && s.ActWindow > 0 && s.ActWindow != q.Window {
		w := *q
		w.Window = s.ActWindow
		c.Int8 = &w
	}
	if q := c.Int8; q != nil {
		c.unsignedDot = q.UnsignedDot != 0 && d.ISA&q.UnsignedDot == q.UnsignedDot
	}
	return c
}

// gridRows reports whether op o reads or writes a value of rows of its own
// (graph.Value.Rows).
func gridRows(g *graph.Graph, o graph.Op) bool {
	for _, v := range append(slices.Clone(o.In), o.Out...) {
		if g.Values[v].Rows > 0 {
			return true
		}
	}
	return false
}

// gridRowsOf is the rows of the grid op o's values state.
func gridRowsOf(g *graph.Graph, o graph.Op) int {
	for _, v := range append(slices.Clone(o.In), o.Out...) {
		if r := g.Values[v].Rows; r > 0 {
			return r
		}
	}
	return 0
}

// gridRowWise reports whether op o computes each row of its output from
// that row of its inputs alone, so that it runs over a grid of rows of its
// own as over the step's: the kinds a gate holds over a strided grid
// (TestComposedCPUGridRows, TestComposedGPUGridRows, TestComposedCPUGridAttend,
// TestComposedGPUGridAttend).
func gridRowWise(o graph.Op) bool {
	switch o.Kind {
	case graph.OpNorm, graph.OpLayerNorm, graph.OpHeadNorm, graph.OpAdd, graph.OpExpr, graph.OpMap, graph.OpKVAppend:
		return true
	case graph.OpMatVec:
		return len(o.In) == 1
	}
	return false
}

// ownsItsGrid reports whether an op of kind k reads its input's rows and
// writes its output's itself, each value at its own grid's rows.
func ownsItsGrid(k graph.OpKind) bool {
	switch k {
	case graph.OpConv, graph.OpUnfold, graph.OpTakeRows, graph.OpRegroup, graph.OpFold, graph.OpUntile, graph.OpDropRows,
		graph.OpRowBreak, graph.OpWrapRows:
		return true
	}
	return false
}

// decomposeOps is every op of c's graph as primitives, owned by part of.
func (r *Registry) decomposeOps(c *Context, of int) ([]Prim, error) {
	var prims []Prim
	sums, folded := sliceSums(c.Graph)
	for i, o := range c.Graph.Ops {
		if folded[i] {
			continue
		}
		d, ok := r.Decompositions[o.Kind]
		if !ok {
			return nil, fmt.Errorf("op %d (%s) has no decomposition", i, o.Kind)
		}
		oc := c
		if !ownsItsGrid(o.Kind) && gridRows(c.Graph, o) {
			// A row-wise op over a grid of rows of its own runs at that
			// grid's rows (Verify holds its every value to one grid; a
			// convolution, an unfold, a row take, a regroup, a fold and a row
			// drop read their input's rows and write their own themselves): the
			// step's rows are the grid's for its decomposition. An op that
			// reads rows across values of other grids is not formed.
			if !gridRowWise(o) {
				return nil, fmt.Errorf("op %d (%s) over a grid of rows of its own", i, o.Kind)
			}
			g := *c
			g.Step.Rows = gridRowsOf(c.Graph, o)
			oc = &g
		}
		ps, err := d(o, oc)
		if s, ok := sums[i]; ok {
			ps, err = decomposeSliceSum(o, s, oc), nil
		}
		if err != nil {
			return nil, fmt.Errorf("op %d (%s): %w", i, o.Kind, err)
		}
		for j := range ps {
			ps[j].own(of)
		}
		prims = append(prims, ps...)
	}
	return prims, nil
}

// own marks p's operands as part of's, the step's tables aside, and a
// part's own table that a decomposition named another part's (a shared
// history) left that part's.
func (p *Prim) own(of int) {
	mark := func(o *Operand) {
		if o.Kind == OperandTable && o.PerPart && o.Of != 0 {
			return
		}
		if o.Kind != OperandTable || o.PerPart {
			o.Of = of
		}
	}
	for i := range p.In {
		mark(&p.In[i])
	}
	mark(&p.Out)
	if p.Carry != nil {
		mark(&p.CarryOut)
	}
	for i := range p.Stores {
		mark(&p.Stores[i].Out)
	}
}
