package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
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
}

// Lowered is a kernel in its backend's form.
type Lowered struct {
	GPU *ir.Kernel
	CPU []byte
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
	// Rotaries is the part's segment's rotaries, each a table of the step,
	// the first named the model's RotaryBase'th (tableName).
	Rotaries   []graph.Rotary
	RotaryBase int
	// Of is the part's index in the model's parts, the blocks consecutive.
	Of     int
	formed map[graph.ValueID]bool
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
		Lowerings:      map[DeviceKind]Lowering{DeviceGPU: gpuLowering{}, DeviceCPU: cpuLowering{Shuffles: true}},
	}
}

func (r *Registry) decompose(f Model, part Part, of int, s Step, d Device) ([]Prim, error) {
	view, rots, base := f, f.Rotaries(), 0
	if part.Segment == SegmentVision {
		view, base = f.Segment(), len(textRotaries(f))
		rots = view.Rotaries()
	}
	s = s.stepOf(part.Segment)
	c := &Context{Graph: part.Graph, Part: part.Kind, Formats: formats(view, part), Step: s, Int8: r.Profiles[d.Kind].Int8,
		Rotaries: rots, RotaryBase: base, Of: of, formed: map[graph.ValueID]bool{}}
	if q := c.Int8; q != nil && s.ActWindow > 0 && s.ActWindow != q.Window {
		w := *q
		w.Window = s.ActWindow
		c.Int8 = &w
	}
	var prims []Prim
	for i, o := range part.Graph.Ops {
		d, ok := r.Decompositions[o.Kind]
		if !ok {
			return nil, fmt.Errorf("op %d (%s) has no decomposition", i, o.Kind)
		}
		ps, err := d(o, c)
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
