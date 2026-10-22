// Package jit is the composer: a model and its placement in, for every
// device the kernels the model runs there and the plan that launches them.
// The model's graph is decomposed into primitives, each generated as naive
// IR, the IR rewritten by registered passes, and after the last pass lowered
// by the device's backend.
package jit

import (
	"errors"
	"fmt"
)

// Step is the shape of one step the engine runs: rows a step, the history's
// positions a page, and the longest history a row attends.
type Step struct {
	Rows, Page, Keys int
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
	// Args is a host call's argument words.
	Args []CallArg
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
	parts, err := Front(f)
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
	for i := 0; i < len(parts); {
		j := i + 1
		for j < len(parts) && plan.Parts[j].Device == plan.Parts[i].Device {
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
// passed as one.
func (r *Registry) Kernels(f Model, run []PartPlan, first int, s Step) ([]*Kernel, error) {
	d := run[0].Device
	var ks []*Kernel
	var errs []error
	for i, pp := range run {
		prims, err := r.decompose(f, pp.Part, first+i, s, d)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", pp.Part, err))
			continue
		}
		for _, p := range prims {
			k, err := generate(p)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", pp.Part, err)
			}
			k.Part, k.Of = pp.Part.Kind, first+i
			ks = append(ks, k)
		}
	}
	prof, ok := r.Profiles[d.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no profile for %s", d.Kind)
	}
	ks, err := prof.run(ks, d)
	if err != nil {
		return nil, err
	}
	return ks, errors.Join(errs...)
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
		c := Code{Name: k.Name, Passes: k.Passes, Lowered: l, Groups: k.groups()}
		for _, a := range k.Args {
			if a.Buf >= 0 {
				b := k.Bufs[a.Buf]
				ca := CallArg{Field: a.Field, Operand: b.Operand, Words: b.Words, Word: b.Word > 0}
				if b.Offset > 0 {
					if nd := k.Nodes[b.Offset-1]; nd.Op == OpConst {
						ca.Offset = int(nd.Imm)
					} else {
						ca.Moving = true
					}
				}
				c.Args = append(c.Args, ca)
				continue
			}
			nd := k.Nodes[a.Node-1]
			c.Args = append(c.Args, CallArg{Field: a.Field, Value: true, Length: nd.Op == OpLength, Extent: int(nd.Imm)})
		}
		for _, b := range k.Bufs {
			if b.Kind != BufShared {
				c.Binds = append(c.Binds, Binding{Operand: b.Operand, Writes: b.Kind == BufOut, Bytes: b.bytes(), Words: b.Words})
			}
		}
		codes = append(codes, PartCode{Code: c, Part: k.Of})
	}
	return codes, errors.Join(errs...)
}
