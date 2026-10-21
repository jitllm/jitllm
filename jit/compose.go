// Package jit is the composer: a model and its placement in, for every
// device the kernels the model runs there and the plan that launches them.
// The model's graph is decomposed into primitives, each generated as naive
// IR, the IR rewritten by registered passes, and after the last pass lowered
// by the device's backend.
package jit

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// Step is the shape of one step the engine runs: rows a step, the history's
// positions a page, and the longest history a row attends.
type Step struct {
	Rows, Page, Keys int
}

// Placement is the device each part of the model runs on.
type Placement func(Part) Device

// Code is one kernel as its backend runs it, and the passes that formed it.
type Code struct {
	Name   string
	Passes []string
	Lowered
}

// PartPlan is one part of the model on its device: its kernels in launch
// order.
type PartPlan struct {
	Part   Part
	Device Device
	Codes  []Code
}

// Plan is the model on its placement.
type Plan struct {
	Parts []PartPlan
}

// Compose forms the model in f at step s on placement p. A part it cannot
// form is an error, beside the parts it formed.
func (r *Registry) Compose(f *jlm.File, s Step, p Placement) (*Plan, error) {
	parts, err := front(f)
	if err != nil {
		return nil, err
	}
	plan := &Plan{}
	var errs []error
	for _, part := range parts {
		d := p(part)
		codes, err := r.form(f, part, s, d)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", part, err))
		}
		plan.Parts = append(plan.Parts, PartPlan{Part: part, Device: d, Codes: codes})
	}
	return plan, errors.Join(errs...)
}

// form is one part on device d: decomposed, generated, passed and lowered.
func (r *Registry) form(f *jlm.File, part Part, s Step, d Device) ([]Code, error) {
	prims, err := r.decompose(f, part, s, d)
	if err != nil {
		return nil, err
	}
	var ks []*Kernel
	for _, p := range prims {
		k, err := generate(p)
		if err != nil {
			return nil, err
		}
		k.Part = part.Kind
		ks = append(ks, k)
	}
	prof, ok := r.Profiles[d.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no profile for %s", d.Kind)
	}
	ks, err = prof.run(ks, d)
	if err != nil {
		return nil, err
	}
	low, ok := r.Lowerings[d.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no lowering for %s", d.Kind)
	}
	var codes []Code
	for _, k := range ks {
		l, err := low.Lower(k, d)
		if err != nil {
			return codes, fmt.Errorf("jit: lowering %s: %w", k.Name, err)
		}
		codes = append(codes, Code{Name: k.Name, Passes: k.Passes, Lowered: l})
	}
	return codes, nil
}
