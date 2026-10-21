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
	for _, part := range parts {
		plan.Parts = append(plan.Parts, PartPlan{Part: part, Device: p(part)})
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
	return plan, errors.Join(errs...)
}

// formRun is a run of consecutive parts on one device: each decomposed and
// generated, the run passed as one, each part's kernels lowered.
func (r *Registry) formRun(f *jlm.File, run []PartPlan, first int, s Step) error {
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
				return fmt.Errorf("%s: %w", pp.Part, err)
			}
			k.Part, k.Of = pp.Part.Kind, first+i
			ks = append(ks, k)
		}
	}
	codes, err := r.form(ks, d)
	if err != nil {
		errs = append(errs, err)
	}
	for _, c := range codes {
		run[c.of-first].Codes = append(run[c.of-first].Codes, c.Code)
	}
	return errors.Join(errs...)
}

type ownedCode struct {
	Code
	of int
}

// form is a run's kernels on device d: passed and lowered.
func (r *Registry) form(ks []*Kernel, d Device) ([]ownedCode, error) {
	prof, ok := r.Profiles[d.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no profile for %s", d.Kind)
	}
	ks, err := prof.run(ks, d)
	if err != nil {
		return nil, err
	}
	low, ok := r.Lowerings[d.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no lowering for %s", d.Kind)
	}
	var codes []ownedCode
	var errs []error
	for _, k := range ks {
		l, err := low.Lower(k, d)
		if err != nil {
			errs = append(errs, fmt.Errorf("jit: lowering %s: %w", k.Name, err))
			continue
		}
		codes = append(codes, ownedCode{Code: Code{Name: k.Name, Passes: k.Passes, Lowered: l}, of: k.Of})
	}
	return codes, errors.Join(errs...)
}
