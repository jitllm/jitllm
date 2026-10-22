package model

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// WithPlanDevice runs every State's decode through the composer's plan on
// dev, gpu the composer's description of it (docs/design/composer-plan.md,
// step 7.3), in place of the host and the tier: Forward and Prefill step
// through engine/run, and every other entry point refuses until the runner
// covers it.
func WithPlanDevice(dev backend.Device, gpu jit.Device) Option {
	return func(l *loadOpts) { l.opt.planDev, l.opt.planGPU = dev, gpu }
}

// composing is the model as the composer reads it: its container, and each
// block's graph as buildGraphs built it from the container.
type composing struct {
	*jlm.File
	m *Model
}

func (c composing) BlockGraph(li int) *graph.Graph { return c.m.BlockGraph(li) }

// Composing is the model as the composer and the plan's runner read it: one
// mapping from the container to the block graphs, the one this package's
// own graph path runs.
func (m *Model) Composing() run.Source { return composing{m.container, m} }

// runner is the model's plan at a capacity of keys positions, composed and
// compiled once.
func (m *Model) runner(keys int) (*run.Runner, error) {
	m.planMu.Lock()
	defer m.planMu.Unlock()
	if r := m.runners[keys]; r != nil {
		return r, nil
	}
	r, err := run.New(m.Composing(), m.opt.planDev, m.opt.planGPU, keys, m.rope)
	if err != nil {
		return nil, err
	}
	if m.runners == nil {
		m.runners = map[int]*run.Runner{}
	}
	m.runners[keys] = r
	return r, nil
}

func (m *Model) closeRunners() {
	m.planMu.Lock()
	defer m.planMu.Unlock()
	for k, r := range m.runners {
		r.Close()
		delete(m.runners, k)
	}
}

// openPlan gives s a session on the runner at its capacity; an error waits
// for s's first step, since NewState returns none.
func (s *State) openPlan() {
	r, err := s.m.runner(s.maxSeq)
	if err == nil {
		s.planSess, err = r.NewSession()
	}
	s.planErr = err
}

// planForward is Forward on the runner.
func (s *State) planForward(token int32) ([]float32, error) {
	if s.planErr != nil {
		return nil, s.planErr
	}
	logits, _, err := s.planSess.Step(token, s.bpos[0])
	if err != nil {
		return nil, err
	}
	copy(s.logits, logits)
	s.tok = token
	s.bpos[0]++
	s.pos = max(s.pos, s.bpos[0])
	return s.logits, nil
}

// planPrefill is Prefill on the runner: a step a token.
func (s *State) planPrefill(tokens []int32) ([]float32, error) {
	var out []float32
	for _, t := range tokens {
		var err error
		if out, err = s.planForward(t); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// planRefuses is the error of an entry point the runner does not cover, nil
// for a State not on it.
func (s *State) planRefuses(what string) error {
	if s.planSess == nil && s.planErr == nil {
		return nil
	}
	return fmt.Errorf("model: %s does not run on the composer's plan yet; Forward and Prefill do", what)
}
