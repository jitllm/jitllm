package model

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
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

// BlockGraph is block li's graph: an encoder's own (encgraph.go) where the
// model is one, nil where its family states none.
func (c composing) BlockGraph(li int) *graph.Graph {
	if e := c.m.enc; e != nil {
		if li < len(e.graphs) && e.graphs[li] != nil {
			return e.graphs[li].g
		}
		return nil
	}
	return c.m.BlockGraph(li)
}

// HeadGraph is the vocabulary's head, or an encoder's pooling into its
// embedding.
func (c composing) HeadGraph() *graph.Graph {
	if c.m.enc != nil {
		p := graph.Pooled{NEmbd: c.m.Cfg.NEmbd, Pooling: uint8(c.m.Cfg.Pooling)}
		return p.Build()
	}
	// A decoder that embeds pools its normed rows, through its dense heads
	// where it has them (embedstate.go).
	if cfg := c.m.Cfg; cfg.Pooling != jlm.PoolNone {
		p := graph.Pooled{NEmbd: cfg.NEmbd, Pooling: uint8(cfg.Pooling), OutNorm: true, Eps: cfg.RMSEps}
		if e := c.m.dense1.e; e != nil {
			p.Dense = c.m.dense1.rows
		}
		return p.Build()
	}
	return c.m.HeadGraph()
}

// EmbedGraph is the input rows: an encoder's are the token's row, the
// token type's first, the position's where it has a table, LayerNormed.
func (c composing) EmbedGraph() *graph.Graph {
	if e := c.m.enc; e != nil {
		cfg := c.m.Cfg
		g := graph.Embed{NEmbd: cfg.NEmbd, NVocab: cfg.NVocab, TypeRow: e.typ0 != nil, Norm: true, Eps: cfg.RMSEps}
		if e.rope == nil {
			g.Positions = e.pos.rows
		}
		return g.Build()
	}
	return c.m.EmbedGraph()
}

func (c composing) ActWindow() int { return c.m.ActWindow() }

// Rotaries is the model's rotaries as its rotary ops state them: the
// model's own, then the sliding layers' where it trains two.
func (c composing) Rotaries() []graph.Rotary {
	var out []graph.Rotary
	for _, r := range c.Ropes() {
		out = append(out, graph.Rotary{NRot: r.NRot, Base: r.Base, Freqs: len(r.Freqs) > 0})
	}
	return out
}

// Ropes is the engine's rotaries in Rotaries' order, each table's planes:
// an encoder's own where it turns q and k.
func (c composing) Ropes() []nn.Rope {
	if e := c.m.enc; e != nil && e.rope != nil {
		return []nn.Rope{*e.rope}
	}
	out := []nn.Rope{c.m.rope}
	if c.m.ropeSWA != nil {
		out = append(out, *c.m.ropeSWA)
	}
	return out
}

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
	r, err := run.New(m.Composing(), m.opt.planDev, m.opt.planGPU, keys)
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
