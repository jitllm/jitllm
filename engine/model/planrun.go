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
	if e := c.m.enc; e != nil && e.mb != nil {
		// ModernBERT's trunk or an LFM2 encoder's (d1-omni's), then the
		// decision head's blocks.
		mb, trunk := e.mb, c.trunk()
		switch {
		case li < trunk && e.lf != nil:
			return e.lf.blocks[li].g
		case li < trunk:
			return mb.blocks[li].g
		case li-trunk < len(mb.head):
			return mb.head[li-trunk].g
		}
		return nil
	}
	if e := c.m.enc; e != nil {
		if li < len(e.graphs) && e.graphs[li] != nil {
			return e.graphs[li].g
		}
		return nil
	}
	return c.m.BlockGraph(li)
}

// Blocks is the config's layers, and a ModernBERT decision head's blocks
// after them.
func (c composing) Blocks() int {
	if e := c.m.enc; e != nil && e.mb != nil {
		return c.trunk() + len(e.mb.head)
	}
	return c.m.Cfg.NLayer
}

// trunk is a decision model's encoder blocks: an LFM2 encoder's or
// ModernBERT's.
func (c composing) trunk() int {
	if e := c.m.enc; e.lf != nil {
		return len(e.lf.blocks)
	}
	return len(c.m.enc.mb.blocks)
}

// SeamGraph is ModernBERT's decision prologue before its head's first
// block: the trunk's final norm and the question type's row.
func (c composing) SeamGraph(li int) *graph.Graph {
	e := c.m.enc
	if e == nil || e.mb == nil || len(e.mb.head) == 0 || li != c.trunk() {
		return nil
	}
	d := c.m.Cfg.NEmbd
	g := &graph.Graph{Block: -1}
	x := g.Input("x", d)
	g.Out = []graph.ValueID{graph.DecisionPrologue(d, len(e.mb.types)/d, c.m.Cfg.RMSEps, e.lf != nil)(g, x)}
	return g
}

// HeadGraph is the vocabulary's head, or an encoder's pooling into its
// embedding.
func (c composing) HeadGraph() *graph.Graph {
	if e := c.m.enc; e != nil && e.mb != nil && len(e.mb.head) > 0 {
		s := graph.Scorer{NEmbd: c.m.Cfg.NEmbd, Eps: c.m.Cfg.RMSEps}
		return s.Build()
	}
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
	if e := c.m.enc; e != nil && e.lf != nil {
		// An LFM2 encoder's: the token's row alone.
		g := graph.Embed{NEmbd: c.m.Cfg.NEmbd, NVocab: c.m.Cfg.NVocab}
		return g.Build()
	}
	if e := c.m.enc; e != nil && e.mb != nil {
		// ModernBERT's: the token's row, LayerNormed without a bias.
		cfg := c.m.Cfg
		g := graph.Embed{NEmbd: cfg.NEmbd, NVocab: cfg.NVocab, Norm: true, NoNormBias: true, Eps: cfg.RMSEps}
		return g.Build()
	}
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

// Tower is false: the text model's view.
func (c composing) Tower() bool { return false }

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
	if e := c.m.enc; e != nil && e.lf != nil {
		return []nn.Rope{e.lf.rope}
	}
	if e := c.m.enc; e != nil && e.mb != nil {
		return []nn.Rope{e.mb.ropeG, e.mb.ropeL}
	}
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

// WithPlanChunk runs a planned model's prompt n tokens a step (n of 2 or
// more), on a plan of n rows sharing the decode plan's sequence. The
// default is a token a step: the composed step of several rows has no
// matrix-matrix form yet, and on the V100 Qwen3-0.6B prefills at 141 tok/s
// in chunks of 32 against 294 a token at a time.
func WithPlanChunk(n int) Option { return func(l *loadOpts) { l.opt.planChunk = n } }

// chunkRunner is the model's plan at a capacity of keys positions in steps
// of WithPlanChunk's rows, composed and compiled once; a model whose plan
// has no step of several rows says why, once.
func (m *Model) chunkRunner(keys int) (*run.Runner, error) {
	return m.planRunner(keys, m.opt.planChunk)
}

// planRunner is the model's plan at a capacity of keys positions in steps
// of rows, composed and compiled once, or why it has none.
func (m *Model) planRunner(keys, rows int) (*run.Runner, error) {
	m.planMu.Lock()
	defer m.planMu.Unlock()
	at := [2]int{keys, rows}
	if r := m.chunkRunners[at]; r != nil {
		return r, nil
	}
	if err := m.chunkErrs[at]; err != nil {
		return nil, err
	}
	if m.chunkRunners == nil {
		m.chunkRunners, m.chunkErrs = map[[2]int]*run.Runner{}, map[[2]int]error{}
	}
	r, err := run.New(m.Composing(), m.opt.planDev, m.opt.planGPU, keys, run.WithRows(rows))
	if err != nil {
		m.chunkErrs[at] = err
		return nil, err
	}
	m.chunkRunners[at] = r
	return r, nil
}

func (m *Model) closeRunners() {
	m.planMu.Lock()
	defer m.planMu.Unlock()
	for k, r := range m.runners {
		r.Close()
		delete(m.runners, k)
	}
	for k, r := range m.chunkRunners {
		r.Close()
		delete(m.chunkRunners, k)
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

// planPrefill is Prefill on the runner: a step a token, or with
// WithPlanChunk up to its rows a step on the chunk runner, its session the
// decode session's sequence, where the plan composes a step of several.
func (s *State) planPrefill(tokens []int32) ([]float32, error) {
	if s.planErr != nil {
		return nil, s.planErr
	}
	chunk := s.m.opt.planChunk
	if chunk > 1 && len(tokens) > 1 && s.planChunk == nil {
		if r, err := s.m.chunkRunner(s.maxSeq); err == nil {
			if s.planChunk, err = r.NewSessionOf(s.planSess); err != nil {
				return nil, err
			}
		}
	}
	var out []float32
	for len(tokens) > 0 {
		if s.planChunk == nil || len(tokens) == 1 {
			var err error
			if out, err = s.planForward(tokens[0]); err != nil {
				return nil, err
			}
			s.promptRows++
			tokens = tokens[1:]
			continue
		}
		n := min(len(tokens), chunk)
		logits, _, err := s.planChunk.Rows(tokens[:n], s.bpos[0])
		if err != nil {
			return nil, err
		}
		copy(s.logits, logits)
		out = s.logits
		s.tok = tokens[n-1]
		s.bpos[0] += n
		s.pos = max(s.pos, s.bpos[0])
		s.promptRows += n
		tokens = tokens[n:]
	}
	return out, nil
}

// planScores is an encoder readout's question on the composer's plan: the
// encoder over ids in one bidirectional step and the head's score at each
// marker, into variant v's scratch. A step's rows are a power of two at or
// past the sequence's, each a runner composed once.
func (d *Decider) planScores(ids []int32, qt jlm.QuestionType, markers []int, v int) ([]float32, error) {
	rows := 64
	for rows < len(ids) {
		rows *= 2
	}
	s := d.plans[rows]
	if s == nil {
		r, err := d.m.planRunner(rows, rows)
		if err != nil {
			return nil, err
		}
		if s, err = r.NewSession(); err != nil {
			return nil, err
		}
		if d.plans == nil {
			d.plans = map[int]*run.Session{}
		}
		d.plans[rows] = s
	}
	d.planMarks = d.planMarks[:0]
	for _, m := range markers {
		d.planMarks = append(d.planMarks, int32(m))
	}
	got, err := s.Decide(ids, int(qt), d.planMarks)
	if err != nil {
		return nil, err
	}
	out := d.variantScores(v, len(markers))
	copy(out, got)
	d.input += len(ids)
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
