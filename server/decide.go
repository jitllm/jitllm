package server

import (
	"context"
	"fmt"
	"time"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Decisions: TypeSafe's System One request, answered by a decision model's
// label readout (docs/design/decision-models.md). /v1/systemone calls
// Engine.Decide and nothing else, so an answer served here is the answer
// `jitllm decide` prints for the same model and request.

// DecideOptions is Decide's input.
type DecideOptions struct {
	// Model is a loaded model's id, or the file name it was loaded from.
	Model string
	// State is what every question is about: a string or any JSON value,
	// object keys in the order the request wrote them.
	State jinja.Value
	// Questions are the request's questions in its order.
	Questions []engine.DecisionQuestion
}

// DecideResult is one answer per question, in question order, and the
// response body they make.
type DecideResult struct {
	ModelID     string
	Answers     []engine.DecisionAnswer
	Body        []byte
	InputTokens int
	QueuedFor   time.Duration
	Took        time.Duration
}

// Decide answers o's questions with a decision model, on a Decider of its
// own instance placed where the model's sessions are.
func (e *Engine) Decide(ctx context.Context, o DecideOptions) (*DecideResult, error) {
	lm, err := e.resolveModel(o.Model)
	if err != nil {
		return nil, err
	}
	if lm.m.Container.Config().DecisionLabels == nil {
		return nil, fmt.Errorf("%w: model %q (%s) is not a decision model: its container states no "+
			"label readout, so no score of its forward pass is an answer the weights were trained for",
			ErrInvalid, lm.id, lm.m.Cfg.Arch)
	}
	t0 := time.Now()
	gs := e.gatesFor(lm.gateIDs())
	gs.acquire(e.nextID("dec"))
	defer gs.release()
	d, err := lm.takeDecider(e.defaultMaxSeq(lm))
	if err != nil {
		return nil, err
	}
	defer lm.giveDecider(d)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t1 := time.Now()
	ans, err := d.d.Decide(o.State, o.Questions)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	lm.tokensPrefilled.Add(int64(d.d.InputTokens()))
	return &DecideResult{ModelID: lm.id, Answers: ans, Body: d.d.DecisionResponse(lm.id, o.Questions, ans),
		InputTokens: d.d.InputTokens(), QueuedFor: t1.Sub(t0), Took: time.Since(t1)}, nil
}

// decider is a Decider and the instance it asks through.
type decider struct {
	in *engine.Instance
	d  *engine.Decider
}

func (x *decider) close() { x.in.Close() }

// deciders is a model's idle deciders, as embedders are its embedders: one
// per concurrent request, built when none is idle and kept.
type deciders struct {
	free   []*decider
	busy   int
	closed bool
	idle   chan struct{}
}

func (lm *LoadedModel) takeDecider(maxSeq int) (*decider, error) {
	lm.decMu.Lock()
	if lm.dec.closed {
		lm.decMu.Unlock()
		return nil, fmt.Errorf("%w: model %q was unloaded", ErrNotFound, lm.id)
	}
	lm.dec.busy++
	if n := len(lm.dec.free); n > 0 {
		x := lm.dec.free[n-1]
		lm.dec.free = lm.dec.free[:n-1]
		lm.decMu.Unlock()
		return x, nil
	}
	lm.decMu.Unlock()
	in, err := engine.NewInstance(lm.m, maxSeq, lm.io...)
	if err == nil && lm.device() != nil {
		if err = in.SetDeviceLayers(lm.device(), lm.deviceBlocks()); err != nil {
			in.Close()
		}
	}
	if err != nil {
		lm.giveDecider(nil)
		return nil, err
	}
	d, err := engine.NewDecider(in)
	if err != nil {
		in.Close()
		lm.giveDecider(nil)
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return &decider{in: in, d: d}, nil
}

// giveDecider returns a decider to the free list, or closes it when the model
// is being unloaded.
func (lm *LoadedModel) giveDecider(x *decider) {
	lm.decMu.Lock()
	defer lm.decMu.Unlock()
	lm.dec.busy--
	if lm.dec.closed {
		if x != nil {
			x.close()
		}
		if lm.dec.busy == 0 {
			close(lm.dec.idle)
		}
		return
	}
	if x != nil {
		lm.dec.free = append(lm.dec.free, x)
	}
}

// closeDeciders closes the idle deciders and waits for the busy ones to come
// back, so none outlives the model.
func (lm *LoadedModel) closeDeciders() {
	lm.decMu.Lock()
	if lm.dec.closed {
		lm.decMu.Unlock()
		return
	}
	lm.dec.closed = true
	free := lm.dec.free
	lm.dec.free = nil
	lm.dec.idle = make(chan struct{})
	busy, idle := lm.dec.busy, lm.dec.idle
	lm.decMu.Unlock()
	for _, x := range free {
		x.close()
	}
	if busy > 0 {
		<-idle
	}
}
