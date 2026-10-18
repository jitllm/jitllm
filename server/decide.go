package server

import (
	"context"
	"fmt"
	"image"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Decisions: TypeSafe's System One request, answered by a decision model
// (docs/design/decision-models.md). /v1/systemone calls Engine.Decide and
// nothing else, so an answer served here is the answer `jitllm decide` prints
// for the same model and request.
//
// A decision runs in one turn on the model's gates (its devices' and the
// host's, as a generate does), like an embedding: its prompts are prefilled
// on the model's Decider, one State the model keeps and reuses, its blocks
// placed on the model's device as a session's are (decideMaxBlocks); the
// requests that arrive while it is busy are answered together after it, as
// one batch (Decide).

// DecideOptions is Decide's input.
type DecideOptions struct {
	// Model is a loaded model's id, or the file name it was loaded from.
	Model string
	// State is what every question is about: a string or any JSON value,
	// object keys in the order the request wrote them.
	State jinja.Value
	// Questions are the request's questions in its order.
	Questions []model.DecisionQuestion
	// Images are the request's pictures in its order, at most
	// model.MaxDecisionImages; only a vision decision model (d1, Clef) reads
	// them.
	Images []image.Image
	// Videos are the request's videos, each its frames (Clef's).
	Videos []model.VideoFrames
	// Audio is the request's voice clip, 16 kHz mono samples (d1-omni's).
	Audio []float32
}

// DecideResult is one answer per question, in question order.
type DecideResult struct {
	ModelID     string
	Answers     []model.DecisionAnswer
	InputTokens int
	QueuedFor   time.Duration
	Took        time.Duration
	// Batched is how many requests ran in the batch this one ran in.
	Batched int
}

// decJob is one request waiting for, or answered by, a batch.
type decJob struct {
	ctx  context.Context
	req  model.DecisionRequest
	t0   time.Time
	done chan struct{}
	res  DecideResult
	err  error
}

// Decide answers o's questions with a decision model.
//
// Requests for one model that arrive while it is answering others are
// answered together: each queues, and whichever takes the model's Decider
// next runs every request queued by then as one batch (Decider.DecideBatch),
// a decoder readout's prompts as rows of one batch session. A request that
// fails its checks fails alone.
func (e *Engine) Decide(ctx context.Context, o DecideOptions) (*DecideResult, error) {
	lm, err := e.resolveModel(o.Model)
	if err != nil {
		return nil, err
	}
	if lm.m.Decision() == jlm.DecisionNone {
		return nil, fmt.Errorf("%w: model %q (%s) is not a decision model: its container states no "+
			"decision readout, so no score of its forward pass is an answer the weights were trained for",
			ErrInvalid, lm.id, lm.m.Cfg.Arch)
	}
	job := &decJob{ctx: ctx, req: model.DecisionRequest{State: o.State, Questions: o.Questions, Images: o.Images,
		Videos: o.Videos, Audio: o.Audio},
		t0: time.Now(), done: make(chan struct{})}
	lm.decQMu.Lock()
	lm.decQ = append(lm.decQ, job)
	lm.decQMu.Unlock()

	lm.decMu.Lock()
	select {
	case <-job.done:
		// Answered in a batch another request ran.
		lm.decMu.Unlock()
	default:
		e.decideQueued(lm)
		lm.decMu.Unlock()
	}
	if job.err != nil {
		return nil, job.err
	}
	return &job.res, nil
}

// decideQueued runs every queued request of lm as one batch, on the model's
// gates, and answers each. The caller holds lm.decMu.
func (e *Engine) decideQueued(lm *LoadedModel) {
	lm.decQMu.Lock()
	jobs := lm.decQ
	lm.decQ = nil
	lm.decQMu.Unlock()
	finish := func(j *decJob, err error) {
		j.err = err
		close(j.done)
	}
	if lm.decClosed {
		for _, j := range jobs {
			finish(j, fmt.Errorf("%w: model %q was unloaded", ErrNotFound, lm.id))
		}
		return
	}
	// A request whose client has gone is not run.
	live := jobs[:0]
	for _, j := range jobs {
		if err := j.ctx.Err(); err != nil {
			finish(j, err)
			continue
		}
		live = append(live, j)
	}
	jobs = live
	if len(jobs) == 0 {
		return
	}
	gs := e.gatesFor(lm.gateIDs())
	gs.acquire(e.nextID("dec"))
	defer gs.release()
	if lm.dec == nil {
		d, err := lm.m.NewDecider(e.defaultMaxSeq(lm))
		if err == nil && lm.device() != nil {
			if err = d.SetDevice(lm.device(), lm.deviceBlocks()); err != nil {
				d.Close()
			}
		}
		// The model's shared pages: a question whose prompt starts as an
		// earlier one did aliases that prefix's pages and prefills the rest.
		if err == nil && lm.ttft.shared != nil {
			if _, err = d.SetPromptCache(lm.ttft.shared, lm.ttft.ns); err != nil {
				d.Close()
			}
		}
		if err != nil {
			for _, j := range jobs {
				finish(j, err)
			}
			return
		}
		lm.dec = d
	}
	reqs := make([]model.DecisionRequest, len(jobs))
	for i, j := range jobs {
		reqs[i] = j.req
	}
	t1 := time.Now()
	r0 := lm.dec.PromptRestored()
	out, err := lm.dec.DecideBatch(reqs)
	took := time.Since(t1)
	restored := int64(lm.dec.PromptRestored() - r0)
	lm.ttft.restored.Add(restored)
	lm.ttft.computed.Add(-restored)
	lm.decBatchMax.Store(max(lm.decBatchMax.Load(), int64(len(jobs))))
	for i, j := range jobs {
		if err != nil {
			finish(j, err)
			continue
		}
		r := out[i]
		if r.Err != nil {
			finish(j, fmt.Errorf("%w: %v", ErrInvalid, r.Err))
			continue
		}
		lm.tokensPrefilled.Add(int64(r.InputTokens))
		lm.ttft.computed.Add(int64(r.InputTokens))
		j.res = DecideResult{ModelID: lm.id, Answers: r.Answers, InputTokens: r.InputTokens,
			QueuedFor: t1.Sub(j.t0), Took: took, Batched: len(jobs)}
		finish(j, nil)
	}
}

// deviceBlocks is how many blocks a State, Decider or Embedder of this model
// offers its device: the load's cap, or every block (-1).
func (lm *LoadedModel) deviceBlocks() int {
	if lm.maxBlocks != 0 {
		return lm.maxBlocks
	}
	return -1
}

// closeDecider closes the model's Decider; a decision in flight holds decMu
// and finishes first.
func (lm *LoadedModel) closeDecider() {
	lm.decMu.Lock()
	defer lm.decMu.Unlock()
	lm.decClosed = true
	if lm.dec != nil {
		lm.dec.Close()
		lm.dec = nil
	}
}
