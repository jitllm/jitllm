package server

import (
	"context"
	"fmt"
	"image"
	"time"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
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
	// Images are the request's pictures, in the order the template shows
	// them, each encoded through the model's tower.
	Images []image.Image
	// Clips are the request's voice clips, mono samples in [-1, 1] at the
	// audio tower's rate, each encoded through it after the pictures.
	Clips [][]float32
	// Videos are the request's videos, shown after the pictures, each its
	// frames read through the model's tower as it reads a video
	// (Model.VideoSpans).
	Videos []Video
}

// Video is a request's video: its frames in order, and their rate in frames
// a second, 0 where the request states none.
type Video struct {
	Frames []image.Image
	FPS    float64
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
	if lm.m.Container.Config().Decision == jlm.DecisionNone {
		return nil, fmt.Errorf("%w: model %q is not a decision model: its container states no "+
			"readout, so no score of its forward pass is an answer the weights were trained for",
			ErrInvalid, lm.id)
	}
	if len(o.Images) > 0 && !lm.m.SeesPictures() {
		return nil, fmt.Errorf("%w: model %q does not accept images: it carries no vision tower", ErrInvalid, lm.id)
	}
	if len(o.Clips) > 0 && lm.m.Audio == nil {
		return nil, fmt.Errorf("%w: model %q does not accept audio: it carries no audio tower", ErrInvalid, lm.id)
	}
	if len(o.Videos) > 0 && (lm.m.Tower == nil || lm.m.Tower.Video == nil) {
		return nil, fmt.Errorf("%w: model %q does not accept video: its container states no tower that reads one", ErrInvalid, lm.id)
	}
	videos := make([][]model.Span, len(o.Videos))
	for i, v := range o.Videos {
		var err error
		if videos[i], err = lm.m.VideoSpans(v.Frames, v.FPS); err != nil {
			return nil, fmt.Errorf("%w: videos[%d]: %v", ErrInvalid, i, err)
		}
	}
	pics := make([]model.Span, 0, len(o.Images)+len(o.Clips))
	for i, img := range o.Images {
		sp, err := lm.m.PictureSpan(img)
		if err != nil {
			return nil, fmt.Errorf("%w: images[%d]: %v", ErrInvalid, i, err)
		}
		pics = append(pics, sp)
	}
	for i, clip := range o.Clips {
		sp, err := lm.m.ClipSpan(clip)
		if err != nil {
			return nil, fmt.Errorf("%w: files[%d]: %v", ErrInvalid, i, err)
		}
		pics = append(pics, sp)
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
	var ans []engine.DecisionAnswer
	var asked error
	err = d.ms.Run(ctx, engine.PhasePrefill, func(in *engine.Instance) error {
		if d.d == nil {
			if d.d, asked = engine.NewDecider(in); asked != nil {
				return asked
			}
		}
		d.d.Use(in)
		ans, asked = d.d.DecideMedia(o.State, o.Questions, pics, videos)
		return asked
	})
	if asked != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, asked)
	}
	if err != nil {
		return nil, err
	}
	lm.tokensPrefilled.Add(int64(d.d.InputTokens()))
	return &DecideResult{ModelID: lm.id, Answers: ans, Body: d.d.DecisionResponse(lm.id, o.Questions, ans),
		InputTokens: d.d.InputTokens(), QueuedFor: t1.Sub(t0), Took: time.Since(t1)}, nil
}

// decider is a Decider and the session it asks through: a history on the
// model's instance, which a question leases (Session.Run), so a decider
// composes nothing. Its Decider is made on its first lease.
type decider struct {
	ms *engine.Session
	d  *engine.Decider
}

func (x *decider) close() { x.ms.Close() }

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
	lm.attachMu.RLock()
	ms, err := lm.pool.NewSession(maxSeq)
	lm.attachMu.RUnlock()
	if err != nil {
		lm.giveDecider(nil)
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return &decider{ms: ms}, nil
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
