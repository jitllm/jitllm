package model

import (
	"fmt"
	"image"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Decisions in a batch: several requests answered at once, as a server
// answers the requests that arrived while it was busy with one. A decoder
// readout's prompts -- every question and every variant of every request --
// run as rows of one batch session (Scheduler), a step reading the weights
// once for every prompt in it, and each prompt's label logits are read where
// its last row comes out. An encoder readout (Laya) runs its sequences one
// encoder pass each: its blocks are non-causal over the call's rows, so
// sequences packed into one call would attend to each other.

// DecisionRequest is one request's state and questions.
type DecisionRequest struct {
	State     jinja.Value
	Questions []DecisionQuestion
	// Images are the request's pictures, in its order (d1's, Clef's); a
	// request with pictures runs on its own (Decide), its tower once and its
	// picture rows from the image cache for every question after the first.
	Images []image.Image
	// Videos are the request's videos (Clef's); a request with one runs on
	// its own too.
	Videos []VideoFrames
}

// decisionBatchRows is the most prompts one batch session runs at once.
const decisionBatchRows = 16

// DecisionResult is one request's part of a batch: its answers and the
// tokens its prompts ran, or why it failed alone.
type DecisionResult struct {
	Answers     []DecisionAnswer
	InputTokens int
	Err         error
}

// batchJob is one prompt of a batch: which request, question and variant it
// asks, what its readout reads, and the scores it came out with. A Decider
// keeps its jobs from batch to batch (Decider.jobs), each with its sequence
// and the function that reads the sequence's logits made once, so a warm
// batch's engine half allocates nothing.
type batchJob struct {
	d       *Decider
	r, q, v int
	qt      jlm.QuestionType
	ids     []int32
	groups  [][]int32
	markers []int
	// clef is a Clef request's prompt: one job answers the whole request.
	clef   *clefPrompt
	scores []float32
	seq    Seq
	sample func([]float32) int32
	err    error
}

// read is the job's sequence's Sample: the label scores at the prompt's last
// row, into the job's own scores.
func (j *batchJob) read(lg []float32) int32 {
	vals := j.scores[:0]
	for _, g := range j.groups {
		for _, id := range g {
			if int(id) >= len(lg) {
				j.err = fmt.Errorf("label id %d is past the %d logits", id, len(lg))
				return 0
			}
			vals = append(vals, lg[id])
		}
	}
	j.scores = vals
	s, err := j.d.labelScoresOf(vals, j.groups, 0)
	if err != nil {
		j.err = err
		return 0
	}
	j.scores = append(j.scores[:0], s...)
	return 0
}

// job is the Decider's i-th kept job, cleared for a new prompt.
func (d *Decider) job(i int) *batchJob {
	for len(d.jobs) <= i {
		j := &batchJob{d: d}
		j.sample = j.read
		d.jobs = append(d.jobs, j)
	}
	j := d.jobs[i]
	j.err = nil
	return j
}

// DecideBatch answers every request, in order: out[i] is what Decide would
// answer for reqs[i] alone. A request that fails its checks fails alone
// (out[i].Err); an error running the batch fails it whole.
func (d *Decider) DecideBatch(reqs []DecisionRequest) ([]DecisionResult, error) {
	out := make([]DecisionResult, len(reqs))
	n := 0
	for r := range reqs {
		rq, res := &reqs[r], &out[r]
		if res.Err = d.Prepare(rq.State, rq.Questions, len(rq.Images)); res.Err != nil {
			continue
		}
		// A request alone in its batch whose questions share a trunk runs
		// the trunk once (decideTrunk), and one with media runs alone too;
		// several text requests stay rows of one session, each prompt whole.
		if len(rq.Images) > 0 || len(rq.Videos) > 0 || len(reqs) == 1 && d.trunkShared(rq.Questions) {
			res.Answers, res.Err = d.DecideMedia(rq.State, rq.Questions, rq.Images, rq.Videos)
			res.InputTokens = d.InputTokens()
			continue
		}
		first := n
		if d.kind == jlm.DecisionClef {
			j := d.job(n)
			j.r, j.q, j.v = r, -1, 0
			if res.Err = d.clefJob(j, rq.State, rq.Questions); res.Err != nil {
				continue
			}
			n++
			continue
		}
	questions:
		for qi := range rq.Questions {
			q := &rq.Questions[qi]
			for v := range d.variants(q) {
				j := d.job(n)
				j.r, j.q, j.v, j.qt = r, qi, v, q.Type
				var err error
				if d.kind == jlm.DecisionLaya {
					j.ids, j.markers, err = d.layaSequence(rq.State, q)
				} else {
					j.ids, j.groups, err = d.prompt(rq.State, q, v)
				}
				if err != nil {
					res.Err = fmt.Errorf("questions.%s: %w", q.ID, err)
					n = first
					break questions
				}
				n++
			}
		}
	}
	jobs := d.jobs[:n]
	d.batchN = n
	for _, j := range jobs {
		out[j.r].InputTokens += len(j.ids)
	}
	if err := d.runJobs(jobs); err != nil {
		return nil, err
	}
	// The answers, each from its question's variants, in the request's order.
	at := 0
	for r := range reqs {
		res := &out[r]
		// A request that failed alone, or that ran alone (its media), has
		// its answer already and no job in the batch.
		if res.Err != nil || res.Answers != nil {
			continue
		}
		qs := reqs[r].Questions
		if d.kind == jlm.DecisionClef {
			res.Answers, res.Err = d.clefAnswers(qs, jobs[at].scores)
			at++
			continue
		}
		res.Answers = make([]DecisionAnswer, len(qs))
		for qi := range qs {
			var all [][]float32
			for ; at < len(jobs) && jobs[at].r == r && jobs[at].q == qi; at++ {
				all = append(all, jobs[at].scores)
			}
			a, err := d.answer(&qs[qi], all)
			if err != nil {
				res.Answers, res.Err = nil, fmt.Errorf("questions.%s: %w", qs[qi].ID, err)
				break
			}
			res.Answers[qi] = a
		}
		for at < len(jobs) && jobs[at].r == r {
			at++
		}
	}
	return out, nil
}

// runJobs runs a batch's prompts and leaves each job's scores: the engine's
// half of a batch, which makes no heap allocation once warm
// (TestDecisionDoesNotAllocate). An encoder's run one pass each; a decoder's
// as rows of the batch session.
func (d *Decider) runJobs(jobs []*batchJob) error {
	d.input = 0
	for _, j := range jobs {
		d.input += len(j.ids)
	}
	if d.kind == jlm.DecisionClef {
		// One pass a request, every row of it kept for the head: a batch
		// session's rows give back their last row alone.
		for _, j := range jobs {
			if cap(j.scores) < len(j.clef.opts) {
				j.scores = make([]float32, len(j.clef.opts))
			}
			j.scores = j.scores[:len(j.clef.opts)]
			if err := d.clefForward(j.clef, j.scores); err != nil {
				return err
			}
		}
		d.input = 0
		for _, j := range jobs {
			d.input += len(j.ids)
		}
		return nil
	}
	if d.kind == jlm.DecisionLaya {
		for _, j := range jobs {
			s, err := d.layaForward(j.ids, j.qt, j.markers, 0)
			if err != nil {
				return err
			}
			j.scores = append(j.scores[:0], s...)
		}
		return nil
	}
	if len(jobs) == 0 {
		return nil
	}
	sc, err := d.batchSession(min(len(jobs), decisionBatchRows))
	if err != nil {
		return err
	}
	for _, j := range jobs {
		j.seq = Seq{Prompt: j.ids, MaxTokens: 1, Sample: j.sample, Out: j.seq.Out[:0]}
		sc.Submit(&j.seq)
	}
	for sc.Pending() > 0 || sc.Live() > 0 {
		done, err := sc.Step()
		if err != nil {
			return err
		}
		for _, s := range done {
			if s.Err != nil {
				return s.Err
			}
		}
	}
	for _, j := range jobs {
		if j.err != nil {
			return j.err
		}
	}
	return nil
}

// batchSession is the Decider's batch session, at least rows wide, placed on
// the Decider's device as its own State is. It is kept between batches; a
// wider batch replaces it.
func (d *Decider) batchSession(rows int) (*Scheduler, error) {
	if d.batch != nil && len(d.batch.live) >= rows {
		return d.batch, nil
	}
	if d.batch != nil {
		d.batch.Close()
		d.batch = nil
	}
	sc := NewScheduler(d.m, rows, d.st.MaxSeq())
	if d.dev != nil {
		if err := sc.st.SetDeviceLayers(d.dev, d.devMax); err != nil {
			sc.Close()
			return nil, err
		}
	}
	d.batch = sc
	return sc, nil
}

// BatchDeviceBlocks is how many blocks the Decider's batch session runs on a
// device: 0 before a batch has run.
func (d *Decider) BatchDeviceBlocks() int {
	if d.batch == nil {
		return 0
	}
	return d.batch.st.GPULayers()
}

// BatchStats is what the Decider's batch session has run.
func (d *Decider) BatchStats() SchedulerStats {
	if d.batch == nil {
		return SchedulerStats{}
	}
	return d.batch.Stats()
}

// setBatchDevice follows the Decider's device with its batch session.
func (d *Decider) setBatchDevice(dev nn.Device, max int) error {
	d.dev, d.devMax = dev, max
	if d.batch == nil {
		return nil
	}
	return d.batch.st.SetDeviceLayers(dev, max)
}
