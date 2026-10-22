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
// its last row comes out. An encoder readout (Laya) packs its sequences into
// encoder passes of up to layaPackRows rows: each sequence's rows attend
// inside it and count their positions from its start, on the host and on a
// device (nn.SeqRunsDevice), and its type row and markers are its own.

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
	// Audio is the request's voice clip, 16 kHz mono samples (d1-omni's);
	// a request with one runs on its own (DecideAudio).
	Audio []float32
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
	n, alone := 0, 0
	for r := range reqs {
		rq, res := &reqs[r], &out[r]
		if res.Err = d.Prepare(rq.State, rq.Questions, len(rq.Images)); res.Err != nil {
			continue
		}
		// On a device a request runs alone too: a batch step there carries a
		// prompt chunk as ragged rows, which attend through the decode plan a
		// row at a time, step a recurrent block's state through each run in
		// one sequential launch and stage a rotary table a row, where the
		// request's own prefill runs the tiled prefill attention and the
		// chunked recurrence. Measured on the M4 (Metal, every block placed),
		// both requests as one batch took 19.8x (d1) and 19.1x (lev) the
		// requests alone. A batch on the host is faster than alone (0.64x d1,
		// 0.84x lev) and stays one.
		if len(rq.Audio) > 0 {
			res.Answers, res.Err = d.DecideAudio(rq.State, rq.Questions, rq.Audio)
			res.InputTokens = d.InputTokens()
			continue
		}
		if len(rq.Images) > 0 || len(rq.Videos) > 0 || d.batchAlone() {
			res.Answers, res.Err = d.DecideMedia(rq.State, rq.Questions, rq.Images, rq.Videos)
			res.InputTokens = d.InputTokens()
			alone += res.InputTokens
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
				} else if d.kind == jlm.DecisionD1Omni {
					j.ids, j.markers, err = d.d1OmniSequence(rq.State, q, d.d1OmniMaxLen())
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
	// InputTokens counts the requests that ran alone too.
	d.input += alone
	// The answers, each from its question's variants, in the request's order.
	at := 0
	for r := range reqs {
		res := &out[r]
		// A request that failed alone, or that ran alone (its media, or on a device), has
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
	if (d.kind == jlm.DecisionLaya || d.kind == jlm.DecisionD1Omni) && d.m.opt.planDev != nil {
		// The plan's step is one sequence: each job its own.
		for _, j := range jobs {
			s, err := d.planScores(j.ids, j.qt, j.markers, 0)
			if err != nil {
				return err
			}
			j.scores = append(j.scores[:0], s...)
		}
		return nil
	}
	if d.kind == jlm.DecisionLaya || d.kind == jlm.DecisionD1Omni {
		return d.layaPacked(jobs)
	}
	if len(jobs) == 0 {
		return nil
	}
	if (len(jobs) == 1 || d.nonCausal()) && d.violation != "batch-alone" {
		// One prompt has no rows to share a step with: it runs as Decide
		// runs it, on the Decider's own State, where the head reads back
		// the label logits alone (forwardSpans) rather than a row's every
		// logit, and no batch session is stepped. A non-causal model's
		// prompts each run so too: a batch slot's rows attend causally
		// (bidirRuns), which is not its graph.
		d.input = 0
		for _, j := range jobs {
			s, err := d.forward(j.ids, j.groups, 0)
			if err != nil {
				return err
			}
			j.scores = append(j.scores[:0], s...)
		}
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

// batchAlone reports whether DecideBatch runs each request alone: a decoder
// readout whose blocks sit on a device. A gate turns it off to hold the
// device's batch session to the requests alone, and "batch-alone", which
// puts a batch of one into the batch session, turns it off with the rest.
//
// A Decider with a prompt cache (SetPromptCache) runs its requests alone
// as well: a batch session's page holds every row's slots, so no row's
// tokens name it (State.SetCacheKey), and alone each prompt restores what it
// shares with the last and prefills the rest.
func (d *Decider) batchAlone() bool {
	if d.kind == jlm.DecisionLaya || d.kind == jlm.DecisionClef || d.st == nil || d.violation == "batch-alone" {
		return false
	}
	return ((d.st.GPULayers() > 0 || d.nonCausal() || d.kind == jlm.DecisionKev) && d.violation != "device-batch") ||
		d.st.CacheNamespace() != ""
}

// layaPackRows is the most rows a packed encoder pass holds: a device's
// non-causal set is built for 1024 rows at placement (tier visStart), so a
// pass at or under it does not grow the set.
const layaPackRows = 1024

// layaPacked runs an encoder readout's jobs packed into passes of up to
// layaPackRows rows, a sequence longer than that alone in its pass, and
// leaves each job's scores.
func (d *Decider) layaPacked(jobs []*batchJob) error {
	e := d.emb
	defer func() { e.seqs, e.seqTypes = e.seqs[:0], e.seqTypes[:0] }()
	for len(jobs) > 0 {
		n, k := 0, 0
		for k < len(jobs) && (k == 0 || n+len(jobs[k].ids) <= layaPackRows) {
			n += len(jobs[k].ids)
			k++
		}
		pass := jobs[:k]
		jobs = jobs[k:]
		ids, marks := d.packIDs[:0], d.packMarks[:0]
		e.seqs, e.seqTypes = e.seqs[:0], e.seqTypes[:0]
		for _, j := range pass {
			lo := len(ids)
			for _, m := range j.markers {
				marks = append(marks, lo+m)
			}
			ids = append(ids, j.ids...)
			e.seqs = append(e.seqs, nn.KeyRun{Lo: lo, Hi: len(ids)})
			e.seqTypes = append(e.seqTypes, j.qt)
		}
		d.packIDs, d.packMarks = ids, marks
		if len(pass) == 1 {
			e.seqs, e.seqTypes = e.seqs[:0], e.seqTypes[:0]
		}
		var err error
		if d.kind == jlm.DecisionD1Omni {
			err = e.lfm2Encode(ids, nil)
		} else {
			err = e.mbEncode(ids)
		}
		if err != nil {
			return err
		}
		out := d.variantScores(0, len(marks))
		if err := e.layaScores(out, len(ids), pass[0].qt, marks); err != nil {
			return err
		}
		d.packs++
		for _, j := range pass {
			j.scores = append(j.scores[:0], out[:len(j.markers)]...)
			out = out[len(j.markers):]
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
