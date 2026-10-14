package model

import (
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
)

// Scheduler runs many sequences over one batch session: it admits queued
// prompts into free rows, steps every live row through one pass over the
// weights, and retires rows that finish.
//
// It is driven, not self-running: no goroutine, channel or lock, and Step does
// one batch step and returns. The caller owns concurrency, cancellation and
// backpressure; this owns which sequence occupies which row, and when.
//
// Admission does not stall the rows already decoding. A new sequence's prompt
// rides in the steps: each Step carries every decoding row's next token and up
// to a budget of prompt tokens (WithPromptBudget), a chunk of consecutive
// positions of one sequence laid out as rows beside them. The step reads the
// weights once for all of it, so a long prompt costs the decoding rows one
// step's worth of wait per chunk rather than the whole prefill.
//
// A hybrid's chunk rides the same way, on the host and on a device: a
// recurrent block steps a chunk's rows in position order through its
// sequence's one state, the k-th row reading the state the row before it left
// (the device's run forms, tier.Stats.RecChainRows). Rows with no sequence in
// them cost nothing: only the live sequences' rows run. Changing the width
// means a new State and KV cache.
type Scheduler struct {
	st   *State
	live []*Seq // one entry per row; nil is free
	q    []*Seq
	qh   int // the first queued sequence of q not yet admitted
	// budget is the prompt tokens one step may carry; see WithPromptBudget.
	budget   int
	admitted int // admissions so far, which orders the prompts
	stats    SchedulerStats
	// chunkPos, when set, is where a prompt chunk's row j goes, given the
	// chunk's first position and its length. Nil in every Scheduler but a
	// gate's violation: it is how the gate shows that a chunk laid out at the
	// wrong positions fails it.
	chunkPos func(base, j, k int) int
	// A step's rows, kept from step to step (stepMixed).
	tok           []int32
	seq, pos      []int
	want, feeding []*Seq
	chunks        []promptChunk
	done          []*Seq // what Step returns
}

// SchedulerStats is what Steps have run: a caller -- or a gate -- sees
// admission working rather than infers it from the tokens.
type SchedulerStats struct {
	// Steps is the batch steps run.
	Steps int
	// PromptRows is prompt tokens run inside steps. MixedRows is those of them
	// that shared their step with at least one decoding row: admission that
	// stalled nobody. MaxChunk is the most rows one prompt put into a single
	// step beside a decoding row.
	PromptRows, MixedRows, MaxChunk int
	// MaxSeqs is the most sequences one step carried rows of.
	MaxSeqs int
}

// SchedulerOption configures a Scheduler (NewScheduler).
type SchedulerOption func(*Scheduler)

// WithPromptBudget caps the prompt tokens one Step carries beside the
// decoding rows. Smaller keeps each step -- the decoding rows' time between
// tokens -- short while a prompt is admitted; larger admits it in fewer
// steps. n < 1 is the default.
//
// The default fills the step to the widest batch one pass takes -- the
// device's batch width where a device runs blocks, the host's prefill scratch
// otherwise -- less the rows that may be decoding. A prompt admitted that way
// costs the same passes over the weights its prefill would have cost alone, at
// the same width, and each decoding row waits one pass rather than the whole
// prompt.
func WithPromptBudget(n int) SchedulerOption {
	return func(sc *Scheduler) { sc.budget = n }
}

// Seq is one sequence's progress through the scheduler.
type Seq struct {
	// Prompt is consumed once, over the steps after admission.
	Prompt []int32
	// MaxTokens bounds the generation. Zero means unbounded, which for a
	// server is a mistake and for a test is convenient.
	MaxTokens int
	// Stop reports whether an emitted token ends the sequence. Nil never stops.
	Stop func(id int32) bool
	// Sample turns a row's logits into the next token. Nil means greedy.
	Sample func(logits []float32) int32

	// Out is the tokens generated so far, in order.
	Out []int32
	// Done is set when the sequence retired, and Err says why if it failed.
	Done bool
	Err  error

	slot int
	// fed is how much of Prompt has run; the first token is sampled once it
	// all has.
	fed int
	// adm orders admissions, so the oldest prompt takes the budget first.
	adm int
}

// NewScheduler wraps a batch session of nseq rows. A step runs only the rows
// that hold a sequence, so a free row needs no token.
func NewScheduler(m *Model, nseq, maxSeq int, opts ...SchedulerOption) *Scheduler {
	sc := &Scheduler{st: m.NewBatch(nseq, maxSeq), live: make([]*Seq, nseq)}
	for _, o := range opts {
		o(sc)
	}
	return sc
}

// Submit queues a sequence. It is admitted by a later Step, when a row frees.
func (sc *Scheduler) Submit(s *Seq) { sc.q = append(sc.q, s) }

// Pending is how many sequences are queued but not yet admitted.
func (sc *Scheduler) Pending() int { return len(sc.q) - sc.qh }

// Live is how many rows hold a sequence.
func (sc *Scheduler) Live() int {
	n := 0
	for _, s := range sc.live {
		if s != nil {
			n++
		}
	}
	return n
}

// Stats is what the Steps so far have run.
func (sc *Scheduler) Stats() SchedulerStats { return sc.stats }

// Close releases the session.
func (sc *Scheduler) Close() { sc.st.Close() }

// Step admits what it can, runs one batch step, and returns the sequences that
// retired during it.
//
// A step with no live rows and nothing queued does no work and returns none,
// so a caller may poll it. The slice is the Scheduler's, good until the next
// Step.
func (sc *Scheduler) Step() ([]*Seq, error) {
	// A sequence whose admission failed has retired too, with Err set: it is
	// returned here, or a caller waiting for it would wait forever.
	clear(sc.done[:cap(sc.done)])
	done := sc.admit(sc.done[:0])
	if sc.Live() == 0 {
		sc.done = done
		return done, nil
	}
	done, err := sc.stepMixed(done)
	sc.done = done
	return done, err
}

// stepMixed is one step of every decoding row and the budget's prompt rows.
// The rows that want logits lead -- each decoding row, and the last row of
// each chunk that ends its prompt -- so the head projects only them; a chunk's
// other rows follow. Row order is free: every row names its sequence and
// position (State.forwardRows).
func (sc *Scheduler) stepMixed(done []*Seq) ([]*Seq, error) {
	st := sc.st
	// The step's rows and the sequence each logit row samples for, in the
	// Scheduler's own slices: a step allocates nothing once they are grown.
	tok, seq, pos, want := sc.tok[:0], sc.seq[:0], sc.pos[:0], sc.want[:0]
	defer func() {
		clear(want)
		sc.tok, sc.seq, sc.pos, sc.want = tok, seq, pos, want[:0]
	}()
	row := func(t int32, q, p int) {
		tok, seq, pos = append(tok, t), append(seq, q), append(pos, p)
	}
	decoding := 0
	for i, s := range sc.live {
		if s == nil || s.fed < len(s.Prompt) {
			continue
		}
		// A sequence out of context fails alone rather than failing the step.
		if st.bpos[i] >= st.maxSeq {
			s.Err = errFull{st.maxSeq, st.reqSeq}
			done = append(done, sc.retire(i))
			continue
		}
		row(s.Out[len(s.Out)-1], i, st.bpos[i])
		want = append(want, s)
		decoding++
	}
	// The prompts, oldest admission first, each taking what is left of the
	// budget.
	feeding := sc.feeding[:0]
	defer func() { clear(feeding); sc.feeding = feeding[:0] }()
	for _, s := range sc.live {
		if s != nil && s.fed < len(s.Prompt) {
			feeding = append(feeding, s)
		}
	}
	slices.SortFunc(feeding, byAdmission)
	chunks := sc.chunks[:0]
	defer func() { clear(chunks); sc.chunks = chunks[:0] }()
	left := sc.promptBudget()
	for _, s := range feeding {
		if left == 0 {
			break
		}
		k := min(len(s.Prompt)-s.fed, left)
		left -= k
		chunks = append(chunks, promptChunk{s, k, s.fed+k == len(s.Prompt)})
	}
	at := func(ch promptChunk, j int) int {
		base := st.bpos[ch.s.slot]
		if sc.chunkPos != nil {
			return sc.chunkPos(base, j, ch.k)
		}
		return base + j
	}
	for _, ch := range chunks {
		if ch.ends {
			j := ch.k - 1
			row(ch.s.Prompt[ch.s.fed+j], ch.s.slot, at(ch, j))
			want = append(want, ch.s)
		}
	}
	nlogit := len(tok)
	for _, ch := range chunks {
		for j := range ch.k {
			if ch.ends && j == ch.k-1 {
				continue
			}
			row(ch.s.Prompt[ch.s.fed+j], ch.s.slot, at(ch, j))
		}
	}
	if len(tok) == 0 {
		return done, nil // every live row retired above
	}
	lg, err := st.forwardRows(tok, seq, pos, nlogit)
	if err != nil {
		return done, err
	}
	sc.stats.Steps++
	sc.stats.MaxSeqs = max(sc.stats.MaxSeqs, decoding+len(chunks))
	for _, ch := range chunks {
		ch.s.fed += ch.k
		sc.stats.PromptRows += ch.k
		if decoding > 0 {
			sc.stats.MixedRows += ch.k
			sc.stats.MaxChunk = max(sc.stats.MaxChunk, ch.k)
		}
	}
	nv := st.m.Cfg.NVocab
	for r, s := range want {
		s.Out = append(s.Out, s.pick(lg[r*nv:(r+1)*nv]))
		if s.finished() {
			done = append(done, sc.retire(s.slot))
		}
	}
	return done, nil
}

// byAdmission orders sequences oldest admission first.
func byAdmission(a, b *Seq) int { return a.adm - b.adm }

// promptChunk is one prompt's rows in a step: k of them, the last of its
// prompt when ends is set.
type promptChunk struct {
	s    *Seq
	k    int
	ends bool
}

// promptBudget is the prompt tokens a step may carry; see WithPromptBudget.
func (sc *Scheduler) promptBudget() int {
	if sc.budget > 0 {
		return sc.budget
	}
	width := MaxPrefillChunk
	if sc.st.devCount() > 0 {
		width = nn.MaxDevicePrefillChunk
	}
	return max(width-len(sc.live), 1)
}

// admit fills free rows from the queue, oldest first, and returns the
// sequences that failed admission. A prompt runs in the steps that follow.
func (sc *Scheduler) admit(done []*Seq) []*Seq {
	for i := range sc.live {
		if sc.Pending() == 0 {
			return done
		}
		if sc.live[i] != nil {
			continue
		}
		// Taken from the front by index, and the slice rewound once empty,
		// so a queue that refills allocates nothing past its widest.
		s := sc.q[sc.qh]
		sc.q[sc.qh] = nil
		if sc.qh++; sc.qh == len(sc.q) {
			sc.q, sc.qh = sc.q[:0], 0
		}
		switch {
		case len(s.Prompt) == 0:
			s.Err = errEmptyPrompt{}
		case len(s.Prompt) > sc.st.maxSeq:
			s.Err = errFull{sc.st.maxSeq, sc.st.reqSeq}
		}
		if s.Err != nil {
			s.Done = true
			done = append(done, s)
			continue // the row stays free; the sequence fails without taking it
		}
		// A row that looks free may still carry a position, and on a hybrid a
		// summary: the new sequence starts from neither.
		sc.st.Retire(i)
		s.slot, s.fed, s.adm = i, 0, sc.admitted
		sc.admitted++
		sc.live[i] = s
	}
	return done
}

func (sc *Scheduler) retire(i int) *Seq {
	s := sc.live[i]
	s.Done = true
	sc.live[i] = nil
	sc.st.Retire(i)
	return s
}

func (s *Seq) pick(logits []float32) int32 {
	if s.Sample != nil {
		return s.Sample(logits)
	}
	return Greedy(logits)
}

func (s *Seq) finished() bool {
	if s.MaxTokens > 0 && len(s.Out) >= s.MaxTokens {
		return true
	}
	return s.Stop != nil && s.Stop(s.Out[len(s.Out)-1])
}
