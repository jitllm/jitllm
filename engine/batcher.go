package engine

import (
	"context"
	"errors"
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/nn"
)

// Batcher is continuous batching over a model's served instances (Serve).
// A queued sequence is admitted as a session the model's scheduler leases an
// instance to; every step carries each decoding session's next token and a
// budget of prompt tokens beside them, one StepRuns per instance; a sequence
// that finishes retires, its lease handed back and its history freed. A
// sequence whose prompt has run leases again for generation, which the
// scheduler may hand another instance.
//
// It is driven, not self-running: no goroutine, channel or lock of its own,
// and Step does one step on every instance and returns. The caller owns
// concurrency, cancellation and backpressure; this owns which sequence runs
// on which instance, and when.
type Batcher struct {
	m *model.Model
	// pool leases the batcher's sessions their instances.
	pool   *Pool
	maxSeq int
	// width is the most sequences admitted at once; budget the prompt
	// tokens one step of an instance may carry (WithPromptBudget).
	width, budget int
	live          []*Seq
	q             []*Seq
	qh            int
	admitted      int
	stats         BatcherStats
	// none is a context already done: an admission that would wait for
	// room leaves the sequence queued instead.
	none context.Context
	// setup is WithSessionSetup's.
	setup func(*Session) error
	// chunkPos, when set, is where a prompt chunk's row j goes, given the
	// chunk's first position and its length: a gate's violation.
	chunkPos func(base, j, k int) int
	// A step's groups and lists, kept from step to step.
	groups  []batchGroup
	feeding []*Seq
	done    []*Seq
}

// batchGroup is one instance's part of a step: its runs and, run by run,
// the sequence each is of.
type batchGroup struct {
	in       *Instance
	runs     []Run
	of       []*Seq
	left     int
	decoding int
}

// BatcherStats is what Steps have run: a caller -- or a gate -- sees
// admission working rather than infers it from the tokens.
type BatcherStats struct {
	// Steps is the instance steps run.
	Steps int
	// PromptRows is prompt tokens run inside steps. MixedRows is those of
	// them that shared their step with at least one decoding row:
	// admission that stalled nobody. MaxChunk is the most rows one prompt
	// put into a single step beside a decoding row.
	PromptRows, MixedRows, MaxChunk int
	// MaxSeqs is the most sequences one step carried rows of.
	MaxSeqs int
	// KVStored, KVFetched and KVEvicted are the KV pages the retired
	// sessions stored, fetched and evicted (Session.KVPageStats).
	KVStored, KVFetched, KVEvicted int64
	// RoomToCPU is the steps whose device had no room for them
	// (nn.RowsRoomer), so blocks moved to the CPU device before the step,
	// over the instances the batcher has stepped.
	RoomToCPU int
}

// BatcherOption configures a Batcher (NewBatcher).
type BatcherOption func(*Batcher)

// WithPromptBudget caps the prompt tokens one instance's step carries beside
// its decoding rows. Smaller keeps each step -- the decoding rows' time
// between tokens -- short while a prompt is admitted; larger admits it in
// fewer steps. n < 1 is the default: the widest step the instance takes,
// less the rows that may be decoding.
func WithPromptBudget(n int) BatcherOption {
	return func(b *Batcher) { b.budget = n }
}

// WithSessionSetup calls f on each sequence's session when it is admitted,
// before its prompt runs: its KV store, budget or cache key.
func WithSessionSetup(f func(*Session) error) BatcherOption {
	return func(b *Batcher) { b.setup = f }
}

// Seq is one sequence's progress through the batcher.
type Seq struct {
	// Prompt is consumed once, over the steps after admission.
	Prompt []int32
	// MaxTokens bounds the generation. Zero means unbounded, which for a
	// server is a mistake and for a test is convenient.
	MaxTokens int
	// Stop reports whether an emitted token ends the sequence. Nil never stops.
	Stop func(id int32) bool
	// Sample turns a sequence's logits into its next token. Nil means greedy.
	Sample func(logits []float32) int32

	// Out is the tokens generated so far, in order.
	Out []int32
	// Done is set when the sequence retired, and Err says why if it failed.
	Done bool
	Err  error

	sess *Session
	// fed is how much of Prompt has run; the first token is sampled once it
	// all has.
	fed int
	// adm orders admissions, so the oldest prompt takes the budget first.
	adm int
}

// NewBatcher batches up to width sequences of m of up to maxSeq positions at
// once over pool's instances, leased by pool (MaxStepRows when width < 1).
func NewBatcher(m *model.Model, pool *Pool, width, maxSeq int, opts ...BatcherOption) *Batcher {
	if width < 1 {
		width = MaxStepRows
	}
	none, cancel := context.WithCancel(context.Background())
	cancel()
	b := &Batcher{m: m, pool: pool, width: width, maxSeq: min(maxSeq, m.Cfg.NCtx), none: none}
	for _, o := range opts {
		o(b)
	}
	return b
}

// Submit queues a sequence. It is admitted by a later Step, when there is
// room.
func (b *Batcher) Submit(s *Seq) { b.q = append(b.q, s) }

// Pending is how many sequences are queued but not yet admitted.
func (b *Batcher) Pending() int { return len(b.q) - b.qh }

// Live is how many sequences are admitted.
func (b *Batcher) Live() int { return len(b.live) }

// Stats is what the Steps so far have run.
func (b *Batcher) Stats() BatcherStats {
	st := b.stats
	for i := range b.groups {
		st.RoomToCPU += b.groups[i].in.roomToCPU
	}
	return st
}

// Close retires every admitted sequence, its history freed.
func (b *Batcher) Close() {
	for _, s := range b.live {
		b.end(s)
	}
	clear(b.live)
	b.live = b.live[:0]
}

// Step admits what it can, runs one step on every instance with work, and
// returns the sequences that retired during it. A step with nothing live and
// nothing queued does no work and returns none, so a caller may poll it. The
// slice is the Batcher's, good until the next Step.
func (b *Batcher) Step() ([]*Seq, error) {
	clear(b.done[:cap(b.done)])
	done, err := b.admit(b.done[:0])
	if err != nil || len(b.live) == 0 {
		b.done = done
		return done, err
	}
	for i := range b.groups {
		g := &b.groups[i]
		clear(g.runs)
		clear(g.of)
		g.runs, g.of, g.decoding = g.runs[:0], g.of[:0], 0
		g.left = -1
	}
	// The decoding sequences first: one token each.
	live := b.live[:0]
	for _, s := range b.live {
		if s.fed < len(s.Prompt) {
			live = append(live, s)
			continue
		}
		if s.sess.pos >= s.sess.limit {
			// A sequence out of context fails alone rather than failing the step.
			s.Err = errFull{s.sess.limit, s.sess.limit}
			done = append(done, b.end(s))
			continue
		}
		if s.sess.lease == nil {
			// Its prompt ran and no instance had room for its generation:
			// it waits a step.
			if err := s.sess.Acquire(b.none, PhaseGenerate); err != nil {
				if !errors.Is(err, context.Canceled) {
					s.Err = err
					done = append(done, b.end(s))
					continue
				}
				live = append(live, s)
				continue
			}
		}
		g := b.group(s.sess.lease.Instance)
		g.runs = append(g.runs, Run{Session: s.sess, Tokens: s.Out[len(s.Out)-1:], Logits: true})
		g.of = append(g.of, s)
		g.decoding++
		live = append(live, s)
	}
	clear(b.live[len(live):])
	b.live = live
	// The prompts, oldest admission first, each taking what is left of its
	// instance's budget.
	feeding := b.feeding[:0]
	for _, s := range b.live {
		if s.fed < len(s.Prompt) {
			feeding = append(feeding, s)
		}
	}
	slices.SortFunc(feeding, byAdmission)
	for _, s := range feeding {
		g := b.group(s.sess.lease.Instance)
		if g.left < 0 {
			g.left = b.promptBudget(g)
		}
		k := min(len(s.Prompt)-s.fed, g.left)
		if k == 0 {
			continue
		}
		g.left -= k
		g.runs = append(g.runs, Run{Session: s.sess, Tokens: s.Prompt[s.fed : s.fed+k], Logits: s.fed+k == len(s.Prompt)})
		g.of = append(g.of, s)
	}
	clear(feeding)
	b.feeding = feeding[:0]
	for i := range b.groups {
		g := &b.groups[i]
		if len(g.runs) == 0 {
			continue
		}
		if b.chunkPos != nil {
			g.in.opt.stepPosFault = b.chunkPos
		}
		lg, err := StepRuns(g.runs)
		if b.chunkPos != nil {
			g.in.opt.stepPosFault = nil
		}
		if err != nil {
			b.done = done
			return done, err
		}
		b.stats.Steps++
		b.stats.MaxSeqs = max(b.stats.MaxSeqs, len(g.runs))
		for j, r := range g.runs {
			s := g.of[j]
			if s.fed < len(s.Prompt) {
				k := len(r.Tokens)
				s.fed += k
				b.stats.PromptRows += k
				if g.decoding > 0 {
					b.stats.MixedRows += k
					b.stats.MaxChunk = max(b.stats.MaxChunk, k)
				}
				if !r.Logits {
					continue
				}
			}
			tok, err := s.pick(g.in, lg[j])
			if err != nil {
				s.Err = err
				done = append(done, b.retire(s))
				continue
			}
			s.Out = append(s.Out, tok)
			if s.finished() {
				done = append(done, b.retire(s))
				continue
			}
			if err := s.sess.Acquire(b.none, PhaseGenerate); err != nil && !errors.Is(err, context.Canceled) {
				s.Err = err
				done = append(done, b.retire(s))
			}
		}
	}
	b.done = done
	return done, nil
}

// group is instance in's group of the step.
func (b *Batcher) group(in *Instance) *batchGroup {
	for i := range b.groups {
		if b.groups[i].in == in {
			return &b.groups[i]
		}
	}
	b.groups = append(b.groups, batchGroup{in: in, left: -1})
	return &b.groups[len(b.groups)-1]
}

// promptBudget is the prompt tokens g's step may carry; see WithPromptBudget.
func (b *Batcher) promptBudget(g *batchGroup) int {
	if b.budget > 0 {
		return b.budget
	}
	in := g.in
	width := MaxPrefillChunk
	if in.devCount() > 0 {
		width = nn.MaxDevicePrefillChunk
		// A device whose budget held a narrower batched scratch refuses a
		// wider step rather than splitting it: keep under what it holds.
		if rw, ok := in.ld.(nn.RowsWidther); ok {
			if w := rw.RowsWidth(width); w > 0 {
				width = w
			}
		}
	}
	return max(width-g.decoding, 1)
}

// admit leases sessions to queued sequences, oldest first, while there is
// room, and returns those that failed admission. A prompt runs in the steps
// that follow.
func (b *Batcher) admit(done []*Seq) ([]*Seq, error) {
	for b.Pending() > 0 && len(b.live) < b.width {
		s := b.q[b.qh]
		switch {
		case len(s.Prompt) == 0:
			s.Err = errEmptyPrompt{}
		case len(s.Prompt) > b.maxSeq:
			s.Err = errFull{b.maxSeq, len(s.Prompt)}
		}
		if s.Err == nil && s.sess == nil {
			sess, err := b.pool.NewSession(b.maxSeq)
			if err != nil {
				return done, err
			}
			s.sess = sess
			if b.setup != nil {
				s.Err = b.setup(sess)
			}
		}
		if s.Err == nil {
			if err := s.sess.Acquire(b.none, PhasePrefill); err != nil {
				if errors.Is(err, context.Canceled) {
					return done, nil // no instance has room: it waits in the queue
				}
				s.Err = err
			}
		}
		// Taken from the front by index, and the slice rewound once empty,
		// so a queue that refills allocates nothing past its widest.
		b.q[b.qh] = nil
		if b.qh++; b.qh == len(b.q) {
			b.q, b.qh = b.q[:0], 0
		}
		if s.Err != nil {
			if s.sess != nil {
				s.sess.Close()
				s.sess = nil
			}
			s.Done = true
			done = append(done, s)
			continue
		}
		s.fed, s.adm = 0, b.admitted
		b.admitted++
		b.live = append(b.live, s)
	}
	return done, nil
}

// retire ends s and drops it from live.
func (b *Batcher) retire(s *Seq) *Seq {
	if i := slices.Index(b.live, s); i >= 0 {
		b.live = slices.Delete(b.live, i, i+1)
	}
	return b.end(s)
}

// end retires s: its pages counted, its lease handed back and its history
// freed.
func (b *Batcher) end(s *Seq) *Seq {
	st, fe, ev := s.sess.KVPageStats()
	b.stats.KVStored, b.stats.KVFetched, b.stats.KVEvicted = b.stats.KVStored+st, b.stats.KVFetched+fe, b.stats.KVEvicted+ev
	s.Done = true
	s.sess.Close()
	s.sess = nil
	return s
}

// byAdmission orders sequences oldest admission first.
func byAdmission(a, b *Seq) int { return a.adm - b.adm }

func (s *Seq) pick(in *Instance, logits []float32) (int32, error) {
	if s.Sample != nil {
		return s.Sample(logits), nil
	}
	return in.Sample(nil, logits)
}

func (s *Seq) finished() bool {
	if s.MaxTokens > 0 && len(s.Out) >= s.MaxTokens {
		return true
	}
	return s.Stop != nil && s.Stop(s.Out[len(s.Out)-1])
}
