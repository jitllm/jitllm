package model

import (
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
)

// Speculative decoding with the model's own multi-token-prediction block
// (jlm.Config.NMTP). Draft row q carries the token x_q and the trunk's normed
// hidden state h_{q-1} (zero at q = 0) and predicts the token at q+1 through
// eh_proj, one ordinary block of the architecture and the head; a drafted
// token's row carries the draft's own g as its hidden state.
//
// A round drafts d_1 .. d_k, verifies [y, d_1 .. d_k] in one trunk pass,
// accepts the longest agreeing prefix (greedy) or by speculative rejection
// sampling, takes one more token from the trunk at the first disagreement,
// and rolls the trunk back (attention by position, a recurrence by
// SpecRollback). Greedy decoding through a Speculator is greedy decoding:
// every emitted token is the trunk's argmax over exactly the tokens plain
// decode would have fed.
//
// The graph, the round, and the row pairing chosen against vLLM's:
// docs/engineering-history/model-correctness.md, "engine/model/spec.go".

// SpecRollback is how a rejected verification takes a hybrid's recurrent state
// back. Attention needs no choice: its rows are forgotten by position.
type SpecRollback int

const (
	// SpecRollbackAuto keeps a copy per verified row where every linear block
	// runs on the host, and replays where any runs on a device, which can
	// rewind its state by one step and no further.
	SpecRollbackAuto SpecRollback = iota
	// SpecRollbackRows copies every linear block's state after each verified
	// row and restores the last accepted row's: k copies a round, no extra
	// rows. Host linear blocks only.
	SpecRollbackRows
	// SpecRollbackReplay keeps the state from before the verification and,
	// after a rejection, returns to it and runs the accepted rows again as the
	// head of the next round's verification: one copy a round, and a few more
	// rows in the next pass instead of a pass of their own.
	SpecRollbackReplay
)

// String is the rollback's name: "auto", "rows" or "replay".
func (r SpecRollback) String() string {
	switch r {
	case SpecRollbackRows:
		return "rows"
	case SpecRollbackReplay:
		return "replay"
	}
	return "auto"
}

type specOpts struct {
	draft    int
	minP     float64
	rollback SpecRollback
	// sched, when a gate sets it, is the draft count of round i at
	// sched[i%len(sched)], in place of choose's: rounds of no draft (and
	// their deferred catch-up) beside rounds that draft, whatever this host's
	// timings would choose.
	sched []int
	// drafter is who proposes (SpecDrafter); ngMin and ngMax the n-gram
	// lengths prompt lookup matches (WithSpecNGram).
	drafter      SpecDrafter
	ngMin, ngMax int
	// stop is WithSpecStop's set.
	stop func(int32) bool
}

// SpecOption configures a Speculator.
type SpecOption func(*specOpts)

// WithSpecDraft fixes how many tokens a round drafts. Zero, the default,
// chooses per round from what this session has measured: the acceptance at
// each draft position and the time a draft step and a verification take,
// maximising tokens per second (Speculator.choose).
func WithSpecDraft(k int) SpecOption { return func(o *specOpts) { o.draft = max(k, 0) } }

// WithSpecMinP stops a round's drafting at the first draft the prediction
// block gives less than p of its probability (Strata's spec_min_p): a draft it
// is unsure of is likely rejected, and the rows it would add to the
// verification cost more than they return. Zero, the default, drafts every
// round to the chosen count.
func WithSpecMinP(p float64) SpecOption { return func(o *specOpts) { o.minP = p } }

// WithSpecRollback chooses how a hybrid's recurrent state is taken back; see
// SpecRollback.
func WithSpecRollback(r SpecRollback) SpecOption { return func(o *specOpts) { o.rollback = r } }

// SpecDrafter is who proposes a round's drafts.
type SpecDrafter int

const (
	// SpecDraftAuto drafts with the model's prediction block where it carries
	// one and by prompt lookup where it does not.
	SpecDraftAuto SpecDrafter = iota
	// SpecDraftMTP drafts with the prediction block (jlm.Config.NMTP); a
	// model without one is refused.
	SpecDraftMTP
	// SpecDraftLookup drafts by prompt lookup: the tokens that followed the
	// latest earlier occurrence of the sequence's last n-gram (speclookup.go).
	// No model runs to draft, so it speculates on every architecture.
	SpecDraftLookup
)

// String is the drafter's name: "auto", "mtp" or "lookup".
func (d SpecDrafter) String() string {
	switch d {
	case SpecDraftMTP:
		return "mtp"
	case SpecDraftLookup:
		return "lookup"
	}
	return "auto"
}

// WithSpecDrafter chooses who drafts; see SpecDrafter.
func WithSpecDrafter(d SpecDrafter) SpecOption { return func(o *specOpts) { o.drafter = d } }

// WithSpecNGram sets the n-gram lengths prompt lookup matches the sequence's
// tail at, longest first; the defaults are specNGramMin and specNGramMax.
func WithSpecNGram(lo, hi int) SpecOption {
	return func(o *specOpts) { o.ngMin, o.ngMax = max(lo, 1), max(hi, lo, 1) }
}

// WithSpecStop names the tokens that end a generation (an end-of-generation
// id): a draft in the set is never accepted, so a round that reaches one
// returns it last, from the trunk's own logits, and the trunk has not run it
// -- where plain decode would stand after sampling it.
func WithSpecStop(stop func(int32) bool) SpecOption { return func(o *specOpts) { o.stop = stop } }

// SpecStats is what a Speculator has done.
type SpecStats struct {
	// Rounds is the verification passes; Drafted and Accepted the tokens the
	// prediction block proposed and the trunk confirmed; Emitted every token
	// generated after the prompt's first (Accepted plus one a round).
	Rounds, Drafted, Accepted, Emitted int64
	// Replayed is the rows run a second time to rebuild a recurrent state
	// (SpecRollbackReplay); Restored the rejections a per-row copy undid;
	// Commits the replays that ran as a pass of their own because the pass
	// carrying them rejected too.
	Replayed, Restored, Commits int64
	// The wall time in each phase.
	Draft, Verify, CatchUp time.Duration
	// Retired is a session the adaptive draft count stopped speculating in:
	// no count could beat decode with every draft accepted (choose).
	Retired bool
	// Rollback is the mode the session runs, Auto resolved.
	Rollback SpecRollback
	// Drafter is who drafts, Auto resolved.
	Drafter SpecDrafter
}

// TokensPerRound is how many tokens a verification pass yields.
func (s SpecStats) TokensPerRound() float64 {
	if s.Rounds == 0 {
		return 0
	}
	return float64(s.Emitted) / float64(s.Rounds)
}

// Speculator drafts with a model's prediction block and verifies with the
// trunk. It drives a single-sequence State it does not own, and a draft State
// over the prediction block that it does.
type Speculator struct {
	t, d *State
	opt  specOpts
	mode SpecRollback
	// mtp is the prediction block's own weights; mli its block index. Both
	// unset, and d nil, where prompt lookup drafts.
	mtp *mtpWeights
	mli int
	// hist is every decided token, the prompt's first, where prompt lookup
	// drafts: what its n-grams are matched against (speclookup.go).
	hist []int32

	// The round's starting point: replay is the decided tokens whose trunk
	// rows a replay rollback still owes (at positions bpos[0]..), next the
	// decided token after them, not yet fed, and h the trunk's hidden state at
	// the position before next.
	replay []int32
	next   int32
	h      []float32
	// dlog and dhid are the draft's logits and hidden state at next's row:
	// the proposal for the round's first draft.
	dlog, dhid []float32
	started    bool

	// Scratch, grown to the widest round.
	tout, dout rowsOut
	snaps      []recSnap
	pre        recSnap
	eh, ze     []float32
	rowTok     []int32
	dist, qd   specDist
	qdists     []specDist
	smark      samplerMark

	// What choose reads: per draft position, the drafts offered and accepted
	// there, and the times a draft step and a verification took.
	offered, accepted []int64
	tDraft, tCatch    time.Duration
	nDraft, nCatch    int64
	// vRecent and vN are the latest verification times (up to specWindow)
	// and the count of them, by rows verified (choose). A round is counted
	// only after specSettle rounds in a row verified as many rows (lastRows,
	// streak): the first round of a count compiles its kernels and records
	// its graph, and the rounds after a different count pay the switch -- on
	// a device the two decode rounds after a drafting round can cost several
	// times the third, decode's own, and counting the second would price
	// decode too high.
	// That is a cost of alternating, not of the count, and a session that
	// settles on a count does not pay it; so a re-check or a re-test runs
	// specSettle+1 rounds of its count (again). A count is priced at the
	// median of its latest rounds: not the mean, which one slow round holds
	// up, and not the fastest, which falls with the number of rounds and so
	// favours whichever count is chosen most.
	vRecent          [][]time.Duration
	vN               []int64
	lastRows, streak int
	// again is the count the next againLeft rounds draft: the rest of a
	// re-check or re-test, the last of them counted.
	again, againLeft int
	// spun is how many rounds in a row choose has chosen drafts, and checkAt
	// how many it waits before a round of none: decode's price is otherwise
	// measured only while it is chosen, and a session that drafts from its
	// first rounds would compare against the decode of its first rounds.
	spun, checkAt int

	// owedTok and owedHid are the catch-up rows rounds of no draft left the
	// prediction block (their tokens and the trunk's hidden states): a round
	// that drafts nothing needs no proposal, so its catch-up waits for the
	// next round that drafts, or for a chunk's worth of rows.
	owedTok []int32
	owedHid []float32
	// idle is how many rounds in a row choose has chosen no draft, and
	// probeAt how many it waits before drafting one anyway: a choice of
	// none stops the measurements that could overturn it, so it is tested
	// again after 1, 3, 7, 15 ... rounds -- doubling, a logarithmic cost
	// where speculation does not pay, and a session that one unlucky early
	// rejection talked out of it finds its way back.
	idle, probeAt, retests int
	// retired is a session where no draft count can beat decode even with
	// every draft accepted, on the costs measured (choose): the prediction
	// block stops running, its catch-up with it, and rounds are decode.
	retired bool

	// limit caps the tokens a round returns (Limit); zero is no cap.
	limit int

	stats SpecStats
	// fault is a gate's deliberate break of the contract (specFault); zero in
	// every session a caller makes.
	fault specFault
	// oracle, when a gate sets it, replaces the block's greedy proposals with
	// the token plain decode emits at that position (oracle[i] is the i-th
	// generated token), every corrupt-th one swapped for another: the
	// prediction block still runs every row, and the gate exercises
	// acceptance, partial acceptance and rejection at rates it chooses -- the
	// only way a fixture with random weights, whose block never agrees with
	// its trunk, reaches the accepting paths (Strata's spec_oracle and
	// spec_corrupt).
	oracle  []int32
	corrupt int
	drafted int64 // proposals made, which corrupt counts
	// qOracle is the sampled twin of oracle: the logits to draft from, given
	// the round's drafts so far -- a gate passes the trunk's own, so a draft
	// is accepted with probability one and the accepting path runs (every
	// corrupt-th draft keeps the block's logits). nil in every caller's
	// session.
	qOracle func(drafts []int32) []float32
}

// specFault names the three breaks the speculation gates must catch: each
// leaves the output fluent, and only a comparison against plain decode sees
// it (TestSpecGateDiscriminates).
type specFault int

const (
	specFaultNone specFault = iota
	// faultAcceptUnchecked takes one draft past the first rejection.
	faultAcceptUnchecked
	// faultKeepRejectedKV leaves the rejected rows in the trunk's history.
	faultKeepRejectedKV
	// faultKeepRejectedRec leaves the recurrent state stepped over them.
	faultKeepRejectedRec
)

// Speculate makes a Speculator over this session. The session must be a
// single sequence with no history yet, and the model must carry a prediction
// block. A device the session is on is attached to the draft as well, so the
// prediction block runs where the trunk does when it fits.
func (s *State) Speculate(opts ...SpecOption) (*Speculator, error) {
	m, c := s.m, s.c
	var o specOpts
	for _, f := range opts {
		f(&o)
	}
	if o.drafter == SpecDraftAuto {
		o.drafter = SpecDraftMTP
		if c.NMTP == 0 {
			o.drafter = SpecDraftLookup
		}
	}
	mtp := o.drafter == SpecDraftMTP
	switch {
	case mtp && c.NMTP == 0:
		return nil, fmt.Errorf("model: %s carries no multi-token-prediction block to draft with", c.Arch)
	case c.DSV4():
		// Its prediction blocks are hyper-connected as its trunk is, and the
		// draft path runs an ordinary block; and its pooled history folds rows
		// in as they arrive, which State.rewind does not take back. Not
		// built, so refused by name, whoever drafts.
		return nil, fmt.Errorf("model: speculation on %s is not built", c.Arch)
	case c.Q4E():
		// Its prediction block is hyper-connected as its trunk is, which the
		// draft path's ordinary block does not run, and a rewind takes the
		// device's n-gram history back but not the host's predecessors.
		// Not built, so refused by name, whoever drafts.
		return nil, fmt.Errorf("model: speculation on %s is not built", c.Arch)
	case s.batched || s.nseq != 1:
		return nil, fmt.Errorf("model: speculation drives a single sequence, not a batch")
	case s.pos != 0:
		return nil, fmt.Errorf("model: speculation starts on an empty session")
	case s.lo != 0:
		return nil, fmt.Errorf("model: speculation drives a trunk session")
	}
	sp := &Speculator{t: s, opt: o}
	sp.stats.Drafter = o.drafter
	if mtp {
		sp.mli, sp.mtp = c.NLayer, m.layers[c.NLayer].mtp
		// The draft's positions run one past the trunk's: a round drafts only
		// where the trunk can verify, and the round's last token -- decided,
		// not yet fed -- has its draft row at the position the trunk will feed
		// it at, which on a full context is one past the trunk's last.
		sp.d = m.newStateRange(1, s.maxSeq+1, c.NLayer, c.NLayer+1)
		sp.d.relocate = false
		if s.device != nil {
			if err := sp.d.SetDevice(s.device); err != nil {
				sp.d.Close()
				return nil, err
			}
		}
	}
	sp.mode = sp.opt.rollback
	if sp.mode == SpecRollbackAuto {
		sp.mode = SpecRollbackRows
		if s.devLinear() {
			sp.mode = SpecRollbackReplay
		}
	}
	if sp.mode == SpecRollbackRows && s.devLinear() {
		sp.Close()
		return nil, fmt.Errorf("model: per-row recurrent snapshots need every linear block on the host; " +
			"a device keeps its own state and rewinds it one step (SpecRollbackReplay)")
	}
	sp.stats.Rollback = sp.mode
	sp.h = make([]float32, c.NEmbd)
	sp.dlog = make([]float32, c.NVocab)
	sp.dhid = make([]float32, c.NEmbd)
	if s.recurrent() {
		sp.pre = s.newRecSnap()
	}
	if !mtp {
		sp.lookupInit()
	}
	return sp, nil
}

// Limit caps how many tokens each later round returns, at least one: a
// caller that wants n more tokens says so, and the trunk runs no row past
// them. Zero lifts the cap.
func (sp *Speculator) Limit(n int) { sp.limit = max(n, 0) }

// Draft is the draft session, for its placement and counters.
// It is nil where prompt lookup drafts.
func (sp *Speculator) Draft() *State { return sp.d }

// Stats is what this Speculator has done so far.
func (sp *Speculator) Stats() SpecStats { return sp.stats }

// Close releases the draft session. The trunk session is the caller's.
func (sp *Speculator) Close() error {
	if sp.d == nil {
		return nil
	}
	return sp.d.Close()
}

// Start runs the prompt through the trunk and the prediction block and returns
// the first generated token, chosen by sm (nil is greedy). It observes the
// token into sm, as Next does every token it returns.
func (sp *Speculator) Start(prompt []int32, sm *Sampler) (int32, error) {
	t, c := sp.t, sp.t.m.Cfg
	if sp.started {
		return 0, fmt.Errorf("model: a Speculator starts once")
	}
	if len(prompt) == 0 {
		return 0, errEmptyPrompt{}
	}
	defer t.m.enterPager()()
	n := len(prompt)
	logits, hid, err := sp.trunkPrompt(prompt)
	if err != nil {
		return 0, err
	}
	y := sm.Sample(logits)
	sm.Observe(y)
	sp.stats.Emitted++
	if sp.d == nil {
		sp.hist = append(append(sp.hist[:0], prompt...), y)
		sp.next = y
		copy(sp.h, hid[(n-1)*c.NEmbd:])
		sp.started = true
		return y, nil
	}
	// The draft over the prompt and the first token: row q is (x_q, h_{q-1}),
	// and the last row, (y, h_{n-1}), proposes the first draft.
	toks := append(append([]int32(nil), prompt...), y)
	hs := make([]float32, (n+1)*c.NEmbd)
	copy(hs[c.NEmbd:], hid)
	sp.dout = rowsOut{from: n, logits: sp.dlog, hidden: sp.dhid, prompt: true}
	if err := sp.draftRows(toks, hs, &sp.dout); err != nil {
		return 0, err
	}
	sp.next = y
	copy(sp.h, hid[(n-1)*c.NEmbd:])
	sp.started = true
	return y, nil
}

// trunkPrompt runs the prompt through the trunk and returns the last
// position's logits and the trunk's hidden state -- the output norm's result --
// at every position: the prompt is the draft's history too, and each of its
// rows reads the trunk's state at the row before.
func (sp *Speculator) trunkPrompt(prompt []int32) ([]float32, []float32, error) {
	t, c := sp.t, sp.t.m.Cfg
	n := len(prompt)
	t.kv.note(t.seqPos(0), prompt...)
	hid := make([]float32, n*c.NEmbd)
	logits := make([]float32, c.NVocab)
	if n == 1 {
		if err := t.stepRows(embedSrc(t, prompt), &rowsOut{logits: logits, hidden: hid}); err != nil {
			return nil, nil, err
		}
		return logits, hid, nil
	}
	// The residual rows of every chunk, normed here: the folded head would
	// keep the last chunk's on the device, so the tail projects instead.
	resid := make([]float32, 0, n*c.NEmbd)
	t.rowTap = func(_ int, rows []float32) { resid = append(resid, rows...) }
	t.noHeadFold = true
	lg, err := t.prefillSrc(0, embedSrc(t, prompt), prompt, nil)
	t.rowTap, t.noHeadFold = nil, false
	if err != nil {
		return nil, nil, err
	}
	copy(logits, lg)
	if len(resid) != len(hid) {
		return nil, nil, fmt.Errorf("model: the prompt's %d rows gave %d hidden values", n, len(resid))
	}
	t.batchNormB(hid, resid, t.outNorm, t.m.outNormB, c.NEmbd, n)
	return logits, hid, nil
}

// embedSrc is a prompt's rows as the trunk's prefill reads them.
func embedSrc(s *State, ids []int32) []func(dst []float32) error {
	src := make([]func(dst []float32) error, len(ids))
	for i, id := range ids {
		id := id
		src[i] = func(dst []float32) error { return s.embedInto(dst, id) }
	}
	return src
}

// draftRows runs rows of the draft at its next positions -- toks[i] with the
// hidden state hs[i], row-major -- into out: the rows from out.from on, which
// is the last alone for every step but a gate's.
func (sp *Speculator) draftRows(toks []int32, hs []float32, out *rowsOut) error {
	d, c, w := sp.d, sp.d.m.Cfg, sp.mtp
	n := len(toks)
	// The block's page: its eh_proj is read on the host even where the block
	// itself runs on a device, and the page stays (offerRange keeps it).
	if err := d.m.pageIn(sp.mli); err != nil {
		return err
	}
	// [enorm(embed(x)) ; hnorm(h)] per row, then eh_proj over every row at
	// once: one read of the matrix for the whole step.
	sp.eh = grow32(sp.eh, n*2*c.NEmbd)
	sp.ze = grow32(sp.ze, n*c.NEmbd)
	e := sp.ze // the embeddings, before eh_proj overwrites them
	for i, x := range toks {
		if err := d.embedMTP(e[i*c.NEmbd:(i+1)*c.NEmbd], x); err != nil {
			return err
		}
	}
	for i := 0; i < n; i++ {
		row := sp.eh[i*2*c.NEmbd : (i+1)*2*c.NEmbd]
		d.rmsnorm(row[:c.NEmbd], e[i*c.NEmbd:(i+1)*c.NEmbd], w.enorm, c.RMSEps)
		d.rmsnorm(row[c.NEmbd:], hs[i*c.NEmbd:(i+1)*c.NEmbd], w.hnorm, c.RMSEps)
	}
	d.jit.NewInput()
	if err := d.mm(sp.ze, w.ehProj, sp.eh, n); err != nil {
		return err
	}
	src := make([]func(dst []float32) error, n)
	for i := range src {
		z := sp.ze[i*c.NEmbd : (i+1)*c.NEmbd]
		src[i] = func(dst []float32) error { copy(dst, z); return nil }
	}
	return d.stepRows(src, out)
}

// embedMTP is token x's row of the table the prediction block reads: its own
// where it carries one, the trunk's otherwise. llama.cpp's graph_mtp applies
// no embedding scale here, and neither does this.
func (s *State) embedMTP(dst []float32, x int32) error {
	e := &s.m.layers[s.lo].mtp.embd
	if int(x) < 0 || int(x) >= e.rows {
		return errToken{x, s.c.NVocab}
	}
	if e.packed != nil && len(e.packed.QS) != 0 {
		nn.RowPacked32JIT(dst, e.typ, e.packed, int(x), e.rows, e.k)
		return nil
	}
	nn.Row32JIT(dst, e.typ, e.data, int(x), e.k)
	return nil
}

// Next runs one round and returns the tokens it decided, in order: the
// drafts the trunk accepted and one token from the trunk's own logits. sm
// chooses as Sample would (nil is greedy) and observes every returned token.
func (sp *Speculator) Next(sm *Sampler) ([]int32, error) {
	if !sp.started {
		return nil, fmt.Errorf("model: Next before Start")
	}
	t, d, c := sp.t, sp.d, sp.t.m.Cfg
	greedy := sm == nil || sm.Temp <= 0
	if sp.retired && len(sp.replay) == 0 {
		return sp.decode(sm, greedy)
	}
	defer t.m.enterPager()()
	P := t.bpos[0]
	r := len(sp.replay)
	// The verification writes P..P+r+k: it must fit the trunk's context, and
	// the draft writes up to P+r+k-1.
	k := 0
	switch {
	case sp.d == nil:
		k = min(sp.lookupCount(greedy), t.maxSeq-1-(P+r))
	case !sp.retired:
		k = min(sp.choose(), t.maxSeq-1-(P+r))
	}
	if sp.limit > 0 {
		k = min(k, sp.limit-1)
	}
	if k < 0 {
		return nil, errFull{t.maxSeq, t.reqSeq}
	}
	if k > 0 && len(sp.owedTok) > 0 {
		if err := sp.catchUp(true); err != nil {
			return nil, err
		}
	}

	// Draft. The first proposal is the catch-up's last row; each later one
	// runs the block on the previous draft and the block's own hidden state.
	t0 := time.Now()
	drafts := sp.rowTok[:0]
	// The sampler's history already holds next (every returned token is
	// observed); the drafts are observed past it for the round and taken back.
	if !greedy {
		sm.mark(&sp.smark)
		if len(sp.qdists) < k {
			sp.qdists = append(sp.qdists, make([]specDist, k-len(sp.qdists))...)
		}
	}
	if sp.d == nil {
		drafts = sp.propose(drafts, max(k, 0))
		k = 0 // the loop below is the prediction block's
	}
	for i := 0; i < k; i++ {
		if i > 0 {
			sp.dout = rowsOut{from: 0, logits: sp.dlog, hidden: sp.dhid}
			if err := sp.draftRows([]int32{drafts[i-1]}, sp.dhid, &sp.dout); err != nil {
				return nil, err
			}
		}
		var x int32
		var px float64
		if greedy {
			x = Greedy(sp.dlog)
			if sp.opt.minP > 0 {
				px = float64(topProb(sp.dlog, &sp.dist))
			}
			x = sp.oracleAt(int(sp.stats.Emitted)+i, x)
		} else {
			dl := sp.dlog
			if sp.qOracle != nil {
				if sp.drafted++; sp.corrupt == 0 || sp.drafted%int64(sp.corrupt) != 0 {
					dl = sp.qOracle(drafts)
				}
			}
			q := &sp.qdists[i]
			sm.dist(dl, q)
			x = sm.Sample(dl)
			px = float64(q.prob(x))
		}
		if sp.opt.minP > 0 && px < sp.opt.minP {
			break
		}
		drafts = append(drafts, x)
		if !greedy {
			sm.Observe(x)
		}
	}
	sp.rowTok = drafts
	kd := len(drafts)
	sp.stats.Drafted += int64(kd)
	if !greedy {
		sm.restore(&sp.smark)
	}
	if k > 1 {
		sp.tDraft += time.Since(t0)
		sp.nDraft += int64(k - 1)
	}
	sp.stats.Draft += time.Since(t0)

	// Verify: the owed replay rows, next, and the drafts, in one pass. A
	// rollback needs the state from before it (replay) or after each row
	// (rows).
	t1 := time.Now()
	rows := make([]int32, 0, r+1+kd)
	rows = append(append(append(rows, sp.replay...), sp.next), drafts...)
	nr := len(rows)
	// The rows wanted are next's and the drafts', behind the replayed ones.
	from := r
	sp.tout.from = from
	sp.tout.logits = grow32(sp.tout.logits, (nr-from)*c.NVocab)
	sp.tout.hidden = grow32(sp.tout.hidden, (nr-from)*c.NEmbd)
	sp.tout.snap = nil
	sp.tout.argmax = greedy
	rr, devRec := t.ld.(nn.RecRewinder)
	devRec = devRec && t.devLinear()
	if t.recurrent() && kd > 0 {
		switch sp.mode {
		case SpecRollbackRows:
			for len(sp.snaps) < nr-1 {
				sp.snaps = append(sp.snaps, t.newRecSnap())
			}
			sp.tout.snap = sp.snaps[:nr-1]
		case SpecRollbackReplay:
			t.saveRec(&sp.pre)
			if devRec {
				rr.RecMark()
			}
		}
	}
	t.kv.note(P, rows...)
	if err := t.stepRows(embedSrc(t, rows), &sp.tout); err != nil {
		return nil, err
	}
	sp.stats.Verify += time.Since(t1)
	sp.stats.Rounds++

	// Accept. Row j of the outputs is the trunk at next's position + j: its
	// logits judge draft j+1.
	lg := func(j int) []float32 { return sp.tout.logits[j*c.NVocab : (j+1)*c.NVocab] }
	a := 0
	var y int32
	if greedy {
		// The device's per-row argmax where the step read back no logits.
		tok := func(j int) int32 {
			if sp.tout.tokens != nil {
				return sp.tout.tokens[j]
			}
			return Greedy(lg(j))
		}
		for a < kd && tok(a) == drafts[a] && (sp.opt.stop == nil || !sp.opt.stop(drafts[a])) {
			a++
		}
		if sp.fault == faultAcceptUnchecked && a < kd {
			a++
		}
		if y = sp.tout.token; y < 0 {
			y = tok(a)
		}
	} else {
		sm.mark(&sp.smark)
		for a < kd {
			sm.dist(lg(a), &sp.dist)
			px, qx := sp.dist.prob(drafts[a]), sp.qdists[a].prob(drafts[a])
			if sp.fault == faultAcceptUnchecked || qx > 0 && sm.uniform() < float64(px)/float64(qx) {
				sm.Observe(drafts[a])
				a++
				continue
			}
			break
		}
		if a < kd {
			y = sp.residual(sm, lg(a), a, drafts[a])
		} else {
			y = sm.Sample(lg(a))
		}
		sm.restore(&sp.smark)
	}
	sp.offer(kd, a)
	sp.stats.Accepted += int64(a)

	// Rollback: the trunk keeps the rows through the last accepted draft.
	keep := P + r + 1 + a
	nextReplay := sp.replay[:0]
	if a < kd && sp.fault == faultKeepRejectedKV {
		// The rejected rows stay in the history: positions move on as if
		// every row were kept, and the next round attends over them.
	} else if a < kd {
		switch {
		case !t.recurrent():
			t.rewind(keep)
		case sp.fault == faultKeepRejectedRec:
			t.rewind(keep)
		case sp.mode == SpecRollbackRows:
			t.loadRec(&sp.snaps[r+a])
			t.rewind(keep)
			sp.stats.Restored++
		default:
			// Back to the state before this pass, and the accepted rows go
			// again at the head of the next one.
			t.loadRec(&sp.pre)
			if devRec && !rr.RecRewind() {
				why := "no reason given"
				if e, ok := t.ld.(nn.ErrReporter); ok {
					why = e.Err()
				}
				return nil, fmt.Errorf("model: the device could not rewind its recurrent state: %s", why)
			}
			t.rewind(P)
			nextReplay = rows[:r+1+a]
			sp.stats.Replayed += int64(len(nextReplay))
			// A pass that carried owed rows and rejected again owes them all
			// once more, and deferring again would let the debt grow with
			// every rejection. They are run now instead, as one chunk with no
			// head: one more pass, and the next round starts clean.
			if r > 0 {
				commit := rowsOut{noHead: true}
				if err := t.stepRows(embedSrc(t, nextReplay), &commit); err != nil {
					return nil, err
				}
				nextReplay = nextReplay[:0]
				sp.stats.Commits++
			}
		}
	}
	// What this round's verification of nr rows cost, the commit pass
	// included: what choose prices a round of that many rows at.
	for len(sp.vN) <= nr {
		sp.vRecent, sp.vN = append(sp.vRecent, nil), append(sp.vN, 0)
	}
	if nr != sp.lastRows {
		sp.lastRows, sp.streak = nr, 0
	} else if sp.streak++; sp.streak >= specSettle {
		d := time.Since(t1)
		if len(sp.vRecent[nr]) == specWindow {
			sp.vRecent[nr] = append(sp.vRecent[nr][:0], sp.vRecent[nr][1:]...)
		}
		sp.vRecent[nr] = append(sp.vRecent[nr], d)
		sp.vN[nr]++
	}

	// Catch-up: the draft rewinds to next's row + 1 (behind any rows still
	// owed) and runs the accepted drafts and y against the trunk's own hidden
	// states, its last row proposing the next round's first draft. A round
	// that drafted nothing leaves them owed (Speculator.owedTok).
	hid := func(j int) []float32 { return sp.tout.hidden[j*c.NEmbd : (j+1)*c.NEmbd] }
	if sp.d != nil && !sp.retired {
		back := P + r + 1 - len(sp.owedTok)
		if sp.fault == faultKeepRejectedKV {
			back = min(back, d.bpos[0]) // the trunk ran ahead of what it emitted
		}
		d.rewind(back)
		sp.owedTok = append(append(sp.owedTok, drafts[:a]...), y)
		for j := 0; j <= a; j++ {
			sp.owedHid = append(sp.owedHid, hid(j)...)
		}
	}
	copy(sp.h, hid(a))
	out := append(append([]int32(nil), drafts[:a]...), y)
	sp.replay = append(nextReplay[:0:0], nextReplay...)
	sp.next = y
	if sp.d == nil {
		sp.hist = append(sp.hist, out...)
	} else if !sp.retired && (k > 0 || len(sp.owedTok) >= d.chunkWidth()) {
		if err := sp.catchUp(k == 0); err != nil {
			return nil, err
		}
	}
	for _, x := range out {
		sm.Observe(x)
	}
	sp.stats.Emitted += int64(len(out))
	return out, nil
}

// decode is a retired session's round: plain decode of next, as
// ForwardGreedy (the device's argmax where the head is placed) or Forward and
// the sampler, with no hidden state read back and no prediction block run.
func (sp *Speculator) decode(sm *Sampler, greedy bool) ([]int32, error) {
	t0 := time.Now()
	var y int32
	if greedy {
		var err error
		if y, err = sp.t.ForwardGreedy(sp.next); err != nil {
			return nil, err
		}
	} else {
		lg, err := sp.t.Forward(sp.next)
		if err != nil {
			return nil, err
		}
		y = sm.Sample(lg)
	}
	sm.Observe(y)
	sp.next = y
	sp.stats.Rounds++
	sp.stats.Emitted++
	sp.stats.Verify += time.Since(t0)
	return []int32{y}, nil
}

// catchUp runs the owed rows through the prediction block, the last row's
// proposal in sp.dlog. A round's own catch-up is one step. The debt of rounds
// that drafted nothing (debt) has a length that is new nearly every time, and
// where the block runs ragged rows on a device a new row count compiles and
// records a step of its own -- flushing it in one step
// costs more than the decode it deferred, and as a prompt chunk the block's
// batch scratch was rebuilt beside the trunk's. So the debt goes in pieces no
// longer than the longest step this session already verifies as rows. Only a
// round's own catch-up is what choose prices (tCatch): the debt is the price
// of rounds that drafted nothing, paid a chunk at a time.
func (sp *Speculator) catchUp(debt bool) error {
	t0 := time.Now()
	n, w := len(sp.owedTok), sp.t.m.Cfg.NEmbd
	piece := n
	if debt && sp.d.rowsVerifiable() {
		piece = max(2, len(sp.vN)-1)
	}
	var err error
	for i := 0; i < n && err == nil; i += piece {
		j := min(i+piece, n)
		// Only the last piece's last row proposes; the others write KV.
		sp.dout = rowsOut{from: j - i - 1, logits: sp.dlog, hidden: sp.dhid}
		if j < n && j-i > 1 {
			sp.dout = rowsOut{noHead: true}
		}
		err = sp.draftRows(sp.owedTok[i:j], sp.owedHid[i*w:j*w], &sp.dout)
	}
	sp.owedTok, sp.owedHid = sp.owedTok[:0], sp.owedHid[:0]
	sp.stats.CatchUp += time.Since(t0)
	if !debt {
		sp.tCatch += time.Since(t0)
		sp.nCatch++
	}
	return err
}

// residual draws the corrected token after the trunk rejected draft i, x, at
// a row with logits lg: from max(0, p - q) normalised, p the trunk's
// distribution at the row (sp.dist) and q the draft's (qdists[i]). It draws z
// from p and keeps it with probability max(0, 1 - q(z)/p(z)), which is that
// distribution exactly and needs no vocabulary-sized arithmetic beyond what
// Sample runs.
//
// The rejection says the residual's mass is at least q(x) - p(x) > 0, so the
// expected number of draws is at most 1/(q(x) - p(x)). After enough draws that
// a residual of that mass is missed with probability below double precision's
// epsilon (2^-53), the last draw is taken as it stands.
func (sp *Speculator) residual(sm *Sampler, lg []float32, i int, x int32) int32 {
	q := &sp.qdists[i]
	mass := float64(q.prob(x)) - float64(sp.dist.prob(x))
	limit := 1
	if mass > 0 {
		limit = int(math.Ceil(53 * math.Ln2 / mass))
	}
	var z int32
	for n := 0; n < limit; n++ {
		z = sm.Sample(lg)
		pz, qz := float64(sp.dist.prob(z)), float64(q.prob(z))
		if pz > 0 && sm.uniform() < 1-qz/pz {
			return z
		}
	}
	return z
}

// oracleAt is a gate's proposal for generated token g (see Speculator.oracle),
// or x where no gate set one.
func (sp *Speculator) oracleAt(g int, x int32) int32 {
	if sp.oracle == nil || g >= len(sp.oracle) {
		return x
	}
	sp.drafted++
	if sp.corrupt > 0 && sp.drafted%int64(sp.corrupt) == 0 {
		return (sp.oracle[g] + 1) % int32(sp.t.m.Cfg.NVocab)
	}
	return sp.oracle[g]
}

// offer records a round's drafts: kd offered, the first a accepted.
func (sp *Speculator) offer(kd, a int) {
	for len(sp.offered) < kd {
		sp.offered, sp.accepted = append(sp.offered, 0), append(sp.accepted, 0)
	}
	for i := 0; i < kd; i++ {
		sp.offered[i]++
		if i < a {
			sp.accepted[i]++
		}
		if i >= a {
			break // a draft after the first rejection was never judged
		}
	}
}

// choose is how many tokens this round drafts: the fixed count when one was
// asked for (WithSpecDraft), and otherwise the count that maximises tokens
// per second on what this session has measured.
//
// A round of k drafts yields 1 + sum_{i<=k} prod_{j<=i} alpha_j tokens, alpha_j
// the acceptance at draft position j given position j-1 was accepted (counted
// in offer, with Laplace's rule so an untried position is neither certain nor
// hopeless), and costs a verification of r+1+k rows (r the replayed ones), k-1
// draft steps (the first draft is the previous round's catch-up) and a
// catch-up -- except at k = 0, whose catch-up waits (owedTok) and runs a chunk
// at a time, so a session where speculation does not pay decodes at nearly
// decode's rate.
//
// The verification is priced per row count, because what an extra row costs
// is the device's and the model's: nearly nothing where a pass is bound by
// its weights, as much as a pass where it is bound by launches or by a
// recurrence stepped row by row. Nor is it reliably monotonic in the
// rows. So a count not measured is
// interpolated between measured ones where it has both, and past the deepest
// measured is priced as that one -- optimistic, so it gets tried -- never
// extrapolated from the counts below it, which can price a winning count
// out of reach. The first rounds measure one
// row and two (the first specSettle rounds of each are not counted), so the
// search always has plain decode to beat. A position not yet tried borrows the
// last tried one's alpha; the search runs one past the deepest tried and no
// further than specMaxDraft, so a model that keeps accepting is let draft
// deeper a round at a time and one that stops is not. A choice of no draft
// is re-tested on a doubling schedule (Speculator.probeAt).
func (sp *Speculator) choose() int {
	if s := sp.opt.sched; s != nil {
		return s[int(sp.stats.Rounds)%len(s)]
	}
	if sp.opt.draft > 0 {
		return sp.opt.draft
	}
	if sp.againLeft > 0 {
		sp.againLeft--
		return sp.again
	}
	settled := func(k int) int {
		sp.again, sp.againLeft = k, specSettle
		return k
	}
	measured := func(rows int) bool { return rows < len(sp.vN) && sp.vN[rows] > 0 }
	base := len(sp.replay) + 1
	switch {
	case !measured(1) || base > 1 && !measured(2):
		return 0 // a round with replayed rows drafts nothing until rows of one are priced
	case !measured(2):
		return 1
	}
	priced := func(j int) float64 {
		v := slices.Clone(sp.vRecent[j])
		slices.Sort(v)
		return float64(v[len(v)/2])
	}
	tv := func(rows int) float64 {
		lo, hi := -1, -1 // the measured counts nearest rows from below and above
		for j := range sp.vN {
			if sp.vN[j] == 0 {
				continue
			}
			if j <= rows {
				lo = j
			} else if hi < 0 {
				hi = j
			}
		}
		switch {
		case lo == rows:
			return priced(rows)
		case lo >= 0 && hi >= 0:
			return priced(lo) + (priced(hi)-priced(lo))*float64(rows-lo)/float64(hi-lo)
		case hi >= 0:
			return priced(hi)
		}
		return priced(lo) // past the deepest measured: optimistic, so it is tried
	}
	deepest := min(len(sp.offered)+1, specMaxDraft)
	// A draft step is one row of one block; until one is measured it is
	// priced as a catch-up, which is one draft step of a few rows.
	tc := float64(sp.tCatch) / float64(sp.nCatch)
	td := tc
	if sp.nDraft > 0 {
		td = float64(sp.tDraft) / float64(sp.nDraft)
	}
	alpha := func(j int) float64 {
		if j >= len(sp.offered) {
			j = len(sp.offered) - 1
		}
		if j < 0 {
			return 0.5
		}
		return float64(sp.accepted[j]+1) / float64(sp.offered[j]+2)
	}
	best, bestK := 1/tv(base), 0
	yield, run := 1.0, 1.0
	for k := 1; k <= deepest; k++ {
		run *= alpha(k - 1)
		yield += run
		if r := yield / (tv(base+k) + float64(k-1)*td + tc); r > best {
			best, bestK = r, k
		}
	}
	if bestK > 0 {
		sp.idle = 0
		if sp.spun++; sp.spun > sp.checkAt {
			sp.spun, sp.checkAt = 0, 2*sp.checkAt+1
			return settled(0)
		}
		return bestK
	}
	sp.spun = 0
	// Acceptance is the one thing a later round can change; the costs are
	// the device's. kOpt is the count that would win with every draft
	// accepted, if any does, searched to specMaxDraft rather than one past the
	// deepest tried: what a re-test drafts, so a re-test measures the count
	// most likely to win rather than only the cheapest. If none does
	// after retireAfter re-tests -- each a counted round, so a price is a
	// median of several -- none ever will here, and keeping the
	// block in step costs a draft row a token.
	kOpt, opt := 0, best
	for k := 1; k <= specMaxDraft; k++ {
		if r := float64(k+1) / (tv(base+k) + float64(k-1)*td + tc); r > opt {
			kOpt, opt = k, r
		}
	}
	if base == 1 && kOpt == 0 && sp.retests >= retireAfter {
		sp.retired, sp.stats.Retired = true, true
		sp.owedTok, sp.owedHid = nil, nil
		return 0
	}
	if sp.idle++; sp.idle > sp.probeAt {
		sp.idle, sp.probeAt = 0, 2*sp.probeAt+1
		sp.retests++
		return settled(max(kOpt, 1))
	}
	return 0
}

// retireAfter is how many re-tests (Speculator.probeAt) a session that no
// count can win in takes before it stops speculating: three counted rounds
// of the counts that matter, so one slow round on a busy host does not
// decide it.
const retireAfter = 3

// specMaxDraft is the deepest the adaptive search drafts: it bounds what the
// optimistic price of an unmeasured count explores, every count up to it
// tried a few rounds before the search trusts its prices. Three drafts won
// where speculation paid; one more is explored past it. WithSpecDraft fixes any depth.
const specMaxDraft = 4

// specSettle is how many rounds in a row of one count come before the first
// that prices it: the switch from another count costs the two after it
// (Speculator.vRecent).
const specSettle = 2

// specWindow is how many of a count's latest verification times its price
// is the median of: enough that a slow round or two does not move it, few
// enough to follow a history that grows (attention lengthens with it).
const specWindow = 8

// topProb is the largest probability of a softmax over logits, the draft's
// confidence in its argmax.
func topProb(logits []float32, d *specDist) float32 {
	d.pall = grow32(d.pall, len(logits))
	copy(d.pall, logits)
	nn.Softmax32JIT(d.pall, len(logits))
	return d.pall[Greedy(d.pall)]
}
