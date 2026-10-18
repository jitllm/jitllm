package server

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// stepLoop is continuous batching for one loaded model, on its devices or on
// the host.
//
// A generate on a session whose every block and head are on one device, or
// whose every block and head are on the host, does not take the gates for
// itself. It becomes a ROW of the model's step loop: one goroutine that owns
// every live row's model.State, and each step gathers each row's next token
// and runs them through model.StepRuns -- one device step for all of them
// (nn.SessionStepper), or one host pass over the weights (State.HostSteppable),
// every block's weights read once, each row's history its own session's. Each
// row then
// samples its own logits with its own sampler, exactly as the one-at-a-time
// path does, and its events go to an outbox its request goroutine drains, so a
// slow client never holds up a step.
//
// The loop is recorded on the model's gates while it has rows. Nothing queues
// on a gate (gate.go), so the loop admits a waiting request as a row whenever
// it has room (Config.MaxBatchRows), and a session on the one-at-a-time path runs
// beside it, the two interleaving a step at a time rather than one waiting for
// the other to finish.
//
// A joint step is not always the faster: for each row count the loop times it
// against running the rows one session after another, in situ, and runs the
// faster (batchchoose.go; Config.JointSteps forces either).
//
// A new row's prompt is fed a chunk at a time inside the same steps
// (promptUnits): its chunk rides beside the decoding rows' tokens, so
// admission never stalls them by more than the rows it adds to a step. A
// prompt whose State pipelines its prefill across several cards takes that
// path instead, a pipelined chunk at a time, beside the joint step.
//
// The Go work of a step that does not feed it -- each sampled token's text,
// its stop-string check where the row has none, its event -- runs on a helper
// goroutine while the step runs (stepPost), so it leaves the critical path.
type stepLoop struct {
	e  *Engine
	lm *LoadedModel
	// id is what the loop is called in its gates' queues.
	id string

	width       int
	promptChunk int

	ctx  context.Context
	quit context.CancelFunc
	done chan struct{}
	// hold is read-held while the loop serves; a test write-holds it so the
	// requests it sends pile up and are admitted together.
	hold sync.RWMutex

	mu      sync.Mutex
	waiting []*row
	// rows is every admitted row, prompting or decoding, in admission order.
	// Only the loop goroutine changes it, under mu, so the loop reads it
	// without the lock and telemetry reads it with.
	rows    []*row
	stopped bool
	wake    chan struct{}

	// shape counts admissions and retirements: a joint step the device
	// refused is not retried until the rows it was refused for change.
	shape      uint64
	refusedFor uint64
	refused    bool

	// choice says, per row count, whether a decode step runs joint or one
	// session after another (batchchoose.go).
	choice *jointChoice

	// budget is how many prompt tokens a step carries beside decoding rows
	// (stepBudget).
	budget *stepBudget

	// post is the helper that turns the tokens a step feeds into text and
	// events while the step runs.
	post *stepPost
	// The lists an iteration reuses: the rows it walks (finish changes
	// lp.rows) and the units it steps.
	walk         []*row
	units, joint []unit
	solo         []unit
	runs         []model.Run
	// The prompting rows promptUnits walks, and their allotments.
	prompts []*row
	alloc   []int
	// sealing is the rows whose prompt a step just completed and whose
	// history the memory cache is yet to be offered (sealPrompts), with the
	// logits the prompt ended on.
	sealing []sealJob

	// chunkOf is how many prompt tokens a row's State takes in one prefill:
	// its pipelined chunk when the State pipelines (State.PromptChunk). A
	// gate replaces it to drive the pipelined arm without several cards.
	chunkOf func(*row) int

	// A gate's violations, false but in a test: feedNext feeds each decoding
	// row the token sampled for the row after it; dropMaxTokens never
	// retires a row for reaching its max_tokens; admitAll admits every
	// waiting request whatever the card has room for.
	feedNext, dropMaxTokens, admitAll bool
	// parkDropsLogits zeroes the logits a parked row holds, a violation of
	// the time slice's answer; onStarted, when set, sees each row as its
	// prompt is in.
	parkDropsLogits bool
	onStarted       func(*row)
	// beforeQueueWait, when set, sees each row after it is submitted and
	// before its request waits for admission.
	beforeQueueWait func(*row)
	// parkFault makes every park of a time slice fail, as a device that
	// cannot take a session's history home does.
	parkFault error

	// fair is the fairness level, 0 to 100 (batchfair.go); rr turns the
	// even split of the prompt budget so its remainder rotates.
	fair, rr int
	// The moving averages of a park's, a resume's and a step's time, in
	// nanoseconds, that bound the quantum from below (stepLoop.quantum).
	parkNs, resumeNs, stepNs float64

	stats batchCounters
}

// batchCounters is what BatchStats reports; see telemetry.proto.
type batchCounters struct {
	steps, rowsStepped, jointSteps, jointRows, soloRows atomic.Int64
	maxRows, maxSessions                                atomic.Int32
	promptChunks, promptSteps                           atomic.Int64
	admissions, admitWait                               atomic.Int64
	refusals, separateSteps                             atomic.Int64
	// kvWaits counts admissions put off because the card's room for history
	// was promised to the rows already admitted (stepLoop.kvRoom).
	kvWaits atomic.Int64
	// pipelinedChunks is prompt chunks that ran as a pipelined prefill
	// beside the step rather than as rows of it.
	pipelinedChunks atomic.Int64
	// parks and resumes count time slices: rows parked for a waiting
	// request, and parked rows brought back.
	parks, resumes atomic.Int64
	// parkFreed is the card's room for history, in positions, that parking
	// gave back.
	parkFreed atomic.Int64
	// shortResumes counts resumes the card took back only in part, parked
	// again to wait.
	shortResumes atomic.Int64
	// parkRefusals counts time-slice parks the session refused: the row
	// kept its place and its request ran on.
	parkRefusals atomic.Int64
	// prefixHolds counts rows held back from admission while an admitted row
	// computed the prefix they share (stepLoop.holdsForPrefix), and
	// prefixHoldTimeouts those released because the producer took too long.
	prefixHolds, prefixHoldTimeouts atomic.Int64
	// promptRowsFed counts, per step, the prompting rows it fed: a gate reads
	// it against the prompting rows the step had.
	promptRowsFed, promptRowsHad atomic.Int64
	// postedTokens is tokens whose text and event the helper produced while
	// their step ran.
	postedTokens atomic.Int64
	// maxPromptBeside is the most prompt tokens one step carried beside a
	// decoding row: the bound stepBudget keeps.
	maxPromptBeside atomic.Int64
	// The budget's trajectory over the steps beside decoding rows: how many,
	// its sum, least and most.
	budgetSteps, budgetSum, budgetMin, budgetMax atomic.Int64
	lastRefusal                                  atomic.Pointer[string]
}

// errUnloaded ends a request still waiting for a row when its model goes.
// errPromptCancelled ends a row whose request left before its prompt was in
// and whose sequence started with it: no GenerateStarted is sent, the State
// is reset, and the request returns its context's error, as a prefill
// interrupted alone does (Generate).
var errPromptCancelled = errors.New("server: the request left during its prompt")

var errUnloaded = fmt.Errorf("%w: the model was unloaded before this request got a row", ErrNotFound)

// batchWidth is the most requests the loop holds rows for: the configured
// width, bounded by the widest step a device runs across sessions
// (model.MaxStepRows rows, each decoding session taking one). The KV each row
// needs is not part of it: a session holds its own history from creation, and
// one the device has no room for is not wholly on the device and never
// becomes a row.
func batchWidth(cfg int) int {
	if cfg <= 0 || cfg > model.MaxStepRows {
		return model.MaxStepRows
	}
	return cfg
}

func newStepLoop(e *Engine, lm *LoadedModel) *stepLoop {
	ctx, quit := context.WithCancel(context.Background())
	lp := &stepLoop{
		e:           e,
		lm:          lm,
		id:          "batch:" + lm.id,
		width:       batchWidth(e.cfg.MaxBatchRows),
		promptChunk: e.cfg.PromptChunk,
		choice:      newJointChoice(e.cfg.JointSteps),
		budget:      newStepBudget(e.cfg.StepPromptTokens, e.cfg.PromptChunk, e.cfg.StepCost),
		fair:        fairLevel(e.cfg.Fairness),
		ctx:         ctx,
		quit:        quit,
		done:        make(chan struct{}),
		wake:        make(chan struct{}, 1),
	}
	lp.post = newStepPost(&lp.stats.postedTokens)
	lp.chunkOf = func(r *row) int { return r.s.st.PromptChunk() }
	go lp.run()
	return lp
}

// row is one generate inside the loop.
type row struct {
	s         *Session
	ids       []int32
	reset     bool
	echo      bool
	ephemeral bool
	// tools reads the row's tool calls (Started.Tools).
	tools     *model.ToolStream
	sampler   model.Sampler
	logprobs  *model.Logprobs // nil unless the request asked for logprobs
	maxTokens int
	ignoreEOS bool
	// stops says the row has stop strings: a token's text then decides
	// whether it is fed, so it is produced before the step, not beside it.
	stops    bool
	stream   *streamText
	enqueued time.Time
	depth    int32
	// restored is the prompt positions the memory cache gave the row before
	// it was queued (State.RestorePrefix); its prompt is fed from there. seal
	// has the row's prompt offered to the store once it has run.
	restored int
	seal     bool
	// heldSince is when the row was first held back for a producer of its
	// prefix (holdsForPrefix), and holdSpent says that wait ran out; the
	// loop's own.
	heldSince time.Time
	holdSpent bool
	// fresh says the row's sequence starts with its prompt (no
	// continue_session), so a prompt cut short is reset rather than kept.
	fresh bool
	// prio is the request's priority (1 for jitllm_priority "high"); since
	// is when it began waiting: its arrival, or when it was last parked.
	prio  int
	since time.Time

	// admitted closes when the loop takes the row; done when it lets go of
	// the row for good, after its last event is in the outbox.
	admitted  chan struct{}
	done      chan struct{}
	cancelled atomic.Bool

	// The loop goroutine's own half, written before admitted or done closes
	// and read by the request goroutine only after.
	waited      time.Duration
	fed         int
	begun       bool
	logits      []float32
	next        [1]int32
	out         []int32
	n           int
	promptStart time.Time
	prefill     time.Duration
	decodeStart time.Time
	result      rowResult
	// A time slice's state (batchfair.go): the tokens since the row was
	// admitted or resumed, r.n at its last resume, whether it is parked, and
	// the logits it holds while parked.
	slice, resumedAt int
	parked           bool
	held             []float32
	// startStep is the loop's step count when the row's prompt was in.
	startStep int64

	omu    sync.Mutex
	events []Event
	notify chan struct{}
}

type rowResult struct {
	reason FinishReason
	stop   string
	n      int
	decode time.Duration
	err    error
}

func (r *row) prompting() bool { return r.fed < len(r.ids) }

func (r *row) push(ev Event) {
	r.omu.Lock()
	r.events = append(r.events, ev)
	r.omu.Unlock()
	select {
	case r.notify <- struct{}{}:
	default:
	}
}

func (r *row) take() []Event {
	r.omu.Lock()
	defer r.omu.Unlock()
	evs := r.events
	r.events = nil
	return evs
}

func (r *row) isDone() bool {
	select {
	case <-r.done:
		return true
	default:
		return false
	}
}

// joins says whether a generate on s runs as a row of the model's step loop.
// The caller holds s.mu. A session that is not wholly on one device runs alone
// on its gates as before. One that is, and whose history then outgrows the
// card, is refused the joint step (the tier cannot append its page) and runs
// that token alone, where State.Forward hands a block to the host
// (SetRelocate); from then on it is stepped alone.
func (e *Engine) joins(s *Session) *stepLoop {
	lp := s.lm.loop
	if lp == nil || !stepsJointly(s.st) {
		return nil
	}
	return lp
}

// hasStop says whether stops holds a stop string that can match: an empty
// one never does (streamText.push skips it).
func hasStop(stops []string) bool {
	for _, x := range stops {
		if x != "" {
			return true
		}
	}
	return false
}

// stepsJointly says whether st can take a run of a joint step: wholly on one
// device that steps sessions together, or wholly on the host.
func stepsJointly(st *model.State) bool { return st.Steppable() || st.HostSteppable() }

// submit queues r for a row, recording in r.depth how many requests were
// ahead -- under the lock, since the loop may admit r and read it at once.
func (lp *stepLoop) submit(r *row) error {
	lp.mu.Lock()
	if lp.stopped {
		lp.mu.Unlock()
		return errUnloaded
	}
	lp.waiting = append(lp.waiting, r)
	r.depth = int32(len(lp.waiting) - 1)
	r.s.queuePos.Store(r.depth + 1)
	lp.mu.Unlock()
	select {
	case lp.wake <- struct{}{}:
	default:
	}
	return nil
}

// withdraw takes r back out of the queue, reporting false when the loop has
// already admitted it.
func (lp *stepLoop) withdraw(r *row) bool {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	for i, w := range lp.waiting {
		if w == r {
			lp.waiting = append(lp.waiting[:i], lp.waiting[i+1:]...)
			lp.renumber()
			return true
		}
	}
	return false
}

// renumber republishes every waiting session's queue position. lp.mu held.
func (lp *stepLoop) renumber() {
	for i, w := range lp.waiting {
		w.s.queuePos.Store(int32(i + 1))
	}
}

// close stops the loop. The model's sessions are closed first (UnloadModel,
// Engine.Close), so no row is live; a request still waiting for one ends with
// errUnloaded.
func (lp *stepLoop) close() {
	lp.mu.Lock()
	lp.stopped = true
	waiting := lp.waiting
	lp.waiting = nil
	lp.mu.Unlock()
	lp.quit()
	<-lp.done
	lp.post.close()
	for _, r := range waiting {
		r.result.err = errUnloaded
		close(r.done)
	}
}

func (lp *stepLoop) run() {
	defer close(lp.done)
	for lp.awaitWork() {
		gs := lp.e.gatesFor(lp.lm.gateIDs())
		lp.hold.RLock()
		gs.acquire(lp.id)
		lp.serve()
		gs.release()
		lp.hold.RUnlock()
	}
}

// awaitWork blocks until a request is waiting, reporting false when the loop
// is stopping.
func (lp *stepLoop) awaitWork() bool {
	for {
		lp.mu.Lock()
		n := len(lp.waiting)
		lp.mu.Unlock()
		if n > 0 {
			return true
		}
		select {
		case <-lp.wake:
		case <-lp.ctx.Done():
			return false
		}
	}
}

// serve runs steps until no row is left, admitting rows as they come.
func (lp *stepLoop) serve() {
	for {
		if lp.ctx.Err() != nil {
			for len(lp.rows) > 0 {
				lp.finish(lp.rows[0], FinishCancelled, "", nil)
			}
			return
		}
		lp.admit()
		if len(lp.rows) == 0 {
			return
		}
		lp.iterate()
	}
}

// admit moves waiting requests into rows while there is room: a free row,
// and on a device room on the card for the row's history (kvRoom).
func (lp *stepLoop) admit() {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	lp.dropParked()
	lp.order()
	room, page, promised := lp.kvRoom()
	blocked := func(need int) bool {
		return len(lp.rows) >= lp.width || page > 0 && len(lp.rows) > 0 && need > room && !lp.admitAll
	}
	if len(lp.waiting) > 0 {
		for len(lp.waiting) > 0 {
			// The first waiter that is not a follower held for a producer.
			wi := 0
			for wi < len(lp.waiting) && lp.holdsForPrefix(lp.waiting[wi]) {
				wi++
			}
			if wi == len(lp.waiting) {
				break
			}
			r := lp.waiting[wi]
			need := rowPages(r, 0, page) * page
			if blocked(need) {
				// A time slice: rows past their quantum make way (batchfair.go).
				vs := lp.victims(r, need, room, page)
				for _, v := range vs {
					if lp.parkRow(v) != nil {
						// The session kept its history where it was, so the
						// row keeps its place and runs another slice before
						// it is asked again; the request waits. A park is a
						// courtesy to the queue, never a reason to fail the
						// request it would have preempted.
						v.slice = 0
						lp.stats.parkRefusals.Add(1)
					}
				}
				if len(vs) > 0 {
					room, page, promised = lp.kvRoom()
					need = rowPages(r, 0, page) * page
				}
			}
			if len(lp.rows) >= lp.width {
				break
			}
			if blocked(need) {
				lp.stats.kvWaits.Add(1)
				break
			}
			lp.waiting = append(lp.waiting[:wi], lp.waiting[wi+1:]...)
			r.s.queuePos.Store(0)
			if r.parked {
				if err := lp.resumeRow(r); err != nil {
					lp.mu.Unlock()
					lp.finish(r, FinishError, "", err)
					lp.mu.Lock()
					continue
				}
				if !stepsJointly(r.s.st) && len(lp.rows) > 0 {
					// The card took back fewer of its blocks than it had: a
					// split row steps alone and re-streams its history every
					// token, and holds none of the card's room, so more would
					// be admitted behind it. It parks again and waits for
					// rows to retire; alone on the card it runs split.
					if lp.e.parkLocked(r.s) == nil {
						r.parked = true
						lp.stats.shortResumes.Add(1)
						lp.waiting = append([]*row{r}, lp.waiting...)
						break
					}
					// Refused, it runs split rather than fail.
					lp.stats.parkRefusals.Add(1)
				}
			} else {
				lp.recheckPrefix(r)
				r.waited = time.Since(r.enqueued)
				lp.stats.admissions.Add(1)
				lp.stats.admitWait.Add(int64(r.waited))
				close(r.admitted)
			}
			room -= need
			promised += need
			r.slice = 0
			lp.rows = append(lp.rows, r)
			lp.shape++
		}
		lp.renumber()
	}
	if page > 0 {
		lp.lm.gpuTier().PromiseKV(promised)
	}
}

// minSharedPrefix is the fewest prompt positions two requests must share
// for the later to wait for the earlier's prompt (prefixHoldMax bounds the
// wait): below it the recompute is cheaper than the queue.
const minSharedPrefix = 16

// prefixHoldMax is the longest a row waits for a producer of its prefix; past
// it the row is admitted and computes its own.
const prefixHoldMax = 2 * time.Second

// holdsForPrefix reports whether waiting row r is held back: a fresh row that
// missed the cache, that no priority exempts, while an admitted row is still
// computing a prompt that starts as r's does, so that r restores that prefix
// instead of starting a second producer of it. The hold is bounded
// (prefixHoldMax) and ends when the producer's prompt is in (it sealed it),
// leaves (cancelled or failed: r is promoted to compute it itself), or r
// would gain under minSharedPrefix positions. Rows that share nothing are
// never held, and one held row does not stop the rows behind it. Callers
// hold lp.mu.
func (lp *stepLoop) holdsForPrefix(r *row) bool {
	if !r.seal || !r.fresh || r.begun || r.parked || r.restored != 0 || r.prio > 0 || r.holdSpent {
		return false
	}
	for _, q := range lp.rows {
		if !q.seal || !q.prompting() || q.s == r.s {
			continue
		}
		if sharedPrefix(q.ids, r.ids)-q.restored < minSharedPrefix {
			continue
		}
		if r.heldSince.IsZero() {
			r.heldSince = time.Now()
			lp.stats.prefixHolds.Add(1)
		} else if time.Since(r.heldSince) > prefixHoldMax {
			r.holdSpent = true
			lp.stats.prefixHoldTimeouts.Add(1)
			return false
		}
		return true
	}
	return false
}

// sharedPrefix is how many leading tokens a and b have in common.
func sharedPrefix(a, b []int32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// recheckPrefix looks the memory cache up again for a row that missed it
// before it was queued and has not run: requests for one prefix that arrive
// together all miss, and the one admitted first leaves the prefix for the
// rest while they wait. Only a row still at position zero can restore
// (State.RestorePrefix), so a row that has begun, was parked, or continues a
// session keeps what it has. A failed restore leaves it a miss.
func (lp *stepLoop) recheckPrefix(r *row) {
	if !r.seal || !r.fresh || r.begun || r.parked || r.restored != 0 || r.s.st.Pos() != 0 {
		return
	}
	n, err := r.s.st.RestorePrefix(r.ids)
	if err != nil || n == 0 {
		return
	}
	r.restored, r.fed, r.reset = n, n, false
}

// admitLookahead is the most generated positions admission promises a row
// beyond its prompt. A row's max_tokens is a bound, not a forecast (an
// unbounded chat request's is the context), so promising all of it would
// admit one row at a time; past the lookahead a row grows into whatever room
// is left, and on a full card its oldest pages go home and stream back
// (kvevict.go), as before there was admission. Parking the newest rows whole
// when the card fills is the upgrade; a time slice (batchfair.go) parks rows
// only for a request waiting to be admitted.
const admitLookahead = 512

// rowPages is the pages of history row r takes once its prompt and its
// promised generation are in, less those of its first pos positions, which
// it already holds. A page of 0 prices nothing.
func rowPages(r *row, pos, page int) int {
	if page <= 0 {
		return 0
	}
	end := len(r.ids) + min(max(r.maxTokens, 0), r.resumedAt+admitLookahead)
	if m := r.s.st.MaxSeq(); m > 0 {
		end = min(end, m)
	}
	return max((end+page-1)/page-(pos+page-1)/page, 0)
}

// kvRoom is admission's budget, in positions: the card's room for history
// (tier.GPU.KVRoom) less promised, what the admitted rows are still promised,
// and the page both are counted in. A page of 0 is no bound: a model on the
// host, or no history on the card to price.
//
// Admission keeps the rows' histories on the card. A row admitted past the
// room would send the oldest pages of others home, and a step with two
// sequences' pages at home is refused as one step: every row then runs alone
// and re-streams its evicted prefix each token -- 172 of 256 requests failed
// at 64 concurrent on a V100 that way. A row that does not fit waits for the
// rows ahead of it to retire, or above fairness level 0 for rows past their
// time slice to be parked (batchfair.go); the first row is
// always admitted, so nothing waits on a card that holds none. The promise
// goes to the tier too (tier.GPU.PromiseKV), so a batched scratch built
// after admission does not take the pages it was made on.
func (lp *stepLoop) kvRoom() (room, page, promised int) {
	g := lp.lm.gpuTier()
	if g == nil {
		return 0, 0, 0
	}
	room, page = g.KVRoom()
	if page == 0 {
		return 0, 0, 0
	}
	for _, r := range lp.rows {
		pos := r.s.st.Pos()
		if r.reset && !r.begun {
			pos = 0 // its State's old history goes at its first chunk
		}
		promised += rowPages(r, pos, page) * page
	}
	return room - promised, page, promised
}

// started queues a row's GenerateStarted (and its echo) once its prompt is in.
func (r *row) started() {
	lm := r.s.lm
	devBlocks := r.s.st.GPULayers()
	r.push(Event{Kind: EventStarted, Started: &Started{
		SessionID:    r.s.id,
		ModelID:      lm.id,
		Ephemeral:    r.ephemeral,
		PromptTokens: len(r.ids),
		Restored:     r.restored,
		QueuedFor:    r.waited,
		QueueDepth:   r.depth,
		DeviceBlocks: devBlocks,
		HostBlocks:   lm.m.Cfg.NLayer - devBlocks,
		DeviceIDs:    lm.devIDs(),
		Prefill:      r.prefill,
		Execution:    ExecutionParallel,
		Batched:      true,
		Tools:        r.tools,
	}})
	if r.echo {
		r.push(Event{Kind: EventToken, Token: &Token{ID: -1, Text: lm.m.Vocab.Decode(r.ids), Index: -1}})
	}
	r.decodeStart = time.Now()
}

// unit is one row's part of a step: its next tokens -- the one it sampled, or
// a chunk of its prompt -- and whether the logits after them are wanted.
type unit struct {
	r      *row
	tokens []int32
	logits bool
	prompt bool
	// alone is a pipelined prompt chunk: it runs as its State's own prefill,
	// never as rows of the joint step.
	alone bool
}

// iterate runs one step: every decoding row's next token, and as much of the
// admitted prompts as the step has room for, in one model.StepRuns.
func (lp *stepLoop) iterate() {
	// A move between devices a decoding row's session asked for is prepared
	// and committed here, between two steps, as the one-at-a-time path does
	// between two tokens (Engine.MoveBlocks): the loop is what holds the
	// row's State.
	for _, r := range lp.rows {
		if !r.prompting() {
			r.s.serveMove(false)
		}
	}
	units := lp.decodeUnits()
	decoding := len(units)
	budget := lp.promptChunk
	if decoding > 0 {
		budget = lp.budget.tokens()
		lp.stats.noteBudget(budget)
	}
	units = lp.promptUnits(units, min(budget, model.MaxStepRows-decoding))
	lp.units = units
	if len(units) == 0 {
		return
	}
	rows := 0
	for _, u := range units {
		rows += len(u.tokens)
	}
	lp.stats.steps.Add(1)
	lp.stats.rowsStepped.Add(int64(rows))
	lp.stats.maxRows.Store(max(lp.stats.maxRows.Load(), int32(rows)))
	lp.stats.maxSessions.Store(max(lp.stats.maxSessions.Load(), int32(len(units))))
	if decoding > 0 && decoding < len(units) {
		lp.stats.promptSteps.Add(1)
	}
	if lp.feedNext && decoding > 1 {
		first := units[0].tokens[0]
		for i := range decoding - 1 {
			units[i].tokens[0] = units[i+1].tokens[0]
		}
		units[decoding-1].tokens[0] = first
	}
	prompt := 0
	for _, u := range units[decoding:] {
		if !u.alone {
			prompt += len(u.tokens)
		}
	}
	if decoding > 0 {
		lp.stats.maxPromptBeside.Store(max(lp.stats.maxPromptBeside.Load(), int64(prompt)))
	}
	// The tokens this step feeds become text and events while it runs.
	lp.post.start()
	t0 := time.Now()
	lp.step(units)
	d := time.Since(t0)
	lp.budget.observe(decoding, prompt, d)
	if decoding > 0 {
		lp.stepNs = ewma(lp.stepNs, float64(d))
	}
	lp.post.wait()
	lp.sealPrompts()
}

// sealJob is a row whose prompt just ran, and the logits it ended on.
type sealJob struct {
	r      *row
	logits []float32
}

// sealPrompts offers each completed prompt's history to the memory cache
// (State.SealPrompt). It runs between steps, outside the time the step budget
// and the fairness quantum are fitted to: nothing has moved on since the prompt
// ended, because the row's next token is sampled at the next step.
func (lp *stepLoop) sealPrompts() {
	for _, j := range lp.sealing {
		if j.r.isDone() {
			continue // the loop never touches a retired row's State
		}
		if err := j.r.s.st.SealPrompt(j.logits); err != nil {
			lp.finish(j.r, FinishError, "", err)
		}
	}
	clear(lp.sealing)
	lp.sealing = lp.sealing[:0]
}

// decodeUnits samples every decoding row's next token from the logits its
// last step left, retires the rows that end here, and returns the rest's
// tokens to step.
//
// The order per row is the one-at-a-time path's per token, so a batched
// generate stops on the same token, with the same position, as one alone: a
// cancel is honoured before sampling, max_tokens before sampling, an EOG
// token is never fed (unless the row ignores it, and then it is a token like
// any other), and a stop string ends the row before its token is fed.
func (lp *stepLoop) decodeUnits() []unit {
	vocab := lp.lm.m.Vocab
	units := lp.units[:0]
	lp.walk = append(lp.walk[:0], lp.rows...)
	for _, r := range lp.walk {
		if r.prompting() {
			continue
		}
		if r.cancelled.Load() {
			lp.finish(r, FinishCancelled, "", nil)
			continue
		}
		if r.n >= r.maxTokens && !lp.dropMaxTokens {
			lp.finish(r, FinishMaxTokens, "", nil)
			continue
		}
		next := r.sampler.Sample(r.logits)
		if lp.e.sampleTap != nil {
			lp.e.sampleTap(r.sampler.Temp, r.logits, next)
		}
		r.sampler.Observe(next)
		tlp := takeLogprob(r.logprobs, vocab, r.logits, next)
		if !r.ignoreEOS && vocab.IsEOG(next) {
			lp.finish(r, FinishEOS, "", nil)
			continue
		}
		r.out = append(r.out, next)
		if !r.stops {
			// No stop string can end the row here, so its text is not
			// needed to decide the step: the helper makes it while the
			// step runs.
			lp.post.add(r, next, r.n, tlp)
		} else {
			chunk, hit, match := r.stream.push(r.out)
			r.push(Event{Kind: EventToken, Token: &Token{ID: next, Text: chunk, Index: r.n, Logprob: tlp}})
			if hit {
				r.n++
				lp.finish(r, FinishStop, match, nil)
				continue
			}
		}
		r.next[0] = next
		units = append(units, unit{r: r, tokens: r.next[:], logits: true})
	}
	return units
}

// promptUnits is admission: the admitted prompts' next chunks, the fairness
// level's share of the budget split evenly across a window of them (allot) and the rest oldest first,
// at most budget tokens in all, or a token a prompt of the window when there
// are more prompts than that. They ride in the step the decoding rows take
// (model.StepRuns), each chunk as rows of its own session at its next
// positions, so a new prompt never holds the decoding rows up by more than the
// rows it adds to one step. A chunk that ends its prompt wants its logits --
// the row samples from them at the next step; one that does not costs no head
// projection.
//
// A row whose State pipelines its prefill across several cards (chunkOf
// wider than a joint step, model.MaxStepRows) takes a pipelined chunk
// instead, run alone beside the joint step, so its cards overlap as they
// would on the prompt alone; the decoding rows wait one pipelined chunk rather than the prompt.
// One such chunk a step, the oldest.
func (lp *stepLoop) promptUnits(units []unit, budget int) []unit {
	piped := false
	lp.walk = append(lp.walk[:0], lp.rows...)
	for _, r := range lp.walk {
		if !r.prompting() {
			continue
		}
		// A request that left mid-prompt is not fed the rest. One whose
		// sequence starts here sends no GenerateStarted and gives its State
		// back reset, as a prefill interrupted alone does; one that continues
		// a sequence, or never began its prompt, keeps its history as it is.
		if r.cancelled.Load() && (r.fresh || !r.begun) {
			if r.fresh {
				r.s.st.Reset()
			}
			lp.finish(r, FinishCancelled, "", errPromptCancelled)
			continue
		}
		// A continuing request that left mid-prompt ends here, its history
		// holding what was fed, as a cancel mid-decode leaves it.
		if r.cancelled.Load() {
			if r.begun {
				r.prefill = time.Since(r.promptStart)
			}
			r.started()
			lp.finish(r, FinishCancelled, "", nil)
			continue
		}
		pipe := lp.chunkOf(r)
		k := 0
		if !piped && pipe > model.MaxStepRows && len(r.ids)-r.fed > lp.promptChunk {
			piped = true
			k = -min(pipe, len(r.ids)-r.fed) // a pipelined chunk, marked negative
		}
		lp.prompts = append(lp.prompts, r)
		lp.alloc = append(lp.alloc, k)
	}
	lp.allot(budget)
	for i, r := range lp.prompts {
		k, pipelined := lp.alloc[i], lp.alloc[i] < 0
		if pipelined {
			k = -k
		}
		if k == 0 {
			continue
		}
		if !r.begun {
			if r.reset {
				r.s.st.Reset()
			}
			r.begun, r.promptStart = true, time.Now()
		}
		units = append(units, unit{r: r, tokens: r.ids[r.fed : r.fed+k], logits: r.fed+k == len(r.ids),
			prompt: true, alone: pipelined})
	}
	if n := len(lp.prompts); n > 0 {
		fed := 0
		for _, k := range lp.alloc {
			if k != 0 {
				fed++
			}
		}
		lp.stats.promptRowsHad.Add(int64(n))
		lp.stats.promptRowsFed.Add(int64(fed))
	}
	clear(lp.prompts)
	lp.prompts, lp.alloc = lp.prompts[:0], lp.alloc[:0]
	return units
}

// allot shares budget prompt tokens across lp.prompts into lp.alloc (a
// pipelined chunk, already negative, takes none).
// The fairness level's share (fairShare) goes evenly to the prompts admitted
// after the oldest, as many as its window (fairWindow), at least a token
// each, its remainder turning round from step to step; the rest goes oldest
// first, as at level 0, so a prompt whose rest fits its turn ends at its last
// row. Prompts past the window wait their turn as at level 0.
func (lp *stepLoop) allot(budget int) {
	var plain int
	for _, k := range lp.alloc {
		if k == 0 {
			plain++
		}
	}
	if win := min(plain-1, fairWindow(lp.fair)); win > 0 {
		share := min(max(fairShare(lp.fair, budget), win), max(budget, win))
		budget = max(budget, win)
		per, rem := share/win, share%win
		j := -1 // the oldest plain prompt is not in the window
		for i, r := range lp.prompts {
			if lp.alloc[i] != 0 {
				continue
			}
			if j++; j == 0 {
				continue
			}
			if j > win {
				break
			}
			k := per
			if (j-1+win-lp.rr%win)%win < rem {
				k++
			}
			k = min(k, len(r.ids)-r.fed, budget)
			lp.alloc[i] = k
			budget -= k
		}
		lp.rr++
	}
	for i, r := range lp.prompts {
		if budget <= 0 {
			break
		}
		if lp.alloc[i] < 0 {
			continue
		}
		k := min(budget, len(r.ids)-r.fed-lp.alloc[i])
		lp.alloc[i] += k
		budget -= k
	}
}

// step runs the units: those of sessions wholly on the device as one
// model.StepRuns, any other alone, and moves each row on by what it ran.
func (lp *stepLoop) step(units []unit) {
	joint, solo, npiped := lp.joint[:0], lp.solo[:0], 0
	defer func() { lp.joint, lp.solo = joint[:0], solo[:0] }()
	for _, u := range units {
		switch {
		case u.alone:
			npiped++
		case stepsJointly(u.r.s.st):
			joint = append(joint, u)
		default:
			solo = append(solo, u)
		}
	}
	// A pipelined chunk runs first and alone: its prefill is the State's own
	// path across the cards, and the decoding rows step after it.
	for _, u := range units {
		if !u.alone {
			continue
		}
		out, err := model.StepRuns([]model.Run{{State: u.r.s.st, Tokens: u.tokens, Logits: u.logits}})
		if err != nil {
			lp.finish(u.r, FinishError, "", err)
			continue
		}
		lp.stats.pipelinedChunks.Add(1)
		lp.stats.soloRows.Add(int64(len(u.tokens)))
		lp.ran(u, out[0])
	}
	if (lp.refused && lp.refusedFor == lp.shape) || len(joint) < 2 {
		solo, joint = append(solo, joint...), nil
	}
	// A decode step that could go joint asks the measured choice. A step
	// carrying a prompt chunk always goes joint: admission is what keeps a new
	// prompt from holding the decoding rows up, and alone its chunk would.
	chose, timed := true, len(joint) >= 2 && len(solo) == 0 && !anyPrompt(joint)
	if timed {
		if chose = lp.choice.pick(len(joint)); !chose {
			lp.stats.separateSteps.Add(1)
			solo, joint = joint, nil
		}
	}
	t0 := time.Now()
	if len(joint) > 0 && !lp.stepJoint(joint) {
		solo = append(solo, joint...)
	}
	for _, u := range solo {
		out, err := model.StepRuns([]model.Run{{State: u.r.s.st, Tokens: u.tokens, Logits: u.logits}})
		if err != nil {
			lp.finish(u.r, FinishError, "", err)
			continue
		}
		lp.stats.soloRows.Add(int64(len(u.tokens)))
		lp.ran(u, out[0])
	}
	// A step the device refused ran its rows alone, which is neither arm.
	if timed && !(lp.refused && lp.refusedFor == lp.shape) {
		lp.choice.observe(len(units)-npiped, chose, time.Since(t0))
	}
}

func anyPrompt(units []unit) bool {
	for _, u := range units {
		if u.prompt {
			return true
		}
	}
	return false
}

// stepJoint runs units as one model.StepRuns. It reports false, with no row
// moved, when the device refused the step before running any of it; those
// units then run alone, and the refusal stands until the rows change.
func (lp *stepLoop) stepJoint(units []unit) bool {
	runs := lp.runs[:0]
	defer func() { clear(runs); lp.runs = runs[:0] }()
	rows := 0
	for _, u := range units {
		runs = append(runs, model.Run{State: u.r.s.st, Tokens: u.tokens, Logits: u.logits})
		rows += len(u.tokens)
	}
	gpu := lp.lm.gpuTier()
	if gpu == nil {
		return lp.stepHost(units, runs, rows)
	}
	before := gpu.Stats().SessionRows
	out, err := model.StepRuns(runs)
	if err != nil {
		// A hybrid's linear block keeps a summary that a step advances in
		// place: once any advanced, running the tokens again would apply them
		// twice, so those rows end with the device's error.
		advanced := false
		for _, u := range units {
			advanced = advanced || u.r.s.st.RecurrenceAdvanced()
		}
		if advanced {
			for _, u := range units {
				lp.finish(u.r, FinishError, "", err)
			}
			return true
		}
		why := err.Error()
		lp.stats.refusals.Add(1)
		lp.stats.lastRefusal.Store(&why)
		lp.refused, lp.refusedFor = true, lp.shape
		return false
	}
	// The tier's own count says whether the rows really shared a step:
	// StepRuns runs States one after another, correctly, whenever they cannot.
	if gpu.Stats().SessionRows-before == rows {
		lp.stats.jointSteps.Add(1)
		lp.stats.jointRows.Add(int64(rows))
	} else {
		lp.stats.soloRows.Add(int64(rows))
	}
	for i, u := range units {
		lp.ran(u, out[i])
	}
	return true
}

// stepHost is stepJoint for a model on the host: the runs as one host pass
// (model.StepRuns' host arm). The host refuses no step it was offered, so an
// error ends every row in it: a hybrid's rows may have advanced their
// recurrent state, and running them again would apply their tokens twice.
func (lp *stepLoop) stepHost(units []unit, runs []model.Run, rows int) bool {
	m := lp.lm.m
	before := m.HostStepRows()
	out, err := model.StepRuns(runs)
	if err != nil {
		for _, u := range units {
			lp.finish(u.r, FinishError, "", err)
		}
		return true
	}
	if m.HostStepRows()-before == int64(rows) {
		lp.stats.jointSteps.Add(1)
		lp.stats.jointRows.Add(int64(rows))
	} else {
		lp.stats.soloRows.Add(int64(rows))
	}
	for i, u := range units {
		lp.ran(u, out[i])
	}
	return true
}

// ran moves u's row on by the tokens it just ran: a decoding row by one
// token, a prompting one by its chunk -- and a prompt that is in becomes a
// decoding row, its GenerateStarted queued.
func (lp *stepLoop) ran(u unit, logits []float32) {
	r := u.r
	if !u.prompt {
		r.logits = logits
		r.n++
		r.slice++
		return
	}
	r.fed += len(u.tokens)
	lp.stats.promptChunks.Add(1)
	if r.prompting() {
		return
	}
	r.logits = logits
	if r.seal {
		// The store gets this prompt as PrefillCached would have left it, once
		// the step is over: sealing reads the whole history home from the card
		// and copies it into the store, which costs several times the prompt's
		// own prefill, and inside the step it was counted as the prompt's
		// time -- stepBudget read prompt tokens as dear and fed every later
		// prompt a handful of tokens a step.
		lp.sealing = append(lp.sealing, sealJob{r: r, logits: logits})
	}
	r.prefill = time.Since(r.promptStart)
	r.s.prefilled.Add(int64(len(r.ids)))
	lp.lm.tokensPrefilled.Add(int64(len(r.ids)))
	lp.lm.ttft.restored.Add(int64(r.restored))
	lp.lm.ttft.computed.Add(int64(len(r.ids) - r.restored))
	r.started()
	r.startStep = lp.stats.steps.Load()
	if lp.onStarted != nil {
		lp.onStarted(r)
	}
}

// finish retires r: its held-back text is flushed as the one-at-a-time path
// flushes it, and from here on the loop never touches its State again.
func (lp *stepLoop) finish(r *row, reason FinishReason, stop string, err error) {
	// The helper may hold r's last token; its event goes before the end.
	lp.post.wait()
	if err == nil && (reason == FinishMaxTokens || reason == FinishEOS || reason == FinishCancelled) {
		if tail := r.stream.flush(); tail != "" {
			r.push(Event{Kind: EventToken, Token: &Token{ID: -1, Text: tail, Index: r.n}})
		}
	}
	var decode time.Duration
	if !r.decodeStart.IsZero() {
		decode = time.Since(r.decodeStart)
	}
	r.result = rowResult{reason: reason, stop: stop, n: r.n, decode: decode, err: err}
	lp.mu.Lock()
	for i, x := range lp.rows {
		if x == r {
			lp.rows = append(lp.rows[:i], lp.rows[i+1:]...)
			break
		}
	}
	lp.shape++
	lp.mu.Unlock()
	close(r.done)
}

// generateBatched is Generate's arm for a session that runs as a row. The
// caller holds s.mu, so nothing but the loop touches s.st until r.done.
func (e *Engine) generateBatched(ctx context.Context, lp *stepLoop, s *Session, o GenerateOptions,
	ids []int32, ephemeral, store bool, emit func(Event) error) error {
	lm := s.lm
	sampler := s.sampling
	if o.Sampling != nil {
		sampler = *o.Sampling
	}
	// The row's prompt starts where the session is, or at 0 when it resets.
	start := 0
	if o.Continue {
		start = s.st.Pos()
	}
	maxTokens := tokenLimit(o.MaxTokens, s.st.MaxSeq()-start-len(ids))
	// The memory cache restores before the row is queued, on this goroutine
	// (the caller reset the State and attached the store), so the loop feeds
	// only what it did not hold. It leaves at least one token to run: the row
	// needs that token's logits.
	restored := 0
	if store {
		n, err := s.st.RestorePrefix(ids)
		if err != nil {
			return err
		}
		restored = n
	}
	now := time.Now()
	r := &row{
		tools:     lm.toolStream(o),
		s:         s,
		ids:       ids,
		restored:  restored,
		fed:       restored,
		seal:      store,
		reset:     !o.Continue && restored == 0,
		fresh:     !o.Continue,
		echo:      o.Echo,
		ephemeral: ephemeral,
		sampler:   sampler,
		logprobs:  newLogprobs(o),
		maxTokens: maxTokens,
		ignoreEOS: o.IgnoreEOS,
		stops:     hasStop(o.Stop),
		stream:    newStreamText(lm.m.Vocab.NewChatStream().Next, o.Stop),
		enqueued:  now,
		since:     now,
		prio:      rowPriority(o.Priority),
		admitted:  make(chan struct{}),
		done:      make(chan struct{}),
		notify:    make(chan struct{}, 1),
		out:       make([]int32, 0, maxTokens),
	}
	if err := lp.submit(r); err != nil {
		return err
	}
	defer s.queuePos.Store(0)
	if lp.beforeQueueWait != nil {
		lp.beforeQueueWait(r)
	}

	// ---- the queue: until a row is free, the request can still leave.
	var timeout <-chan time.Time
	if o.QueueTimeout > 0 {
		t := time.NewTimer(o.QueueTimeout)
		defer t.Stop()
		timeout = t.C
	}
	select {
	case <-r.admitted:
	case <-r.done:
		// A row the loop admitted and ran to its end before this select
		// looked has both channels closed, and select picks either. Only a
		// row that ended in the queue (the loop closed) ends here; one that
		// ran drains its outbox and ends with its Finished event below.
		select {
		case <-r.admitted:
		default:
			return r.result.err
		}
	case <-ctx.Done():
		if lp.withdraw(r) {
			return ctx.Err()
		}
		r.cancelled.Store(true)
	case <-timeout:
		if lp.withdraw(r) {
			return ErrQueueTimeout
		}
	}
	s.running.Store(true)
	defer s.running.Store(false)
	s.lastUsed.Store(time.Now().UnixMilli())

	// ---- the row: drain its outbox until the loop lets go of it. A cancel or
	// a failed write marks the row, and the loop retires it at its next step.
	// The cancel is marked the moment it happens, not when this drain next
	// selects: the loop does not wait for the client, so while an emit blocks
	// it keeps stepping the row, and a fast device can run the row to
	// max_tokens before the drain would have looked.
	stopMark := context.AfterFunc(ctx, func() { r.cancelled.Store(true) })
	defer stopMark()
	var emitErr error
	cancelled := ctx.Done()
	for {
		fin := r.isDone()
		for _, ev := range r.take() {
			if emitErr != nil {
				break
			}
			if err := emit(ev); err != nil {
				emitErr = err
				r.cancelled.Store(true)
			}
		}
		if fin {
			break
		}
		select {
		case <-r.notify:
		case <-r.done:
		case <-cancelled:
			r.cancelled.Store(true)
			cancelled = nil
		}
	}

	res := r.result
	s.generated.Add(int64(res.n))
	lm.finished(res.n, r.prefill, res.decode)
	s.lastUsed.Store(time.Now().UnixMilli())
	s.refresh()
	// The history grew by what this generate committed.
	e.applyPageBudget(s.lm, nil)
	if emitErr != nil {
		return emitErr
	}
	if errors.Is(res.err, errPromptCancelled) {
		return context.Cause(ctx)
	}
	if res.err != nil {
		return res.err
	}
	rate := 0.0
	if res.decode > 0 && res.n > 0 {
		rate = float64(res.n) / res.decode.Seconds()
	}
	return emit(Event{Kind: EventFinished, Finished: &Finished{
		Reason:           res.reason,
		StopMatched:      res.stop,
		PromptTokens:     len(ids),
		CompletionTokens: res.n,
		Prefill:          r.prefill,
		Decode:           res.decode,
		TokensPerSecond:  rate,
		BytesPerToken:    lm.m.BytesPerToken(),
		Position:         s.st.Pos(),
		Restored:         r.restored,
	}})
}

// snapshot is the loop's queue as GetDeviceQueue reports it: the sessions in
// rows, then the ones waiting for one.
func (lp *stepLoop) snapshot() (live, waiting []string) {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	for _, r := range lp.rows {
		live = append(live, r.s.id)
	}
	for _, r := range lp.waiting {
		waiting = append(waiting, r.s.id)
	}
	return live, waiting
}

// deviceQueue is a device's queue as GetDeviceQueue and ListDevices report it:
// the gate's holders and waiters, with a model's step loop -- one entry in the
// gate's own queue -- expanded into the sessions in its rows and the requests
// waiting for one. key is a gate key (Engine.gateKey), not a device id as
// written.
func (e *Engine) deviceQueue(key string) (queue []string, running, waiting int) {
	g := e.gate(key)
	gq, running, waiting := g.snapshot()
	loops := map[string]*stepLoop{}
	for _, lm := range e.Models() {
		if lm.loop != nil && hasString(lm.gateIDs(), key) {
			loops[lm.loop.id] = lm.loop
		}
	}
	for _, x := range gq {
		lp, ok := loops[x]
		if !ok {
			queue = append(queue, x)
			continue
		}
		delete(loops, x)
		live, wait := lp.snapshot()
		queue = append(append(queue, live...), wait...)
		waiting += len(wait)
		running += len(live) - 1
	}
	// A loop between taking a request and recording itself is in no queue yet;
	// its requests are still waiting.
	for _, lp := range loops {
		_, wait := lp.snapshot()
		queue = append(queue, wait...)
		waiting += len(wait)
	}
	return queue, max(0, running), max(0, waiting)
}

// sessionMode is how a generate on s runs: beside every other session, as a
// row of a step it shares or interleaved a step at a time.
func sessionMode(*Session) ExecutionMode { return ExecutionParallel }

// pb is the loop's BatchStats, or nil for a model with no loop.
func (lp *stepLoop) pb() *v1.BatchStats {
	if lp == nil {
		return nil
	}
	live, wait := lp.snapshot()
	c := &lp.stats
	out := &v1.BatchStats{
		Width:               int32(lp.width),
		LiveRows:            int32(len(live)),
		Waiting:             int32(len(wait)),
		Steps:               c.steps.Load(),
		RowsStepped:         c.rowsStepped.Load(),
		JointSteps:          c.jointSteps.Load(),
		JointRows:           c.jointRows.Load(),
		SoloRows:            c.soloRows.Load(),
		MaxRowsPerStep:      c.maxRows.Load(),
		MaxSessionsPerStep:  c.maxSessions.Load(),
		PromptChunks:        c.promptChunks.Load(),
		PromptSteps:         c.promptSteps.Load(),
		Admissions:          c.admissions.Load(),
		AdmissionWaitMillis: time.Duration(c.admitWait.Load()).Milliseconds(),
		JointRefusals:       c.refusals.Load(),
		SeparateSteps:       c.separateSteps.Load(),
	}
	for _, ch := range lp.choice.snapshot() {
		out.Choices = append(out.Choices, &v1.JointChoice{
			Rows:    int32(ch.rows),
			Settled: ch.settled,
			Joint:   ch.joint,
			// The medians are per row; a step of the bucket's widest count
			// is what the fields report.
			JointStepMillis:    float64(ch.jointMedian) * float64(ch.rows) / float64(time.Millisecond),
			SeparateStepMillis: float64(ch.separateMedian) * float64(ch.rows) / float64(time.Millisecond),
			Probes:             int32(ch.probes),
		})
	}
	if why := c.lastRefusal.Load(); why != nil {
		out.LastRefusal = *why
	}
	return out
}

// noteBudget records the budget of one step beside decoding rows.
func (c *batchCounters) noteBudget(b int) {
	n := int64(b)
	if c.budgetSteps.Add(1) == 1 || n < c.budgetMin.Load() {
		c.budgetMin.Store(n)
	}
	c.budgetMax.Store(max(c.budgetMax.Load(), n))
	c.budgetSum.Add(n)
}
