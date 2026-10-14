package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/engine/model"
)

// What a request pays before its first token, and the four things here that
// cut it:
//
//   - The memory cache. Every loaded model keeps one bounded model.MemStore,
//     and every generate that starts its sequence prefills through it
//     (State.PrefillCached), so a chat's second turn or a system prompt
//     another request already ran restores its prefix instead of computing
//     it. The store is a share of the model's host budget, taken out of
//     what the pager may hold (applyPageBudget).
//   - The State pool. A model_id request ran on a State built for it and
//     closed after it; Close quiesces the device and retires its recordings,
//     and a fresh State grows its batch scratch again. A finished request's
//     State is reset and kept for the next, up to a bound.
//   - Warm-up at load (LoadOptions.Warm): a short prefill and a decode step
//     on a pooled State, so the first request does not pay the JIT's
//     emission and the device's first recordings.
//   - Admission: a bounded count of requests per model, past which a request
//     is refused with ErrOverloaded (429 and Retry-After over HTTP) rather
//     than queued without end; a solo prefill stops between chunks when its
//     client goes.

// ErrOverloaded is a request refused because its model already holds as many
// as Config.MaxQueue allows. Nothing was run; retrying later is correct. The
// error returned wraps it in an *OverloadedError carrying how long to wait.
var ErrOverloaded = errors.New("server: the model's request queue is full")

// OverloadedError is ErrOverloaded with the wait an HTTP refusal names in
// Retry-After.
type OverloadedError struct {
	Model      string
	Held       int
	RetryAfter time.Duration
}

func (o *OverloadedError) Error() string {
	return fmt.Sprintf("%v: model %q holds %d request(s)", ErrOverloaded, o.Model, o.Held)
}

// Unwrap makes errors.Is(err, ErrOverloaded) hold.
func (o *OverloadedError) Unwrap() error { return ErrOverloaded }

// retryAfter sets Retry-After from an *OverloadedError, in whole seconds,
// at least one.
func retryAfter(w http.ResponseWriter, err error) {
	var o *OverloadedError
	if !errors.As(err, &o) {
		return
	}
	w.Header().Set("Retry-After", strconv.Itoa(max(1, int((o.RetryAfter+time.Second-1)/time.Second))))
}

// admissionChecker is a Backend that can say, before a stream opens, that a
// generate would be refused -- once the stream's 200 is written a refusal can
// only be an error frame. It reserves nothing: Generate decides again, and a
// request that loses the race in between gets the frame.
type admissionChecker interface {
	CheckAdmission(o GenerateOptions) error
}

// admits asks b whether o would be admitted, when b can say.
func admits(b Backend, o GenerateOptions) error {
	if a, ok := b.(admissionChecker); ok {
		return a.CheckAdmission(o)
	}
	return nil
}

// modelOf is the model a generate of o runs on, or nil when o names none
// that is loaded (Generate reports that).
func (e *Engine) modelOf(o GenerateOptions) *LoadedModel {
	if o.SessionID != "" {
		s, err := e.Session(o.SessionID)
		if err != nil {
			return nil
		}
		return s.lm
	}
	lm, err := e.Model(o.ModelID)
	if err != nil {
		return nil
	}
	return lm
}

// CheckAdmission reports the ErrOverloaded a generate of o would get now,
// and the refusal of a picture by a model with no vision tower.
func (e *Engine) CheckAdmission(o GenerateOptions) error {
	lm := e.modelOf(o)
	if lm == nil {
		return nil
	}
	if o.Prompt.Chat != nil && len(o.Prompt.Chat.Images) > 0 {
		if err := acceptsImages(lm); err != nil {
			return err
		}
	}
	if t := &lm.ttft; t.queueCap > 0 && int(t.inflight.Load()) >= t.queueCap {
		t.refused.Add(1)
		return e.overloaded(lm)
	}
	return nil
}

func (e *Engine) overloaded(lm *LoadedModel) error {
	ra := e.cfg.RetryAfter
	if ra <= 0 {
		ra = time.Second
	}
	return &OverloadedError{Model: lm.id, Held: lm.ttft.queueCap, RetryAfter: ra}
}

// defaultMaxQueue is Config.MaxQueue when it is zero.
const defaultMaxQueue = 64

// storeShare is the fraction of a model's host share its memory cache may
// hold, when Config.MemCacheBytes does not say: an eighth, the same slice
// a backgrounded model keeps (backgroundShare).
const storeShare = 8

// MemCacheStats is a model's memory cache and State pool, as counted.
type MemCacheStats struct {
	Store model.MemStoreStats
	// Restored and Computed are prompt positions taken from the store and
	// prefilled, over every generate that started its sequence.
	Restored, Computed int64
	// StatesCreated, StatesReused and StatesClosed count model.States built,
	// handed out again from the pool, and closed.
	StatesCreated, StatesReused, StatesClosed int64
	Pooled                                    int
	// Warmed says a warm-up ran at load.
	Warmed bool
	// Refused counts requests turned away with ErrOverloaded.
	Refused int64
}

// ttft is a LoadedModel's share of the above.
type ttft struct {
	store *model.MemStore // nil with Config.NoMemCache
	// kv is what a State is pointed at: the memory cache, or the memory cache
	// over Config.PromptStore (layered) when the server has one.
	kv model.KVStore
	ns string

	// idle is the pool: reset States waiting for a request. Guarded by the
	// Engine's mu, as lm.sessions is.
	idle    []*model.State
	poolCap int

	inflight atomic.Int32
	queueCap int

	restored, computed                        atomic.Int64
	statesCreated, statesReused, statesClosed atomic.Int64
	refused                                   atomic.Int64
	warmed                                    atomic.Bool
}

// initTTFT sets lm's memory cache, pool and queue bound from the config.
func (e *Engine) initTTFT(lm *LoadedModel, o LoadOptions) {
	t := &lm.ttft
	if !e.cfg.NoMemCache && !lm.m.IsEncoder() {
		t.store = model.NewBoundedMemStore(1)
		// The namespace names this load: the store is the model's own, and
		// the KV width and page geometry are fixed for its life.
		t.ns = fmt.Sprintf("jitllmd/%s/%d", lm.id, lm.loadedAt.UnixNano())
		t.kv = t.store
		if e.cfg.PromptStore != nil {
			t.kv = &layered{mem: t.store, disk: e.cfg.PromptStore}
		}
	}
	switch {
	case e.cfg.SessionPool < 0:
		t.poolCap = 0
	case e.cfg.SessionPool > 0:
		t.poolCap = e.cfg.SessionPool
	default:
		// As many as the device reserved a history for: one more pooled
		// State would take its history on the host.
		t.poolCap = max(o.Sessions, 1)
	}
	switch {
	case e.cfg.MaxQueue < 0:
		t.queueCap = 0
	case e.cfg.MaxQueue > 0:
		t.queueCap = e.cfg.MaxQueue
	default:
		t.queueCap = defaultMaxQueue
	}
}

// storeLimit is the memory cache's bound under a host share: what the config
// names, else an eighth of the share -- and never more than an eighth of what
// the cache holds plus what the host has available now.
//
// The share is divided from what the host offered at load, and a cache grows
// for as long as distinct prompts arrive: an eighth of a 190 GB share is
// 24 GB for a 0.8 GB model, and four such servers bound to one 128 GB NUMA
// node may each grow to it. Re-read after every generate, the second bound
// follows the host: as other processes (or other servers) fill it, each cache
// stops growing: each settles where it holds an eighth of what it holds plus
// what is free, so n caches on one host hold at most n/(n+7) of the memory
// that was free (four: 36%).
func (e *Engine) storeLimit(share, held uint64) uint64 {
	if e.cfg.MemCacheBytes > 0 {
		return e.cfg.MemCacheBytes
	}
	lim := share / storeShare
	if now := e.availNow(); now > 0 {
		lim = min(lim, (held+now)/storeShare)
	}
	return max(lim, 1)
}

// admitQueue counts a request against its model's bound, or refuses it. The
// returned func gives the slot back.
func (e *Engine) admitQueue(lm *LoadedModel) (func(), error) {
	t := &lm.ttft
	n := t.inflight.Add(1)
	if t.queueCap > 0 && int(n) > t.queueCap {
		t.inflight.Add(-1)
		t.refused.Add(1)
		return nil, e.overloaded(lm)
	}
	return func() { t.inflight.Add(-1) }, nil
}

// PromptCache reports the model's memory cache, pool and admission counters.
func (e *Engine) MemCacheStats(lm *LoadedModel) MemCacheStats {
	t := &lm.ttft
	var st model.MemStoreStats
	if t.store != nil {
		st = t.store.Stats()
	}
	e.mu.RLock()
	pooled := len(t.idle)
	e.mu.RUnlock()
	return MemCacheStats{
		Store:         st,
		Restored:      t.restored.Load(),
		Computed:      t.computed.Load(),
		StatesCreated: t.statesCreated.Load(),
		StatesReused:  t.statesReused.Load(),
		StatesClosed:  t.statesClosed.Load(),
		Pooled:        pooled,
		Warmed:        t.warmed.Load(),
		Refused:       t.refused.Load(),
	}
}

// takeIdle pops a pooled State, or nil.
func (e *Engine) takeIdle(lm *LoadedModel) *model.State {
	e.mu.Lock()
	defer e.mu.Unlock()
	n := len(lm.ttft.idle)
	if n == 0 {
		return nil
	}
	st := lm.ttft.idle[n-1]
	lm.ttft.idle = lm.ttft.idle[:n-1]
	return st
}

// releaseEphemeral ends a model_id request's session: unregistered as
// CloseSession does, and its State reset into the pool when there is room
// and the model is still loaded, closed otherwise.
func (e *Engine) releaseEphemeral(s *Session) {
	e.mu.Lock()
	delete(e.sessions, s.id)
	delete(s.lm.sessions, s.id)
	e.mu.Unlock()
	if s.closed.Swap(true) {
		return
	}
	s.cancelGeneration()
	s.mu.Lock()
	if s.parkView != nil {
		s.parkView.Release()
		s.parkView = nil
	}
	ok := e.pool(s.lm, s.st)
	s.mu.Unlock()
	if !ok {
		s.lm.ttft.statesClosed.Add(1)
		s.st.Close()
	}
	e.applyPageBudget(s.lm, nil)
}

// pool resets st and keeps it for the next request, reporting false when the
// pool is full or the model is gone (the caller closes it then). A State
// whose blocks a placement call moved is not kept: the next request must get
// the model's default placement, as a fresh State would.
func (e *Engine) pool(lm *LoadedModel, st *model.State) bool {
	// A parked State is not pooled: its history is in a store, not reset.
	if lm.ttft.poolCap == 0 || st.Parked() || !stepsJointly(st) && lm.loop != nil {
		return false
	}
	st.SetPrefillInterrupt(nil)
	st.Reset()
	// The store is attached per request (useStore); an idle State holds none.
	st.SetKVStore(nil)
	st.SetCacheKey("")
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.models[lm.id] != lm || len(lm.ttft.idle) >= lm.ttft.poolCap {
		return false
	}
	lm.ttft.idle = append(lm.ttft.idle, st)
	return true
}

// closeIdle closes every pooled State and the memory cache. The model is
// already unpublished, so nothing returns a State to the pool after it.
func (e *Engine) closeIdle(lm *LoadedModel) {
	e.dropIdle(lm)
	if lm.ttft.store != nil {
		lm.ttft.store.Close()
	}
}

// dropIdle closes every pooled State. A pooled State is reset and holds no
// sequence, but it is still a session of the tier on every block it was
// placed on, and the tier moves a block between devices only when no other
// session is on it (its history would be freed with the source's copy): left
// in the pool, the warm-up's State alone kept every move from running. The
// pool fills again from the next model_id requests, placed where the blocks
// are then.
func (e *Engine) dropIdle(lm *LoadedModel) {
	e.mu.Lock()
	idle := lm.ttft.idle
	lm.ttft.idle = nil
	e.mu.Unlock()
	for _, st := range idle {
		lm.ttft.statesClosed.Add(1)
		st.Close()
	}
}

// idleKV is the history the pooled States hold. e.mu held.
func (lm *LoadedModel) idleKV() uint64 {
	var kv uint64
	for _, st := range lm.ttft.idle {
		kv += st.KVBytes()
	}
	return kv
}

// useStore says whether this generate prefills through the model's prompt
// store: every generate that starts its sequence, unless the session carries
// a store of its own. A row of the step loop restores first and runs the rest
// of its prompt as a row (generateBatched), so the store costs the batch
// nothing.
func (e *Engine) useStore(s *Session, o GenerateOptions) bool {
	return s.lm.ttft.store != nil && !o.Continue && !s.cached
}

// attachStore points s's State at the model's store for one generate, or
// takes it away. The State is at position 0 (the caller reset it).
func (s *Session) attachStore(on bool) error {
	if !on {
		if s.storeOn {
			s.st.SetKVStore(nil)
			if err := s.st.SetCacheKey(""); err != nil {
				return err
			}
			s.storeOn = false
		}
		return nil
	}
	s.st.SetKVStore(s.lm.ttft.kv)
	if err := s.st.SetCacheKey(s.lm.ttft.ns); err != nil {
		return err
	}
	s.storeOn = true
	return nil
}

// warm runs what a first request would on a pooled State: a short prefill and
// one decode step, so the JIT's kernels for those shapes are emitted and the
// device's first recordings made before a request waits on them. Nothing is
// counted as served and nothing goes into the memory cache.
func (e *Engine) warm(lm *LoadedModel) error {
	if lm.m.IsEncoder() || lm.m.Vocab == nil {
		return nil
	}
	s, err := e.CreateSession(SessionOptions{ModelID: lm.id})
	if err != nil {
		return err
	}
	defer e.releaseEphemeral(s)
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := lm.m.Vocab.Encode("Hello", true)
	if len(ids) == 0 {
		return nil
	}
	start := time.Now()
	lg, err := s.st.Prefill(ids)
	if err != nil {
		return fmt.Errorf("server: warming %q: %w", lm.id, err)
	}
	if _, err := s.st.Forward(model.Greedy(lg)); err != nil {
		return fmt.Errorf("server: warming %q: %w", lm.id, err)
	}
	lm.ttft.warmed.Store(true)
	lm.warmTook = time.Since(start)
	return nil
}

// layered is the memory cache over the server's prompt store
// (Config.PromptStore, jitllmd's -kv-cache directory): a page is looked for in
// memory first and on disk second, and a disk hit is copied up so the next
// restore of it is a memory one; a page offered is kept in both, so the disk
// keeps what the memory cache evicts. Drop forgets both.
//
// A model_id request's namespace names its load (initTTFT), so what it leaves
// on disk serves this process only; a session created with prompt_cache names
// its own, and its pages outlive the process as the disk store's always did.
type layered struct {
	mem  *model.MemStore
	disk model.KVStore
}

func (l *layered) Get(cacheId string, layer, index int, page io.Writer) error {
	err := l.mem.Get(cacheId, layer, index, page)
	if !errors.Is(err, model.ErrNoPage) {
		return err
	}
	var b bytes.Buffer
	if err := l.disk.Get(cacheId, layer, index, &b); err != nil {
		return err
	}
	if err := l.mem.Set(cacheId, layer, index, bytes.NewReader(b.Bytes())); err != nil {
		return err
	}
	_, err = page.Write(b.Bytes())
	return err
}

func (l *layered) Set(cacheId string, layer, index int, page io.Reader) error {
	b, err := io.ReadAll(page)
	if err != nil {
		return err
	}
	if err := l.mem.Set(cacheId, layer, index, bytes.NewReader(b)); err != nil {
		return err
	}
	return l.disk.Set(cacheId, layer, index, bytes.NewReader(b))
}

func (l *layered) Drop(cacheId string) error {
	if err := l.mem.Drop(cacheId); err != nil {
		return err
	}
	return l.disk.Drop(cacheId)
}
