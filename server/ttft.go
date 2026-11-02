package server

import (
	"bytes"
	"context"
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
//     (Instance.PrefillCached), so a chat's second turn or a system prompt
//     another request already ran restores its prefix instead of computing
//     it. The store is a share of the model's host budget, taken out of
//     what the pager may hold (applyPageBudget).
//   - The instance pool. A model_id request ran on an instance built for it and
//     closed after it; Close quiesces the device and retires its recordings,
//     and a fresh instance grows its batch scratch again. A finished request's
//     instance is reset and kept for the next, up to a bound.
//   - Warm-up at load (LoadOptions.Warm): a short prefill and a decode step
//     on a pooled instance, so the first request does not pay the JIT's
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

// MemCacheStats is a model's memory cache and its sessions, as counted.
type MemCacheStats struct {
	Store model.MemStoreStats
	// Restored and Computed are prompt positions taken from the store and
	// prefilled, over every generate that started its sequence.
	Restored, Computed int64
	// SessionsOpened and SessionsClosed count the model.Sessions opened on
	// the model's instance and closed.
	SessionsOpened, SessionsClosed int64
	// Warmed says a warm-up ran at load.
	Warmed bool
	// Refused counts requests turned away with ErrOverloaded.
	Refused int64
}

// ttft is a LoadedModel's share of the above.
type ttft struct {
	store *model.MemStore // nil with Config.NoMemCache
	// shared is the model's pages held once (model.SharedStore), which its
	// Deciders share through: nil with Config.NoMemCache.
	shared *model.SharedStore
	// kv is what an instance is pointed at: the memory cache, or the memory cache
	// over Config.PromptStore (layered) when the server has one.
	kv model.KVStore
	ns string

	inflight atomic.Int32
	queueCap int

	restored, computed             atomic.Int64
	sessionsOpened, sessionsClosed atomic.Int64
	refused                        atomic.Int64
	warmed                         atomic.Bool
}

// initTTFT sets lm's memory cache and queue bound from the config.
func (e *Engine) initTTFT(lm *LoadedModel, o LoadOptions) {
	t := &lm.ttft
	if !e.cfg.NoMemCache && !lm.m.IsEncoder() {
		t.store = model.NewBoundedMemStore(1)
		t.shared = model.NewSharedStore()
		// The namespace names this load: the store is the model's own, and
		// the KV width and page geometry are fixed for its life.
		t.ns = fmt.Sprintf("jitllmd/%s/%d", lm.id, lm.loadedAt.UnixNano())
		t.kv = t.store
		if e.cfg.PromptStore != nil {
			t.kv = &layered{mem: t.store, disk: e.cfg.PromptStore}
		}
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
	return MemCacheStats{
		Store:          st,
		Restored:       t.restored.Load(),
		Computed:       t.computed.Load(),
		SessionsOpened: t.sessionsOpened.Load(),
		SessionsClosed: t.sessionsClosed.Load(),
		Warmed:         t.warmed.Load(),
		Refused:        t.refused.Load(),
	}
}

// releaseEphemeral ends a model_id request's session: unregistered as
// CloseSession does, and its instance reset into the pool when there is room
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
	s.ms.Close()
	s.mu.Unlock()
	s.lm.ttft.sessionsClosed.Add(1)
	e.applyPageBudget(s.lm, nil)
}

// closeIdle closes the memory cache. The model is already unpublished.
func (e *Engine) closeIdle(lm *LoadedModel) {
	if lm.ttft.store != nil {
		lm.ttft.store.Close()
	}
}

// useStore says whether this generate prefills through the model's prompt
// store: every generate that starts its sequence, unless the session carries
// a store of its own. A row of the step loop restores first and runs the rest
// of its prompt as a row (generateBatched), so the store costs the batch
// nothing.
func (e *Engine) useStore(s *Session, o GenerateOptions) bool {
	return s.lm.ttft.store != nil && !o.Continue && !s.cached
}

// attachStore points s's instance at the model's store for one generate, or
// takes it away. The instance is at position 0 (the caller reset it).
func (s *Session) attachStore(on bool) error {
	if !on {
		if s.storeOn {
			s.ms.SetKVStore(nil)
			if err := s.ms.SetCacheKey(""); err != nil {
				return err
			}
			s.storeOn = false
		}
		return nil
	}
	s.ms.SetKVStore(s.lm.ttft.kv)
	if err := s.ms.SetCacheKey(s.lm.ttft.ns); err != nil {
		return err
	}
	s.storeOn = true
	return nil
}

// warm runs what a first request would on a pooled instance: a short prefill and
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
	ctx := context.Background()
	var next int32
	if err := s.ms.Run(ctx, model.PhasePrefill, func(in *model.Instance) error {
		lg, err := in.Prefill(ids)
		if err == nil {
			next = in.Sample(nil, lg)
		}
		return err
	}); err != nil {
		return fmt.Errorf("server: warming %q: %w", lm.id, err)
	}
	if err := s.ms.Run(ctx, model.PhaseGenerate, func(in *model.Instance) error {
		_, err := in.Forward(next)
		return err
	}); err != nil {
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
