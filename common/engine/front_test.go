package engine

import (
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/common/session"
)

// The tests drive the engine through a front end with nothing on screen: a
// queue the test drains, as a window's frame loop would, and values behind a
// lock in place of the window's signals.

// val is one test Value.
type val[T any] struct {
	mu   sync.Mutex
	v    T
	subs map[int]func(T)
	next int
}

func newVal[T any](v T) *val[T] { return &val[T]{v: v} }

func (x *val[T]) Get() T {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.v
}

// Set stores v and calls every subscriber with it, as a signal's Set does.
func (x *val[T]) Set(v T) {
	x.mu.Lock()
	x.v = v
	subs := make([]func(T), 0, len(x.subs))
	for _, fn := range x.subs {
		subs = append(subs, fn)
	}
	x.mu.Unlock()
	for _, fn := range subs {
		fn(v)
	}
}

// SubscribeForever calls fn on every Set until the returned func is called.
func (x *val[T]) SubscribeForever(fn func(T)) func() {
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.subs == nil {
		x.subs = map[int]func(T){}
	}
	id := x.next
	x.next++
	x.subs[id] = fn
	return func() {
		x.mu.Lock()
		delete(x.subs, id)
		x.mu.Unlock()
	}
}

// testStore holds what the engine publishes, with the window's defaults, and
// the transcript.
type testStore struct {
	Status  *val[string]
	Problem *val[session.Problem]

	ModelPath    *val[string]
	Models       *val[[]session.LoadedModel]
	Active       *val[string]
	ModelSummary *val[string]
	Loaded       *val[bool]
	Busy         *val[bool]
	Streaming    *val[bool]
	Loads        *val[[]session.Loading]

	Revision     *val[int]
	ShowThinking *val[bool]
	Stream       *val[string]
	StreamThink  *val[string]
	Vision       *val[bool]
	Chat         *val[bool]
	ChatCapable  *val[bool]

	PromptTokS  *val[float64]
	DecodeTokS  *val[float64]
	BytesPerTok *val[float64]
	GBs         *val[float64]

	DeviceSpec   *val[string]
	Placement    *val[string]
	MaxSeq       *val[int]
	DeviceReport *val[string]
	BlockMap     *val[[]byte]
	Pager        *val[session.PagerStat]
	Alloc        *val[session.Allocation]
	KVCache      *val[bool]

	mu        sync.Mutex
	turns     []session.Turn
	thinkOpen map[int]*val[bool]
}

func newTestStore() *testStore {
	return &testStore{
		Status:       newVal("ready"),
		Problem:      newVal(session.Problem{}),
		ModelPath:    newVal(""),
		Models:       newVal([]session.LoadedModel(nil)),
		Active:       newVal(""),
		ModelSummary: newVal("no model loaded"),
		Loaded:       newVal(false),
		Busy:         newVal(false),
		Streaming:    newVal(false),
		Loads:        newVal([]session.Loading(nil)),
		Revision:     newVal(0),
		ShowThinking: newVal(false),
		Stream:       newVal(""),
		StreamThink:  newVal(""),
		Vision:       newVal(false),
		Chat:         newVal(false),
		ChatCapable:  newVal(true),
		PromptTokS:   newVal(0.0),
		DecodeTokS:   newVal(0.0),
		BytesPerTok:  newVal(0.0),
		GBs:          newVal(0.0),
		DeviceSpec:   newVal("auto"),
		Placement:    newVal(""),
		MaxSeq:       newVal(4096),
		DeviceReport: newVal(""),
		BlockMap:     newVal([]byte(nil)),
		Pager:        newVal(session.PagerStat{}),
		Alloc:        newVal(session.Allocation{}),
		KVCache:      newVal(true),
	}
}

func (s *testStore) AppendTurn(t session.Turn) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turns = append(s.turns, t)
	return len(s.turns) - 1
}

func (s *testStore) SetTurn(i int, t session.Turn) {
	s.mu.Lock()
	if i >= 0 && i < len(s.turns) {
		s.turns[i] = t
	}
	s.mu.Unlock()
}

func (s *testStore) Turn(i int) session.Turn {
	s.mu.Lock()
	defer s.mu.Unlock()
	if i < 0 || i >= len(s.turns) {
		return session.Turn{}
	}
	return s.turns[i]
}

func (s *testStore) AllTurns() []session.Turn {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]session.Turn(nil), s.turns...)
}

// ThinkOpen is turn i's reasoning disclosure, created at ShowThinking.
func (s *testStore) ThinkOpen(i int) *val[bool] {
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.thinkOpen[i]; ok {
		return v
	}
	if s.thinkOpen == nil {
		s.thinkOpen = map[int]*val[bool]{}
	}
	v := newVal(s.ShowThinking.Get())
	s.thinkOpen[i] = v
	return v
}

func (s *testStore) LoadingOf(path string) (session.Loading, bool) {
	return session.LoadingOf(s.Loads, path)
}

func (s *testStore) Loading() bool { return len(s.Loads.Get()) > 0 }

// testShell is the Front.
type testShell struct {
	Store *testStore
	q     chan func()
	seq   atomic.Uint64
}

func newTestShell() *testShell {
	return &testShell{Store: newTestStore(), q: make(chan func(), 64)}
}

var _ Front = (*testShell)(nil)

// state is the engine's view of the store.
func (sh *testShell) state() State {
	st := sh.Store
	return State{
		Status: st.Status, Problem: st.Problem,
		ModelPath: st.ModelPath, Models: st.Models, Active: st.Active,
		ModelSummary: st.ModelSummary, Loaded: st.Loaded, Busy: st.Busy,
		Streaming: st.Streaming, Loads: st.Loads,
		Revision: st.Revision, ShowThinking: st.ShowThinking, Stream: st.Stream,
		StreamThink: st.StreamThink, Vision: st.Vision, Chat: st.Chat,
		ChatCapable: st.ChatCapable,
		PromptTokS:  st.PromptTokS, DecodeTokS: st.DecodeTokS,
		BytesPerTok: st.BytesPerTok, GBs: st.GBs,
		DeviceSpec: st.DeviceSpec, Placement: st.Placement, MaxSeq: st.MaxSeq, DeviceReport: st.DeviceReport,
		BlockMap: st.BlockMap, Pager: st.Pager, Alloc: st.Alloc,
		KVCache: st.KVCache,
	}
}

// Post drops fn when the queue is full, as the window's does.
func (sh *testShell) Post(fn func()) {
	select {
	case sh.q <- fn:
	default:
	}
}

// DrainQueue runs everything posted so far.
func (sh *testShell) DrainQueue(float64) {
	for {
		select {
		case fn := <-sh.q:
			fn()
		default:
			return
		}
	}
}

func (sh *testShell) SetStatus(msg string)        { sh.Store.Status.Set(msg) }
func (sh *testShell) Alert(title, message string) { sh.SetStatus(title) }

func (sh *testShell) Report(p session.Problem) {
	p.Seq = sh.seq.Add(1)
	sh.Post(func() {
		sh.Store.Problem.Set(p)
		sh.Store.Status.Set(p.Title)
	})
}

func (sh *testShell) GoSettings()                   {}
func (sh *testShell) OfferConvert(string)           {}
func (sh *testShell) OfferReconvert(string)         {}
func (sh *testShell) SetTurn(i int, t session.Turn) { sh.Store.SetTurn(i, t) }
func (sh *testShell) SetThinkOpen(i int, open bool) { sh.Store.ThinkOpen(i).Set(open) }
