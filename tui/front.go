package main

import (
	"sync"

	"github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/common/session"
)

// field names one engine value. An engineMsg carries the fields that changed
// since the last one, so each component redraws only for what it shows.
type field uint8

const (
	fStatus field = iota
	fProblem
	fModels
	fActive
	fLoaded
	fBusy
	fStreaming
	fLoads
	fRevision
	fStream
	fRates
	fBlocks
	fAlloc
	fPager
	fSummary
	fOther
)

type fields uint32

func (fs fields) has(f ...field) bool {
	for _, x := range f {
		if fs&(1<<x) != 0 {
			return true
		}
	}
	return false
}

// engineMsg says which engine values changed. turns is set when the
// transcript itself changed (a turn written, a reasoning block folded).
type engineMsg struct {
	changed fields
	turns   bool
}

// postMsg runs fn on the loop's goroutine: the engine's Post, and a job's
// progress.
type postMsg struct{ fn func() }

// wakeMsg asks the loop to collect what changed.
type wakeMsg struct{}

// value is a session.Value behind a lock; a Set marks its field changed.
type value[T any] struct {
	mu sync.Mutex
	v  T
	f  field
	fr *front
}

func (x *value[T]) Get() T {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.v
}

func (x *value[T]) Set(v T) {
	x.mu.Lock()
	x.v = v
	x.mu.Unlock()
	x.fr.mark(x.f, false)
}

func newValue[T any](fr *front, f field, v T) *value[T] { return &value[T]{v: v, f: f, fr: fr} }

// front is the terminal's half of the engine seam. The engine calls it from
// its worker and the loop from Update, so nothing here blocks on the loop:
// changes accumulate in pending, and one wake message at a time goes through
// queue to a forwarder that hands it to the program.
type front struct {
	queue chan any
	st    engine.State

	mu      sync.Mutex
	pending fields
	turnsCh bool
	woken   bool

	turns     []session.Turn
	thinkOpen map[int]bool
	alert     string
}

func newFront() *front {
	return &front{queue: make(chan any, 256), thinkOpen: map[int]bool{}}
}

func (f *front) mark(fl field, turns bool) {
	f.mu.Lock()
	f.pending |= 1 << fl
	f.turnsCh = f.turnsCh || turns
	wake := !f.woken
	f.woken = true
	f.mu.Unlock()
	if wake {
		f.queue <- wakeMsg{}
	}
}

// collect takes what changed since the last collect.
func (f *front) collect() engineMsg {
	f.mu.Lock()
	defer f.mu.Unlock()
	m := engineMsg{changed: f.pending, turns: f.turnsCh}
	f.pending, f.turnsCh, f.woken = 0, false, false
	return m
}

func (f *front) Post(fn func())       { f.queue <- postMsg{fn} }
func (f *front) SetStatus(msg string) { f.st.Status.Set(msg) }

func (f *front) Alert(title, message string) {
	f.mu.Lock()
	f.alert = title + ": " + message
	f.mu.Unlock()
	f.mark(fProblem, false)
}

func (f *front) Report(p session.Problem) {
	p.Seq = f.st.Problem.Get().Seq + 1
	f.st.Problem.Set(p)
}

// The settings and the converter are screens here; the engine's offers say
// where to find them.
func (f *front) GoSettings() {
	f.SetStatus("the device setting does not fit this machine: see Settings (f5)")
}

func (f *front) OfferConvert(string) {
	f.SetStatus("this file needs converting first: select it on Models (f2) and press enter")
}

func (f *front) OfferReconvert(string) {
	f.SetStatus("this container is outdated: select it on Models (f2) and press enter to rebuild it")
}

func (f *front) SetTurn(i int, t session.Turn) {
	f.mu.Lock()
	if i >= 0 && i < len(f.turns) {
		f.turns[i] = t
	}
	f.mu.Unlock()
	f.mark(fRevision, true)
}

func (f *front) SetThinkOpen(i int, open bool) {
	f.mu.Lock()
	f.thinkOpen[i] = open
	f.mu.Unlock()
	f.mark(fRevision, true)
}

// appendTurn adds t to the transcript and returns its index.
func (f *front) appendTurn(t session.Turn) int {
	f.mu.Lock()
	f.turns = append(f.turns, t)
	i := len(f.turns) - 1
	f.mu.Unlock()
	f.mark(fRevision, true)
	return i
}

// setTurns makes turns the transcript, the chat being shown.
func (f *front) setTurns(turns []session.Turn) {
	f.mu.Lock()
	f.turns, f.thinkOpen, f.alert = append([]session.Turn(nil), turns...), map[int]bool{}, ""
	f.mu.Unlock()
	f.st.Stream.Set("")
	f.st.StreamThink.Set("")
	f.st.Problem.Set(session.Problem{})
	f.mark(fRevision, true)
}

func (f *front) turnCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.turns)
}

func (f *front) snapshot() ([]session.Turn, map[int]bool, string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	open := make(map[int]bool, len(f.thinkOpen))
	for k, v := range f.thinkOpen {
		open[k] = v
	}
	return append([]session.Turn(nil), f.turns...), open, f.alert
}

// newState is the engine's State over locked values, each naming the field
// a change to it marks.
func newState(fr *front, deviceSpec, placement string, maxSeq int, chat, kvCache bool) engine.State {
	return engine.State{
		Status:       newValue(fr, fStatus, ""),
		Problem:      newValue(fr, fProblem, session.Problem{}),
		ModelPath:    newValue(fr, fActive, ""),
		Models:       newValue[[]session.LoadedModel](fr, fModels, nil),
		Active:       newValue(fr, fActive, ""),
		ModelSummary: newValue(fr, fSummary, ""),
		Loaded:       newValue(fr, fLoaded, false),
		Busy:         newValue(fr, fBusy, false),
		Streaming:    newValue(fr, fStreaming, false),
		Loads:        newValue[[]session.Loading](fr, fLoads, nil),
		Revision:     newValue(fr, fRevision, 0),
		ShowThinking: newValue(fr, fRevision, false),
		Stream:       newValue(fr, fStream, ""),
		StreamThink:  newValue(fr, fStream, ""),
		Vision:       newValue(fr, fOther, false),
		Chat:         newValue(fr, fOther, chat),
		ChatCapable:  newValue(fr, fOther, false),
		PromptTokS:   newValue(fr, fRates, 0.0),
		DecodeTokS:   newValue(fr, fRates, 0.0),
		BytesPerTok:  newValue(fr, fRates, 0.0),
		GBs:          newValue(fr, fRates, 0.0),
		DeviceSpec:   newValue(fr, fOther, deviceSpec),
		Placement:    newValue(fr, fOther, placement),
		MaxSeq:       newValue(fr, fOther, maxSeq),
		DeviceReport: newValue(fr, fSummary, ""),
		BlockMap:     newValue[[]byte](fr, fBlocks, nil),
		Pager:        newValue(fr, fPager, session.PagerStat{}),
		Alloc:        newValue(fr, fAlloc, session.Allocation{}),
		KVCache:      newValue(fr, fOther, kvCache),
	}
}
