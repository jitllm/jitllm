// Package play connects the game to a decision model: the questions a frame
// opens go to the model as one request, off the render thread, and its
// answers steer the actors when they reach their junctions. What jevman does
// with its scheduler, against a model in this process instead of a server.
package play

import (
	"slices"
	"sync"
	"time"

	"github.com/jitllm/jitllm/examples/jivman/brain"
	"github.com/jitllm/jitllm/examples/jivman/game"
)

// Decision is one answered question, as the panel shows it.
type Decision struct {
	Key    game.Key
	Actor  int
	Escape bool
	Choice game.Dir
	// Probs is per direction, -1 for a direction that was not an option.
	Probs      [4]float64
	Confidence float64
	Latency    time.Duration
	// Fallback is the greedy rule's answer, given when the model did not
	// answer in time or failed.
	Fallback bool
	When     time.Time
}

// Stats is a run's decisions so far.
type Stats struct {
	Calls     int
	Decisions int
	Fallbacks int
	// Latencies is each call's model time.
	Latencies []time.Duration
	// Tokens is the prompt tokens the model read, over every call.
	Tokens int
	Last   [game.NActors]Decision
	Err    error
}

// Mean is the mean call latency.
func (s *Stats) Mean() time.Duration {
	if len(s.Latencies) == 0 {
		return 0
	}
	var t time.Duration
	for _, l := range s.Latencies {
		t += l
	}
	return t / time.Duration(len(s.Latencies))
}

// P50 is the median call latency.
func (s *Stats) P50() time.Duration {
	if len(s.Latencies) == 0 {
		return 0
	}
	c := slices.Clone(s.Latencies)
	slices.Sort(c)
	return c[len(c)/2]
}

func (s *Stats) record(p *game.Point, a *brain.Answer, took time.Duration) game.Dir {
	d := Decision{Key: p.Key, Actor: p.Actor, Escape: p.Escape, Choice: game.DirOf(a.Choice), Confidence: a.Confidence,
		Latency: took, When: time.Now(), Probs: [4]float64{-1, -1, -1, -1}}
	for i, k := range a.Options {
		if dir := game.DirOf(k); dir != game.NoDir && i < len(a.Probs) {
			d.Probs[dir] = a.Probs[i]
		}
	}
	if !p.Offers(d.Choice) {
		return game.NoDir
	}
	s.Decisions++
	s.Last[p.Actor] = d
	return d.Choice
}

func (s *Stats) fallback(g *game.Game, p *game.Point) game.Dir {
	c := g.Greedy(p, g.Options(p))
	s.Decisions++
	s.Fallbacks++
	s.Last[p.Actor] = Decision{Key: p.Key, Actor: p.Actor, Escape: p.Escape, Choice: c, Fallback: true, When: time.Now(),
		Probs: [4]float64{-1, -1, -1, -1}}
	return c
}

// Sync asks the model each question as it comes and waits for the answer:
// the headless player, where the game's clock stops while the model thinks,
// so a run measures the model's play and not the machine's speed.
type Sync struct {
	B *brain.Brain
	Stats
	answers map[game.Key]game.Dir
	buf     []byte
	ans     []brain.Answer
	ps      []game.Point
}

// Decide implements game.Controls.
func (s *Sync) Decide(g *game.Game, p *game.Point) game.Dir {
	if s.answers == nil {
		s.answers = map[game.Key]game.Dir{}
	}
	if d, ok := s.answers[p.Key]; ok {
		return d
	}
	if len(s.answers) > 1024 {
		clear(s.answers)
	}
	s.ps = append(s.ps[:0], *p)
	s.buf = g.Request(s.buf[:0], s.ps)
	var call brain.Call
	var err error
	s.ans, call, err = s.B.Decide(s.buf, s.ans)
	took := call.Took
	s.Calls++
	s.Tokens += call.Tokens
	s.Latencies = append(s.Latencies, took)
	d := game.NoDir
	if err == nil && len(s.ans) == 1 {
		d = s.record(p, &s.ans[0], took)
	} else if err != nil {
		s.Err = err
	}
	if d == game.NoDir {
		d = s.fallback(g, p)
	}
	s.answers[p.Key] = d
	return d
}

// Async is the window's player: Frame sends the open questions as one
// request whenever the model is free, a goroutine runs it, and Decide answers
// from what has come back. An actor at a junction whose answer is not back
// waits there; after Timeout it takes the greedy rule's way (0: never).
type Async struct {
	B       *brain.Brain
	Timeout time.Duration
	// Actors is a bit per actor id this model plays.
	Actors uint8
	Stats

	answers map[game.Key]game.Dir
	asked   map[game.Key]time.Time
	ps      []game.Point
	buf     []byte

	jobs chan job
	mu   sync.Mutex
	done []result
	busy bool
}

type job struct {
	body []byte
	ps   []game.Point
}

type result struct {
	ps   []game.Point
	ans  []brain.Answer
	call brain.Call
	err  error
}

// NewAsync starts the model's goroutine; Close stops it.
func NewAsync(b *brain.Brain, timeout time.Duration, actors uint8) *Async {
	a := &Async{B: b, Timeout: timeout, Actors: actors, answers: map[game.Key]game.Dir{}, asked: map[game.Key]time.Time{}, jobs: make(chan job, 1)}
	go a.run()
	return a
}

func (a *Async) run() {
	for j := range a.jobs {
		ans, call, err := a.B.Decide(j.body, nil)
		a.mu.Lock()
		a.done = append(a.done, result{ps: j.ps, ans: ans, call: call, err: err})
		a.mu.Unlock()
	}
}

// Close stops the goroutine once its call returns.
func (a *Async) Close() { close(a.jobs) }

// Busy says a request is with the model.
func (a *Async) Busy() bool { return a.busy }

// Frame takes back what the model answered and, when it is free, asks the
// questions now open. Call it on the game's thread once a frame.
func (a *Async) Frame(g *game.Game) {
	a.mu.Lock()
	done := a.done
	a.done = nil
	a.mu.Unlock()
	for _, r := range done {
		a.busy = false
		a.Calls++
		a.Latencies = append(a.Latencies, r.call.Took)
		a.Tokens += r.call.Tokens
		if r.err != nil {
			// A failed call is not retried: its questions take the greedy
			// rule's answer, as jevman's do on an error.
			a.Err = r.err
			for i := range r.ps {
				a.answers[r.ps[i].Key] = a.fallback(g, &r.ps[i])
			}
			continue
		}
		for i := range r.ps {
			p := &r.ps[i]
			for j := range r.ans {
				if r.ans[j].ID != p.QuestionID() {
					continue
				}
				if _, ok := a.answers[p.Key]; ok {
					break
				}
				if d := a.record(p, &r.ans[j], r.call.Took); d != game.NoDir {
					a.answers[p.Key] = d
				}
			}
		}
	}
	if a.busy || g.Status != game.Playing {
		return
	}
	if len(a.answers) > 1024 {
		clear(a.answers)
		clear(a.asked)
	}
	a.ps = a.ps[:0]
	for id := 0; id < game.NActors; id++ {
		if a.Actors&(1<<id) != 0 {
			a.ps = g.Questions(id, a.ps)
		}
	}
	open := a.ps[:0]
	for _, p := range a.ps {
		if _, ok := a.answers[p.Key]; ok {
			continue
		}
		if t, ok := a.asked[p.Key]; ok && time.Since(t) < 3*a.timeout() {
			continue
		}
		open = append(open, p)
	}
	if len(open) == 0 {
		return
	}
	now := time.Now()
	for _, p := range open {
		if _, ok := a.asked[p.Key]; !ok {
			a.asked[p.Key] = now
		}
	}
	// The body and the points go to the goroutine; they are its own copies.
	a.buf = g.Request(a.buf[:0], open)
	a.busy = true
	a.jobs <- job{body: slices.Clone(a.buf), ps: slices.Clone(open)}
}

func (a *Async) timeout() time.Duration {
	if a.Timeout <= 0 {
		return time.Hour
	}
	return a.Timeout
}

// Decide implements game.Controls.
func (a *Async) Decide(g *game.Game, p *game.Point) game.Dir {
	if d, ok := a.answers[p.Key]; ok {
		return d
	}
	if p.Escape {
		// Pac-Man never stops for an escape question.
		return game.NoDir
	}
	if t, ok := a.asked[p.Key]; ok && a.Timeout > 0 && time.Since(t) > a.Timeout {
		d := a.fallback(g, p)
		a.answers[p.Key] = d
		return d
	}
	return game.NoDir
}

// Waiting says actor id is stopped for its answer.
func (a *Async) Waiting(g *game.Game, id int) bool {
	return g.Status == game.Playing && g.Driven(id) && g.Actors[id].Waiting
}
