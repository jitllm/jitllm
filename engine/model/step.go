package model

import (
	"errors"
	"fmt"
	"slices"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit"
)

// Step decodes one token for each instance at once, tokens[i] for states[i], and
// returns each session's logits, which stay valid until that instance's next step.
// It is StepRuns with every run one token long and every run's logits wanted.
// The returned slice is states[0]'s, as StepRuns' is its first run's.
func Step(states []*Instance, tokens []int32) ([][]float32, error) {
	if len(states) != len(tokens) {
		return nil, fmt.Errorf("model: %d states and %d tokens", len(states), len(tokens))
	}
	if len(states) == 0 {
		return StepRuns(nil)
	}
	s0 := states[0]
	s0.stepRuns = slices.Grow(s0.stepRuns[:0], len(states))[:len(states)]
	runs := s0.stepRuns
	for i, st := range states {
		runs[i] = Run{Instance: st, Tokens: tokens[i : i+1], Logits: true}
	}
	out, err := StepRuns(runs)
	clear(runs)
	return out, err
}

// Run is one instance's part of a step across sessions (StepRuns): Tokens are
// the instance's next len(Tokens) positions -- one token for a session that is
// decoding, a chunk of its prompt for one being admitted -- and Logits says
// whether the caller wants the logits after the last of them. A decoding
// session does; a prompt chunk that does not end its prompt does not, and its
// rows then cost no projection of the head.
//
// A run may carry a picture instead, or as well: Encode is a picture the
// Instance encodes in its vision segment in this step -- part of the tower or all
// of it -- and Spans, in place of Tokens, is a chunk of a mixed prompt (tokens
// and a picture's rows) the instance prefills in this step.
//
// A run names its session (Session), which holds a lease on the instance
// the step runs on (Session.Acquire); every session of one StepRuns leases
// the same instance, and StepRuns sets the run's instance to it. A run of an
// instance's own session names the instance (instance) instead.
type Run struct {
	Instance   *Instance
	Session *Session
	Tokens  []int32
	Spans   []Span
	Logits  bool
	Encode  *Encode
}

// Encode is a picture encoded in the steps of StepRuns: its instance's vision
// segment runs the tower's blocks over the picture's rows in the same step the
// other sessions decode in, and an encode may be spread over several steps by
// blocks -- never by rows, which attend to each other.
type Encode struct {
	// Picture is the picture to encode (Instance.Picture).
	Picture *Picture
	// Blocks is the most tower blocks one step runs; zero runs the rest.
	Blocks int
	// Image is the picture's rows, grid and name once Done.
	Image Image
	Done  bool
	// Ran is the tower blocks this encode has run over every step: zero for
	// a picture the image cache answered.
	Ran int

	begun       bool
	tile, tiles int
	rows        []float32
}

// step advances the encode by one step's blocks on st's vision segment.
func (e *Encode) step(st *Instance) error {
	if e.Done {
		return nil
	}
	vs, err := st.Vision()
	if err != nil {
		return err
	}
	defer vs.m.enterPager()()
	v, p := vs.vis, e.Picture
	c := v.t.Cfg
	clean := v.fault == towerFaultNone && c.fault == towerFaultNone && !qwenNoUnwindow
	if !e.begun {
		if clean {
			if hit, ok := vs.m.imgCache.get(p.Key); ok {
				v.cacheHits++
				e.Image, e.Done = p.image(append([]float32(nil), hit.rows...), c.ProjDim), true
				return nil
			}
		}
		one := p.GH * c.PatchSz * p.GW * c.PatchSz * 3
		e.tiles = len(p.Pixels) / one
		e.begun, e.tile, e.rows = true, 0, e.rows[:0]
		if err := e.begin(vs, one); err != nil {
			return err
		}
	}
	before := v.hostBlocks + v.devBlocks
	done, err := vs.encodeBlocks(e.Blocks)
	e.Ran += int(v.hostBlocks + v.devBlocks - before)
	if err != nil {
		v.prog = encodeProg{}
		return err
	}
	if !done {
		return nil
	}
	rows, err := vs.encodeFinish()
	if err != nil {
		return err
	}
	e.rows = append(e.rows, rows...)
	if e.tile++; e.tile < e.tiles {
		return e.begin(vs, p.GH*c.PatchSz*p.GW*c.PatchSz*3)
	}
	if clean {
		vs.m.imgCache.put(p.Key, e.rows, p.Grid)
	}
	e.Image, e.Done = p.image(append([]float32(nil), e.rows...), c.ProjDim), true
	return nil
}

// begin starts the encode's current pass: the whole picture for every tower
// but a tiling one, whose squares are each their own pass.
func (e *Encode) begin(vs *Instance, one int) error {
	p := e.Picture
	return vs.encodeBegin(p.Pixels[e.tile*one:(e.tile+1)*one], p.GH, p.GW)
}

// MaxStepRows is the most rows -- every run's tokens together -- one joint
// step carries; a StepRuns over more goes one instance after another.
const MaxStepRows = nn.MaxDevicePrefillChunk

// StepRuns runs one step for several instances, each taking its run of tokens at
// its own next positions, and returns, run by run, the session's logits after
// the run's last token where the run wants them and nil where it does not. The
// logits stay valid until that instance's next step. Each session's position moves
// on by its run's length.
//
// instances are separate sessions -- separate users -- and by themselves they
// take the device one after another, each reading every block's weights for
// its own tokens. When they share a model and a device that holds every block
// and the head, the runs go as rows of one ragged step (nn.SessionStepper):
// the weights are read once for every row, each row's history -- its attention
// pages, and a hybrid's recurrent state -- stays its own instance's, and a
// session's chunk sees its own earlier rows causally, as its prefill would.
// So a server can admit a new session's prompt in chunks inside the steps the
// other sessions decode in, and none of them waits for the whole prompt.
//
// A picture is work of the same step: a run's Encode runs its instance's vision
// segment -- the tower's blocks over the picture's rows, which are different
// weights from the text rows' and so their own submission -- and a run's Spans
// prefill a mixed chunk, before the token runs step together. Each answer is
// the one the run would have had alone.
//
// The slice StepRuns returns is its first run's instance's and is reused by the
// next StepRuns that instance leads; the logits in it are each instance's own, as
// above. A step allocates nothing once its sessions are warm.
//
// An instance appears in at most one run, and every run has at least one token, a
// span or a picture. Anything that cannot step jointly -- one instance, an instance
// on the host, a batched instance, more than MaxStepRows rows -- runs one instance
// after another (Prefill for a chunk, Forward for one token), which is the
// same answer.
func StepRuns(runs []Run) ([][]float32, error) {
	return stepRuns(runs, false)
}

// StepJoint is StepRuns for runs a caller grouped itself -- the server's
// decode groups: a lone Steppable session's tokens go through the joint arm too,
// as a step of one session, where StepRuns runs it on the one-at-a-time arm.
// A row then takes the same step in a group of one as among others.
func StepJoint(runs []Run) ([][]float32, error) {
	return stepRuns(runs, true)
}

func stepRuns(runs []Run, lone bool) ([][]float32, error) {
	in, err := leasedOn(runs)
	if err != nil {
		return nil, err
	}
	if in != nil {
		in.mu.Lock()
		defer in.mu.Unlock()
		for i := range runs {
			if err := in.use(runs[i].Session); err != nil {
				return nil, err
			}
			runs[i].Instance = in
		}
		if err := in.use(runs[0].Session); err != nil {
			return nil, err
		}
	}
	return stepRunsOn(runs, lone)
}

// leasedOn is the instance the runs' sessions lease, nil for runs of
// instances' own sessions.
func leasedOn(runs []Run) (*Instance, error) {
	var in *Instance
	for i, r := range runs {
		if r.Session == nil {
			if in != nil || i > 0 && runs[0].Session != nil {
				return nil, fmt.Errorf("model: run %d names no session beside runs that do", i)
			}
			continue
		}
		if i > 0 && runs[0].Session == nil {
			return nil, fmt.Errorf("model: run %d names a session beside runs that do not", i)
		}
		l := r.Session.lease
		if l == nil {
			return nil, fmt.Errorf("model: run %d's session holds no lease (Session.Acquire)", i)
		}
		if in != nil && l.Instance != in {
			return nil, fmt.Errorf("model: run %d's session leases another instance: a step runs on one", i)
		}
		in = l.Instance
	}
	return in, nil
}

// sess is the session whose history the run reads and writes.
func (r *Run) sess() *Session {
	if r.Session != nil {
		return r.Session
	}
	return r.Instance.Session
}

// full is the error a run past its session's room is.
func (r *Run) full() error {
	if r.Session != nil {
		return errFull{r.Session.limit, r.Session.limit}
	}
	return errFull{r.Instance.maxSeq, r.Instance.reqSeq}
}

// stepRunsOn is StepRuns with every run's instance its instance and every
// session where the instance places its blocks.
func stepRunsOn(runs []Run, lone bool) ([][]float32, error) {
	rows, pictures := 0, false
	for i := range runs {
		r := &runs[i]
		if r.Instance == nil || (len(r.Tokens) == 0 && r.Spans == nil && r.Encode == nil) {
			return nil, fmt.Errorf("model: run %d has no instance, or no tokens, spans or picture", i)
		}
		for j := range runs[:i] {
			if runs[j].sess() == r.sess() {
				return nil, fmt.Errorf("model: run %d's session is in an earlier run: a session takes one run a step", i)
			}
		}
		if r.Spans != nil && len(r.Tokens) > 0 {
			return nil, fmt.Errorf("model: run %d carries tokens and spans; a chunk is one or the other", i)
		}
		rows += len(r.Tokens)
		pictures = pictures || r.Spans != nil || r.Encode != nil
	}
	if pictures {
		return stepPictures(runs)
	}
	if ss, ok := stepper(runs, rows); ok {
		return stepTogether(runs, rows, ss)
	}
	if lone && len(runs) == 1 && rows <= MaxStepRows && runs[0].Instance.Steppable() {
		return stepTogether(runs, rows, runs[0].Instance.ldStep)
	}
	if hostStepper(runs, rows) {
		return stepHost(runs, rows)
	}
	if len(runs) == 0 {
		return nil, nil
	}
	out := runs[0].Instance.stepOut(len(runs))
	for i, r := range runs {
		var lg []float32
		var err error
		if r.Session != nil {
			if err := r.Instance.use(r.Session); err != nil {
				return nil, err
			}
		}
		if len(r.Tokens) == 1 {
			lg, err = r.Instance.Forward(r.Tokens[0])
		} else {
			lg, err = r.Instance.Prefill(r.Tokens)
		}
		if err != nil {
			return nil, err
		}
		if r.Logits {
			out[i] = lg
		}
	}
	return out, nil
}

// stepPictures is StepRuns for a step that holds pictures: each run's encode
// advances by its blocks, each mixed chunk prefills, and the token runs then
// step as StepRuns steps them.
func stepPictures(runs []Run) ([][]float32, error) {
	out := make([][]float32, len(runs))
	var toks []Run
	var at []int
	for i, r := range runs {
		if r.Session != nil {
			if err := r.Instance.use(r.Session); err != nil {
				return nil, err
			}
		}
		if r.Encode != nil {
			if err := r.Encode.step(r.Instance); err != nil {
				return nil, err
			}
		}
		switch {
		case r.Spans != nil:
			lg, err := r.Instance.PrefillMixed(r.Spans...)
			if err != nil {
				return nil, err
			}
			if r.Logits {
				out[i] = lg
			}
		case len(r.Tokens) > 0:
			toks = append(toks, Run{Instance: r.Instance, Session: r.Session, Tokens: r.Tokens, Logits: r.Logits})
			at = append(at, i)
		}
	}
	if len(toks) == 0 {
		return out, nil
	}
	lg, err := stepRunsOn(toks, false)
	if err != nil {
		return nil, err
	}
	for j, i := range at {
		out[i] = lg[j]
	}
	return out, nil
}

// stepper is the device that can take the runs as one step, when there is
// one.
func stepper(runs []Run, rows int) (nn.SessionStepper, bool) {
	if len(runs) < 2 || rows > MaxStepRows {
		return nil, false
	}
	s0 := runs[0].Instance
	for _, r := range runs {
		if st := r.Instance; !st.Steppable() || st.m != s0.m || st.ldCand != s0.ldCand {
			return nil, false
		}
	}
	return s0.ldStep, true
}

// Steppable reports whether this instance can take a run of StepRuns' joint arm:
// a single sequence whose every block and head run on a device that steps rows
// of different sessions together. Steppable instances of one Model on one device
// step as one; StepRuns runs any other instance alone. A caller batching sessions
// asks it per instance, so one instance placed differently does not send the whole
// step down the one-at-a-time arm. StepRefusal says why an instance is not.
func (s *Instance) Steppable() bool { return s.StepRefusal() == nil }

// StepRefusal is why this instance cannot take a run of StepRuns' joint arm, nil
// when it can (Steppable). StepRuns runs a refused instance alone, which is the
// same answer at a pass over the weights per session; this names which
// condition put it there, so a model that never steps as rows says so rather
// than reading as one that does. It allocates nothing: a step asks it of every
// Instance.
func (s *Instance) StepRefusal() error {
	switch {
	case s.ldStep == nil:
		return errStepNoDevice
	case s.batched:
		return errStepBatched
	case s.emb != nil:
		return errStepEmbedder
	}
	return s.rowsRefusal()
}

// The conditions StepRefusal names.
var (
	errStepNoDevice = errors.New("model: the instance's device does not step rows of different sessions (no nn.SessionStepper)")
	errStepBatched  = errors.New("model: a batched instance steps its own rows, not other sessions'")
	errStepEmbedder = errors.New("model: an embedder runs no decode step")
	errStepNoRows   = errors.New("model: the instance's device runs no ragged rows (no nn.RowsDevice)")
	errStepNoHead   = errors.New("model: the head is not on a device")
	errStepSplit    = errors.New("model: not every block of the instance is on the device")
)

// hostStepper reports whether the runs can go as one host step across their
// sessions (stepHost): two or more instances of one model, each HostSteppable,
// the rows within the host's widest batch.
func hostStepper(runs []Run, rows int) bool {
	if len(runs) < 2 || rows > MaxPrefillChunk {
		return false
	}
	s0 := runs[0].Instance
	for _, r := range runs {
		if !r.Instance.HostSteppable() || r.Instance.m != s0.m {
			return false
		}
	}
	return true
}

// HostSteppable reports whether this instance can take a run of StepRuns' host
// arm: a single sequence whose every block and head run on the host. Such
// instances of one Model step as rows of one pass over the weights (stepHost),
// each row reading and writing its own session's history and recurrent state;
// StepRuns runs any other instance alone, which is the same answer. HostRefusal
// says why an instance is not. It allocates nothing.
func (s *Instance) HostSteppable() bool { return s.HostRefusal() == nil }

// HostRefusal is why this instance cannot take a run of StepRuns' host arm, nil
// when it can (HostSteppable).
func (s *Instance) HostRefusal() error {
	c := s.c
	switch {
	case s.batched:
		return errStepBatched
	case s.emb != nil:
		return errStepEmbedder
	case s.devCount() > 0 || s.head != nil:
		return errHostOnDevice
	case s.lo != 0 || s.hi != c.NLayer:
		return errHostPartial
	case c.MLA() || c.Indexer() || c.MSA() || c.DSV4():
		// Their cached rows and selections are read through the batch's own
		// slots (mlaProjectRows, idxRows, msaRows, ds4Block); a row of
		// another session there would read the wrong history.
		return errHostCache
	}
	return nil
}

// The conditions HostRefusal names beside StepRefusal's.
var (
	errHostOnDevice = errors.New("model: a device runs some of the instance's blocks or its head")
	errHostPartial  = errors.New("model: the instance runs part of the model (a prediction block)")
	errHostCache    = errors.New("model: the architecture's cached rows are read per batch slot (MLA, an indexer, MSA, DeepSeek V4)")
)

// PromptChunk is how many prompt tokens this instance's Prefill takes in one
// chunk: a pipelined chunk when every block is on a device that crosses
// several cards (deviceChunk), a device chunk on a device, the host's chunk
// otherwise. A caller feeding a prompt in pieces feeds pieces this wide to
// keep the path Prefill would take on the whole prompt.
func (s *Instance) PromptChunk() int {
	if s.devCount() > 0 {
		return s.deviceChunk()
	}
	return MaxPrefillChunk
}

// HostStepRows is the rows StepRuns has run as host steps across sessions,
// which says whether sessions really shared passes over the weights.
func (m *Model) HostStepRows() int64 { return atomic.LoadInt64(&m.hostStepRows) }

// stepHost is StepRuns' host arm: the runs' rows as one ragged pass over the
// weights (rowsHost, the batch's own step), each row embedded and rotated at
// its own session's next position, its K and V written into and its attention
// and recurrent state read from that session's Instance. Each session's logits and
// position then move on as Prefill and Forward move them. The rows whose
// logits are wanted lead, so the head projects only them.
func stepHost(runs []Run, rows int) ([][]float32, error) {
	s0 := runs[0].Instance
	c := s0.c
	defer s0.m.enterPager()()
	toks, pos, own, win, seq := s0.stepTok[:0], s0.stepPos[:0], s0.rowOwn[:0], s0.rowWin[:0], s0.rowSeq[:0]
	// The lists keep their capacity; the sessions are let go of, so a step
	// pins no instance past it.
	defer func() {
		clear(own)
		s0.stepTok, s0.stepPos, s0.rowOwn, s0.rowWin, s0.rowSeq = toks[:0], pos[:0], own[:0], win[:0], seq[:0]
	}()
	addRow := func(r *Run, j int) error {
		se, tok := r.sess(), r.Tokens[j]
		p := se.bpos[0] + j
		if stepPos != nil {
			p = stepPos(se.bpos[0], j, len(r.Tokens))
		}
		if int(tok) < 0 || int(tok) >= c.NVocab {
			return errToken{tok, c.NVocab}
		}
		if p >= se.limit {
			return r.full()
		}
		if stepShare {
			se = runs[0].sess()
		}
		toks, pos, own = append(toks, tok), append(pos, p), append(own, se)
		win, seq = append(win, j == len(r.Tokens)-1), append(seq, 0)
		return nil
	}
	nlogit := 0
	for i := range runs {
		if r := &runs[i]; r.Logits {
			if err := addRow(r, len(r.Tokens)-1); err != nil {
				return nil, err
			}
			nlogit++
		}
	}
	for i := range runs {
		r := &runs[i]
		for j := range r.Tokens {
			if r.Logits && j == len(r.Tokens)-1 {
				continue
			}
			if err := addRow(r, j); err != nil {
				return nil, err
			}
		}
	}
	s0.rowOwn, s0.rowWin = own, win
	// The step's scratch is the model's, lent to whichever instance leads.
	s0.borrowStep()
	defer s0.returnStep()
	lg, err := s0.rowsHost(toks, seq, pos, nlogit, len(runs))
	if err != nil {
		return nil, err
	}
	atomic.AddInt64(&s0.m.hostStepRows, int64(rows))
	out := s0.stepOut(len(runs))
	at := 0
	for i, r := range runs {
		st, k := r.Instance, len(r.Tokens)
		st.Session = r.sess()
		st.kv.note(st.bpos[0], r.Tokens...)
		if r.Logits {
			copy(st.logits, lg[at*c.NVocab:(at+1)*c.NVocab])
			out[i] = st.logits
			at++
		}
		st.advance(0, k)
		for range k {
			st.hot.endToken()
		}
		st.reap()
		// A decoding row steps its seam tuner as stepTogether's do. A tuner
		// that moves blocks onto a card takes the instance out of the host
		// step (HostRefusal), and StepRuns runs it alone from then on.
		if k == 1 {
			st.seamStep()
		}
		if err := st.kvCheck(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// stepShare, when set, gives every row of a host step the first run's instance
// as its history. False but in a gate's violation: it is how the gate shows
// that rows reading one session's history fail it.
var stepShare bool

// stepPos, when set, is where a run's row j goes, given the instance's next
// position and the run's length. Nil but in a gate's violation: it is how the
// gate shows that a chunk laid out at the wrong positions fails it.
var stepPos func(base, j, k int) int

// stepTogether is StepRuns' joint arm: each row embedded and rotated at its
// own position, one device step, and each session's logits and position moved on
// as Prefill and Forward move them. The rows whose logits are wanted -- each
// such run's last -- lead, so the head projects only them (nn.Head.LogitRows).
func stepTogether(runs []Run, rows int, ss nn.SessionStepper) ([][]float32, error) {
	s0 := runs[0].Instance
	m, c := s0.m, s0.m.Cfg
	defer m.enterPager()()
	nrot := c.RopeW()
	s0.growBatch(rows)
	x, cs, csSWA := s0.bx[:rows*c.ResidW()], s0.bcs[:rows*nrot], s0.bswaTable(rows, c.NRotSWA)
	// The step's lists are s0's, reused: its rows' sessions and positions, the
	// wanted rows' logits and the slice handed back.
	sess, pos := s0.stepSess[:0], s0.stepPos[:0]
	s0.stepTok = s0.stepTok[:0]
	defer func() { s0.stepSess, s0.stepPos = clearDevs(sess), pos[:0] }()
	addRow := func(r *Run, j int) error {
		se, i := r.sess(), len(pos)
		tok, p := r.Tokens[j], se.bpos[0]+j
		if stepPos != nil {
			p = stepPos(se.bpos[0], j, len(r.Tokens))
		}
		if int(tok) < 0 || int(tok) >= c.NVocab {
			return errToken{tok, c.NVocab}
		}
		if p >= se.limit {
			return r.full()
		}
		s0.ropeRow(i, se.ropeOf(0, p))
		sess, pos = append(sess, se.ld), append(pos, p)
		s0.stepTok = append(s0.stepTok, tok)
		return nil
	}
	nlogit := 0
	for i := range runs {
		if r := &runs[i]; r.Logits {
			if err := addRow(r, len(r.Tokens)-1); err != nil {
				return nil, err
			}
			nlogit++
		}
	}
	for i := range runs {
		r := &runs[i]
		for j := range r.Tokens {
			if r.Logits && j == len(r.Tokens)-1 {
				continue
			}
			if err := addRow(r, j); err != nil {
				return nil, err
			}
		}
	}
	// The rows' input rows, scale, positions and per-layer inputs through the
	// composed graph (cpuembed.go), or the old engine path where it declines.
	var bple []float32
	if c.PLEDim != 0 {
		s0.growBPLE(rows)
		bple = s0.bple[:rows*c.pleWidth()]
	}
	if s0.cpuEmbed(x[:rows*c.NEmbd], bple, s0.stepTok, pos) != jit.DeclineServes {
		for i, tok := range s0.stepTok {
			row := x[i*c.NEmbd : (i+1)*c.NEmbd]
			if err := m.embedRow(row, int(tok)); err != nil {
				return nil, err
			}
			if c.EmbdScale != 1 {
				s0.scale(row, float32(c.EmbdScale))
			}
			if err := m.addPos(row, s0.prow, pos[i]); err != nil {
				return nil, err
			}
			if c.PLEDim != 0 {
				w := c.pleWidth()
				if err := s0.pleInputs(bple[i*w:(i+1)*w], tok, row); err != nil {
					return nil, err
				}
			}
		}
	}
	// AltUp's other streams, from each row's embedding (altup.go), or
	// DeepSeek V4's copies of it.
	if c.streamHead() {
		if err := s0.expandStreams(x, rows); err != nil {
			return nil, err
		}
	}
	// The head's Logits is s0's single-sequence destination; the step borrows
	// it for the wanted rows and hands it back. A step no row wants logits
	// from runs no head. AltUp's head reads the streams' mean, which the host
	// takes: the rows are read back and the host projects them.
	var h *nn.Head
	var logits []float32
	if nlogit > 0 && c.streamHead() {
		s0.stepLogits = slices.Grow(s0.stepLogits[:0], nlogit*c.NVocab)[:nlogit*c.NVocab]
		logits = s0.stepLogits
	} else if nlogit > 0 {
		h = s0.head
		s0.stepLogits = slices.Grow(s0.stepLogits[:0], nlogit*c.NVocab)[:nlogit*c.NVocab]
		logits = s0.stepLogits
		keep := h.Logits
		h.RowLogits, h.Logits, h.LogitRows = true, logits, nlogit
		defer func() { h.RowLogits, h.Logits, h.LogitRows = false, keep, 0 }()
	}
	s0.layerInputs(s0.bple[:rows*c.pleWidth()])
	s0.tokenIDs(s0.stepTok)
	s0.rowAxes(pos[0], 0, rows)
	if !ss.LayersSessions(s0.lo, s0.hi, sess, pos, x, cs, csSWA, h) {
		why := "no reason given"
		if e, ok := s0.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return nil, fmt.Errorf("model: the device refused a step of %d rows across %d sessions: %s",
			rows, len(runs), why)
	}
	headed := false
	if nlogit > 0 && c.streamHead() {
		if err := s0.collapseStreams(x, rows); err != nil {
			return nil, err
		}
		headed = s0.cpuHeadRows(s0.bx, logits, nil, nlogit) == jit.DeclineServes
		if !headed {
			s0.altHeadNorm(nlogit)
			if err := s0.mm(logits, *s0.outW, s0.bh, nlogit); err != nil {
				return nil, err
			}
		}
	}
	out := s0.stepOut(len(runs))
	at := 0
	for i, r := range runs {
		st, k := r.Instance, len(r.Tokens)
		st.Session = r.sess()
		st.kv.note(st.bpos[0], r.Tokens...)
		if r.Logits {
			copy(st.logits, logits[at*c.NVocab:(at+1)*c.NVocab])
			switch {
			case headed:
			case c.streamHead():
				st.finishLogits(st.logits)
			default:
				st.finishApplied(st.logits, h.Applied)
			}
			out[i] = st.logits
			at++
		}
		st.advance(0, k)
		for range k {
			st.hot.endToken()
		}
		st.reap()
		// A decoding row's token is complete, so it steps the seam tuner as
		// Forward does: a session stepping as a row of a server's step loop
		// would otherwise never measure. A prompt chunk steps it no more than
		// Prefill does.
		if k == 1 {
			st.seamStep()
		}
		if err := st.kvCheck(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// stepOut is n nil logits slots in s's reused result slice (StepRuns).
func (s *Instance) stepOut(n int) [][]float32 {
	s.stepRes = slices.Grow(s.stepRes[:0], n)[:n]
	clear(s.stepRes)
	return s.stepRes
}

// clearDevs empties a reused list of sessions, so it pins no device the
// Instance has let go of.
func clearDevs(ds []nn.LayerDevice) []nn.LayerDevice {
	clear(ds)
	return ds[:0]
}
