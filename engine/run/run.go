// Package run executes the composer's plan of a model on one device: the
// engine's side of the buffer plan (docs/design/composer-plan.md, step 7).
//
// A Runner is the plan composed and compiled once, at a capacity, with what
// every sequence shares uploaded once -- the container's weights and norms,
// the plan's word blocks, the model's rotary planes. A Session is one
// sequence's buffers: its history pools and page table, the step's tables,
// its input row and the step's own values and scratch. Each Step writes the
// step's tables and input, launches every kernel in order and reads back the
// logits and the token.
package run

import (
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// Source is a model the runner composes and fills: the composer's input,
// and the container's bytes behind each entry.
type Source interface {
	jit.Model
	// ActWindow is how many activations share one int8 scale: the
	// engine's decision for this model (Model.ActWindow).
	ActWindow() int
	Span(e *jlm.Entry) (qs, d, sc []byte)
	EnsurePage(i int) error
	BlockOf(e *jlm.Entry) int
}

// Runner is a model's plan on one device.
type Runner struct {
	f      Source
	dev    backend.Device
	step   jit.Step
	plan   *jit.Plan
	planes []float32
	guard  int
	reg    *jit.Registry
	direct bool
	runs   []launch
	// shared is, by storage, the buffer every session binds: a weight, a
	// word block, the rotary planes; nil for a session's own.
	shared []backend.Buf
	// sizes is each storage's bytes at the capacity.
	sizes          []int64
	logits, token  int
	nvocab, nembd  int
	input, kvPages int
}

type launch struct {
	k             backend.Kernel
	groups, width int
	at            []int
}

// Option configures a Runner.
type Option func(*Runner)

// WithComposer composes the plan with r in place of jit.Default(): another
// profile of passes, for a gate holding one pass to its absence.
func WithComposer(r *jit.Registry) Option { return func(rn *Runner) { rn.reg = r } }

// WithDirectLaunches launches every kernel of every step instead of
// replaying a recording: a profile's view, each kernel timed on its own.
func WithDirectLaunches() Option { return func(rn *Runner) { rn.direct = true } }

// WithGuard has every buffer n poisoned bytes past its size, which
// Session.Overrun checks: a gate's view of a kernel writing past what the
// plan states (RULE 13).
func WithGuard(n int) Option { return func(r *Runner) { r.guard = n } }

// New composes the model in f for gpu, the composer's description of dev, at
// a capacity of keys positions a sequence, compiles it on dev and uploads what
// every sequence shares. rope is the model's rotary.
func New(f Source, dev backend.Device, gpu jit.Device, keys int, rope nn.Rope, opts ...Option) (*Runner, error) {
	c := f.Config()
	r := &Runner{f: f, dev: dev, step: jit.Step{Rows: 1, Page: 64, Keys: keys, ActWindow: f.ActWindow()}, logits: -1, token: -1, input: -1,
		nvocab: int(c.NVocab), nembd: int(c.NEmbd)}
	for _, o := range opts {
		o(r)
	}
	r.planes = rope.TabPlanes(int(c.NRot) / 2)
	if r.reg == nil {
		r.reg = jit.Default()
	}
	plan, err := r.reg.Compose(f, r.step, func(jit.Part) jit.Device { return gpu })
	if err != nil {
		return nil, err
	}
	r.plan = plan
	r.kvPages = r.step.Rows * ((r.step.Keys + r.step.Page - 1) / r.step.Page)
	r.shared = make([]backend.Buf, len(plan.Storages))
	r.sizes = make([]int64, len(plan.Storages))
	for i, st := range plan.Storages {
		n, err := r.bytes(st)
		if err != nil {
			r.Close()
			return nil, err
		}
		r.sizes[i] = n
		if st.Source == jit.SourceInput {
			r.input = i
		}
	}
	for _, pp := range plan.Parts {
		for _, code := range pp.Codes {
			if code.GPU == nil {
				r.Close()
				return nil, fmt.Errorf("run: %v formed %s for the host", pp.Part, code.Name)
			}
			k := *code.GPU
			k.Name = fmt.Sprintf("p%02d_%s", len(r.runs), k.Name)
			kern, err := dev.Compile(&k)
			if err != nil {
				r.Close()
				return nil, fmt.Errorf("run: %s: %w", k.Name, err)
			}
			l := launch{k: kern, groups: code.Groups, width: k.Group[0]}
			for _, b := range code.Binds {
				l.at = append(l.at, b.Storage)
				if b.Writes && b.Operand.Kind == jit.OperandValue {
					switch b.Operand.Name {
					case "logits":
						r.logits = b.Storage
					case "token":
						r.token = b.Storage
					}
				}
			}
			r.runs = append(r.runs, l)
		}
	}
	if r.logits < 0 || r.token < 0 {
		r.Close()
		return nil, fmt.Errorf("run: the plan writes no logits or no token")
	}
	for i, st := range plan.Storages {
		var data []byte
		switch st.Source {
		case jit.SourceWeight:
			if data, err = r.weight(st); err != nil {
				r.Close()
				return nil, err
			}
		case jit.SourceWords:
			data = wordBytes(st.Words)
		case jit.SourceRopePlanes:
			data = floatBytes(r.planes)
		default:
			continue
		}
		// A region of the container may be longer than the kernels read
		// of it: it is aligned.
		r.sizes[i] = max(r.sizes[i], int64(len(data)))
		if r.shared[i], err = r.alloc(r.sizes[i], data); err != nil {
			r.Close()
			return nil, err
		}
	}
	return r, nil
}

// bytes is storage st's size at the runner's capacity: what the plan states,
// or for what the engine owns, the engine's own.
func (r *Runner) bytes(st jit.Storage) (int64, error) {
	c := r.f.Config()
	pages := int64(r.step.Rows * ((r.step.Keys + r.step.Page - 1) / r.step.Page))
	switch st.Source {
	case jit.SourcePages:
		return 4 * pages, nil
	case jit.SourceKeys, jit.SourceValues:
		return 4 * pages * int64(r.step.Page) * int64(c.NKVHead*c.HeadDim), nil
	case jit.SourceRopePlanes:
		return int64(4 * len(r.planes)), nil
	case jit.SourceTransfer:
		return 0, fmt.Errorf("run: a plan of one device carries no value between devices")
	}
	if st.Bytes == 0 {
		return 0, fmt.Errorf("run: the plan states no size for %s storage %v", st.Source, st.Key)
	}
	return st.Bytes, nil
}

// weight is the region of a weight a storage names, in the container, its
// block paged in.
func (r *Runner) weight(st jit.Storage) ([]byte, error) {
	e, ok := r.f.Find(st.Role, int32(st.Block), -1)
	if !ok {
		return nil, fmt.Errorf("run: no tensor %v of block %d in the container", st.Role, st.Block)
	}
	region := func() []byte {
		qs, d, sc := r.f.Span(e)
		return [3][]byte{qs, d, sc}[st.Region]
	}
	if st.Region < 0 || st.Region > 2 {
		return nil, fmt.Errorf("run: tensor %v of block %d has no region %d", st.Role, st.Block, st.Region)
	}
	if b := region(); b != nil {
		return b, nil
	}
	if err := r.f.EnsurePage(r.f.BlockOf(e)); err != nil {
		return nil, err
	}
	b := region()
	if b == nil {
		return nil, fmt.Errorf("run: region %d of tensor %v of block %d is not resident", st.Region, st.Role, st.Block)
	}
	return b, nil
}

// alloc is a buffer of n bytes and the runner's guard, poisoned, with data
// at its start.
func (r *Runner) alloc(n int64, data []byte) (backend.Buf, error) {
	b, err := r.dev.Alloc(int(n) + r.guard)
	if err != nil {
		return nil, err
	}
	init := make([]byte, int(n)+r.guard)
	for i := range init {
		init[i] = 0xff
	}
	copy(init, data)
	if err := b.Write(init); err != nil {
		b.Free()
		return nil, err
	}
	return b, nil
}

// Capacity is how many positions a session holds.
func (r *Runner) Capacity() int { return r.step.Keys }

// Close frees the runner's kernels and shared buffers; its sessions must be
// closed first.
func (r *Runner) Close() {
	for _, l := range r.runs {
		l.k.Close()
	}
	r.runs = nil
	for i, b := range r.shared {
		if b != nil {
			b.Free()
			r.shared[i] = nil
		}
	}
}

// Session is one sequence on a runner.
type Session struct {
	r    *Runner
	bufs []backend.Buf
	own  []bool
	args [][]backend.Buf
	// rows, pos and toks are the step's tables, written each step.
	rows, pos, toks []byte
	tok             int32
	at              int
	logits          []float32
	raw, tb         []byte
	err             error
	launchAll       func(backend.Session)
	// rec is the step's launches recorded once, on a device that records
	// (backend.Recorder), and replayed each step.
	rec backend.Recording
}

// NewSession is an empty sequence: its pools, page table and step buffers.
func (r *Runner) NewSession() (*Session, error) {
	st := r.plan.Storages
	s := &Session{r: r, bufs: make([]backend.Buf, len(st)), own: make([]bool, len(st)),
		rows: make([]byte, 4*jit.RowWords), pos: make([]byte, 8), toks: make([]byte, 4), tb: make([]byte, 4),
		logits: make([]float32, r.nvocab)}
	s.raw = unsafe.Slice((*byte)(unsafe.Pointer(&s.logits[0])), 4*len(s.logits))
	for i := range st {
		if r.shared[i] != nil {
			s.bufs[i] = r.shared[i]
			continue
		}
		var data []byte
		switch st[i].Source {
		case jit.SourcePages:
			w := make([]uint32, r.kvPages)
			for p := range w {
				w[p] = uint32(p)
			}
			data = wordBytes(w)
		case jit.SourceQueryOffset:
			data = wordBytes([]uint32{0})
		}
		b, err := r.alloc(r.sizes[i], data)
		if err != nil {
			s.Close()
			return nil, err
		}
		s.bufs[i], s.own[i] = b, true
	}
	for _, l := range r.runs {
		a := make([]backend.Buf, len(l.at))
		for j, at := range l.at {
			a[j] = s.bufs[at]
		}
		s.args = append(s.args, a)
	}
	s.launchAll = s.launch
	return s, nil
}

// Step runs token tok at position pos and returns the step's logits, a
// slice the next Step overwrites, and its token.
func (s *Session) Step(tok int32, pos int) ([]float32, int32, error) {
	r := s.r
	if pos < 0 || pos >= r.step.Keys {
		return nil, 0, fmt.Errorf("run: position %d outside a capacity of %d", pos, r.step.Keys)
	}
	if tok < 0 || int(tok) >= r.nvocab {
		return nil, 0, fmt.Errorf("run: token %d outside a vocabulary of %d", tok, r.nvocab)
	}
	s.tok, s.at, s.err = tok, pos, nil
	r.dev.Session(s.launchAll)
	if s.err != nil {
		return nil, 0, s.err
	}
	return s.logits, int32(binary.LittleEndian.Uint32(s.tb)), nil
}

// launch is a step on the device: the tables and the input written, every
// kernel launched, the logits and the token read back.
func (s *Session) launch(se backend.Session) {
	r := s.r
	if s.err = s.tables(se); s.err != nil {
		return
	}
	// The kernels and their arguments are the session's, and what a step
	// changes is in its tables: a device that records replays the step's
	// launches as one submission, recorded on the first step.
	if rec, ok := se.(backend.Recorder); ok && !r.direct {
		if s.rec == nil {
			if s.rec, s.err = backend.Record(rec, func() { s.err = s.launches(se) }); s.err != nil {
				s.rec = nil
				return
			}
		}
		s.err = rec.Replay(s.rec)
	} else {
		s.err = s.launches(se)
	}
	if s.err != nil {
		return
	}
	if s.err = se.Read(s.bufs[r.logits], s.raw); s.err != nil {
		return
	}
	s.err = se.Read(s.bufs[r.token], s.tb)
}

// launches launches every kernel of the step in order.
func (s *Session) launches(se backend.Session) error {
	for i, l := range s.r.runs {
		if err := se.Launch(l.k, l.groups, l.width, s.args[i]...); err != nil {
			return err
		}
	}
	return nil
}

// tables writes the step's tables and its input: the row table, the
// rotary's positions, the tokens.
func (s *Session) tables(se backend.Session) error {
	r := s.r
	put := binary.LittleEndian.PutUint32
	put(s.rows[4*jit.RowTab:], 0)
	put(s.rows[4*jit.RowStart:], 0)
	put(s.rows[4*jit.RowEnd:], uint32(s.at+1))
	put(s.rows[4*jit.RowWrite:], uint32(s.at))
	put(s.pos, 1)
	put(s.pos[4:], uint32(s.at))
	put(s.toks, uint32(s.tok))
	var err error
	for i, st := range r.plan.Storages {
		switch st.Source {
		case jit.SourceRows:
			err = se.WriteAt(s.bufs[i], 0, s.rows)
		case jit.SourceRopePositions:
			err = se.WriteAt(s.bufs[i], 0, s.pos)
		case jit.SourceTokens:
			err = se.WriteAt(s.bufs[i], 0, s.toks)
		case jit.SourceInput:
			var row []byte
			if row, err = s.inputRow(); err == nil {
				err = se.WriteAt(s.bufs[i], 0, row)
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// inputRow is the token's row of the embedding, as the container stores it.
func (s *Session) inputRow() ([]byte, error) {
	f := s.r.f
	e, ok := f.Find(jlm.RoleTokenEmbd, -1, -1)
	if !ok {
		return nil, errors.New("run: the container has no token embedding")
	}
	if e.Type != jlm.TypeF32 {
		return nil, fmt.Errorf("run: a token embedding of %v; the input row is copied only from f32", e.Type)
	}
	qs, _, _ := f.Span(e)
	if qs == nil {
		return nil, errors.New("run: the token embedding is not resident")
	}
	w := 4 * s.r.nembd
	return qs[int(s.tok)*w : int(s.tok+1)*w], nil
}

// Overrun is the first storage a kernel wrote past its size into the
// runner's guard, or nil; it reads every buffer, so it is a gate's.
func (s *Session) Overrun() error {
	r := s.r
	if r.guard == 0 {
		return errors.New("run: the runner has no guard")
	}
	for i, b := range s.bufs {
		n := r.sizes[i]
		p := make([]byte, int(n)+r.guard)
		if err := b.Read(p); err != nil {
			return err
		}
		for j := n; j < int64(len(p)); j++ {
			if p[j] != 0xff {
				return fmt.Errorf("run: %s storage %v written at byte %d, past its %d", r.plan.Storages[i].Source, r.plan.Storages[i].Key, j, n)
			}
		}
	}
	return nil
}

// Close frees the session's own buffers.
func (s *Session) Close() {
	if s.rec != nil {
		s.rec.Free()
		s.rec = nil
	}
	for i, b := range s.bufs {
		if s.own[i] && b != nil {
			b.Free()
		}
		s.bufs[i] = nil
	}
}

func wordBytes(w []uint32) []byte {
	b := make([]byte, 4*len(w))
	for i, x := range w {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

func floatBytes(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	return append([]byte(nil), unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), 4*len(f))...)
}

// Plan is the runner's composed plan.
func (r *Runner) Plan() *jit.Plan { return r.plan }

// Storage is storage i's bytes as the session's buffer holds them: a
// diagnostic, copied out of the device, which allocates.
func (s *Session) Storage(i int) []byte {
	x := make([]byte, s.r.sizes[i])
	s.bufs[i].Read(x)
	return x
}

// SetStorage writes b over the start of storage i: a diagnostic, with
// StepEach, to hold a kernel to inputs of the caller's choosing.
func (s *Session) SetStorage(i int, b []byte) error { return s.bufs[i].Write(b) }

// StepEach is a step a kernel at a time, waiting for each, after(i) called
// once kernel i has finished: a diagnostic, as Model.Trace is the host's, to
// read what each kernel wrote. Its logits and token are not read back.
func (s *Session) StepEach(tok int32, pos int, after func(i int)) error {
	r := s.r
	s.tok, s.at = tok, pos
	var err error
	r.dev.Session(func(se backend.Session) { err = s.tables(se) })
	if err != nil {
		return err
	}
	for i, l := range r.runs {
		r.dev.Session(func(se backend.Session) {
			if err = se.Launch(l.k, l.groups, l.width, s.args[i]...); err == nil {
				err = se.Sync()
			}
		})
		if err != nil {
			return err
		}
		after(i)
	}
	return nil
}
