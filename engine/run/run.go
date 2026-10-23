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
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
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
	// The container's pager (jlm.File): a page is held while its bytes are
	// copied out, so no other read can evict and refill its frame meanwhile.
	Hold(b int, rs []jlm.Range) (jlm.Lease, error)
	HoldFilled(b int) (jlm.Lease, bool)
	PageBounds(b int) (off, size uint64)
	InDense(e *jlm.Entry) bool
	SpanLen(e *jlm.Entry) (qs, d, sc uint64)
	SheetSpan(e *jlm.Entry) (qs, d, sc, sheets uint64, err error)
	BlockOf(e *jlm.Entry) int
	// A mixture's bank paged an expert at a time (jlm.File.ExpertPaged).
	ExpertPaged(e *jlm.Entry) bool
	ExpertPage(e *jlm.Entry, x int) int
	Sheet(e *jlm.Entry, x int) (qs, d, sc []byte)
	// Ropes is the engine's rotaries in jit.Model.Rotaries' order.
	Ropes() []nn.Rope
}

// Runner is a model's plan on one device.
type Runner struct {
	f      Source
	dev    backend.Device
	step   jit.Step
	plan   *jit.Plan
	planes map[string][]float32
	guard  int
	reg    *jit.Registry
	direct bool
	runs   []launch
	// shared is, by storage, the buffer every session binds: a weight, a
	// word block, the rotary planes; nil for a session's own.
	shared []backend.Buf
	// sizes is each storage's bytes at the capacity.
	sizes         []int64
	logits, token int
	nvocab, nembd int
	// outW is the head's output width: the vocabulary's, or an encoder's
	// embedding's.
	outW          int
	input, pixels int
	// res is the weights on the device, shared where WithResidency gave it
	// (ownRes unset); wst is a weight storage's region there, need each
	// launch's blocks and wargs where its arguments name block weights.
	res    *Residency
	ownRes bool
	wst    []*wstore
	need   [][]int
	wargs  [][]warg
	scan   [2][2]int
}

// warg is argument j of a launch, block weight s.
type warg struct {
	j int
	s *wstore
}

type launch struct {
	k             backend.Kernel
	groups, width int
	at            []int
	// seg is the segment of the part it is of: a picture's step launches
	// the vision segment's, a token's the text model's.
	seg jit.Segment
}

// Option configures a Runner.
type Option func(*Runner)

// WithComposer composes the plan with r in place of jit.Default(): another
// profile of passes, for a gate holding one pass to its absence.
func WithComposer(r *jit.Registry) Option { return func(rn *Runner) { rn.reg = r } }

// WithDirectLaunches launches every kernel of every step instead of
// replaying a recording: a profile's view, each kernel timed on its own.
func WithDirectLaunches() Option { return func(rn *Runner) { rn.direct = true } }

// WithRows composes the plan for steps of up to n rows of one sequence: a
// prompt's chunk at a time (Session.Rows), each row its own position.
func WithRows(n int) Option { return func(r *Runner) { r.step.Rows, r.step.Seqs = max(1, n), 1 } }

// WithPictureRows composes the model's vision segment into the plan for
// pictures of up to n rows (Session.Picture): its parts the one model's, a
// picture's rows written into the text step's input rows at its place.
func WithPictureRows(n int) Option { return func(r *Runner) { r.step.Picture = n } }

// WithGuard has every buffer n poisoned bytes past its size, which
// Session.Overrun checks: a gate's view of a kernel writing past what the
// plan states (RULE 13).
func WithGuard(n int) Option { return func(r *Runner) { r.guard = n } }

// New composes the model in f for gpu, the composer's description of dev, at
// a capacity of keys positions a sequence, compiles it on dev and uploads what
// every sequence shares. Each of the model's rotaries (Source.Ropes) fills
// its table's planes.
func New(f Source, dev backend.Device, gpu jit.Device, keys int, opts ...Option) (*Runner, error) {
	c := f.Config()
	r := &Runner{f: f, dev: dev, step: jit.Step{Rows: 1, Page: 64, Keys: keys, ActWindow: f.ActWindow()}, logits: -1, token: -1, input: -1, pixels: -1,
		nvocab: int(c.NVocab), nembd: int(c.NEmbd)}
	for _, o := range opts {
		o(r)
	}
	r.planes = map[string][]float32{}
	plane := func(i int) string {
		if i == 0 {
			return "rope.planes"
		}
		return fmt.Sprintf("rope.planes.%d", i)
	}
	rots := f.Rotaries()
	for i, rope := range f.Ropes() {
		r.planes[plane(i)] = rope.TabPlanes(rope.NRot / 2)
		if i < len(rots) && rots[i].Grid {
			r.planes[plane(i)] = rope.Planes()
		}
	}
	// The vision segment's rotaries' planes, named after the text model's.
	if r.step.Picture > 0 {
		v, ok := f.Segment().(Source)
		if !ok {
			return nil, fmt.Errorf("run: the model states no vision segment the runner reads")
		}
		base := max(1, len(rots))
		vr := v.Rotaries()
		for i, rope := range v.Ropes() {
			r.planes[plane(base+i)] = rope.TabPlanes(rope.NRot / 2)
			if i < len(vr) && vr[i].Grid {
				r.planes[plane(base+i)] = rope.Planes()
			}
		}
	}
	if r.reg == nil {
		r.reg = jit.Default()
	}
	plan, err := r.reg.Compose(f, r.step, func(jit.Part) jit.Device { return gpu })
	if err != nil {
		return nil, err
	}
	r.plan = plan
	r.shared = make([]backend.Buf, len(plan.Storages))
	r.sizes = make([]int64, len(plan.Storages))
	for i, st := range plan.Storages {
		n, err := r.bytes(st)
		if err != nil {
			r.Close()
			return nil, err
		}
		r.sizes[i] = n
		if st.Source == jit.SourceInput && st.Segment == jit.SegmentVision {
			r.pixels = i
		} else if st.Source == jit.SourceInput {
			r.input = i
		}
	}
	// The logits and the token are the head's and the sample's outputs,
	// whichever kernel writes them: a fusion may join a part's last kernel
	// to the next part's first.
	outOf := map[jit.Operand]jit.PartKind{}
	for i, pp := range plan.Parts {
		if len(pp.Part.Graph.Out) > 0 && pp.Part.Segment == jit.SegmentText {
			outOf[jit.Operand{Kind: jit.OperandValue, Value: pp.Part.Graph.Out[0], Of: i}] = pp.Part.Kind
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
			l := launch{k: kern, groups: code.Groups, width: k.Group[0], seg: pp.Part.Segment}
			for _, b := range code.Binds {
				l.at = append(l.at, b.Storage)
				if kind, ok := outOf[jit.Operand{Kind: jit.OperandValue, Value: b.Operand.Value, Of: b.Operand.Of}]; ok &&
					b.Writes && b.Operand.Kind == jit.OperandValue {
					switch kind {
					case jit.PartHead:
						r.logits = b.Storage
					case jit.PartSample:
						r.token = b.Storage
					}
				}
			}
			r.runs = append(r.runs, l)
		}
	}
	// An encoder's plan samples nothing: its head's output, one row of
	// its width, is the model's.
	sampled := false
	for _, pp := range plan.Parts {
		if pp.Part.Kind == jit.PartHead && pp.Part.Segment == jit.SegmentText && len(pp.Part.Graph.Out) > 0 {
			r.outW = pp.Part.Graph.Values[pp.Part.Graph.Out[0]].Width
		}
		sampled = sampled || pp.Part.Kind == jit.PartSample
	}
	if r.logits < 0 || sampled && r.token < 0 {
		r.Close()
		return nil, fmt.Errorf("run: the plan writes no logits or no token")
	}
	if r.res == nil {
		r.res, r.ownRes = NewResidency(f, dev, nil, 0), true
		r.res.guard = r.guard
	}
	if r.res.guard != r.guard {
		return nil, fmt.Errorf("run: a runner guards %d bytes past its buffers and the weights it binds %d", r.guard, r.res.guard)
	}
	r.wst = make([]*wstore, len(plan.Storages))
	for i, st := range plan.Storages {
		var data []byte
		release := func() {}
		switch st.Source {
		case jit.SourceWeight:
			// On the device's thread, where the weights' state changes.
			r.dev.Session(func(se backend.Session) { r.wst[i], err = r.res.add(se, st) })
			if err != nil {
				r.Close()
				return nil, err
			}
			// A region of the container may be longer than the kernels
			// read of it: it is aligned.
			r.sizes[i] = max(r.sizes[i], int64(r.wst[i].bytes))
			continue
		case jit.SourceWords:
			data = wordBytes(st.Words)
		case jit.SourceRopePlanes:
			p, ok := r.planes[st.Key.Name]
			if !ok {
				r.Close()
				return nil, fmt.Errorf("run: no rotary fills table %q", st.Key.Name)
			}
			data = floatBytes(p)
		default:
			continue
		}
		// A region of the container may be longer than the kernels read
		// of it: it is aligned.
		r.sizes[i] = max(r.sizes[i], int64(len(data)))
		r.shared[i], err = r.alloc(r.sizes[i], data)
		// Copied to the device: its page may go.
		release()
		if err != nil {
			r.Close()
			return nil, err
		}
	}
	r.dev.Session(func(se backend.Session) { err = r.res.fill(se) })
	if err != nil {
		r.Close()
		return nil, err
	}
	for _, l := range r.runs {
		var need []int
		var wa []warg
		for j, at := range l.at {
			s := r.wst[at]
			if s == nil || r.f.InDense(s.e) {
				continue
			}
			wa = append(wa, warg{j, s})
			if b := r.f.BlockOf(s.e); !slices.Contains(need, b) {
				need = append(need, b)
			}
		}
		r.need, r.wargs = append(r.need, need), append(r.wargs, wa)
	}
	// Each segment's step scans its own blocks, in order: a page-in during
	// one evicts the other's first (page.Pick: outside the scan, never used
	// in it again).
	for _, seg := range []jit.Segment{jit.SegmentText, jit.SegmentVision} {
		lo, hi := -1, -1
		for i, l := range r.runs {
			for _, b := range r.need[i] {
				if l.seg == seg {
					if lo < 0 || b < lo {
						lo = b
					}
					hi = max(hi, b+1)
				}
			}
		}
		r.scan[seg] = [2]int{max(lo, 0), max(hi, 0)}
	}
	return r, nil
}

// Residency is the weights the runner's plan binds.
func (r *Runner) Residency() *Residency { return r.res }

// bytes is storage st's size at the runner's capacity: what the plan states,
// or for what the engine owns, the engine's own.
func (r *Runner) bytes(st jit.Storage) (int64, error) {
	switch st.Source {
	case jit.SourcePages:
		// The vision segment's table is a picture's one sequence's, the
		// same in every plan of the model; the text's its step's.
		if st.Segment == jit.SegmentVision {
			return 4 * int64(r.visionPages()), nil
		}
		return 4 * int64(r.step.Rows*((r.step.Keys+r.step.Page-1)/r.step.Page)), nil
	case jit.SourceRopePlanes:
		return int64(4 * len(r.planes[st.Key.Name])), nil
	case jit.SourceTransfer:
		return 0, fmt.Errorf("run: a plan of one device carries no value between devices")
	}
	if st.Bytes == 0 {
		return 0, fmt.Errorf("run: the plan states no size for %s storage %v", st.Source, st.Key)
	}
	return st.Bytes, nil
}

// bank is a paged bank's region whole, its sheets back to back: each
// expert's page held while its plane is copied out, the block's own page
// held throughout where it is in, as the old engine path assembles a bank
// (model.sheetOf). Every expert of the bank is resident on the device;
// paging experts through the device is not done here.
func bank(f Source, e *jlm.Entry, region int) ([]byte, error) {
	if lb, ok := f.HoldFilled(f.BlockOf(e)); ok {
		defer lb.Release()
	}
	var out []byte
	for x := 0; ; x++ {
		p := f.ExpertPage(e, x)
		if p < 0 {
			if x == 0 {
				return nil, fmt.Errorf("run: bank %v of block %d has no expert pages", e.Role, e.Block)
			}
			return out, nil
		}
		off, size := f.PageBounds(p)
		l, err := f.Hold(p, []jlm.Range{{Off: off, N: size}})
		if err != nil {
			return nil, err
		}
		qs, d, sc := f.Sheet(e, x)
		if qs == nil {
			l.Release()
			return nil, fmt.Errorf("run: expert %d of bank %v of block %d is not resident", x, e.Role, e.Block)
		}
		out = append(out, [3][]byte{qs, d, sc}[region]...)
		l.Release()
	}
}

// regionOf is region i of e in the container: in the dense region, or in
// its block's page, held until release runs.
func regionOf(f Source, e *jlm.Entry, i int) (data []byte, release func(), err error) {
	none := func() {}
	if f.ExpertPaged(e) {
		b, err := bank(f, e, i)
		return b, none, err
	}
	region := func() []byte {
		qs, d, sc := f.Span(e)
		return [3][]byte{qs, d, sc}[i]
	}
	if f.InDense(e) {
		if b := region(); b != nil {
			return b, none, nil
		}
		return nil, none, fmt.Errorf("run: region %d of tensor %v is not in the dense region", i, e.Role)
	}
	blk := f.BlockOf(e)
	off, size := f.PageBounds(blk)
	l, err := f.Hold(blk, []jlm.Range{{Off: off, N: size}})
	if err != nil {
		return nil, none, err
	}
	b := region()
	if b == nil {
		l.Release()
		return nil, none, fmt.Errorf("run: region %d of tensor %v of block %d is not resident", i, e.Role, e.Block)
	}
	return b, l.Release, nil
}

// visionPages is the page table a picture's step reads: its one sequence's
// pages.
func (r *Runner) visionPages() int {
	if r.step.Picture == 0 {
		return 0
	}
	return (r.step.Picture + r.step.Page - 1) / r.step.Page
}

// alloc is a buffer of n bytes and the runner's guard, poisoned, with data
// at its start.
// reuse assigns the step's own storages to shared buffers: each lives from
// the first launch that binds it to the last, written before it is read
// (jit.SourceStep), and two whose lives do not meet share one, as large as
// the larger. It returns each storage's buffer and the buffers' sizes. The
// logits and the token are read after the step and keep their own, and a
// runner with a guard shares nothing, so each guard is its storage's.
func (r *Runner) reuse(skip func(at int) bool) (map[int]int, []int64) {
	slot := map[int]int{}
	if r.guard > 0 {
		return slot, nil
	}
	// A storage both segments bind (a picture's projected rows, which the
	// token steps after its own read) lives across steps: its own buffer.
	segs := map[int]map[jit.Segment]bool{}
	for _, l := range r.runs {
		for _, at := range l.at {
			if segs[at] == nil {
				segs[at] = map[jit.Segment]bool{}
			}
			segs[at][l.seg] = true
		}
	}
	first, last := map[int]int{}, map[int]int{}
	for li, l := range r.runs {
		for _, at := range l.at {
			st := r.plan.Storages[at]
			if st.Source != jit.SourceStep || r.shared[at] != nil || at == r.logits || at == r.token || len(segs[at]) > 1 || skip != nil && skip(at) {
				continue
			}
			if _, ok := first[at]; !ok {
				first[at] = li
			}
			last[at] = li
		}
	}
	order := make([]int, 0, len(first))
	for at := range first {
		order = append(order, at)
	}
	slices.SortFunc(order, func(a, b int) int { return cmp.Or(cmp.Compare(first[a], first[b]), cmp.Compare(a, b)) })
	// A slot is free from the launch after the last that used it; a
	// storage takes the least free slot that holds it, else the largest
	// free one, grown, else a new one.
	var sizes []int64
	var busy []int
	for _, at := range order {
		need, k := r.sizes[at], -1
		for j := range sizes {
			if busy[j] >= first[at] {
				continue
			}
			fits, kfits := sizes[j] >= need, k >= 0 && sizes[k] >= need
			switch {
			case k < 0, fits && !kfits, fits && kfits && sizes[j] < sizes[k], !fits && !kfits && sizes[j] > sizes[k]:
				k = j
			}
		}
		if k < 0 {
			k = len(sizes)
			sizes, busy = append(sizes, 0), append(busy, 0)
		}
		slot[at], sizes[k], busy[k] = k, max(sizes[k], need), last[at]
	}
	return slot, sizes
}

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
	if r.ownRes && r.res != nil {
		r.res.Close()
	}
	r.res = nil
}

// Session is one sequence on a runner.
type Session struct {
	r    *Runner
	bufs []backend.Buf
	own  []bool
	args [][]backend.Buf
	// rows, pos and toks are the step's tables, written each step; in is
	// the step's tokens, its live rows, at the first's position.
	rows, pos, toks []byte
	in              []int32
	one             [1]int32
	at              int
	bidir           bool
	// request is a decision's words (jit.SourceRequest); marks its markers.
	request []byte
	marks   []int32
	// pic is a picture's step (Picture), its pixels the input in place of
	// the tokens' rows; blank its tokens, one a live row; wins its rows'
	// window descriptors (jit.SourceWindows).
	pic            *Picture
	blank          []int32
	wins           []byte
	logits         []float32
	raw, tb, input []byte
	err            error
	launchAll      func(backend.Session)
	// rec is the step's launches recorded once, on a device that records
	// (backend.Recorder), and replayed each step.
	// recs is each segment's launches recorded once, on a device that
	// records (backend.Recorder), and replayed each step of that segment;
	// seg is the segment the step in progress launches.
	recs [2]backend.Recording
	seg  jit.Segment
	// recGen is the residency's generation each recording was made at, and
	// bound each launch's: a page moved since is a buffer the recording or
	// the arguments no longer name.
	recGen [2]int
	bound  []int
	// charged is what the session's own buffers cost the device's account.
	charged uint64
	// zero is Reset's zeros, reused.
	zero []byte
	// slots are the buffers the step's own storages share (Runner.reuse).
	slots []backend.Buf
}

// Restart makes s an empty sequence again: its recurrent state zero, its
// other buffers as they are (a history's positions are rewritten as the
// sequence reaches them).
func (s *Session) Restart() error {
	for i, st := range s.r.plan.Storages {
		if st.Source == jit.SourceState {
			if err := s.bufs[i].Write(make([]byte, s.r.sizes[i])); err != nil {
				return err
			}
		}
	}
	return nil
}

// NewSession is an empty sequence: its pools, page table and step buffers.
func (r *Runner) NewSession() (*Session, error) { return r.newSession(nil) }

// NewSessionOf is a session on r of h's sequence: h's history pools and
// recurrent state, each storage of r's plan that one of h's plan matches
// by key, so either runner's steps move the one sequence on. A prompt's
// chunks run on a runner of several rows (WithRows) and its decode on one
// of a row.
func (r *Runner) NewSessionOf(h *Session) (*Session, error) { return r.newSession(h) }

// sequence reports whether a storage of source src is a sequence's own,
// kept from step to step.
func sequence(src jit.Source) bool {
	return src == jit.SourceKeys || src == jit.SourceValues || src == jit.SourceState
}

func (r *Runner) newSession(h *Session) (*Session, error) {
	st := r.plan.Storages
	of := map[jit.Operand]int{}
	if h != nil {
		for j, x := range h.r.plan.Storages {
			if sequence(x.Source) {
				of[x.Key] = j
			}
		}
	}
	// The vision segment is the same in every plan of the model, and a
	// sequence's picture's step runs on its first session: a session of the
	// sequence binds that one's vision storages, its projected rows among
	// them, rather than a copy.
	vis := map[jit.Operand]int{}
	if h != nil {
		for j, x := range h.r.plan.Storages {
			if x.Segment == jit.SegmentVision && x.Source != jit.SourceWeight {
				vis[x.Key] = j
			}
		}
	}
	shareVis := func(at int) bool {
		_, ok := vis[st[at].Key]
		return ok && st[at].Segment == jit.SegmentVision
	}
	R := max(r.step.Rows, r.step.Picture)
	s := &Session{r: r, bufs: make([]backend.Buf, len(st)), own: make([]bool, len(st)),
		rows: make([]byte, 4*jit.RowWords*R), pos: make([]byte, 4*(1+R)), toks: make([]byte, 4*R), tb: make([]byte, 4*r.step.Rows),
		logits: make([]float32, r.step.Rows*r.outW), input: make([]byte, 4*R*r.nembd), request: make([]byte, 4*(jit.RequestRows+R))}
	for _, at := range []int{r.input, r.pixels} {
		if at >= 0 {
			s.input = make([]byte, max(int64(len(s.input)), r.sizes[at]))
		}
	}
	s.raw = unsafe.Slice((*byte)(unsafe.Pointer(&s.logits[0])), 4*len(s.logits))
	slot, slots := r.reuse(shareVis)
	// The session's own buffers are the permanent part: blocks page out to
	// make room for them before they are made, never the other way.
	var own uint64
	for _, n := range slots {
		own += uint64(n)
	}
	for i := range st {
		_, slotted := slot[i]
		_, ofSeq := of[st[i].Key]
		if r.wst[i] == nil && r.shared[i] == nil && !slotted && !(ofSeq && sequence(st[i].Source)) && !shareVis(i) {
			own += uint64(r.sizes[i] + int64(r.guard))
		}
	}
	for range slots {
		own += uint64(r.guard)
	}
	var err error
	r.dev.Session(func(se backend.Session) {
		if err = r.res.reclaim(se, own, -1, 0, 0); err == nil {
			r.res.charge(own)
			s.charged = own
		}
	})
	if err != nil {
		return nil, err
	}
	for _, sz := range slots {
		b, err := r.alloc(sz, nil)
		if err != nil {
			s.Close()
			return nil, err
		}
		s.slots = append(s.slots, b)
	}
	for i := range st {
		if r.wst[i] != nil {
			s.bufs[i] = r.wst[i].buf
			continue
		}
		if r.shared[i] != nil {
			s.bufs[i] = r.shared[i]
			continue
		}
		if shareVis(i) {
			j := vis[st[i].Key]
			if h.r.sizes[j] < r.sizes[i] {
				s.Close()
				return nil, fmt.Errorf("run: vision storage %q is %d bytes in the sequence's plan, %d here", st[i].Key.Name, h.r.sizes[j], r.sizes[i])
			}
			s.bufs[i] = h.bufs[j]
			continue
		}
		if k, ok := slot[i]; ok {
			s.bufs[i] = s.slots[k]
			continue
		}
		if j, ok := of[st[i].Key]; ok && sequence(st[i].Source) {
			if h.r.sizes[j] < r.sizes[i] {
				s.Close()
				return nil, fmt.Errorf("run: %s storage %q is %d bytes in the sequence's plan, %d here", st[i].Source, st[i].Key.Name, h.r.sizes[j], r.sizes[i])
			}
			s.bufs[i] = h.bufs[j]
			continue
		}
		var data []byte
		switch st[i].Source {
		case jit.SourcePages:
			w := make([]uint32, r.sizes[i]/4)
			for p := range w {
				w[p] = uint32(p)
			}
			data = wordBytes(w)
		case jit.SourceQueryOffset:
			data = wordBytes([]uint32{0})
		case jit.SourceState:
			data = make([]byte, r.sizes[i])
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
	s.bound = make([]int, len(r.runs))
	for i := range s.bound {
		s.bound[i] = r.res.gen
	}
	s.launchAll = s.launch
	return s, nil
}

// Step runs token tok at position pos and returns the step's logits, a
// slice the next Step overwrites, and its token.
func (s *Session) Step(tok int32, pos int) ([]float32, int32, error) {
	s.one[0] = tok
	return s.Rows(s.one[:], pos)
}

// Rows runs tokens toks at positions pos, pos+1, ... as one step, as many
// as the runner's rows (WithRows), and returns the last one's logits, a
// slice the next step overwrites, and its token.
func (s *Session) Rows(toks []int32, pos int) ([]float32, int32, error) {
	return s.SpanRows(toks, pos, nil)
}

// step is Rows with the step's picture rows named.
func (s *Session) step(toks []int32, pos int) ([]float32, int32, error) {
	r := s.r
	n := len(toks)
	if n < 1 || n > r.step.Rows {
		return nil, 0, fmt.Errorf("run: %d rows in a step of %d", n, r.step.Rows)
	}
	if pos < 0 || pos+n > r.step.Keys {
		return nil, 0, fmt.Errorf("run: positions %d to %d outside a capacity of %d", pos, pos+n-1, r.step.Keys)
	}
	for _, tok := range toks {
		if tok < 0 || int(tok) >= r.nvocab {
			return nil, 0, fmt.Errorf("run: token %d outside a vocabulary of %d", tok, r.nvocab)
		}
	}
	s.in, s.at, s.err = toks, pos, nil
	r.dev.Session(s.launchAll)
	if s.err != nil {
		return nil, 0, s.err
	}
	return s.logits[(n-1)*r.outW : n*r.outW], int32(binary.LittleEndian.Uint32(s.tb[4*(n-1):])), nil
}

// Decide runs a decision model's request as one bidirectional step: its
// tokens toks, its question type qt and its option markers, rows of toks.
// It returns the scorer's value at each marker, a slice the next step
// overwrites.
func (s *Session) Decide(toks []int32, qt int, markers []int32) ([]float32, error) {
	n := len(toks)
	if len(markers) < 1 || len(markers) > s.r.step.Rows {
		return nil, fmt.Errorf("run: %d markers in a step of %d rows", len(markers), s.r.step.Rows)
	}
	for _, m := range markers {
		if m < 0 || int(m) >= n {
			return nil, fmt.Errorf("run: marker row %d of %d", m, n)
		}
	}
	put := binary.LittleEndian.PutUint32
	put(s.request[4*jit.RequestType:], uint32(qt))
	put(s.request[4*jit.RequestCount:], uint32(len(markers)))
	for i := range s.r.step.Rows {
		put(s.request[4*(jit.RequestRows+i):], uint32(markers[min(i, len(markers)-1)]))
	}
	if _, err := s.Encode(toks, true); err != nil {
		return nil, err
	}
	return s.logits[:len(markers)*s.r.outW], nil
}

// Picture is a picture's step for a vision tower's plan.
type Picture struct {
	// Pixels is the picture, [y][x][channel].
	Pixels []float32
	// Grid is the patch grid's width; Rows the step's live rows, the patches
	// and any rows the tower puts before them.
	Grid, Rows int
	// Pos is each row's position, nil for its index: a tower whose rows are
	// its patches in another order than raster names each row's patch.
	Pos []int32
	// Windows is each row's window, [lo, hi) of the rows, where a block
	// attends inside windows (graph.Attrs.RowWindows).
	Windows []int32
	// Take is each output row's row of the head's last value before it, nil
	// for none (graph.OpTakeRows): window-ordered rows back in raster order.
	Take []int32
}

// Picture runs the model's vision segment over picture p as one
// bidirectional step of the session: its projected rows stay on the
// device, for the token steps that take them at the picture's place
// (SpanRows).
func (s *Session) Picture(p *Picture) error {
	if err := s.picture(p); err != nil {
		return err
	}
	defer s.endPicture()
	s.r.dev.Session(s.launchAll)
	return s.err
}

// PictureEach is Picture a launch at a time, after(i) called once launch i
// has finished, its storages readable.
func (s *Session) PictureEach(p *Picture, after func(i int)) error {
	if err := s.picture(p); err != nil {
		return err
	}
	defer s.endPicture()
	return s.each(after)
}

// endPicture ends a picture's step: the session's next is a token's.
func (s *Session) endPicture() { s.pic, s.bidir, s.seg = nil, false, jit.SegmentText }

// picture readies picture p's step: its pixels the input, a blank token a
// live row, the grid's width and the output's order the request.
func (s *Session) picture(p *Picture) error {
	r, n := s.r, p.Rows
	if r.step.Picture == 0 {
		return fmt.Errorf("run: the plan composes no vision segment (WithPictureRows)")
	}
	if n < 1 || n > r.step.Picture || p.Grid < 1 || p.Pos != nil && len(p.Pos) < n || p.Windows != nil && len(p.Windows) < 2*n ||
		len(p.Take) > r.step.Picture {
		return fmt.Errorf("run: %d rows of a grid %d wide in a picture's step of %d rows (%d positions, %d window bounds, %d taken)",
			n, p.Grid, r.step.Picture, len(p.Pos), len(p.Windows), len(p.Take))
	}
	if r.pixels < 0 || len(p.Pixels) == 0 || 4*len(p.Pixels) > len(s.input) {
		return fmt.Errorf("run: %d pixels; the step takes %d bytes", len(p.Pixels), len(s.input))
	}
	if len(s.blank) < n {
		s.blank = make([]int32, r.step.Picture)
	}
	if s.wins == nil {
		s.wins = make([]byte, len(s.rows))
	}
	put := binary.LittleEndian.PutUint32
	put(s.request[4*jit.RequestGrid:], uint32(p.Grid))
	for i, t := range p.Take {
		put(s.request[4*(jit.RequestRows+i):], uint32(t))
	}
	s.in, s.at, s.err, s.pic, s.bidir, s.seg = s.blank[:n], 0, nil, p, true, jit.SegmentVision
	return nil
}

// SpanRows is Rows over a prompt's span where some rows are a picture's: row
// i takes the picture's projected row from[i] (its last Picture's) in place
// of its token's, a negative from[i] its token's own; from nil is Rows.
func (s *Session) SpanRows(toks []int32, pos int, from []int32) ([]float32, int32, error) {
	r := s.r
	if from != nil && (r.step.Picture == 0 || len(from) != len(toks)) {
		return nil, 0, fmt.Errorf("run: %d picture rows for %d tokens in a plan of %d picture rows", len(from), len(toks), r.step.Picture)
	}
	if r.step.Picture > 0 {
		for i := range r.step.Rows {
			w := jit.SpliceNone
			if from != nil {
				if f := from[min(i, len(from)-1)]; f >= 0 && int(f) < r.step.Picture {
					w = uint32(f)
				}
			}
			binary.LittleEndian.PutUint32(s.request[4*(jit.RequestRows+i):], w)
		}
	}
	return s.step(toks, pos)
}

// Encode runs a sequence toks as one step and returns its head's output:
// the embedding, a slice the next step overwrites. Where bidir every row
// reads every other (an encoder's, a non-causal decoder's); else each reads
// those before it.
func (s *Session) Encode(toks []int32, bidir bool) ([]float32, error) {
	s.bidir = bidir
	defer func() { s.bidir = false }()
	if _, _, err := s.Rows(toks, 0); err != nil {
		return nil, err
	}
	return s.logits[:s.r.outW], nil
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
	// A step whose blocks page binds other buffers from step to step: it is
	// launched, not replayed, while any block is out.
	res := r.res
	if rec, ok := se.(backend.Recorder); ok && !r.direct && res.pagesIn == len(res.blocks) {
		if s.recs[s.seg] != nil && s.recGen[s.seg] != res.gen {
			s.recs[s.seg].Free()
			s.recs[s.seg] = nil
		}
		if s.recs[s.seg] == nil {
			s.recGen[s.seg] = res.gen
			if s.recs[s.seg], s.err = backend.Record(rec, func() { s.err = s.launches(se) }); s.err != nil {
				s.recs[s.seg] = nil
				return
			}
		}
		s.err = rec.Replay(s.recs[s.seg])
	} else {
		s.err = s.launches(se)
	}
	// A picture's step leaves its rows on the device for the token steps
	// that take them; it reads nothing back.
	if s.err != nil || s.seg == jit.SegmentVision {
		return
	}
	if s.err = se.Read(s.bufs[r.logits], s.raw); s.err != nil || r.token < 0 {
		return
	}
	s.err = se.Read(s.bufs[r.token], s.tb)
}

// bind pages in the blocks launch i binds and, where a page moved since it
// last ran, names their buffers in its arguments.
func (s *Session) bind(se backend.Session, i int) error {
	r := s.r
	res := r.res
	if len(r.need[i]) == 0 {
		return nil
	}
	lo, hi := r.scan[r.runs[i].seg][0], r.scan[r.runs[i].seg][1]
	if err := res.ensure(se, r.need[i], lo, hi); err != nil {
		return err
	}
	if s.bound[i] == res.gen {
		return nil
	}
	for _, a := range r.wargs[i] {
		s.args[i][a.j] = a.s.buf
	}
	s.bound[i] = res.gen
	return nil
}

// launches launches every kernel of the step in order.
func (s *Session) launches(se backend.Session) error {
	for i, l := range s.r.runs {
		if l.seg != s.seg {
			continue
		}
		if err := s.bind(se, i); err != nil {
			return err
		}
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
	n := len(s.in)
	put(s.pos, uint32(n))
	rows := r.step.Rows
	if s.seg == jit.SegmentVision {
		rows = r.step.Picture
	}
	// A row past the live ones repeats the last: the threads a launch holds
	// to its live rows do its work again, writing what it wrote.
	for i := range rows {
		j := min(i, n-1)
		w := s.rows[4*jit.RowWords*i:]
		put(w[4*jit.RowTab:], 0)
		put(w[4*jit.RowStart:], 0)
		end := s.at + j + 1
		if s.bidir {
			end = s.at + n
		}
		put(w[4*jit.RowEnd:], uint32(end))
		put(w[4*jit.RowWrite:], uint32(s.at+j))
		put(s.pos[4*(1+i):], uint32(s.at+j))
		put(s.toks[4*i:], uint32(s.in[j]))
		if p := s.pic; p != nil {
			if p.Pos != nil {
				put(s.pos[4*(1+i):], uint32(p.Pos[j]))
			}
			v := s.wins[4*jit.RowWords*i:]
			copy(v[:4*jit.RowWords], w[:4*jit.RowWords])
			if p.Windows != nil {
				put(v[4*jit.RowStart:], uint32(p.Windows[2*j]))
				put(v[4*jit.RowEnd:], uint32(p.Windows[2*j+1]))
			}
		}
	}
	var err error
	// Each table at its storage's own size: one segment's alone may be
	// shorter than the step's buffers.
	at := func(i int, b []byte) []byte { return b[:min(int64(len(b)), r.sizes[i])] }
	for i, st := range r.plan.Storages {
		switch st.Source {
		case jit.SourceRows:
			err = se.WriteAt(s.bufs[i], 0, at(i, s.rows))
		case jit.SourceRopePositions:
			err = se.WriteAt(s.bufs[i], 0, at(i, s.pos))
		case jit.SourceTokens:
			err = se.WriteAt(s.bufs[i], 0, at(i, s.toks))
		case jit.SourceRequest:
			err = se.WriteAt(s.bufs[i], 0, at(i, s.request))
		case jit.SourceWindows:
			if s.wins != nil {
				err = se.WriteAt(s.bufs[i], 0, at(i, s.wins))
			}
		case jit.SourceInput:
			// The vision segment's input is a picture's pixels, written by
			// its step; the text's its tokens' rows, by theirs.
			switch px := s.pic; {
			case st.Segment == jit.SegmentVision && px != nil:
				err = se.WriteAt(s.bufs[i], 0, s.input[:copy(s.input, unsafe.Slice((*byte)(unsafe.Pointer(&px.Pixels[0])), 4*len(px.Pixels)))])
			case st.Segment == jit.SegmentText && px == nil:
				if err = s.inputRows(); err == nil {
					err = se.WriteAt(s.bufs[i], 0, s.input[:r.sizes[i]])
				}
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// inputRows copies each row's token's row of the embedding, as the
// container stores it, into s.input.
func (s *Session) inputRows() error {
	f := s.r.f
	e, ok := f.Find(jlm.RoleTokenEmbd, -1, -1)
	if !ok {
		return errors.New("run: the container has no token embedding")
	}
	if e.Type != jlm.TypeF32 {
		return fmt.Errorf("run: a token embedding of %v; the input row is copied only from f32", e.Type)
	}
	qs, _, _ := f.Span(e)
	if qs == nil {
		return errors.New("run: the token embedding is not resident")
	}
	w := 4 * s.r.nembd
	for i := range s.r.step.Rows {
		tok := int(s.in[min(i, len(s.in)-1)])
		copy(s.input[i*w:(i+1)*w], qs[tok*w:(tok+1)*w])
	}
	return nil
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

// Reset forgets the sequence: its recurrent state zeroed, as a new
// session's is, and its history to be written over from position 0 by the
// next steps. A session of the sequence (NewSessionOf) binds the same state,
// so one reset serves both.
func (s *Session) Reset() error {
	r := s.r
	var err error
	r.dev.Session(func(se backend.Session) {
		for i, st := range r.plan.Storages {
			if st.Source != jit.SourceState || s.bufs[i] == nil {
				continue
			}
			if n := int(r.sizes[i]); len(s.zero) < n {
				s.zero = make([]byte, n)
			}
			if err = se.WriteAt(s.bufs[i], 0, s.zero[:r.sizes[i]]); err != nil {
				return
			}
		}
	})
	return err
}

// Close frees the session's own buffers.
func (s *Session) Close() {
	if s.charged > 0 && s.r.res != nil {
		s.r.dev.Session(func(backend.Session) { s.r.res.refund(s.charged) })
		s.charged = 0
	}
	for i, rec := range s.recs {
		if rec != nil {
			rec.Free()
			s.recs[i] = nil
		}
	}
	for i, b := range s.bufs {
		if s.own[i] && b != nil {
			b.Free()
		}
		s.bufs[i] = nil
	}
	for _, b := range s.slots {
		b.Free()
	}
	s.slots = nil
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
	s.one[0] = tok
	s.in, s.at = s.one[:], pos
	return s.each(after)
}

// each runs the step's launches one at a time, after(i) once launch i has
// finished.
func (s *Session) each(after func(i int)) error {
	r := s.r
	var err error
	r.dev.Session(func(se backend.Session) { err = s.tables(se) })
	if err != nil {
		return err
	}
	for i, l := range r.runs {
		if l.seg != s.seg {
			continue
		}
		r.dev.Session(func(se backend.Session) {
			if err = s.bind(se, i); err != nil {
				return
			}
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
