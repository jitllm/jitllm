package board

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestComposedDecodeRunsOnTheGPU runs the composer's decode steps of the
// board's fixture on every device a backend opens, from the plan alone: each
// of the plan's storages built from the source the plan states for it -- the
// container's weights and norms, the tables the engine writes (the row
// descriptors, the page table, the query offset, the rotary positions and
// planes), the input token's row of the embedding, an empty history the steps
// fill -- at the size the plan states, or the engine's own for what it owns,
// and every kernel launched in order. The plan is composed once, at the
// session's capacity, as the engine runs it: the attention reads each step's
// bounds from the row descriptors, so one plan serves every shorter history.
// The capacities are one inside the attention's first split count and one
// past it. The prompt and then the host's choices are fed: each
// position's logits are held to the engine's host decode of the same tokens,
// and its token to the host's choice.
func TestComposedDecodeRunsOnTheGPU(t *testing.T) {
	path := testmodels.Path(*boardModel + ".jlm")
	f, err := jlm.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.LoadAll(); err != nil {
		t.Fatal(err)
	}
	parts, err := jit.Front(f)
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(prompt, true)
	const more = 6
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	for _, dev := range devs {
		for _, capacity := range []int{16, 300} {
			t.Run(fmt.Sprintf("%s/%d", dev.API(), capacity), func(t *testing.T) {
				decodeOn(t, m, f, parts, dev, ids, more, capacity)
			})
		}
	}
}

// decodeOn is the gate's run on one device at one capacity.
func decodeOn(t *testing.T, m *model.Model, f *jlm.File, parts []jit.Part, dev backend.Device, ids []int32, more, capacity int) {
	st := m.NewState(len(ids) + more + 8)
	defer st.Close()
	run := newStepRun(t, f, parts, dev, jit.Step{Rows: 1, Page: 64, Keys: capacity})
	defer run.free()
	worst := 0.0
	tok := ids[0]
	for pos := range len(ids) + more {
		want, err := st.Forward(tok)
		if err != nil {
			t.Fatal(err)
		}
		got, token := run.step(pos, tok)
		var num, den float64
		best := 0
		for i := range want {
			d := float64(got[i]) - float64(want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
			if want[i] > want[best] {
				best = i
			}
		}
		nmse := num / den
		if !(nmse < 1e-5) {
			t.Fatalf("position %d: logits NMSE %.3g against the host", pos, nmse)
		}
		worst = max(worst, nmse)
		if token != best {
			t.Fatalf("position %d: token %d, the host's %d", pos, token, best)
		}
		tok = int32(best)
		if pos+1 < len(ids) {
			tok = ids[pos+1]
		}
	}
	t.Logf("%d positions on %s, one plan for %d keys, worst logits NMSE %.3g against the host", len(ids)+more, dev.Name(), capacity, worst)
}

// stepRun is a device's storages across steps, by key: one buffer each, the
// history among them, kept from step to step.
type stepRun struct {
	t     *testing.T
	f     *jlm.File
	parts []jit.Part
	dev   backend.Device
	// plan is the step composed once at the capacity s, and its kernels.
	s     jit.Step
	plan  *jit.Plan
	kerns []backend.Kernel
	bufs  map[jit.Operand]backend.Buf
	size  map[jit.Operand]int64
	once  map[jit.Operand]bool
	// each, where set, runs after each kernel, launched alone, with the
	// storages it wrote.
	each func(i int, name string, wrote []jit.Operand)
}

func newStepRun(t *testing.T, f *jlm.File, parts []jit.Part, dev backend.Device, s jit.Step) *stepRun {
	return &stepRun{t: t, f: f, parts: parts, dev: dev, s: s, bufs: map[jit.Operand]backend.Buf{},
		size: map[jit.Operand]int64{}, once: map[jit.Operand]bool{}}
}

func (r *stepRun) free() {
	for _, b := range r.bufs {
		b.Free()
	}
	for _, k := range r.kerns {
		k.Close()
	}
}

// step composes and runs the decode step of token tok at position pos, and
// returns its logits and token.
func (r *stepRun) step(pos int, tok int32) ([]float32, int) {
	t := r.t
	s := r.s
	if pos >= s.Keys {
		t.Fatalf("position %d past the plan's %d keys", pos, s.Keys)
	}
	if r.plan == nil {
		plan, err := jit.Default().Compose(r.f, s, func(jit.Part) jit.Device { return sm70 })
		if err != nil {
			t.Fatal(err)
		}
		r.plan = plan
		for _, pp := range plan.Parts {
			for _, c := range pp.Codes {
				if c.GPU == nil {
					t.Fatalf("part %v formed %s for the host", pp.Part, c.Name)
				}
				k := *c.GPU
				k.Name = fmt.Sprintf("x%02d_%s", len(r.kerns), k.Name)
				kern, err := r.dev.Compile(&k)
				if err != nil {
					t.Fatalf("%s: %v", k.Name, err)
				}
				r.kerns = append(r.kerns, kern)
			}
		}
	}
	plan := r.plan
	for _, st := range plan.Storages {
		r.fill(st, r.bytes(st, s), r.source(st, pos, tok))
	}
	type launch struct {
		k             backend.Kernel
		name          string
		groups, width int
		bufs          []backend.Buf
		wrote         []jit.Operand
	}
	var runs []launch
	var logits, token jit.Operand
	i := 0
	for _, pp := range plan.Parts {
		for _, c := range pp.Codes {
			kern := r.kerns[i]
			i++
			var bs []backend.Buf
			var wrote []jit.Operand
			for _, b := range c.Binds {
				at := plan.Storages[b.Storage].Key
				bs = append(bs, r.bufs[at])
				if !b.Writes {
					continue
				}
				wrote = append(wrote, at)
				switch {
				case b.Operand.Kind == jit.OperandValue && b.Operand.Name == "logits":
					logits = at
				case b.Operand.Kind == jit.OperandValue && b.Operand.Name == "token":
					token = at
				}
			}
			runs = append(runs, launch{kern, c.Name, c.Groups, c.GPU.Group[0], bs, wrote})
		}
	}
	var lerr error
	if r.each != nil {
		for i, l := range runs {
			r.dev.Session(func(se backend.Session) {
				if lerr = se.Launch(l.k, l.groups, l.width, l.bufs...); lerr == nil {
					lerr = se.Sync()
				}
			})
			if lerr != nil {
				t.Fatal(lerr)
			}
			r.each(i, l.name, l.wrote)
		}
	} else {
		r.dev.Session(func(se backend.Session) {
			for _, l := range runs {
				if lerr = se.Launch(l.k, l.groups, l.width, l.bufs...); lerr != nil {
					return
				}
			}
			lerr = se.Sync()
		})
	}
	if lerr != nil {
		t.Fatal(lerr)
	}
	for _, l := range runs {
		for _, at := range l.wrote {
			b := make([]byte, r.size[at]+guard)
			if err := r.bufs[at].Read(b); err != nil {
				t.Fatal(err)
			}
			for i := r.size[at]; i < int64(len(b)); i++ {
				if b[i] != 0xff {
					t.Fatalf("%s wrote byte %d of storage %v, past the %d the plan states", l.name, i, at, r.size[at])
				}
			}
		}
	}
	if r.size[logits] == 0 || r.size[token] == 0 {
		t.Fatal("the step writes no logits or no token")
	}
	raw := make([]byte, r.size[logits])
	if err := r.bufs[logits].Read(raw); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, r.f.Config().NVocab)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	tb := make([]byte, r.size[token])
	if err := r.bufs[token].Read(tb); err != nil {
		t.Fatal(err)
	}
	return out, int(int32(binary.LittleEndian.Uint32(tb)))
}

// bytes is storage st's size at step s: what the plan states, or for what
// the engine owns, the engine's own -- the step's pages of the history, the
// pools those pages are, the model's rotary planes.
func (r *stepRun) bytes(st jit.Storage, s jit.Step) int64 {
	c := r.f.Config()
	pages := int64(s.Rows * ((s.Keys + s.Page - 1) / s.Page))
	switch st.Source {
	case jit.SourcePages:
		return 4 * pages
	case jit.SourceKeys, jit.SourceValues:
		return 4 * pages * int64(s.Page) * int64(c.NKVHead*c.HeadDim)
	case jit.SourceRopePlanes:
		return int64(4 * len(r.planes()))
	}
	if st.Bytes == 0 && st.Source != jit.SourceTransfer {
		r.t.Fatalf("the plan states no size for %s storage %v", st.Source, st.Key)
	}
	return st.Bytes
}

func (r *stepRun) planes() []float32 {
	c := r.f.Config()
	return nn.Rope{NRot: int(c.NRot), Base: float64(c.RopeBase)}.TabPlanes(int(c.NRot) / 2)
}

// guard is the poisoned bytes past each storage's size: a kernel writing
// past what the plan states for it changes them (RULE 13).
const guard = 4096

// fill makes storage st's buffer at least n bytes and a guard, keeping what
// the steps before wrote in it, and writes data into it where data is set.
func (r *stepRun) fill(st jit.Storage, n int64, data []byte) {
	at := st.Key
	n = max(n, int64(len(data)))
	if b, ok := r.bufs[at]; !ok || r.size[at] < n {
		nb, err := r.dev.Alloc(int(n + guard))
		if err != nil {
			r.t.Fatal(err)
		}
		init := make([]byte, n+guard)
		for i := n; i < n+guard; i++ {
			init[i] = 0xff
		}
		if ok {
			old := make([]byte, r.size[at]+guard)
			if err := b.Read(old); err != nil {
				r.t.Fatal(err)
			}
			copy(init, old[:r.size[at]])
			b.Free()
		}
		if err := nb.Write(init); err != nil {
			r.t.Fatal(err)
		}
		r.bufs[at], r.size[at] = nb, n
	}
	if data != nil && !r.once[at] {
		if err := r.bufs[at].WriteAt(0, data); err != nil {
			r.t.Fatal(err)
		}
	}
	switch st.Source {
	case jit.SourceWeight, jit.SourceWords, jit.SourcePages, jit.SourceRopePlanes:
		r.once[at] = true
	}
}

// source is what the engine puts in storage st before the step at position
// pos runs token tok: nil for what the step writes before it reads, and for
// the history, which the steps fill.
func (r *stepRun) source(st jit.Storage, pos int, tok int32) []byte {
	t, f, c := r.t, r.f, r.f.Config()
	row := make([]uint32, jit.RowWords)
	switch st.Source {
	case jit.SourceWeight:
		e, ok := f.Find(st.Role, int32(st.Block), -1)
		if !ok {
			t.Fatalf("no tensor %v of block %d in the container", st.Role, st.Block)
		}
		qs, _, _ := f.Span(e)
		return qs
	case jit.SourceInput:
		e, ok := f.Find(jlm.RoleTokenEmbd, -1, -1)
		if !ok {
			t.Fatal("no token embedding")
		}
		qs, _, _ := f.Span(e)
		w := 4 * int(c.NEmbd)
		return qs[int(tok)*w : int(tok+1)*w]
	case jit.SourceRows:
		row[jit.RowTab], row[jit.RowStart], row[jit.RowEnd], row[jit.RowWrite] = 0, 0, uint32(pos+1), uint32(pos)
		return wordBytes(row)
	case jit.SourcePages:
		w := make([]uint32, 64)
		for i := range w {
			w[i] = uint32(i)
		}
		return wordBytes(w)
	case jit.SourceQueryOffset:
		return wordBytes([]uint32{0})
	case jit.SourceRopePositions:
		return wordBytes([]uint32{1, uint32(pos)})
	case jit.SourceRopePlanes:
		p := r.planes()
		return unsafe.Slice((*byte)(unsafe.Pointer(&p[0])), 4*len(p))
	case jit.SourceWords:
		return wordBytes(st.Words)
	case jit.SourceStep, jit.SourceKeys, jit.SourceValues:
		return nil
	}
	t.Fatalf("a %s storage this gate has no source for", st.Source)
	return nil
}

func wordBytes(w []uint32) []byte {
	b := make([]byte, 4*len(w))
	for i, x := range w {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}
