package board

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestComposedDecodeRunsOnTheGPU runs the composer's decode steps of the
// board's fixture on every device a backend opens, from the plan alone: each
// storage the plan binds built from its source -- the container's weights and
// norms, the tables the engine writes (the row descriptors, the page table,
// the query offset, the rotary positions and planes), the input token's row
// of the embedding, an empty history the steps fill -- and every kernel
// launched in order. The prompt and then the host's choices are fed: each
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
		t.Run(dev.API(), func(t *testing.T) {
			st := m.NewState(len(ids) + more + 8)
			defer st.Close()
			run := newStepRun(t, f, parts, dev)
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
			t.Logf("%d positions on %s, worst logits NMSE %.3g against the host", len(ids)+more, dev.Name(), worst)
		})
	}
}

// stepRun is a device's storages across steps: one buffer per storage the
// plans bind, the history among them, kept from step to step.
type stepRun struct {
	t     *testing.T
	f     *jlm.File
	parts []jit.Part
	dev   backend.Device
	bufs  map[jit.Operand]backend.Buf
	size  map[jit.Operand]int64
	once  map[jit.Operand]bool
	// each, where set, runs after each kernel, launched alone, with the
	// storages it wrote.
	each func(i int, name string, wrote []jit.Operand)
}

func newStepRun(t *testing.T, f *jlm.File, parts []jit.Part, dev backend.Device) *stepRun {
	return &stepRun{t: t, f: f, parts: parts, dev: dev, bufs: map[jit.Operand]backend.Buf{},
		size: map[jit.Operand]int64{}, once: map[jit.Operand]bool{}}
}

func (r *stepRun) free() {
	for _, b := range r.bufs {
		b.Free()
	}
}

// step composes and runs the decode step of token tok at position pos, and
// returns its logits and token.
func (r *stepRun) step(pos int, tok int32) ([]float32, int) {
	t := r.t
	s := jit.Step{Rows: 1, Page: 64, Keys: pos + 1}
	d := cost.Device{Device: sm70}
	ccs, err := cost.EstimateCodes(jit.Default(), r.f, s, func(jit.Part) cost.Device { return d }, true)
	if err != nil {
		t.Fatal(err)
	}
	// Every storage at its widest binding of the step before any launch
	// holds a buffer: one grown between two kernels' bindings would leave the
	// first holding the freed one.
	written := map[jit.Operand]bool{}
	need := map[jit.Operand]int64{}
	for _, cc := range ccs {
		for j, b := range cc.Code.Binds {
			at := cc.Binds[j].Storage
			if b.Writes {
				written[at] = true
			}
			need[at] = max(need[at], cc.Binds[j].Reach)
		}
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
	for i, cc := range ccs {
		if cc.Code.GPU == nil {
			t.Fatalf("part %v formed %s for the host", r.parts[cc.Part], cc.Code.Name)
		}
		k := *cc.Code.GPU
		k.Name = fmt.Sprintf("x%02d_%s", i, k.Name)
		kern, err := r.dev.Compile(&k)
		if err != nil {
			t.Fatalf("%s: %v", k.Name, err)
		}
		defer kern.Close()
		var bs []backend.Buf
		var wrote []jit.Operand
		for j, b := range cc.Code.Binds {
			at := cc.Binds[j].Storage
			if w := cc.Binds[j].Words; w != nil {
				at.Name += fmt.Sprintf("#%d", i)
				r.fill(at, max(cc.Binds[j].Reach, int64(4*len(w))), wordBytes(w))
			} else {
				r.fill(at, need[at], r.source(at, b.Operand.Name, written[at], pos, tok))
			}
			bs = append(bs, r.bufs[at])
			if b.Writes {
				wrote = append(wrote, at)
			}
			if b.Writes && at.Kind == jit.OperandValue {
				switch b.Operand.Name {
				case "logits":
					logits = at
				case "token":
					token = at
				}
			}
		}
		runs = append(runs, launch{kern, k.Name, cc.Code.Groups, k.Group[0], bs, wrote})
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

// fill makes storage at a buffer of at least n bytes and writes data into it
// where data is set.
func (r *stepRun) fill(at jit.Operand, n int64, data []byte) {
	n = max(n, int64(len(data))) + 4096
	if b, ok := r.bufs[at]; !ok || r.size[at] < n {
		nb, err := r.dev.Alloc(int(n))
		if err != nil {
			r.t.Fatal(err)
		}
		init := make([]byte, n)
		if ok {
			// A storage that grows keeps what the steps before wrote: the
			// history is one.
			if err := b.Read(init[:r.size[at]]); err != nil {
				r.t.Fatal(err)
			}
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
	if fixed(at) {
		r.once[at] = true
	}
}

// fixed is a source that does not change from step to step.
func fixed(at jit.Operand) bool {
	return at.Kind == jit.OperandWeight || at.Kind == jit.OperandVector || at.Kind == jit.OperandConsts ||
		at.Kind == jit.OperandTable && (at.Name == "pages" || at.Name == "rope.planes" || at.Name == "kv.k" || at.Name == "kv.v")
}

// source is what the engine puts in storage at, bound as the operand named
// name, before the step at position pos runs token tok: nil for what the step
// writes before it reads.
func (r *stepRun) source(at jit.Operand, name string, written bool, pos int, tok int32) []byte {
	t, f, c := r.t, r.f, r.f.Config()
	u32s := func(w ...uint32) []byte { return wordBytes(w) }
	switch at.Kind {
	case jit.OperandWeight, jit.OperandVector:
		blk := int32(-1)
		if at.Of >= 0 && at.Of < len(r.parts) {
			blk = int32(r.parts[at.Of].Block)
		}
		e, ok := f.Find(at.Role, blk, -1)
		if !ok {
			t.Fatalf("no tensor %v of block %d in the container", at.Role, blk)
		}
		qs, _, _ := f.Span(e)
		return qs
	case jit.OperandTable:
		switch at.Name {
		case "rows":
			return u32s(0, 0, uint32(pos+1), uint32(pos))
		case "pages":
			w := make([]uint32, 64)
			for i := range w {
				w[i] = uint32(i)
			}
			return u32s(w...)
		case "q.off":
			return u32s(0)
		case "rope.pos":
			return u32s(1, uint32(pos))
		case "rope.planes":
			p := nn.Rope{NRot: int(c.NRot), Base: float64(c.RopeBase)}.TabPlanes(int(c.NRot) / 2)
			return unsafe.Slice((*byte)(unsafe.Pointer(&p[0])), 4*len(p))
		case "kv.k", "kv.v", "rope.cs":
			return nil
		}
		t.Fatalf("a table %q this gate has no source for", at.Name)
	case jit.OperandValue:
		if written {
			return nil
		}
		if name != "x" {
			t.Fatalf("value %q is read and nothing in the step writes it", name)
		}
		e, ok := f.Find(jlm.RoleTokenEmbd, -1, -1)
		if !ok {
			t.Fatal("no token embedding")
		}
		qs, _, _ := f.Span(e)
		w := 4 * int(c.NEmbd)
		return qs[int(tok)*w : int(tok+1)*w]
	case jit.OperandScratch:
		return nil
	}
	t.Fatalf("a %v storage this gate has no source for", at.Kind)
	return nil
}

func wordBytes(w []uint32) []byte {
	b := make([]byte, 4*len(w))
	for i, x := range w {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}
