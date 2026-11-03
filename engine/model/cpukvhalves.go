package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// kvHalfRow is the floats a row of the binary16 conversions' forms holds.
const kvHalfRow = 256

// cpuKVHalves is a binary16 history's conversions to and from the floats a
// history leaves and returns as (migrateKVAs), composed for the CPU at the
// instance's placement: the floats packed as the history's pages hold them
// (graph.OpHalves), the halves widened (graph.OpWidenHalves). f and h stage
// the rows a call's floats end inside of.
type cpuKVHalves struct {
	narrow, widen cpuHeadForm
	f             []float32
	h             []uint32
	// ran counts the conversions served.
	ran int
}

// cpuPlaceKV composes the conversions where the instance's history is
// binary16.
func (s *Instance) cpuPlaceKV() {
	st := &s.cpu.kvh
	if s.kvl.fmt != cpu.KVF16 {
		return
	}
	nb := graph.New()
	nb.Out(nb.Emit(graph.OpHalves, []graph.ValueID{nb.In("kv.f", kvHalfRow)},
		[]graph.Result{{Name: "kv.h", Width: kvHalfRow / 2, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)...)
	s.cpuComposeHead(&st.narrow, nb.G, composer.DeclineNoGraph, cpuRowsCap)
	wb := graph.New()
	h := wb.In("kv.h", kvHalfRow/2)
	wb.G.Values[h].Elem = graph.ElemU32
	wb.Out(wb.Emit(graph.OpWidenHalves, []graph.ValueID{h},
		[]graph.Result{{Name: "kv.f", Width: kvHalfRow}}, nil, graph.Attrs{}, nil, nil)...)
	s.cpuComposeHead(&st.widen, wb.G, composer.DeclineNoGraph, cpuRowsCap)
	u := cpuRowsCap
	for _, hf := range []*cpuHeadForm{&st.narrow, &st.widen} {
		if hf.f != nil {
			u = max(u, hf.f.unit, hf.f.rows)
		}
	}
	st.f, st.h = make([]float32, u*kvHalfRow), make([]uint32, u*kvHalfRow/2)
}

// cpuHalves narrows the floats f into the halves h, two a word, or widens h
// into f, through the composed conversions; false where they do not serve.
func (s *Instance) cpuHalves(narrow bool, f []float32, h []uint16) bool {
	st := &s.cpu.kvh
	hf := &st.widen
	if narrow {
		hf = &st.narrow
	}
	form, why := hf.formOf()
	if why != composer.DeclineServes || len(f) != len(h) {
		return false
	}
	run := func(fp, hp unsafe.Pointer, k int) {
		b := &s.cpu.bound
		*b = cpuBound{li: cpuInputRows, own: s.cpuOwn(form), n: k}
		b.io[0], b.io[1] = hp, fp
		if narrow {
			b.io[0], b.io[1] = fp, hp
		}
		s.cpu.call[0] = uint32(k)
		for i := range form.codes {
			s.cpuCall(b, &form.codes[i])
		}
	}
	u := max(form.unit, 1)
	whole := len(f) / kvHalfRow / u * u
	for r0 := 0; r0 < whole; {
		k := min(form.rows/u*u, whole-r0)
		run(unsafe.Pointer(&f[r0*kvHalfRow]), unsafe.Pointer(&h[r0*kvHalfRow]), k)
		r0 += k
	}
	// The rows past a whole number of units, staged in one call of a unit.
	at := whole * kvHalfRow
	if rest := len(f) - at; rest > 0 {
		sf, sh := st.f[:u*kvHalfRow], unsafe.Slice((*uint16)(unsafe.Pointer(&st.h[0])), u*kvHalfRow)
		if narrow {
			clear(sf[copy(sf, f[at:]):])
		} else {
			clear(sh[copy(sh, h[at:]):])
		}
		run(unsafe.Pointer(&sf[0]), unsafe.Pointer(&sh[0]), u)
		if narrow {
			copy(h[at:], sh[:rest])
		} else {
			copy(f[at:], sf[:rest])
		}
	}
	st.ran++
	return true
}
