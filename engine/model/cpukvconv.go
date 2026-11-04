package model

import (
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// kvHalfRow is the floats a row of the binary16 conversions' forms holds.
const kvHalfRow = 256

// cpuKVConv is one of a history's conversions, composed for the CPU: rows of
// fw floats into rows of hw words as the history's pages hold them, and back.
type cpuKVConv struct {
	fw, hw        int
	narrow, widen cpuHeadForm
}

// cpuKVConvs is a history's conversions to and from the floats it leaves and
// returns as (migrateKVAs), composed for the CPU at the instance's placement:
// a binary16 history's halves (graph.OpHalves, graph.OpWidenHalves) in rows
// of kvHalfRow floats, a q8 history's rows (graph.OpQ8Rows,
// graph.OpWidenQ8Rows) one a head width its layers hold. f and h stage the
// rows a call's floats end inside of.
type cpuKVConvs struct {
	conv []cpuKVConv
	f    []float32
	h    []uint32
	// ran counts the conversions served.
	ran int
}

// cpuPlaceKV composes the conversions where the instance's history is
// binary16 or q8.
func (s *Instance) cpuPlaceKV() {
	st := &s.cpu.kvconv
	st.conv = nil
	narrow, widen := graph.OpHalves, graph.OpWidenHalves
	var widths []int
	switch s.kvl.fmt {
	case cpu.KVF16:
		widths = []int{kvHalfRow}
	case cpu.KVQ8:
		narrow, widen = graph.OpQ8Rows, graph.OpWidenQ8Rows
		for li := s.lo; li < s.hi; li++ {
			if hd := s.kvlAt(li).headDim; !slices.Contains(widths, hd) {
				widths = append(widths, hd)
			}
		}
	default:
		return
	}
	nf, nh := 0, 0
	for _, fw := range widths {
		c := cpuKVConv{fw: fw, hw: fw / 2}
		if s.kvl.fmt == cpu.KVQ8 {
			c.hw = graph.Q8RowWords(fw)
		}
		nb := graph.New()
		nb.Out(nb.Emit(narrow, []graph.ValueID{nb.In("kv.f", c.fw)},
			[]graph.Result{{Name: "kv.h", Width: c.hw, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)...)
		s.cpuComposeHead(&c.narrow, nb.G, composer.DeclineNoGraph, cpuRowsCap)
		wb := graph.New()
		h := wb.In("kv.h", c.hw)
		wb.G.Values[h].Elem = graph.ElemU32
		wb.Out(wb.Emit(widen, []graph.ValueID{h}, []graph.Result{{Name: "kv.f", Width: c.fw}}, nil, graph.Attrs{}, nil, nil)...)
		s.cpuComposeHead(&c.widen, wb.G, composer.DeclineNoGraph, cpuRowsCap)
		u := cpuRowsCap
		for _, hf := range []*cpuHeadForm{&c.narrow, &c.widen} {
			if hf.f != nil {
				u = max(u, hf.f.unit, hf.f.rows)
			}
		}
		nf, nh = max(nf, u*c.fw), max(nh, u*c.hw)
		st.conv = append(st.conv, c)
	}
	st.f, st.h = make([]float32, nf), make([]uint32, nh)
}

// cpuKV narrows the floats f of head rows hd wide into the history's bytes
// h, or widens h into f, through the composed conversions; false where they
// do not serve.
func (s *Instance) cpuKV(narrow bool, hd int, f []float32, h []byte) bool {
	st := &s.cpu.kvconv
	for i := range st.conv {
		if c := &st.conv[i]; s.kvl.fmt == cpu.KVF16 || c.fw == hd {
			return s.cpuKVConvert(c, narrow, f, h)
		}
	}
	return false
}

// cpuKVConvert runs conversion c over f and h in calls of whole units, the
// rows past them staged in one call of a unit.
func (s *Instance) cpuKVConvert(c *cpuKVConv, narrow bool, f []float32, h []byte) bool {
	st := &s.cpu.kvconv
	hf := &c.widen
	if narrow {
		hf = &c.narrow
	}
	form, why := hf.formOf()
	if why != composer.DeclineServes || len(h)*c.fw != len(f)*4*c.hw {
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
	whole := len(f) / c.fw / u * u
	for r0 := 0; r0 < whole; {
		k := min(form.rows/u*u, whole-r0)
		run(unsafe.Pointer(&f[r0*c.fw]), unsafe.Pointer(&h[r0*c.hw*4]), k)
		r0 += k
	}
	if at := whole * c.fw; at < len(f) {
		hb := at * 4 * c.hw / c.fw
		sf, sh := st.f[:u*c.fw], unsafe.Slice((*byte)(unsafe.Pointer(&st.h[0])), u*c.hw*4)
		if narrow {
			clear(sf[copy(sf, f[at:]):])
		} else {
			clear(sh[copy(sh, h[hb:]):])
		}
		run(unsafe.Pointer(&sf[0]), unsafe.Pointer(&sh[0]), u)
		if narrow {
			copy(h[hb:], sh)
		} else {
			copy(f[at:], sf)
		}
	}
	st.ran++
	return true
}
