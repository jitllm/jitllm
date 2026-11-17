//go:build amd64 || arm64

package probe

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// former forms the probe's measuring kernels: graphs of ops the composer
// forms for the host, as it forms a model's.
type former struct {
	h   *cpu.Target
	r   *composer.Registry
	dev composer.Device
}

// newFormer is the former for the CPU h states.
func newFormer(h *cpu.Target) former {
	return former{h: h, r: composer.Default(), dev: composer.Device{Kind: composer.DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(h.ISA()), ExactDiv: true}}
}

// formed is a probe's kernel: the composer's codes for its graph in launch
// order, mapped, and the instructions of each class the estimator counts
// them issuing.
type formed struct {
	codes  []composer.Code
	mapped []*cpu.Code
	issued [cost.NumClasses]int64
}

// form is graph g formed at rows rows.
func (fm former) form(g *graph.Graph, rows int) (*formed, error) {
	step := composer.Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}
	ks, codes, err := fm.r.FormKernels(composer.Request{Graph: g, Step: step, Device: fm.dev, Apart: true})
	if err != nil {
		return nil, err
	}
	f := &formed{codes: codes}
	for i, k := range ks {
		n := cost.Issued(k, step, cost.Device{Device: fm.dev})
		for cl := range n {
			f.issued[cl] += n[cl]
		}
		if codes[i].CPU == nil {
			f.close()
			return nil, fmt.Errorf("probe: %s has no host code", codes[i].Name)
		}
		m, err := fm.h.Map(codes[i].CPU)
		if err != nil {
			f.close()
			return nil, err
		}
		f.mapped = append(f.mapped, m)
	}
	return f, nil
}

// close unmaps f's codes.
func (f *formed) close() {
	for _, m := range f.mapped {
		m.Close()
	}
}

// count is the instructions of class cl one run of f issues.
func (f *formed) count(cl cost.Class) float64 { return float64(f.issued[cl]) }

// argWords is an argument block, cpu.Args' layout as words: a word may be a
// count, or an address past a buffer's end, which in one of cpu.Args'
// pointer fields the collector takes for a bad pointer.
type argWords [(unsafe.Sizeof(cpu.Args{}) + 7) / 8]uint64

// bound is f with its argument blocks filled, one a call, for one run.
type bound struct {
	f      *formed
	blocks [][]argWords
	keep   [][]uint32
}

// bind is f's argument blocks, value v of its graph at vals[v]: each call's
// words as the composer states them, on one worker.
func (f *formed) bind(vals map[graph.ValueID]unsafe.Pointer) (*bound, error) {
	b := &bound{f: f, blocks: make([][]argWords, len(f.codes))}
	scratch := map[composer.Operand][]uint32{}
	for ci := range f.codes {
		c := &f.codes[ci]
		n, rows := 1, 0
		if c.Share > 0 {
			rows = c.Share
		} else {
			for _, x := range c.Calls {
				n *= x
			}
		}
		b.blocks[ci] = make([]argWords, n)
		for at := range n {
			w := &b.blocks[ci][at]
			for _, a := range c.Args {
				if a.Field < 0 || a.Field%8 != 0 || a.Field/8 >= len(w) {
					return nil, fmt.Errorf("probe: %s has a call word at %d past the argument block", c.Name, a.Field)
				}
				slot := &w[a.Field/8]
				switch {
				case a.Coord > 0:
					*slot = uint64(coordOf(c.Calls, rows, at, a.Coord-1))
					continue
				case a.Share:
					*slot = uint64(rows)
					continue
				case a.Length || a.Value:
					*slot = uint64(a.Extent)
					continue
				}
				var p unsafe.Pointer
				switch o := a.Operand; {
				case a.Words != nil:
					words := append([]uint32(nil), a.Words...)
					b.keep = append(b.keep, words)
					if a.Word {
						*slot = uint64(words[a.Offset])
						continue
					}
					p = unsafe.Pointer(&words[0])
				case o.Kind == composer.OperandValue && vals[o.Value] != nil:
					p = vals[o.Value]
				case o.Kind == composer.OperandScratch:
					if scratch[o] == nil {
						size := 1 << 16
						for _, bd := range c.Binds {
							if bd.Operand == o && bd.Bytes > 0 {
								size = int(bd.Bytes)/4 + 1
							}
						}
						scratch[o] = make([]uint32, size)
						b.keep = append(b.keep, scratch[o])
					}
					p = unsafe.Pointer(&scratch[o][0])
				default:
					return nil, fmt.Errorf("probe: %s binds %+v, which no probe provides", c.Name, o)
				}
				elem := 4
				if a.Elem > 0 {
					elem = a.Elem
				}
				off := a.Offset
				for ax, per := range a.Per {
					off += coordOf(c.Calls, rows, at, ax) * per
				}
				*slot = uint64(uintptr(p)) + uint64(off*elem)
			}
		}
	}
	return b, nil
}

// coordOf is call at's coordinate along axis ax of calls: 0 for a call of a
// code whose rows are one call's.
func coordOf(calls []int, rows, at, ax int) int {
	if rows > 0 || ax >= len(calls) {
		return 0
	}
	for a := len(calls) - 1; a > ax; a-- {
		at /= calls[a]
	}
	return at % calls[ax]
}

// run calls every code of b once at each of its calls, in launch order.
func (b *bound) run() {
	for ci, m := range b.f.mapped {
		for i := range b.blocks[ci] {
			m.Call((*cpu.Args)(unsafe.Pointer(&b.blocks[ci][i])))
		}
	}
}
