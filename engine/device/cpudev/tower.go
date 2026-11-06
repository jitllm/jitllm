package cpudev

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Tower is a vision tower's parts on the CPU device, apart from its blocks
// (nn.TowerParts): each picture class's entry forms, one a sample format,
// the head's and each tap's, composed once when the instance is made, their
// weights -- the tower's tensors, which no page holds -- bound at each run,
// and the call record their runs read.
type Tower struct {
	x     *Exec
	entry []map[uint8]*Form
	head  *Form
	taps  []*Form
	dense Tensors
	tapsW []Tensors
	bound Bound
}

// NewTower composes p's parts on x, their weights read in dense and each
// tap's in taps.
func NewTower(x *Exec, p TowerPlans, dense Tensors, taps []Tensors) (*Tower, error) {
	t := &Tower{x: x, dense: dense, tapsW: taps}
	for _, cl := range p.Entry {
		forms := map[uint8]*Form{}
		for px, pp := range cl {
			f, err := x.FormGraph(pp)
			if err != nil {
				return nil, fmt.Errorf("the tower's entry: %w", err)
			}
			forms[px] = f
		}
		t.entry = append(t.entry, forms)
	}
	f, err := x.FormGraph(p.Head)
	if err != nil {
		return nil, fmt.Errorf("the tower's head: %w", err)
	}
	t.head = f
	for j, pp := range p.Taps {
		f, err := x.FormGraph(pp)
		if err != nil {
			return nil, fmt.Errorf("the tower's tap %d: %w", j, err)
		}
		t.taps = append(t.taps, f)
	}
	if len(t.taps) != len(taps) {
		return nil, fmt.Errorf("the tower's %d taps read %d tensors", len(t.taps), len(taps))
	}
	return t, nil
}

// run runs f over n rows at request req and positions pos, its weights read
// in w, its picture's words pic, its graph's inputs and outputs io.
func (t *Tower) run(f *Form, w Tensors, n int, req, pos, pic []uint32, io ...[]float32) error {
	ws, why := t.x.Bindings().Bind(f, -1, w)
	if why != nn.DeclineServes {
		return fmt.Errorf("its weights: %v", why)
	}
	b := &t.bound
	*b = Bound{Li: -1, Own: t.x.Device().Own(f), N: n, Req: req, Pos: pos, Pic: pic, Weights: ws}
	for i, x := range io {
		b.IO[i] = unsafe.Pointer(&x[0])
	}
	t.x.Runner().Run(b, f)
	return nil
}

// Entry runs picture class class's entry of sample format px over the
// picture's words pic into a square's n patch rows (nn.TowerParts).
func (t *Tower) Entry(class int, px uint8, pic, req, pos []uint32, n int, rows []float32) error {
	if class < 0 || class >= len(t.entry) || t.entry[class][px] == nil {
		return fmt.Errorf("the tower's entry has no form for class %d's samples of format %d", class, px)
	}
	if err := t.run(t.entry[class][px], t.dense, n, req, pos, pic, rows); err != nil {
		return fmt.Errorf("the tower's entry: %w", err)
	}
	return nil
}

// Head runs the tower's head over a square's n rows the last block left in
// in, into out (nn.TowerParts).
func (t *Tower) Head(n int, req, pos []uint32, in, out []float32) error {
	if err := t.run(t.head, t.dense, n, req, pos, nil, in, out); err != nil {
		return fmt.Errorf("the tower's head: %w", err)
	}
	return nil
}

// Tap runs tap j over the n rows its block left in in, into out
// (nn.TowerParts).
func (t *Tower) Tap(j, n int, req, pos []uint32, in, out []float32) error {
	if j < 0 || j >= len(t.taps) {
		return fmt.Errorf("the tower has no tap %d", j)
	}
	if err := t.run(t.taps[j], t.tapsW[j], n, req, pos, nil, in, out); err != nil {
		return fmt.Errorf("the tower's tap %d: %w", j, err)
	}
	return nil
}
