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
	// band is each class's band forms (graph.Attrs.PicPass), acc the bands'
	// accumulators (a video pair's second frame the second) and bandOut a
	// band form's own value; Bands counts the band runs.
	band    []map[uint8]*Form
	acc     [2][]uint32
	hacc    [2][]uint32
	bandOut []float32
	Bands   int
	head    *Form
	taps    []*Form
	dense   Tensors
	tapsW   []Tensors
	bound   Bound
	// chain is a chained tower's blocks' forms (nn.TowerChain), chainN the
	// rows each runs over, headN the rows of the head's first input, and
	// io a chained head's call's buffers.
	chain  []*Form
	chainN []int
	headN  int
	io     [][]float32
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
	for _, cl := range p.Band {
		var forms map[uint8]*Form
		for px, pp := range cl {
			f, err := x.FormGraph(pp)
			if err != nil {
				return nil, fmt.Errorf("the tower's entry band: %w", err)
			}
			if forms == nil {
				forms = map[uint8]*Form{}
			}
			forms[px] = f
		}
		t.band = append(t.band, forms)
	}
	t.acc = [2][]uint32{make([]uint32, p.Acc), make([]uint32, p.Acc)}
	t.hacc = [2][]uint32{make([]uint32, p.HAcc), make([]uint32, p.HAcc)}
	t.bandOut = make([]float32, p.BandOut)
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
	for i, pp := range p.Chain {
		f, err := x.FormGraph(pp)
		if err != nil {
			return nil, fmt.Errorf("the tower's block %d: %w", i, err)
		}
		t.chain, t.chainN = append(t.chain, f), append(t.chainN, pp.Rows)
	}
	t.headN = p.Head.Rows
	return t, nil
}

// run runs f over n rows at request req and positions pos, its weights read
// in w (block li's, or no block's for li -1), its picture's words pic, its
// graph's inputs and outputs io.
func (t *Tower) run(f *Form, li int, w Tensors, n int, req, pos, pic []uint32, io ...[]float32) error {
	ws, why := t.x.Bindings().Bind(f, li, w)
	if why != nn.DeclineServes {
		return fmt.Errorf("its weights: %v", why)
	}
	b := &t.bound
	*b = Bound{Li: li, Own: t.x.Device().Own(f), N: n, Req: req, Pos: pos, Pic: pic, Acc: t.acc[0], Acc1: t.acc[1], HAcc: t.hacc[0], HAcc1: t.hacc[1], Weights: ws}
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
	if err := t.run(t.entry[class][px], -1, t.dense, n, req, pos, pic, rows); err != nil {
		return fmt.Errorf("the tower's entry: %w", err)
	}
	return nil
}

// Band runs picture class class's band form key over the band req names
// into its accumulator (nn.TowerParts).
func (t *Tower) Band(class int, key uint8, pic, req, pos []uint32, n int) error {
	if class < 0 || class >= len(t.band) || t.band[class][key] == nil {
		return fmt.Errorf("the tower's entry has no band form %d for class %d", key, class)
	}
	if err := t.run(t.band[class][key], -1, t.dense, n, req, pos, pic, t.bandOut); err != nil {
		return fmt.Errorf("the tower's entry band: %w", err)
	}
	t.Bands++
	return nil
}

// Head runs the tower's head over a square's n rows the last block left in
// in, into out (nn.TowerParts).
func (t *Tower) Head(n int, req, pos []uint32, in, out []float32) error {
	if err := t.run(t.head, -1, t.dense, n, req, pos, nil, in, out); err != nil {
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
	if err := t.run(t.taps[j], -1, t.tapsW[j], n, req, pos, nil, in, out); err != nil {
		return fmt.Errorf("the tower's tap %d: %w", j, err)
	}
	return nil
}

// ChainBlock runs a chained tower's block i over in into out, its weights
// block li's tensors w (nn.TowerChain).
func (t *Tower) ChainBlock(i, li int, w Tensors, req, pos []uint32, in, out []float32) error {
	if i < 0 || i >= len(t.chain) {
		return fmt.Errorf("the tower has no chained block %d", i)
	}
	if err := t.run(t.chain[i], li, w, t.chainN[i], req, pos, nil, in, out); err != nil {
		return fmt.Errorf("the tower's block %d: %w", i, err)
	}
	return nil
}

// ChainHead runs a chained tower's head over the last block's rows and the
// earlier blocks' it reads, in, into out (nn.TowerChain).
func (t *Tower) ChainHead(req, pos []uint32, out []float32, in ...[]float32) error {
	t.io = append(append(t.io[:0], in...), out)
	if err := t.run(t.head, -1, t.dense, t.headN, req, pos, nil, t.io...); err != nil {
		return fmt.Errorf("the tower's head: %w", err)
	}
	return nil
}
