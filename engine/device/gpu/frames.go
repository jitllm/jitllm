package gpu

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// cardFrames is a card's frames over the files of the blocks placed on it:
// the card reads a block's page from the file into memory of its own --
// page-locked where its backend pins host memory -- uploads it, and drops
// the page once nothing on the card reads it there again.
type cardFrames struct {
	by map[*jlm.File]*jlm.Frames
	// stage is each frames' staging (stagePages of its largest page), capped
	// the page budget the last plan of its file carried, and held what it
	// keeps beyond its staging: the banks of the blocks this card streams,
	// within cap (sizeFrames).
	stage, capped, held map[*jlm.File]uint64
}

// stagePages is how many of a file's largest pages a card's frames keep
// between uses: the block being uploaded and the next one read beside it.
// More is a host cache the card does not read from (its blocks are on the
// card); fewer serialises every read behind an upload.
const stagePages = 2

// framesOf is this card's frames over f, made the first time a block of f
// is offered: dense entries are read when a block asks for them, never the
// region whole, and a frame given back drops whatever the card keyed on its
// address.
func (g *devTier) framesOf(f *jlm.File) (*jlm.Frames, error) {
	if f == nil {
		return nil, nil
	}
	g.framesMu.Lock()
	defer g.framesMu.Unlock()
	if fr := g.frames.by[f]; fr != nil {
		return fr, nil
	}
	opts := []jlm.FramesOption{jlm.DenseOnDemand()}
	if hp, ok := g.dev.(backend.HostPinner); ok {
		opts = append(opts, jlm.WithFrameMemory(pinnedFrames{hp}))
	}
	fr, err := jlm.NewFrames(f, opts...)
	if err != nil {
		return nil, fmt.Errorf("gpu: %s: frames over %s: %w", g.dev.Name(), f.Path, err)
	}
	var big uint64
	for p := 0; p < f.NPages(); p++ {
		_, n := f.PageBounds(p)
		big = max(big, n)
	}
	fr.SetBudget(stagePages * big)
	if g.frames.stage == nil {
		g.frames.stage, g.frames.capped, g.frames.held = map[*jlm.File]uint64{}, map[*jlm.File]uint64{}, map[*jlm.File]uint64{}
	}
	g.frames.stage[f] = stagePages * big
	fr.OnRecycle(func(p unsafe.Pointer, n uintptr) {
		lo := uintptr(p)
		g.forget(lo, lo+n)
	})
	if g.frames.by == nil {
		g.frames.by = map[*jlm.File]*jlm.Frames{}
	}
	g.frames.by[f] = fr
	return fr, nil
}

// offerFrames is the frames the card reads offered block li into, and w
// bound at them, claimed and unread (nn.LayerWeights.Bind), so admission
// reads shapes and lengths and nothing is read until the card admits. The
// frame is pinned until the caller releases the lease: nothing else read
// into the frames evicts it between the claim and the upload. nil frames
// for weights from no file (a caller that binds its own).
func (g *devTier) offerFrames(li int, p *nn.LayerPlan, w *nn.LayerWeights) (*jlm.Frames, jlm.Lease, error) {
	if w == nil || p.File == nil {
		return nil, jlm.Lease{}, nil
	}
	fr, err := g.framesOf(p.File)
	if err == nil {
		g.framesMu.Lock()
		g.frames.capped[p.File] = p.FrameBudget
		g.framesMu.Unlock()
	}
	if err != nil || w.Bind == nil {
		return fr, jlm.Lease{}, err
	}
	if err := w.Bind(w, fr); err != nil {
		return fr, jlm.Lease{}, err
	}
	l, err := fr.Hold(li, nil)
	return fr, l, err
}

// SetFrameBudget is nn.FrameBudgeter: every card's frames over f capped at
// bytes from here, and re-sized to it.
func (g *GPU) SetFrameBudget(f *jlm.File, bytes uint64) {
	for _, d := range g.devs {
		d.framesMu.Lock()
		_, has := d.frames.by[f]
		if has {
			d.frames.capped[f] = bytes
		}
		d.framesMu.Unlock()
		if has {
			d.sizeFrames(f)
		}
	}
}

// ownWeight is w read into this card's frames over f, where w is a dense
// entry of f -- the head, a table, a stream's tensor -- from the file into
// the card's own staging rather than from another device's memory; release
// lets the staging go once the upload has copied it. w itself, with a
// release that does nothing, for a weight that is no dense entry of f.
func (g *devTier) ownWeight(f *jlm.File, w nn.Weight) (nn.Weight, func(), error) {
	if f == nil || w.Entry == nil || !f.InDense(w.Entry) {
		return w, func() {}, nil
	}
	fr, err := g.framesOf(f)
	if err != nil {
		return w, func() {}, err
	}
	qs, d, sc := fr.Span(w.Entry)
	if len(qs) == 0 {
		return w, func() {}, fmt.Errorf("reading the entry into the card's frames failed")
	}
	w.Data = qs
	if w.Packed != nil {
		p := *w.Packed
		p.QS, p.D, p.SC = qs, d, sc
		w.Packed = &p
	}
	e := w.Entry
	return w, func() { fr.DropDense(e) }, nil
}

// FramesOf is nn.FramesHolder: the named card's frames over f, nil for a
// card it does not have or that has paged nothing of f.
func (g *GPU) FramesOf(dev string, f *jlm.File) *jlm.Frames {
	d, err := g.device(dev)
	if err != nil {
		return nil
	}
	d.framesMu.Lock()
	defer d.framesMu.Unlock()
	return d.frames.by[f]
}

// sizeFrames sizes this card's frames over f to their staging and the banks
// of the blocks it streams -- a streamed block's experts are read every
// token, and frames that keep them are what keeps that read off the disk --
// capped at the plan's page budget.
func (g *devTier) sizeFrames(f *jlm.File) {
	if f == nil {
		return
	}
	var need uint64
	g.mu.Lock()
	for _, l := range g.layers {
		st := l.stream
		if st == nil || st.hybrid {
			continue
		}
		var each int
		for m := range st.sh {
			for p := range st.sh[m] {
				each += st.sh[m][p]
			}
		}
		need += uint64(each) * uint64(st.nExpert)
	}
	g.mu.Unlock()
	g.framesMu.Lock()
	fr := g.frames.by[f]
	if fr == nil {
		g.framesMu.Unlock()
		return
	}
	held := need
	if c := g.frames.capped[f]; c > 0 && held > c {
		held = c
	}
	g.frames.held[f] = held
	b := g.frames.stage[f] + held
	g.framesMu.Unlock()
	fr.SetBudget(b)
}

// framesHeld is what this card's frames keep of the host's memory beyond
// their staging (sizeFrames): host memory the card spends, which
// HostReserved reports so the CPU device's frames give it up.
func (g *devTier) framesHeld() uint64 {
	g.framesMu.Lock()
	defer g.framesMu.Unlock()
	var n uint64
	for _, h := range g.frames.held {
		n += h
	}
	return n
}

// closeFrames closes every frames this card holds: the model whose blocks
// they paged is gone from the card. Not under g.mu: closing tells the card
// of every frame it gives back (forget).
func (g *devTier) closeFrames() {
	g.framesMu.Lock()
	by := g.frames.by
	g.frames = cardFrames{}
	g.framesMu.Unlock()
	for _, fr := range by {
		fr.Close()
	}
}

// pinnedFrames is a card's page-locked host memory as a frames' memory
// (jlm.FrameMemory): a transfer out of it needs no bounce through a staging
// copy.
type pinnedFrames struct{ hp backend.HostPinner }

func (p pinnedFrames) Map(n int) ([]byte, error) { return p.hp.PinHost(n) }
func (p pinnedFrames) Unmap(b []byte)            { p.hp.UnpinHost(b) }
