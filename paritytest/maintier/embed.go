package maintier

import (
	"encoding/binary"
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// embPend is a prompt chunk whose rows EmbedRows promised and has not yet
// produced: the gather runs INSIDE the next submission whose x is dst, writing
// the chunk's scratch directly, so the rows never cross the bus in either
// direction. Anything else that reaches the device first makes it real on the
// host (materialize) before going on.
type embPend struct {
	dst []float32
	ids []int32
}

// same reports whether x is exactly pe's destination -- the slice the caller
// was promised the rows in, handed back to Layers.
func (pe *embPend) same(x []float32) bool {
	return len(x) > 0 && len(x) == len(pe.dst) && unsafe.SliceData(x) == unsafe.SliceData(pe.dst)
}

// EmbedRows promises dst's first len(ids) rows as the token embedding,
// gathered on the card from the resident head (nn.EmbedDevice). It needs a
// head that is also the embedding (nn.Head.Embeds) in a format
// kernels.GetRowsPacked reads; otherwise it returns false and writes nothing.
//
// The gather rides the next submission rather than running here: reading the
// rows back into a fresh prompt buffer cost more than the host lookup it
// replaced. The chunk's Layers call gathers straight into its scratch
// (takeEmbed), which also skips uploading x.
func (g *devTier) EmbedRows(ids []int32, dst []float32) bool {
	n := len(ids)
	g.mu.Lock()
	defer g.mu.Unlock()
	bs := g.bs
	if n == 0 || n > batchWidth || bs == nil || bs.head == nil || !bs.head.ok || !bs.headEmbeds ||
		len(dst) != n*bs.head.k || kernels.IsFloat(bs.head.t) {
		return false
	}
	r := bs.head
	for _, id := range ids {
		if id < 0 || int(id) >= r.nrows {
			g.LastErr = fmt.Sprintf("tier: EmbedRows: token %d outside a vocabulary of %d", id, r.nrows)
			return false
		}
	}
	if !g.embedKernel(bs) {
		return false
	}
	pe := &g.embBuf
	pe.dst, pe.ids = dst, append(pe.ids[:0], ids...)
	g.embPend = pe
	return true
}

// embedKernel makes bs's gather kernel and id buffer if it has none -- here,
// outside any Session, because a Session may not compile. One kernel at the
// widest chunk serves every narrower one (a launch covers fewer rows). A
// scratch rebuilt between EmbedRows and the submission (a KV growth) comes
// back without them, which is why the submission asks again. Callers hold mu.
func (g *devTier) embedKernel(bs *blockScratch) bool {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	if bs.embK != nil {
		return true
	}
	r := bs.head
	es := bs.embScale
	if g.kb.scaleFault == ScaleFaultEmbd {
		es = 1
	}
	kk, err := kernels.GetRowsPacked(r.t, r.nrows, r.k, batchWidth, es)
	var k backend.Kernel
	if err == nil {
		k, err = g.dev.Compile(kk)
	}
	var ib backend.Buf
	if err == nil {
		if ib, err = g.dev.Alloc(batchWidth * 4); err != nil {
			closeKern(k)
		}
	}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	bs.embK, bs.embIds = k, ib
	return true
}

// dropEmbed forgets an unconsumed promise; see layersCall.
func (g *devTier) dropEmbed() {
	g.mu.Lock()
	g.embPend = nil
	g.mu.Unlock()
}

// dropEmbedOf is dropEmbed for session sid, from the router, which holds the
// device's view rather than the session's.
func (g *devTier) dropEmbedOf(sid uint64) {
	g.mu.Lock()
	if ds := g.sess[sid]; ds != nil {
		ds.embPend = nil
	}
	g.mu.Unlock()
}

// takeEmbed settles the promise against a submission of x: it returns it when
// x IS the promised destination (the caller then gathers into the scratch),
// and otherwise makes the rows real on the host first, so a path that uploads
// x -- a row at a time, a narrower chunk -- uploads the embedding.
func (g *devTier) takeEmbed(x []float32) *embPend {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.takeEmbedLocked(x)
}

// takeEmbedLocked is takeEmbed for a caller holding mu -- layersOnce runs
// under it, Session included.
func (g *devTier) takeEmbedLocked(x []float32) *embPend {
	pe := g.embPend
	g.embPend = nil
	if pe == nil {
		return nil
	}
	if pe.same(x) {
		return pe
	}
	g.materializeLocked(pe)
	return nil
}

// materializeLocked runs a promised gather on its own and reads the rows back
// into the promised slice. Callers hold mu.
func (g *devTier) materializeLocked(pe *embPend) {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	bs := g.bs
	if bs == nil || bs.head == nil || !g.embedKernel(bs) {
		return
	}
	r, kern, ib := bs.head, bs.embK, bs.embIds
	n := len(pe.ids)
	if bs.embOut == nil {
		b, err := g.dev.Alloc(batchWidth * r.k * 4)
		if err != nil {
			g.LastErr = err.Error()
			return
		}
		bs.embOut = b
	}
	ob := bs.embOut
	var err error
	g.dev.Session(func(s backend.Session) {
		g.subBytes.embIDs = embIDs(g.subBytes.embIDs, pe.ids, batchWidth, r.nrows)
		if err = s.Write(ib, g.subBytes.embIDs); err != nil {
			return
		}
		threads := kernels.GetRowsThreads(r.t, r.k, n)
		if err = s.Launch(kern, (threads+127)/128, 128, r.qs, r.d, r.sc, ib, ob); err != nil {
			return
		}
		err = s.Read(ob, f32b(pe.dst[:n*r.k]))
	})
	if err != nil {
		g.LastErr = err.Error()
		return
	}
	g.EmbedLaunches++
}

// embIDs is the first rows entries of the id buffer: the promised ids, then
// nrows -- a padding row, which the gather writes as zeros. A launch whose last
// group can run past its rows (materialize) writes the whole buffer, so it
// reads padding and never a stale id; one that cannot writes only its rows.
func embIDs(raw []byte, ids []int32, rows, nrows int) []byte {
	raw = slices.Grow(raw[:0], rows*4)[:rows*4]
	for i := 0; i < rows; i++ {
		v := uint32(nrows)
		if i < len(ids) {
			v = uint32(ids[i])
		}
		binary.LittleEndian.PutUint32(raw[4*i:], v)
	}
	return raw
}
