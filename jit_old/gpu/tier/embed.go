package tier

import (
	"encoding/binary"
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// embPend is a prompt chunk whose rows EmbedRows promised and has not yet
// produced: the gather runs inside the next submission whose x is dst,
// writing the chunk's scratch directly, so the rows never cross the bus.
// Anything else that reaches the device first makes them real on the host
// (materializeLocked) before going on.
type embPend struct {
	dst []float32
	ids []int32
}

// same reports whether x is exactly pe's destination.
func (pe *embPend) same(x []float32) bool {
	return len(x) > 0 && len(x) == len(pe.dst) && unsafe.SliceData(x) == unsafe.SliceData(pe.dst)
}

// EmbedRows promises dst's first len(ids) rows as the token embedding,
// gathered on the card from the resident head (nn.EmbedDevice): a head that
// is also the embedding, in a packed format the gather reads (a float head
// is the host's lookup, which reads the same floats). The gather rides the
// next submission (takeEmbedLocked).
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

// embedKernel makes bs's gather (compose.GatherPacked at batchWidth rows,
// through the tier's kernel cache) and its id buffer where it has none,
// outside any Session. A launch over fewer rows runs fewer workgroups.
// Callers hold mu.
func (g *devTier) embedKernel(bs *blockScratch) bool {
	defer g.scratchWin().close()
	if bs.emb != nil {
		return true
	}
	r := bs.head
	sub := r.t.Info().Sub
	if sub <= 0 || r.k%sub != 0 {
		return false
	}
	row := compose.Row{Width: r.k / sub, Rows: batchWidth}
	w := compose.MatVecWeight{T: r.t, K: r.k, Rows: r.nrows, QS: row.Buf(compose.BufU32), D: row.Buf(compose.BufU32), SC: row.Buf(compose.BufU32)}
	ids, out := row.Buf(compose.BufU32), row.Buf(compose.BufOut)
	if err := row.GatherPacked(w, ids, out, bs.embScale); err != nil {
		g.LastErr = err.Error()
		return false
	}
	k := g.rowKern(&row)
	if k.k == nil {
		return false
	}
	ib, err := g.dev.Alloc(batchWidth * 4)
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	bs.emb, bs.embIds = &k, ib
	return true
}

// embedThreads is the gather's threads over rows tokens.
func embedThreads(r *resident, rows int) int { return rows * (r.k / r.t.Info().Sub) }

// dropEmbed forgets an unconsumed promise; see layersCall.
func (g *devTier) dropEmbed() {
	g.mu.Lock()
	g.embPend = nil
	g.mu.Unlock()
}

// dropEmbedOf is dropEmbed for session sid, from the router.
func (g *devTier) dropEmbedOf(sid uint64) {
	g.mu.Lock()
	if ds := g.sess[sid]; ds != nil {
		ds.embPend = nil
	}
	g.mu.Unlock()
}

// takeEmbed settles the promise against a submission of x: it returns it
// when x is the promised destination (the caller gathers into the scratch),
// and otherwise makes the rows real on the host first.
func (g *devTier) takeEmbed(x []float32) *embPend {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.takeEmbedLocked(x)
}

// takeEmbedLocked is takeEmbed for a caller holding mu.
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

// materializeLocked runs a promised gather on its own and reads the rows
// back into the promised slice. Callers hold mu.
func (g *devTier) materializeLocked(pe *embPend) {
	defer g.scratchWin().close()
	bs := g.bs
	if bs == nil || bs.head == nil || !g.embedKernel(bs) {
		return
	}
	r, kern, ib := bs.head, bs.emb.k, bs.embIds
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
		threads := embedThreads(r, n)
		if err = s.Launch(kern, (threads+bs.emb.group-1)/bs.emb.group, bs.emb.group, r.qs, r.d, r.sc, ib, ob); err != nil {
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

// embIDs is the first rows entries of the id buffer: the promised ids,
// then nrows -- a padding row, which the gather writes as zeros.
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
