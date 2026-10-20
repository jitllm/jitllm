package tier

import (
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// PrepHead uploads the output norm and the vocabulary projection.
func (g *devTier) PrepHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	bs := g.bs
	if bs == nil || h == nil {
		return false
	}
	// The head is the model's: a second State on this tier finds it placed.
	// Building it again allocated a second norm and logits buffer over the
	// first ones, leaking both and charging the budget twice.
	if bs.head != nil && bs.head.ok && bs.mvHead.rows == h.W.Rows {
		return g.adoptHead(h)
	}
	g.dropHeadScratch(bs)
	q, ok := quantOf(h.W.T)
	if !ok || h.W.K != bs.p.NEmbd || h.W.Rows <= 0 || len(h.W.Data) == 0 {
		return false
	}
	// Under paging the projection and a block slot compete for the same bytes,
	// and the head is permanent while a slot turns over every block. The head
	// runs once per token and a block NLayer times, so the projection is taken
	// only when it costs no slot.
	if len(g.stream) > 0 {
		want := uint64(len(h.W.Data))
		if now, after := g.slots(), g.slotsIfPerm(want); after < now {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and "+
				"taking them would leave %d block slot(s) instead of %d, so it "+
				"stays on the host: a slot turns over every block and the "+
				"projection runs once a token", want, after, now)
			return false
		}
	}
	// -1: the output projection is not a block, so it is not released with one.
	r := g.resident(-1, 0, q, h.W.Data, h.W.Rows, h.W.K, h.W.Packed)
	if r == nil || !r.ok {
		return false
	}
	split := g.tuneSplit(q, h.W.Rows, h.W.K, r)
	// HeadSplit pins the head's split alone, leaving every block matvec on the
	// tuner: the head's row count is far beyond any block matvec's, and a global
	// split cannot test it in isolation.
	if g.HeadSplit > 0 {
		split = g.HeadSplit
	}
	// The output projection has no bias in any architecture here; false is a
	// statement about nn.Head, not a default.
	kern := g.kernel(q, h.W.K, h.W.Rows, split, 1, false)
	if kern == nil {
		return false
	}
	var red backend.Kernel
	if split > 1 {
		if red = g.reduceKernel(h.W.Rows, split); red == nil {
			split = 1
			if kern = g.kernel(q, h.W.K, h.W.Rows, 1, 1, false); kern == nil {
				return false
			}
		}
	}
	if !g.sizePart(h.W.Rows * split) {
		return false
	}
	var err error
	// A head with no norm (Qwen3.8-Flash-Next's reads its mix as it is)
	// leaves hNorm nil, and the emit copies the row where the norm would
	// have written.
	if len(h.Norm) > 0 {
		f := make([]float32, len(h.Norm))
		for i, v := range h.Norm {
			f[i] = float32(v)
		}
		if bs.hNorm, err = g.dev.Alloc(len(f) * 4); err != nil {
			return false
		}
		if bs.hNorm.Write(f32b(f)) != nil {
			return false
		}
	} else if bs.copyRows == nil {
		g.LastErr = "a head with no norm, on a scratch with no copy for it"
		return false
	}
	// A LayerNorm head needs its bias; an RMSNorm one has none.
	if lnBlock(&bs.p) {
		if len(h.NormB) != len(h.Norm) {
			g.LastErr = fmt.Sprintf("a LayerNorm head with a %d-wide bias for a %d-wide norm",
				len(h.NormB), len(h.Norm))
			return false
		}
		if bs.hNormB, err = g.dev.Alloc(len(h.NormB) * 4); err != nil {
			return false
		}
		if bs.hNormB.Write(f32b(h.NormB)) != nil {
			return false
		}
		g.charge(uint64(len(h.NormB) * 4))
	}
	if bs.logits, err = g.dev.Alloc(h.W.Rows * 4); err != nil {
		return false
	}
	bs.hRaw = make([]byte, h.W.Rows*4)
	bs.hOut = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hRaw[0])), h.W.Rows)
	bs.head, bs.mvHead = r, mv{kern: kern, red: red, rows: h.W.Rows, split: split}
	g.shapeChanged()
	g.headSrc, g.headNormHost = &h.W.Data[0], slices.Clone(h.Norm)
	g.charge(uint64(len(h.Norm)*4 + h.W.Rows*4))
	bs.headEmbeds, bs.embScale = h.Embeds, h.EmbdScale
	if bs.embScale == 0 {
		bs.embScale = 1
	}
	// The final softcap, which the contract on nn.Head.Softcap makes this
	// tier's to apply once it accepts the head -- so a failure here is a
	// refusal of the head, not a head without its cap.
	if h.Softcap != 0 {
		ck, e := oldkernels.Softcap(h.W.Rows, h.Softcap)
		if e != nil {
			return false
		}
		c, e := g.dev.Compile(ck)
		if e != nil {
			return false
		}
		b, e := g.dev.Alloc(h.W.Rows * 4)
		if e != nil {
			g.closeKern(c)
			return false
		}
		bs.headCap, bs.logitsCap = c, b
		g.charge(uint64(h.W.Rows * 4))
	}
	// The greedy token on the device (nn.Head.ArgmaxOnly). Without it the head
	// still runs and every caller gets its logits, so a failure is not one.
	if g.argmaxK == nil {
		if ak, e := oldkernels.Argmax(h.W.Rows); e == nil {
			if c, e := g.dev.Compile(ak); e == nil {
				w := g.scratchWin() // the device's own word (scratch.go)
				if b, e := g.dev.Alloc(4); e == nil {
					g.argmaxK, g.argmaxOut = c, b
				} else {
					g.closeKern(c)
				}
				w.close()
			}
		}
	}
	g.prewarmHeadKernels()
	return true
}
