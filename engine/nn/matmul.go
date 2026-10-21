//go:build amd64 || arm64

package nn

import (
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// MatMul is the GGUF row-major prefill path: one pass over the weights serving
// a whole batch of tokens. Decode is memory-bound, but prefill reuses each
// weight across the batch, so arithmetic intensity rises and the kernel
// becomes the constraint.
//
// x is [ntok][k] and out is [ntok][nrows], both token-major, because that is
// what the graph wants either side of this call. The kernel works in
// [rows][token], so the transpose happens here, once, on ntok*nrows floats --
// against ntok*nrows*k multiply-accumulates, which is a 1/k tax.
//
// Returns false whenever it cannot help, which is a supported answer: the
// caller falls back to per-token MatVec and the model still produces tokens.
func (f *JIT) MatMul(out []float32, t quant.Type, w []byte, x []float32, nrows, k, ntok int) bool {
	countRowMajor(t) // engine/nn/rowmajor.go: the GGUF-block layout, counted
	if f == nil || ntok < 2 || nrows < 1 || k < 1 || f.cfg.NoGEMM {
		return false
	}
	// The GEMM emits its own kernels, so it must ask whether this CPU has the
	// instructions (VNNI) itself; otherwise a pre-VNNI host takes a SIGILL.
	// There is no pre-VNNI GGUF GEMM: a container never uses this family.
	if !cpu.SupportedNative(t) {
		return false
	}
	// And the GEMM exists only on the tiers that have it: the SSE tier has no
	// GGUF-block GEMM (a container never decodes through it), so a float
	// router that the SSE float matvec serves must not reach an AVX2 GEMM
	// here -- Map would refuse it, but only after emitting it.
	if f.em == nil || !f.em.RowMajorGGUF {
		return false
	}
	if k%cpu.Q8Block != 0 || k%int(t.BlockElems()) != 0 {
		return false
	}
	if len(x) < ntok*k || len(out) < ntok*nrows {
		return false
	}
	// Two different "blocks": the activation scales are per 32 elements for
	// every format, while the weight block is 32 for Q4_0/Q8_0 but a
	// 256-element super-block for the k-quants.
	nbAct := k / cpu.Q8Block
	nb := cpu.BlocksPerRow(t, k)
	f.ensureTuners()
	tile := f.gemmTile(t, nb, nrows, ntok)
	start := time.Now()
	if tile.mr == 0 {
		return false
	}
	// Even the narrowest tile is eight tokens, so a much shorter batch is
	// mostly zeros and loses to the per-token loop. Decline at half padding or
	// worse; a seven-token tail wastes one column of eight and still wins.
	tok := cpu.GEMMTokens(tile.nr)
	if ntok*2 <= tok {
		return false
	}
	main, tail := f.gemmKernels(t, nb, tile)
	if main == nil {
		return false
	}

	// Buffers sized for one token tile. Prefill batches are bounded by the tile
	// width, not by the sequence length, so these do not grow with the prompt.
	f.growGEMM(tok, k, nbAct, nrows)

	groups := nrows / tile.mr
	rest := nrows - groups*tile.mr
	rowBytes := cpu.RowBytes(t, k)
	konst := f.konst[t]
	if konst == nil {
		konst = cpu.KernelConst(t)
		f.konst[t] = konst
	}

	for base := 0; base < ntok; base += tok {
		n := min(tok, ntok-base)
		// Always pack a full tile: the packed layout interleaves by k-group, so
		// a short tile would lay the real tokens down at the wrong stride. A
		// short final batch is copied into a zero-padded staging buffer and its
		// junk columns are not read back out.
		src := x[base*k : (base+n)*k]
		if n < tok {
			copy(f.gx[:n*k], src)
			for i := n * k; i < tok*k; i++ {
				f.gx[i] = 0
			}
			src = f.gx[:tok*k]
		}
		// Packed across the pool by k-block, not by token: the layout is
		// k-group major, so one k-group's tok tokens share a cache line and a
		// token split would hand every worker a piece of every line. The packer
		// is a generated kernel (TestPackActMatchesGoPacker).
		var packErr error
		tp := time.Now()
		pcode, pk := f.packKernel(t, tok)
		f.pool.Do(nbAct, max(1, nbAct/(2*f.pool.N())), func(_, blo, bhi int) {
			if err := cpu.PackActJIT(pcode, t, pk, f.gq[:tok*k], f.gscale[:nbAct*tok],
				f.gsum[:nbAct*tok], f.ghalf[:2*nbAct*tok], src, tok, k, blo, bhi, f.actWindow); err != nil {
				packErr = err
			}
		})
		if f.cfg.MatMulProf {
			mmSince(tp, &mmPack)
			mmRegions.Add(1)
		}
		if packErr != nil {
			return false
		}
		tk := time.Now()
		if groups > 0 {
			chunk := max(1, groups/(4*f.pool.N()))
			f.pool.DoLabeled(regionLabel(t, "/gemm"), groups, chunk, func(worker, lo, hi int) {
				args := cpu.Args{
					Out: &f.gout[lo*tile.mr*tok], W: &w[lo*tile.mr*rowBytes],
					A: &f.gq[0], AScale: &f.gscale[0],
					Rows: int64(hi - lo), K: int64(nb), RowStr: int64(rowBytes),
					Scr: &konst[0], Cols: int64(tok), OutStr: int64(tok * 4), ASum: &f.gsum[0],
					AHalfSum: &f.ghalf[0],
					Scratch:  (*byte)(unsafe.Pointer(&f.gscratch[worker*gemmScratch])),
				}
				main.Call(&args)
			})
		}
		if rest > 0 && tail != nil {
			args := cpu.Args{
				Out: &f.gout[groups*tile.mr*tok], W: &w[groups*tile.mr*rowBytes],
				A: &f.gq[0], AScale: &f.gscale[0],
				Rows: int64(rest), K: int64(nb), RowStr: int64(rowBytes),
				Scr: &konst[0], Cols: int64(tok), OutStr: int64(tok * 4), ASum: &f.gsum[0],
				AHalfSum: &f.ghalf[0],
				Scratch:  (*byte)(unsafe.Pointer(&f.gscratch[0])),
			}
			tail.Call(&args)
		}
		if f.cfg.MatMulProf {
			mmSince(tk, &mmKernel)
			mmRegions.Add(1)
		}
		tt := time.Now()
		// [rows][tok] -> [token][rows], across the pool, split by row.
		f.pool.Do(nrows, max(1, nrows/(4*f.pool.N())), func(_, rlo, rhi int) {
			// Blocked, because neither pure order works: row-outer writes at
			// stride nrows (a line per element), and token-outer re-reads all
			// of gout n times (a clear regression measured on prefill).
			// Each block of eight rows is one sequential copy of gout into a
			// small buffer that stays in L1, then transposed out in 64-byte
			// runs.
			const rb = 8
			var buf [rb * 8 * maxGEMMNr]float32
			for r0 := rlo; r0 < rhi; r0 += rb {
				r1 := min(r0+rb, rhi)
				copy(buf[:(r1-r0)*tok], f.gout[r0*tok:r1*tok])
				for i := 0; i < n; i++ {
					dst := out[(base+i)*nrows : (base+i+1)*nrows]
					for r := r0; r < r1; r++ {
						dst[r] = buf[(r-r0)*tok+i]
					}
				}
			}
		})
		if f.cfg.MatMulProf {
			mmSince(tt, &mmTrans)
			mmRegions.Add(1)
		}
	}
	if f.cfg.MatMulProf {
		mmSince(start, &mmTotal)
		mmCalls.Add(1)
	}
	// One real call, one sample. The tuner compares rates rather than
	// durations because this shape is shared by matmuls of very different row
	// counts.
	macs, took := float64(nrows)*float64(ntok)*float64(k), time.Since(start)
	f.gemmObserve(t, nb, nrows, ntok, tile, macs, took)

	return true
}

// The phases of MatMul (pack, kernel, transpose), timed separately, enabled by
// WithMatMulProfile so the ordinary path pays one predictable branch. As with
// ProfileNanos, the totals are process-wide by API and the gate
// (Config.MatMulProf) is per JIT.
var (
	mmPack    atomic.Int64
	mmKernel  atomic.Int64
	mmTrans   atomic.Int64
	mmTotal   atomic.Int64
	mmCalls   atomic.Int64
	mmRegions atomic.Int64
)

// MatMulProfile returns the nanoseconds spent in each phase and the number of
// pool regions dispatched, then resets. Zero everywhere unless it is armed.
func MatMulProfile() (pack, kernel, transpose, total, calls, regions int64) {
	return mmPack.Swap(0), mmKernel.Swap(0), mmTrans.Swap(0),
		mmTotal.Swap(0), mmCalls.Swap(0), mmRegions.Swap(0)
}

func mmSince(t time.Time, c *atomic.Int64) {
	c.Add(int64(time.Since(t)))
}

// maxGEMMNr bounds the tile's token width so the transpose buffer can be a
// fixed-size array: tok is 8*nr and gemmCandidates tops out at nr=6.
const maxGEMMNr = 6

// gemmScratch is the per-participant staging area, in float32 lanes. It is
// cpu.GEMMScratchMax, derived beside the emitter's layout so it cannot drift.
const gemmScratch = cpu.GEMMScratchMax

func (f *JIT) growGEMM(tok, k, nb, nrows int) {
	if len(f.gq) < tok*k {
		f.gq = make([]int8, tok*k)
		f.gx = make([]float32, tok*k)
	}
	if len(f.gscale) < nb*tok {
		f.gscale = make([]float32, nb*tok)
		f.gsum = make([]int32, nb*tok)
		f.ghalf = make([]int32, 2*nb*tok)
	}
	if len(f.gout) < nrows*tok {
		f.gout = make([]float32, nrows*tok)
	}
	f.ensureScratch()
}

// ensureScratch sizes the per-worker staging area. Decode needs it too: the
// amd64 k-quant kernels stage sub-block scales in the red zone below RSP, and
// AAPCS64 has no red zone, so the arm64 kernels take the space through
// Args.Scratch. Each worker gets its own slice; sharing one would be a data
// race whose symptom is a slightly wrong logit.
func (f *JIT) ensureScratch() {
	if n := f.pool.Max() * gemmScratch; len(f.gscratch) < n {
		f.gscratch = make([]float32, n)
	}
}

// scratchFor is worker w's private staging area.
func (f *JIT) scratchFor(w int) *byte {
	return (*byte)(unsafe.Pointer(&f.gscratch[w*gemmScratch]))
}

// gemmKernels returns the tiled kernel and, when the row count is not a
// multiple of the tile height, a one-row kernel for the remainder. Padding the
// weights instead is not an option: they are a read-only mapping.
func (f *JIT) gemmKernels(t quant.Type, nb int, tl gemmTile) (main, tail *cpu.Code) {
	main = f.gemmKernel(t, nb, tl.mr, tl.nr)
	if main == nil {
		return nil, nil
	}
	if tl.mr > 1 {
		tail = f.gemmKernel(t, nb, 1, tl.nr)
	}
	return main, tail
}

func (f *JIT) gemmKernel(t quant.Type, nb, mr, nr int) *cpu.Code {
	key := gemmKey{t: t, nb: nb, mr: mr, nr: nr}
	if c, ok := f.gemm[key]; ok {
		return c
	}
	// The kernel is told the activation window: at 256 the arm64 k-quant fold
	// loads each super-block's activation scale once, at 32 it must not. The
	// packer is given the same f.actWindow, which keeps the two in agreement.
	b, err := cpu.EmitGEMMWindow(t, mr, nr, nb, f.actWindow)
	if err != nil {
		f.gemm[key] = nil
		return nil
	}
	c, err := cpu.MapNamed(b, t.String()+"_gemm"+itoaN(mr)+"x"+itoaN(nr)+"_k"+itoaN(nb))
	if err != nil {
		f.gemm[key] = nil
		return nil
	}
	f.gemm[key] = c
	return c
}

// packKernel returns the generated activation packer for this (type, tile
// width), emitting it on first use and caching it for the life of the JIT.
//
// Keyed on (tok, needs-half-sums), which is what the kernel bakes; the quant
// type enters only through the bias constant, which is data.
func (f *JIT) packKernel(t quant.Type, tok int) (*cpu.Code, []float32) {
	half := cpu.NeedsHalfSums(t)
	f.packMu.Lock()
	defer f.packMu.Unlock()
	if f.packKonst == nil {
		f.packKonst = map[quant.Type][]float32{}
	}
	k, ok := f.packKonst[t]
	if !ok {
		k = cpu.PackActConsts(t)
		f.packKonst[t] = k
	}
	key := packKey{tok, half}
	if c, ok := f.packCode[key]; ok {
		return c, k
	}
	if f.packCode == nil {
		f.packCode = map[packKey]*cpu.Code{}
	}
	c, err := cpu.MapNamed(cpu.EmitPackAct(tok, half), "packact")
	if err != nil {
		c = nil
	}
	f.packCode[key] = c
	return c, k
}

type packKey struct {
	tok  int
	half bool
}
