//go:build amd64 || arm64

package nn

import (
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The generated attention path. The single-head kernels serve every head
// dimension and cannot decline; the paired and tiled ones are optimisations
// over them and report whether they ran.

// KVWidthPaysOff decides whether a binary16 KV cache is worth it for this
// (head dim, stride), by emitting both kernels and comparing their sizes.
//
// An f16 cache halves the bytes the attention walk reads; the cost is widening
// a half. amd64 widens for free (VCVTPH2PS from memory replaces the load), so
// the kernel does not grow; arm64 needs LDRd plus FCVTL and grows ~25%. This is
// a tie-break, not a rate prediction: a kernel that does not grow and reads half
// the bytes cannot be slower, and the one measurement of a growing kernel
// (arm64) lost, so a growing kernel declines. The 5% band separates the two
// observed populations.
func (f *JIT) KVWidthPaysOff(hd, kvStride int) bool {
	if f == nil || hd <= 0 {
		return false
	}
	// The tier's own kernels: the SSE tier has no F16C, so its f16 kernels are
	// much larger and this declines the f16 cache there.
	a32, err := f.em.AttnScores(hd, kvStride, cpu.KVF32)
	if err != nil {
		return false
	}
	a16, err := f.em.AttnScores(hd, kvStride, cpu.KVF16)
	if err != nil {
		return false
	}
	b32, err := f.em.AttnAcc(hd, kvStride, cpu.KVF32)
	if err != nil {
		return false
	}
	b16, err := f.em.AttnAcc(hd, kvStride, cpu.KVF16)
	if err != nil {
		return false
	}
	grow := float64(len(a16)+len(b16)) / float64(len(a32)+len(b32))
	return grow <= 1.05
}

// AddAttn generates the attention kernels for one (head dimension, KV stride);
// they are the JIT's default set, which AttnScores and the rest run. kv is
// the KV cache's format (f32, binary16 or q8_0): the same kernels with a
// widening load and a narrower stride, baked because the cache's format is
// constant for the session. A model whose layers attend at two geometries (Gemma 4) takes a
// second set from AttnSetFor.
func (f *JIT) AddAttn(hd, kvStride int, kv cpu.KVFmt) { f.AddAttnKV(hd, hd, kvStride, kv) }

// AddAttnKV is AddAttn with the key and value widths given separately. They
// differ only under MLA, where the cached row is KVLoraRank + NRot wide and the
// value is its first KVLoraRank floats, read from the same buffer at the same
// stride.
func (f *JIT) AddAttnKV(hdK, hdV, kvStride int, kv cpu.KVFmt) {
	if f == nil || hdK <= 0 || hdV <= 0 {
		return
	}
	f.attnKV = kv
	f.attn.emit(f, hdK, hdV, kvStride, kv)
}

// AttnSet is one attention geometry's kernels: the single-head score and
// accumulate, the accumulating twin a paged cache sums pages with, and the
// paired twins. Every one bakes (key width, value width, KV stride, cache
// width), so a set serves exactly one geometry.
type AttnSet struct {
	key                           attnKey
	scores, acc, accInto, scores2 *cpu.Code
	acc2, acc2Into                *cpu.Code
}

type attnKey struct {
	hdK, hdV, stride int
	kv               cpu.KVFmt
}

func (a *AttnSet) emit(f *JIT, hdK, hdV, kvStride int, kv cpu.KVFmt) {
	must := func(name string, hd int, b []byte, err error) *cpu.Code {
		if err != nil {
			panic(err)
		}
		return mustMap(name+"_hd"+itoaN(hd), b)
	}
	a.key = attnKey{hdK, hdV, kvStride, kv}
	b, err := f.em.AttnScores(hdK, kvStride, kv)
	a.scores = must("attn_scores", hdK, b, err)
	b, err = f.em.AttnAcc(hdV, kvStride, kv)
	a.acc = must("attn_acc", hdV, b, err)
	// The accumulating twin, for a KV cache split into pages: page 0 overwrites
	// Out and every page after it adds in. See cpu.EmitAttnAccInto for why the
	// result is bit-identical to one contiguous call rather than merely close.
	b, err = f.em.AttnAccInto(hdV, kvStride, kv)
	a.accInto = must("attn_acc_into", hdV, b, err)
	// The paired kernels halve the pass over K and V, on both architectures.
	b, err = f.em.AttnScores2(hdK, kvStride, kv)
	a.scores2 = must("attn_scores2", hdK, b, err)
	b, err = f.em.AttnAcc2(hdV, kvStride, kv)
	a.acc2 = must("attn_acc2", hdV, b, err)
	b, err = f.em.AttnAcc2Into(hdV, kvStride, kv)
	a.acc2Into = must("attn_acc2_into", hdV, b, err)
}

func (a *AttnSet) close() {
	for _, c := range []*cpu.Code{a.scores, a.acc, a.accInto, a.scores2, a.acc2, a.acc2Into} {
		if c != nil {
			c.Close()
		}
	}
	*a = AttnSet{}
}

// Attn is the default set AddAttnKV provisioned.
func (f *JIT) Attn() *AttnSet { return &f.attn }

// AttnSetFor is the kernel set for one geometry: the default set when it is
// that geometry, otherwise one emitted on first ask and kept until Close. The
// set is emitted at placement time, never during a token.
func (f *JIT) AttnSetFor(hdK, hdV, kvStride int, kv cpu.KVFmt) *AttnSet {
	k := attnKey{hdK, hdV, kvStride, kv}
	if f.attn.key == k && f.attn.scores != nil {
		return &f.attn
	}
	for _, a := range f.attnSets {
		if a.key == k {
			return a
		}
	}
	a := &AttnSet{}
	a.emit(f, hdK, hdV, kvStride, kv)
	f.attnSets = append(f.attnSets, a)
	return a
}

// AttnTiledSet is the qt-wide score kernel for one geometry. Every stride is
// baked, so a set is keyed on the exact geometry its caller uses. Only a
// bidirectional caller (a vision segment) can use it: its queries arrive as a
// block against one K, where decode has one query at a time.
type AttnTiledSet struct {
	key    tiledKeyA
	qt     int
	scores *cpu.Code
}

type tiledKeyA struct {
	hd, kvStride, qStride, scoreStride, qt int
	kv                                     cpu.KVFmt
}

// AttnTiledFor is the tiled set for one geometry, emitted on first ask and
// kept until Close; nil when the tier has no such kernel (qt < 2, or the
// emitter declines the tile), which a caller treats as "run the paired
// kernels". A refusal is kept too, so it is asked once.
func (f *JIT) AttnTiledFor(hd, kvStride, qStride, scoreStride, qt int, kv cpu.KVFmt) *AttnTiledSet {
	if f == nil || qt < 2 {
		return nil
	}
	k := tiledKeyA{hd, kvStride, qStride, scoreStride, qt, kv}
	for _, a := range f.attnTiled {
		if a.key == k {
			if a.scores == nil {
				return nil
			}
			return a
		}
	}
	a := &AttnTiledSet{key: k}
	f.attnTiled = append(f.attnTiled, a)
	b, err := f.em.AttnScoresTiled(hd, kvStride, qStride, scoreStride, qt, kv)
	if err != nil {
		return nil
	}
	c, err := cpu.MapNamed(b, "attn_scores_t"+itoaN(qt)+"_hd"+itoaN(hd))
	if err != nil {
		return nil
	}
	a.qt, a.scores = qt, c
	return a
}

// Qt is the query tile the set holds, 0 for a nil set.
func (a *AttnTiledSet) Qt() int {
	if a == nil {
		return 0
	}
	return a.qt
}

// Scores computes scores[j][t] = dot(q[j], K[t]) for the baked qt queries in
// one pass over K. scores and q are the row-0 slices; the strides were baked
// at emit time.
func (a *AttnTiledSet) Scores(scores, k, q []float32, npos int) bool {
	if a == nil || a.scores == nil || npos <= 0 || len(scores) < npos {
		return false
	}
	args := cpu.Args{
		Out: &scores[0], W: (*byte)(unsafe.Pointer(&k[0])),
		Rows: int64(npos), Q32: &q[0],
	}
	a.scores.Call(&args)
	return true
}

// AddAttnTiled makes the tiled set for this geometry the JIT's default, the
// one AttnTiledQt and AttnScoresTiled run.
func (f *JIT) AddAttnTiled(hd, kvStride, qStride, scoreStride, qt int, kv cpu.KVFmt) {
	if f == nil {
		return
	}
	f.attnT = f.AttnTiledFor(hd, kvStride, qStride, scoreStride, qt, kv)
}

// AttnTiledQt is the query tile the JIT's default tiled set holds, 0 when it
// has none.
func (f *JIT) AttnTiledQt() int {
	if f == nil {
		return 0
	}
	return f.attnT.Qt()
}

// AttnScoresTiled runs the default tiled set; see AttnTiledSet.Scores.
func (f *JIT) AttnScoresTiled(scores, k, q []float32, npos int) bool {
	if f == nil {
		return false
	}
	return f.attnT.Scores(scores, k, q, npos)
}

// AttnScores2 computes two heads' score rows from one walk of the K cache:
// s0[t] = dot(q0, k[t*kvStride:]) and s1[t] likewise, for t in [0, npos).
// Returns false when there is no kernel, and the caller makes two single calls.
func (a *AttnSet) AttnScores2(s0, s1, k, q0, q1 []float32, npos int) bool {
	if a == nil || a.scores2 == nil || npos <= 0 || len(s0) < npos || len(s1) < npos {
		return false
	}
	args := cpu.Args{
		Out: &s0[0], W: (*byte)(unsafe.Pointer(&k[0])),
		Rows: int64(npos), Q32: &q0[0], Out2: &s1[0], Q2: &q1[0],
	}
	a.scores2.Call(&args)
	return true
}

// AttnAcc2 accumulates one walk of the V cache into two heads' outputs:
// o0[i] = sum_t w0[t]*v[t*kvStride+i], and o1 likewise.
func (a *AttnSet) AttnAcc2(o0, o1, v, w0, w1 []float32, npos int) bool {
	return a.acc2Run(o0, o1, v, w0, w1, npos, false)
}

// AttnAcc2Into is AttnAcc2 that adds into both outputs, so a paired window split
// across KV pages sums to what one call would have given.
func (a *AttnSet) AttnAcc2Into(o0, o1, v, w0, w1 []float32, npos int) bool {
	return a.acc2Run(o0, o1, v, w0, w1, npos, true)
}

func (a *AttnSet) acc2Run(o0, o1, v, w0, w1 []float32, npos int, into bool) bool {
	if a == nil || npos <= 0 || len(w0) < npos || len(w1) < npos {
		return false
	}
	k := a.acc2
	if into {
		k = a.acc2Into
	}
	if k == nil {
		return false
	}
	args := cpu.Args{
		Out: &o0[0], W: (*byte)(unsafe.Pointer(&v[0])),
		AScale: &w0[0], Rows: int64(npos), Out2: &o1[0], AScale2: &w1[0],
	}
	k.Call(&args)
	return true
}

// AttnScores computes scores[t] = dot(q, k[t*kvStride:]) for t in [0, npos).
func (a *AttnSet) AttnScores(scores []float32, k, q []float32, npos int) {
	if npos <= 0 {
		return
	}
	if len(scores) < npos {
		lengthPanic("attn_scores", npos, len(scores))
	}
	args := cpu.Args{
		Out: &scores[0], W: (*byte)(unsafe.Pointer(&k[0])),
		Rows: int64(npos), Q32: &q[0],
	}
	a.scores.Call(&args)
}

// AttnAcc computes out[i] = sum over t of att[t] * v[t*kvStride+i].
func (a *AttnSet) AttnAcc(out, v, att []float32, npos int) { attnAccRun(out, v, att, npos, a.acc) }

// AttnAccInto is AttnAcc that adds into out. A paged cache calls AttnAcc for the
// first page and this for every page after it, which sums to exactly what one
// call over the whole window would have produced.
func (a *AttnSet) AttnAccInto(out, v, att []float32, npos int) {
	attnAccRun(out, v, att, npos, a.accInto)
}

func attnAccRun(out, v, att []float32, npos int, k *cpu.Code) {
	if npos <= 0 {
		return
	}
	if len(att) < npos {
		lengthPanic("attn_acc", npos, len(att))
	}
	args := cpu.Args{
		Out: &out[0], W: (*byte)(unsafe.Pointer(&v[0])),
		AScale: &att[0], Rows: int64(npos),
	}
	k.Call(&args)
}

// The JIT's own attention calls run its default set (AddAttnKV).

// AttnScores2 is the default set's AttnScores2.
func (f *JIT) AttnScores2(s0, s1, k, q0, q1 []float32, npos int) bool {
	if f == nil {
		return false
	}
	return f.attn.AttnScores2(s0, s1, k, q0, q1, npos)
}

// AttnAcc2 is the default set's AttnAcc2.
func (f *JIT) AttnAcc2(o0, o1, v, w0, w1 []float32, npos int) bool {
	if f == nil {
		return false
	}
	return f.attn.AttnAcc2(o0, o1, v, w0, w1, npos)
}

// AttnAcc2Into is the default set's AttnAcc2Into.
func (f *JIT) AttnAcc2Into(o0, o1, v, w0, w1 []float32, npos int) bool {
	if f == nil {
		return false
	}
	return f.attn.AttnAcc2Into(o0, o1, v, w0, w1, npos)
}

// AttnScores is the default set's AttnScores.
func (f *JIT) AttnScores(scores []float32, k, q []float32, npos int) {
	f.attn.AttnScores(scores, k, q, npos)
}

// AttnAcc is the default set's AttnAcc.
func (f *JIT) AttnAcc(out, v, att []float32, npos int) { f.attn.AttnAcc(out, v, att, npos) }

// AttnAccInto is the default set's AttnAccInto.
func (f *JIT) AttnAccInto(out, v, att []float32, npos int) { f.attn.AttnAccInto(out, v, att, npos) }
