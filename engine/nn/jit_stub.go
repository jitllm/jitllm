//go:build !(amd64 || arm64)

package nn

import (
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// JIT is the generated-code tier. This platform has no code generator, so it
// is always nil and a model cannot run here (Available reports false).
//
// This file must track the amd64/arm64 JIT method for method: it is the only
// definition of *JIT on other platforms, so a missing method fails to compile
// in model, far from here. Every method below is the f == nil branch of its
// sibling.
type JIT struct{}

// Option and the With...() constructors exist here so model/ compiles: a
// caller's knobs are the same words on every platform, and there is simply no
// tier for them to configure. See engine/nn/options.go for what each one means.
type Option func(*Config)

// Config is engine/nn/options.go's, minus the fields only an emitter reads.
type Config struct{}

func NewJIT(maxK, maxRows int, types []quant.Type, _ ...Option) *JIT { return nil }

// TuneMode is engine/nn/options.go's; there is nothing to tune here.
type TuneMode int

const (
	TuneAuto TuneMode = iota
	TuneOff
	TuneForce
)

func WithSched(...sched.Option) Option { return func(*Config) {} }
func WithProfile(bool) Option          { return func(*Config) {} }
func WithMatMulProfile(bool) Option    { return func(*Config) {} }
func WithTune(TuneMode) Option         { return func(*Config) {} }
func WithQuietTuner(bool) Option       { return func(*Config) {} }
func WithTileTokens(int) Option        { return func(*Config) {} }
func WithChunkBytes(int) Option        { return func(*Config) {} }
func WithGEMMTokens(int) Option        { return func(*Config) {} }
func WithGEMMRows(int) Option          { return func(*Config) {} }
func WithGEMMExact(bool) Option        { return func(*Config) {} }
func WithFusedKSplit(int) Option       { return func(*Config) {} }
func WithFusedKSlices(int) Option      { return func(*Config) {} }
func WithMatVecPick(int) Option        { return func(*Config) {} }
func WithMixtureBatch(int) Option      { return func(*Config) {} }
func WithPackWidth(int) Option         { return func(*Config) {} }
func WithAccs(int) Option              { return func(*Config) {} }
func WithPrefetch(int) Option          { return func(*Config) {} }
func WithA64Prefetch(int) Option       { return func(*Config) {} }
func WithGEMMTile(string) Option       { return func(*Config) {} }
func WithPrefillChunk(int) Option      { return func(*Config) {} }
func WithParticipants(int) Option      { return func(*Config) {} }
func WithPrefillCoreSet(string) Option { return func(*Config) {} }
func WithoutGEMM(bool) Option          { return func(*Config) {} }
func WithActWindow(int) Option         { return func(*Config) {} }

// Argmax32JIT declines: there is no code generator on this platform at all.
func Argmax32JIT(x []float32) (int32, bool) { return 0, false }

// ProfileNanos has nothing to report: no generated code runs on this platform.
func ProfileNanos() (quantize, kernel, out int64) { return 0, 0, 0 }

// ResetForTest is a no-op without an emitter.
func ResetForTest() {}

func (f *JIT) Types() []quant.Type { return nil }
func (f *JIT) Close() error        { return nil }
func (f *JIT) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	countRowMajor(t)
	return false
}

// NewInput marks the activation buffer as overwritten. Nothing is cached here,
// so there is nothing to invalidate.
func (f *JIT) NewInput() {}

// Parallel runs fn over [0,n). There is no pinned pool without sched, so it
// runs inline on the calling goroutine — the same shape the amd64 version takes
// when f is nil.
func (f *JIT) Parallel(n, chunk int, fn func(lo, hi int)) {
	if n > 0 {
		fn(0, n)
	}
}

// Workers is the pool size, 1 when unthreaded.
func (f *JIT) Workers() int { return 1 }

// AddShape generates a shape-specialized kernel. There are none here.
func (f *JIT) AddShape(t quant.Type, k int) {}

// MatMul is the batched prefill path. There is no generated code here.
func (f *JIT) MatMul(out []float32, t quant.Type, w []byte, x []float32, nrows, k, ntok int) bool {
	countRowMajor(t)
	return false
}

// PrefillChunk has nothing to tune: there is no batched path here.
func (f *JIT) PrefillChunk() int { return 32 }

// ObserveChunk has nothing to tune.
func (f *JIT) ObserveChunk(width, ntok int, d time.Duration) {}
func (f *JIT) BeginPrefill()                                 {}
func (f *JIT) EndPrefill()                                   {}

// AddAttn, AttnScores and AttnAcc have no generated code here.
func (f *JIT) AddAttn(hd, kvStride int, kv cpu.KVFmt)         {}
func (f *JIT) AddAttnKV(hdK, hdV, kvStride int, kv cpu.KVFmt) {}

// AddDelta, Conv1d and GatedDelta likewise: no generated code here.
func (f *JIT) AddDelta(n int)                                                     {}
func (f *JIT) AddConv1d(taps, chans int)                                          {}
func (f *JIT) Conv1d(out, state, wT, x []float32, taps, chans int)                { unsupported() }
func (f *JIT) AddDWConv(s cpu.DWShape)                                            {}
func (f *JIT) DWConvRow(out, in, w []float32, s cpu.DWShape)                      { unsupported() }
func (f *JIT) GatedDelta(o, state, k, q, v []float32, decay, gate float32, n int) { unsupported() }
func (f *JIT) AddDeltaChan(n int)                                                 { unsupported() }
func (f *JIT) AddSSD(n int)                                                       { unsupported() }
func (f *JIT) AddSelScan(n int)                                                   { unsupported() }
func (f *JIT) SelScan(o, state, b, c, x, dt, a, d []float32, rows, n int)         { unsupported() }
func (f *JIT) GatedSSD(o, state, b, c, x []float32, decay, dt, d float32, rows, n int) {
	unsupported()
}
func (f *JIT) GatedDeltaChan(o, state, k, q, v, decay []float32, gate float32, n int) {
	unsupported()
}
func (f *JIT) KVWidthPaysOff(hd, kvStride int) bool                       { return false }
func (f *JIT) AttnScores(scores []float32, k, q []float32, npos int) bool { return false }
func (f *JIT) AttnAcc(out, v, att []float32, npos int) bool               { return false }
func (f *JIT) AttnAccInto(out, v, att []float32, npos int) bool           { return false }

func (f *JIT) AttnScores2(s0, s1, k, q0, q1 []float32, npos int) bool  { return false }
func (f *JIT) AttnAcc2(o0, o1, v, w0, w1 []float32, npos int) bool     { return false }
func (f *JIT) AttnAcc2Into(o0, o1, v, w0, w1 []float32, npos int) bool { return false }

// AttnSet is one attention geometry's kernels; none are generated here.
type AttnSet struct{}

func (f *JIT) Attn() *AttnSet                                           { return nil }
func (f *JIT) AttnSetFor(hdK, hdV, kvStride int, kv cpu.KVFmt) *AttnSet { return nil }
func (a *AttnSet) AttnScores(scores []float32, k, q []float32, npos int) {
	unsupported()
}
func (a *AttnSet) AttnAcc(out, v, att []float32, npos int)                 { unsupported() }
func (a *AttnSet) AttnAccInto(out, v, att []float32, npos int)             { unsupported() }
func (a *AttnSet) AttnScores2(s0, s1, k, q0, q1 []float32, npos int) bool  { return false }
func (a *AttnSet) AttnAcc2(o0, o1, v, w0, w1 []float32, npos int) bool     { return false }
func (a *AttnSet) AttnAcc2Into(o0, o1, v, w0, w1 []float32, npos int) bool { return false }

// TokenStart and TokenEnd bracket a decode step for the width tuner; there is
// nothing to tune here.
func (f *JIT) TokenStart() {}
func (f *JIT) TokenEnd()   {}

// DiscardToken has no duel to tell.
func (f *JIT) DiscardToken() {}

// MaxGroup, MVGroup and MatVecPackedGroup: no packed kernels here.
const MaxGroup = 3

type MVGroup struct {
	N     int
	Out   [MaxGroup][]float32
	T     [MaxGroup]quant.Type
	P     [MaxGroup]*Packed
	Rows  [MaxGroup]int
	X     []float32
	K     int
	ActUp []float32
	Act   ActKind
	Resid []float32
	Alpha float32
}

func (f *JIT) MatVecPackedGroup(g *MVGroup) bool { return false }

// AheadSet, SetAhead, SetLookahead and AheadLines: nothing fuses here, so
// there is no lookahead.
type AheadSet struct {
	P [MaxGroup]*Packed
}

func (f *JIT) SetAhead(s *AheadSet) {}
func (f *JIT) SetLookahead(on bool) {}
func (f *JIT) AheadLines() int64    { return 0 }

// Fuse, Unfuse and Phases: no pool here, so nothing fuses.
func (f *JIT) Fuse() bool    { return false }
func (f *JIT) Unfuse()       {}
func (f *JIT) Phases() int64 { return 0 }

// Shapes is how many shape-specialized kernels were generated.
func (f *JIT) Shapes() int { return 0 }

func (f *JIT) AddAttnTiled(hd, kvStride, qStride, scoreStride, qt int, kv cpu.KVFmt) {}
func (f *JIT) AttnTiledQt() int                                                      { return 0 }

// AttnTiledSet is one geometry's tiled score kernel; none is generated here.
type AttnTiledSet struct{}

func (f *JIT) AttnTiledFor(hd, kvStride, qStride, scoreStride, qt int, kv cpu.KVFmt) *AttnTiledSet {
	return nil
}
func (a *AttnTiledSet) Qt() int                                      { return 0 }
func (a *AttnTiledSet) Scores(scores, k, q []float32, npos int) bool { return false }
func (f *JIT) AttnScoresTiled(scores, k, q []float32, npos int) bool { return false }

// DeltaGate32JIT has no generated code here.
func DeltaGate32JIT(decay, beta, alpha, dt, b, a []float32) bool { return false }

// DeltaDecayBound32JIT has no generated code here.
func DeltaDecayBound32JIT(decay, alpha, dt, a []float32, lb float32) { unsupported() }

// SSDGate32JIT has no generated code here.
func SSDGate32JIT(decay, dt, alpha, bias, a []float32) { unsupported() }

// Available reports whether this build has a code generator: it does not.
func Available() bool { return false }

// unsupported stops a caller that reached generated code on a host with no
// emitter. There is no interpreted tier to finish the work in.
func unsupported() { panic("jit: no code generator for this architecture") }

// QuantizeKVRows and ReserveKVQuant have no generated code here.
func (f *JIT) QuantizeKVRows(dst, src []float32, hd, n int) { unsupported() }
func (f *JIT) ReserveKVQuant(hd int)                        {}
func (f *JIT) WidenKVRows(dst, src []float32, hd, n int)    { unsupported() }

// MatMulFloat is the float GEMM. There is no generated code here.
func (f *JIT) MatMulFloat(out []float32, t quant.Type, w []byte, x []float32, nrows, k, ntok int) bool {
	return false
}
