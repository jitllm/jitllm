package nn

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// Rope is a model's rotary configuration: per-model constants that travel
// together, so the pair layout cannot be passed wrong at one call site.
type Rope struct {
	NRot int
	Base float64
	// Neox rotates (i, i+NRot/2) instead of (2i, 2i+1). llama is NORM; gemma,
	// qwen and phi3 are NEOX. See the note on Apply.
	Neox bool
	// Freqs divides theta per pair, from rope_freqs.weight (llama 3.1+) or
	// rope_factors_short.weight (phi3 LongRoPE). nil is no rescaling.
	Freqs []float32
	// Scale multiplies cos and sin -- ggml's mscale, from
	// rope.scaling.attn_factor. Zero means one.
	Scale float64
	// Runs makes the position multi-axis: run u's pairs are turned by
	// coordinate u.Axis (see ropeaxes.go and TableAt). nil is a plain index.
	// Table ignores it, which is exact for a row whose coordinates agree and
	// whose runs are one frequency sequence (M-RoPE's text rows).
	Runs []RopeRun

	// first and stride select a frequency sequence other than 0, 1, 2, ...
	// for one run's folded planes (runRope); zero values are the plain table.
	first, stride int
}

// YarnFreqs is YaRN's rotary scaling as per-pair divisors for Rope.Freqs:
// pair p's angle pos*base^(-2p/nrot) becomes that times
// (1-e)/factor + e, where e is 1 below the correction range, 0 above it and a
// linear ramp between.
//
// The range ends are corr(b) = nrot*ln(origCtx/(2*pi*b)) / (2*ln(base)) for
// b = betaFast and betaSlow. llama.cpp and transformers' default floor the
// first and ceil the second; exact skips that, as gpt-oss's config
// (truncate false) asks. The magnitude correction is not here -- it scales
// cos and sin, which Rope.Scale does.
func YarnFreqs(nrot int, base, factor float64, origCtx int, betaFast, betaSlow float64, exact bool) []float32 {
	corr := func(b float64) float64 {
		return float64(nrot) * math.Log(float64(origCtx)/(2*math.Pi*b)) / (2 * math.Log(base))
	}
	lo, hi := corr(betaFast), corr(betaSlow)
	if !exact {
		lo, hi = math.Floor(lo), math.Ceil(hi)
	}
	lo, hi = math.Max(lo, 0), math.Min(hi, float64(nrot-1))
	if hi == lo {
		hi += 0.001
	}
	out := make([]float32, nrot/2)
	for p := range out {
		ramp := math.Min(math.Max((float64(p)-lo)/(hi-lo), 0), 1)
		e := 1 - ramp
		out[p] = float32(1 / ((1-e)/factor + e))
	}
	return out
}

// RopeTable is Table through this JIT, which lets a test build (Config.RopeGo)
// select the float64 Go table per model. Every caller in model/ takes it.
func (f *JIT) RopeTable(r Rope, cs []float32, pos int) {
	if f != nil && f.cfg.RopeGo {
		npairs := r.NRot / 2
		if len(cs)/2 < npairs {
			npairs = len(cs) / 2
		}
		if npairs > 0 {
			ropeGoTable(r, cs, pos, npairs)
		}
		return
	}
	r.Table(cs, pos)
}

// Table fills the cosines and sines one position needs, interleaved, for a
// caller that applies the rotation elsewhere. It is a generated kernel on
// every tier; a tier without one panics naming the shape.
//
// The frequencies are folded mod four quarter-turns in float64 once at load
// (buildRopeTab); the kernel does the position-dependent part. It lands within
// one float32 ulp at 1.0 of the float64 table at every position tested, and
// the device kernel (kernels.RopeTable) transcribes it from the same planes
// (TabPlanes). See jit_old/cpu/ropetab.go for the precision analysis and
// engine/nn/ropetab_test.go for the gate.
func (r Rope) Table(cs []float32, pos int) {
	npairs := r.NRot / 2
	if len(cs)/2 < npairs {
		npairs = len(cs) / 2
	}
	if npairs <= 0 {
		return
	}
	if pos < 0 || pos >= cpu.RopeTabMaxPos {
		panic(fmt.Sprintf("jit: rotary table at position %d, outside the %d the "+
			"kernel's digit decomposition covers", pos, cpu.RopeTabMaxPos))
	}
	code := ropeTabCodeFor(npairs)
	if code == nil {
		panic(fmt.Sprintf("jit: no rotary-table kernel for %d pair(s) on tier %v -- "+
			"every tier generates one, so this is a wiring bug and not a fallback",
			npairs, cpu.HostTier()))
	}
	tab := ropeTabFor(r, npairs)
	ropeTabCalls.Add(1)
	args := cpu.Args{
		Out:    &cs[0],
		AScale: &tab.block[0],
		Scr:    (*byte)(unsafe.Pointer(&ropeTabConsts[0])),
		K:      int64(pos),
	}
	code.Call(&args)
}

// TabPlanes is the folded constant block a generated rotary-table kernel reads
// -- the four head planes, the four residual planes and mscale, in
// kernels/ropetab_const.go's layout.
//
// It is exported for the device, which uploads the planes once per model like
// a weight, so every tier derives the table from the same fold.
//
// The returned slice is a copy: the cache behind it is shared by every
// sequence on this configuration.
func (r Rope) TabPlanes(npairs int) []float32 {
	if npairs <= 0 {
		return nil
	}
	return append([]float32(nil), ropeTabFor(r, npairs).block...)
}

// ropeTabCalls counts tables the kernel has built. A gate cannot prove the
// kernel ran by asserting it differs from a reference, since it often matches
// to the last bit; a counter answers that.
var ropeTabCalls atomic.Int64

// RopeTableCalls is how many tables the generated kernel has built.
func RopeTableCalls() int64 { return ropeTabCalls.Load() }

// SetRopeGo flips this JIT's rotary dose (Config.RopeGo). The Go table only
// exists under the jitllmtest build tag (ropego.go), so a release binary has no
// path to it. It is a setter rather than an option because the dose harness
// (engine/model/ropedose_test.go) toggles it mid-run so both arms share one JIT.
// Not safe to call while another goroutine is using this JIT.
func (f *JIT) SetRopeGo(on bool) {
	if f != nil {
		f.cfg.RopeGo = on
	}
}

var ropeTabConsts = cpu.RopeTabConsts()

// ropeTab is one model's folded rotary constants, in the layout
// ropetab_const.go defines.
//
// freqs is kept ALIVE rather than merely keyed on, which is what makes the
// pointer in ropeTabKey sound: while the entry lives the backing array cannot
// be collected, so its address cannot be reused by a different slice.
type ropeTab struct {
	block []float32
	freqs []float32
}

// ropeTabKey identifies one rotary configuration at one pair count.
//
// The frequency slice is keyed by identity, not content, so a token does not
// hash NRot/2 floats. A model builds Freqs once and never writes it; the
// address, length and end words pin it, and the entry holds a reference so the
// address cannot be recycled.
type ropeTabKey struct {
	nrot, npairs int
	// runFirst and runStride are a multi-axis run's frequency sequence
	// (runRope).
	runFirst, runStride int
	base, scale         uint64
	nfreq               int
	freqs               uintptr
	first, last         uint32
}

func (r Rope) tabKey(npairs int) ropeTabKey {
	k := ropeTabKey{
		nrot:      r.NRot,
		npairs:    npairs,
		runFirst:  r.first,
		runStride: max(r.stride, 1),
		base:      math.Float64bits(r.Base),
		scale:     math.Float64bits(r.Scale),
		nfreq:     len(r.Freqs),
	}
	if len(r.Freqs) > 0 {
		k.freqs = uintptr(unsafe.Pointer(unsafe.SliceData(r.Freqs)))
		k.first = math.Float32bits(r.Freqs[0])
		k.last = math.Float32bits(r.Freqs[len(r.Freqs)-1])
	}
	return k
}

// The folded constants and the kernels, both keyed and both published as a
// whole snapshot, which is engine/nn/moetopk.go's pattern: a model asks for exactly
// one key for its whole life, so neither map grows past the number of loaded
// configurations and the read path takes no lock.
var ropeTabs struct {
	mu   sync.Mutex
	snap atomic.Pointer[map[ropeTabKey]*ropeTab]
}

type ropeCodeKey struct {
	t      cpu.Tier
	npairs int
}

var ropeCodes struct {
	mu   sync.Mutex
	snap atomic.Pointer[map[ropeCodeKey]*cpu.Code]
}

func init() {
	t := map[ropeTabKey]*ropeTab{}
	ropeTabs.snap.Store(&t)
	c := map[ropeCodeKey]*cpu.Code{}
	ropeCodes.snap.Store(&c)
}

func ropeTabFor(r Rope, npairs int) *ropeTab {
	key := r.tabKey(npairs)
	if t, ok := (*ropeTabs.snap.Load())[key]; ok {
		return t
	}
	ropeTabs.mu.Lock()
	defer ropeTabs.mu.Unlock()
	old := *ropeTabs.snap.Load()
	if t, ok := old[key]; ok {
		return t
	}
	t := buildRopeTab(r, npairs)
	next := make(map[ropeTabKey]*ropeTab, len(old)+1)
	for k, v := range old {
		next[k] = v
	}
	next[key] = t
	ropeTabs.snap.Store(&next)
	return t
}

// buildRopeTab folds one configuration's frequencies into the kernel's planes.
//
// Each digit's weight is folded progressively and exactly: the next weight is
// this one's times 128, and multiplying by a power of two and taking mod 4 of
// an exact value are exact in float64. The only rounding is the first
// quarter-turn conversion and the split into two float32 words.
//
// The frequency uses the float64 table's own recurrence (f *= step, runFreqs),
// so a comparison between the two is about the kernel, not the frequencies.
func buildRopeTab(r Rope, npairs int) *ropeTab {
	blk := make([]float32, cpu.RopeTabBlock(npairs))
	freqs := runFreqs(r, npairs)
	for p := 0; p < npairs; p++ {
		w := math.Mod(freqs[p]*(2/math.Pi), 4)
		for d := 0; d < cpu.RopeTabDigits; d++ {
			if d > 0 {
				w = math.Mod(w*(1<<cpu.RopeTabDigitBits), 4)
			}
			h, l := cpu.RopeTabHeadSplit(w)
			blk[cpu.RopeTabPlaneOff(d, npairs)+p] = h
			blk[cpu.RopeTabPlaneOff(cpu.RopeTabDigits+d, npairs)+p] = l
		}
	}
	// The pad past npairs is never read -- the ragged tail re-runs the LAST
	// whole vector rather than a padded one -- but a zero there is one less
	// claim to re-derive, and make has already put one there.
	mscale := r.Scale
	if mscale == 0 {
		mscale = 1
	}
	blk[cpu.RopeTabScaleOff(npairs)] = float32(mscale)
	return &ropeTab{block: blk, freqs: r.Freqs}
}

// ropeTabCodeFor emits and maps one kernel under the lock and publishes a new
// snapshot. A failure is cached as a nil entry rather than retried: an emitter
// that refuses this shape refuses it every token.
//
// It is keyed by tier as well as by shape, so a forced-SSE run is not handed
// an AVX2 kernel an earlier caller mapped.
func ropeTabCodeFor(npairs int) *cpu.Code {
	key := ropeCodeKey{cpu.HostTier(), npairs}
	if c, ok := (*ropeCodes.snap.Load())[key]; ok {
		return c
	}
	ropeCodes.mu.Lock()
	defer ropeCodes.mu.Unlock()
	old := *ropeCodes.snap.Load()
	if c, ok := old[key]; ok {
		return c
	}
	var c *cpu.Code
	if b, err := cpu.EmittersFor(key.t).RopeTable(npairs); err == nil {
		if mapped, err := cpu.MapNamed(b, "rope_table"); err == nil {
			c = mapped
		}
	}
	next := make(map[ropeCodeKey]*cpu.Code, len(old)+1)
	for k, v := range old {
		next[k] = v
	}
	next[key] = c
	ropeCodes.snap.Store(&next)
	return c
}
