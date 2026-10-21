package cpu

import (
	"sync"
	"unsafe"
)

// Features is what this CPU can execute, probed once and asked by the emitters.
//
// There is no option, environment variable or build tag for any of it: an
// engine that has to be told its host's capabilities is wrong by default on
// every unconfigured machine. The probe is cheap (CPUID, once per process), so
// it is not cached to disk; what is cached is the expensive decision (pack
// width, split), keyed on the bytes EmitNative produces, which moves when the
// capability set moves. See docs/design/inferred-kernels.md.
type Features struct {
	// The SSE family, all from CPUID leaf 1 (SSE2 is EDX bit 26; the rest are
	// ECX). They are what the SSE tier is built on (tier.go): SSSE3 carries
	// PMADDUBSW, PSHUFB and PSIGNB, SSE4.1 carries PMOVZX/PMOVSX. SSE2 is part
	// of x86-64 itself and is probed anyway, so a hypervisor that masks it is a
	// named refusal rather than an assumption.
	//
	// They need no OS half: long mode cannot run without the OS managing XMM
	// state, so there is no XCR0 question for a 128-bit register.
	SSE2   bool
	SSE3   bool
	SSSE3  bool
	SSE41  bool
	SSE42  bool
	POPCNT bool

	AVX2 bool
	FMA  bool
	F16C bool

	// AVXVNNI is the VEX-encoded VPDPBUSD every quantized kernel emits:
	// one instruction for a four-byte int8 dot product. Alder Lake and Zen 4
	// have it; Skylake-SP, Haswell and everything older do not.
	AVXVNNI bool

	// AVX512VNNI is the same operation EVEX-encoded, a different instruction on
	// a different generation. A part can have one, both or neither (Alder Lake
	// has AVX-VNNI and no AVX-512; Skylake-SP has AVX-512 and neither VNNI), so
	// each is asked separately.
	AVX512VNNI bool

	AVX512F  bool
	AVX512BW bool
	AVX512DQ bool
	AVX512VL bool

	// OSAVX and OSAVX512 are the operating system's half of the answer, read
	// from XCR0: the OS saves the state across a context switch. Without it
	// (noxsave, an old hypervisor) the CPUID bit is set and the first
	// instruction raises #UD. Every accessor below requires both.
	OSAVX    bool
	OSAVX512 bool
}

// UsableAVX512 reports whether 512-bit vectors and the 32 ZMM registers can
// actually be used here: the core four subsets and the OS saving their state.
// BW is required because it carries the byte and word ops (VPMADDUBSW,
// VPMADDWD) the quantized kernels need at ZMM width.
func (f Features) UsableAVX512() bool {
	return f.OSAVX512 && f.AVX512F && f.AVX512BW && f.AVX512VL && f.AVX512DQ
}

// UsableAVX2 is AVX2 plus the OS saving YMM state.
func (f Features) UsableAVX2() bool { return f.OSAVX && f.AVX2 }

var (
	featuresOnce sync.Once
	features     Features
	// hostTier is tierOf(features), computed inside the same Once so the hot
	// callers (nn picks its elementwise kernels by tier on every call) pay an
	// atomic load and nothing else.
	hostTier Tier

	// probeMu serialises the TEST instruments that re-probe (ForceNoVNNIForTest,
	// ForceTierForTest, retier). The shipping path never takes it: it reads the
	// probe through featuresOnce and nothing else.
	probeMu sync.Mutex
)

// CPU returns this host's capability set, probed once.
func CPU() Features {
	featuresOnce.Do(probeFeatures)
	return features
}

// HostTier is the instruction-set tier this host runs: TierAVX2 when AVX2, FMA
// and F16C are usable, TierSSE when the SSE floor holds instead, and TierNone
// below that. See tier.go.
func HostTier() Tier {
	featuresOnce.Do(probeFeatures)
	return hostTier
}

// cpuid runs the JIT-compiled CPUID probe and returns EAX, EBX, ECX, EDX
// (see feature_amd64.go).
func cpuid(leaf, sub int32) (eax, ebx, ecx, edx uint32, ok bool) {
	code, err := mapExec(cpuidProbe(leaf, sub))
	if err != nil {
		return 0, 0, 0, 0, false
	}
	defer code.Close()
	var out [8]uint64
	args := Args{Out: (*float32)(unsafe.Pointer(&out[0]))}
	code.Call(&args)
	return uint32(out[0]), uint32(out[1]), uint32(out[2]), uint32(out[3]), true
}

// xcr0Probe emits: ECX = 0, XGETBV, store EDX:EAX through Args.Out.
func xcr0Probe() []byte {
	var a Buf
	a.MOVLoad(R10, At(RDI, 0))
	a.MOVimm32(RCX, 0)
	a.XGETBV()
	a.MOVStore(At(R10, 0), RAX)
	a.MOVStore(At(R10, 8), RDX)
	a.RET()
	return a.Bytes()
}

func probeFeatures() {
	max, _, _, _, ok := cpuid(0, 0)
	if !ok || max < 1 {
		return
	}
	// Leaf 1 ECX: OSXSAVE bit 27, AVX 28, FMA 12, F16C 29.
	_, _, ecx1, edx1, ok := cpuid(1, 0)
	if !ok {
		return
	}
	// Leaf 1 EDX bit 26 SSE2; ECX bit 0 SSE3 (/proc/cpuinfo's "pni"), 9 SSSE3,
	// 19 SSE4.1, 20 SSE4.2, 23 POPCNT.
	features.SSE2 = edx1&(1<<26) != 0
	features.SSE3 = ecx1&(1<<0) != 0
	features.SSSE3 = ecx1&(1<<9) != 0
	features.SSE41 = ecx1&(1<<19) != 0
	features.SSE42 = ecx1&(1<<20) != 0
	features.POPCNT = ecx1&(1<<23) != 0
	features.FMA = ecx1&(1<<12) != 0
	features.F16C = ecx1&(1<<29) != 0
	osxsave := ecx1&(1<<27) != 0
	avx := ecx1&(1<<28) != 0

	// XGETBV is only legal when OSXSAVE is set; otherwise it raises #UD.
	if osxsave {
		if code, err := mapExec(xcr0Probe()); err == nil {
			var out [2]uint64
			args := Args{Out: (*float32)(unsafe.Pointer(&out[0]))}
			code.Call(&args)
			code.Close()
			x := uint32(out[0])
			// XCR0 bit 1 SSE, 2 YMM_Hi128; 5 opmask, 6 ZMM_Hi256, 7 Hi16_ZMM.
			features.OSAVX = avx && x&0x6 == 0x6
			features.OSAVX512 = features.OSAVX && x&0xE0 == 0xE0
		}
	}

	if max >= 7 {
		// Leaf 7.0 EBX: AVX2 5, AVX512F 16, AVX512DQ 17, AVX512BW 30,
		// AVX512VL 31. ECX: AVX512VNNI 11.
		_, ebx7, ecx7, _, ok := cpuid(7, 0)
		if ok {
			features.AVX2 = ebx7&(1<<5) != 0
			features.AVX512F = ebx7&(1<<16) != 0
			features.AVX512DQ = ebx7&(1<<17) != 0
			features.AVX512BW = ebx7&(1<<30) != 0
			features.AVX512VL = ebx7&(1<<31) != 0
			features.AVX512VNNI = ecx7&(1<<11) != 0
		}
		// Leaf 7.1 EAX bit 4: AVX-VNNI, the VEX encoding.
		if eax71, _, _, _, ok := cpuid(7, 1); ok {
			features.AVXVNNI = eax71&(1<<4) != 0
		}
	}

	// The test instruments, not knobs: see feature_force.go.
	if forceNoVNNI {
		features.AVXVNNI = false
		features.AVX512VNNI = false
	}
	if tierForced {
		applyForcedTier(&features, forceTier)
	}
	hostTier = tierOf(features)
}

// applyForcedTier clears what a host of tier t would not have. Forcing can
// only take capabilities AWAY -- a host cannot be made to execute an
// instruction it lacks -- so TierAVX2 (the top) clears nothing.
func applyForcedTier(f *Features, t Tier) {
	if t >= TierAVX2 {
		return
	}
	// Below the AVX2 tier: no VEX-encoded unit of any kind is usable.
	f.AVX2, f.FMA, f.F16C = false, false, false
	f.AVXVNNI, f.AVX512VNNI = false, false
	f.AVX512F, f.AVX512BW, f.AVX512DQ, f.AVX512VL = false, false, false, false
	f.OSAVX, f.OSAVX512 = false, false
	if t >= TierSSE {
		return
	}
	// Below the SSE floor: the two extensions the SSE tier assumes, and the
	// one above them, so the refuse-by-name path can be exercised.
	f.SSSE3, f.SSE41, f.SSE42 = false, false, false
}

// retier makes the probe report tier t (or the host's own answer, for
// TierAVX2), re-probing immediately, and returns the tier the host reported
// before. It is the test instruments' one implementation: ForceTierForTest
// (jitllmtest) and this package's own tests call it, and nothing on the
// shipping path does.
//
// It resets every cache derived from the probe, not just the probe: Baseline
// caches its verdict behind its own Once, and a forced run must not read the
// real CPU's floor.
func retier(t Tier) Tier {
	probeMu.Lock()
	defer probeMu.Unlock()
	old := HostTier()
	forceTier, tierForced = t, t < TierAVX2
	featuresOnce = sync.Once{}
	features = Features{}
	hostTier = TierNone
	baselineOnce = sync.Once{}
	baselineErr = nil
	CPU()
	return old
}
