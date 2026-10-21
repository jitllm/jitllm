package cpu

import "strings"

// Tier is the instruction-set tier a host runs and a kernel is generated for.
//
// It is one answer, chosen once from the probe, and every emitter asks it
// through EmittersFor(tier). amd64's two tiers share nothing at the encoding
// level: the AVX2 tier is VEX-encoded 256-bit code with FMA and F16C, and a
// Goldmont Atom (SSE4.2 and nothing above) faults on the first VEX prefix. The
// SSE tier is legacy-encoded 128-bit code (sse.go).
//
// VNNI is not a tier: it is a DotKind inside the AVX2 tier (prevnni.go). The
// SSE tier has its own packed emitters and takes no DotKind, because any value
// other than DotVEX handed to an AVX2 emitter emits VPDPBUSD.
type Tier uint8

const (
	// TierNone is a host below every floor this package emits for: an x86-64
	// without SSSE3 and SSE4.1, or an architecture with no emitter.
	TierNone Tier = iota
	// TierSSE is amd64 with SSE2, SSE3, SSSE3 and SSE4.1 and no usable AVX2:
	// legacy-encoded 128-bit XMM code, two-operand destructive forms, no FMA,
	// no F16C, no VEX anywhere.
	TierSSE
	// TierAVX2 is amd64 with AVX2, FMA and F16C and the OS saving YMM state:
	// every kernel this package emitted before the SSE tier existed.
	TierAVX2
	// TierNEON is arm64 (ARMv8-A NEON). Its dispatch is its own -- the A64
	// emitters answer to the same names -- and it has no second tier.
	TierNEON

	numTiers
)

// NumTiers is how many Tier values exist, for a caller that keeps one of
// something per tier in an array.
const NumTiers = int(numTiers)

func (t Tier) String() string {
	switch t {
	case TierNone:
		return "none"
	case TierSSE:
		return "sse"
	case TierAVX2:
		return "avx2"
	case TierNEON:
		return "neon"
	}
	return "tier?"
}

// ISA is a set of x86 instruction-set extensions: what a kernel needs, or what
// a host has.
//
// It is carried in the kernel's bytes: an SSE-tier kernel begins with a long
// NOP whose displacement spells the set (DeclareISA, isa.go), so Map can check
// a buffer against this host with no x86 decoder. A kernel with no
// declaration is an AVX2-tier kernel and needs AVX2|FMA|F16C.
type ISA uint16

const (
	ISASSE2 ISA = 1 << iota
	ISASSE3
	ISASSSE3
	ISASSE41
	ISASSE42
	ISAPOPCNT
	// ISAAVX2 is AVX2 WITH the OS saving YMM state (Features.UsableAVX2): the
	// silicon having the unit is not enough to execute it.
	ISAAVX2
	ISAFMA
	ISAF16C
	// ISAVNNI is AVX-VNNI, the VEX-encoded VPDPBUSD.
	ISAVNNI
)

// ISATierSSE is the floor of the SSE tier: what every SSE-tier kernel may
// assume. SSSE3 carries PMADDUBSW/PSHUFB/PSIGNB (the int8 dot and the code
// tables) and SSE4.1 carries PMOVZX/PMOVSX (the scale unpack); every x86-64
// part since ~2008 has them. SSE4.2 and POPCNT are
// probed and named but not assumed.
const ISATierSSE = ISASSE2 | ISASSE3 | ISASSSE3 | ISASSE41

// ISATierAVX2 is what an UNDECLARED kernel needs: the AVX2 tier's floor.
const ISATierAVX2 = ISAAVX2 | ISAFMA | ISAF16C

var isaNames = [...]string{
	"sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt",
	"avx2", "fma", "f16c", "avx-vnni",
}

// Names lists the extensions in the set, in a fixed order.
func (s ISA) Names() []string {
	var out []string
	for i, n := range isaNames {
		if s&(1<<i) != 0 {
			out = append(out, n)
		}
	}
	return out
}

func (s ISA) String() string {
	if s == 0 {
		return "none"
	}
	return strings.Join(s.Names(), ", ")
}

// ISA is the set of extensions this host can EXECUTE.
//
// The AVX bits require the OS half: AVX2 and its companions are reported only
// when XCR0 says the OS saves YMM state. The SSE bits need no such check,
// since x86-64 long mode requires the OS to manage XMM state.
func (f Features) ISA() ISA {
	var s ISA
	set := func(ok bool, b ISA) {
		if ok {
			s |= b
		}
	}
	set(f.SSE2, ISASSE2)
	set(f.SSE3, ISASSE3)
	set(f.SSSE3, ISASSSE3)
	set(f.SSE41, ISASSE41)
	set(f.SSE42, ISASSE42)
	set(f.POPCNT, ISAPOPCNT)
	avx := f.UsableAVX2()
	set(avx, ISAAVX2)
	set(avx && f.FMA, ISAFMA)
	set(avx && f.F16C, ISAF16C)
	set(avx && f.AVXVNNI, ISAVNNI)
	return s
}

// missingNames names what need requires and f lacks, spelling out the case a
// bare feature name would hide: AVX2 present in the silicon with the OS not
// saving its state.
func missingNames(f Features, need ISA) string {
	miss := need &^ f.ISA()
	var out []string
	for _, n := range miss.Names() {
		switch {
		case n == "avx2" && f.AVX2 && !f.OSAVX:
			out = append(out, "avx2 (present, but the OS does not save YMM state)")
		case (n == "fma" || n == "f16c" || n == "avx-vnni") && !f.UsableAVX2() && featureBit(f, n):
			out = append(out, n+" (present, but unusable without OS YMM state)")
		default:
			out = append(out, n)
		}
	}
	return strings.Join(out, ", ")
}

func featureBit(f Features, n string) bool {
	switch n {
	case "fma":
		return f.FMA
	case "f16c":
		return f.F16C
	case "avx-vnni":
		return f.AVXVNNI
	}
	return false
}

// tierOf is the tier a feature set supports on amd64: AVX2 when it holds, and
// the SSE tier when that fails and the SSE floor holds.
func tierOf(f Features) Tier {
	s := f.ISA()
	switch {
	case s&ISATierAVX2 == ISATierAVX2:
		return TierAVX2
	case s&ISATierSSE == ISATierSSE:
		return TierSSE
	}
	return TierNone
}
