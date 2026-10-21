//go:build !amd64

package cpu

import "runtime"

// Features is the amd64 capability set. On every other architecture the answer
// is "none of these x86 units", and the emitters there ask their own probes
// (arm64's DotProd, via hasA64DotProd). It exists on every architecture so
// callers need no build tag.
type Features struct {
	SSE2       bool
	SSE3       bool
	SSSE3      bool
	SSE41      bool
	SSE42      bool
	POPCNT     bool
	AVX2       bool
	FMA        bool
	F16C       bool
	AVXVNNI    bool
	AVX512VNNI bool
	AVX512F    bool
	AVX512BW   bool
	AVX512DQ   bool
	AVX512VL   bool
	OSAVX      bool
	OSAVX512   bool
}

func (f Features) UsableAVX512() bool { return false }
func (f Features) UsableAVX2() bool   { return false }

// CPU reports an empty set: this is not an x86.
func CPU() Features { return Features{} }

// HostTier is TierNEON on arm64, whose A64 emitters answer to the same names
// the AVX2 tier's do, and TierNone on an architecture with no emitter.
func HostTier() Tier {
	if runtime.GOARCH == "arm64" {
		return TierNEON
	}
	return TierNone
}
