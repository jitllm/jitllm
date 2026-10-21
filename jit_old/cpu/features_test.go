//go:build amd64 && linux

package cpu

import (
	"os"
	"strings"
	"testing"
)

// TestFeaturesAgreeWithTheKernel checks the JIT's CPUID probe against an
// independent source: /proc/cpuinfo, which the kernel filled from its own CPUID
// walk, so a bit read from the wrong leaf, register or position disagrees. On
// any host some flags are true and some false, so a probe stuck at either value
// fails.
func TestFeaturesAgreeWithTheKernel(t *testing.T) {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		t.Skipf("no /proc/cpuinfo: %v", err) // visible, and the rest still runs
	}
	var flags string
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(ln, "flags") {
			flags = " " + strings.TrimSpace(strings.SplitN(ln, ":", 2)[1]) + " "
			break
		}
	}
	if flags == "" {
		t.Fatal("no flags line in /proc/cpuinfo: this gate proved nothing")
	}
	has := func(f string) bool { return strings.Contains(flags, " "+f+" ") }

	f := CPU()
	for _, c := range []struct {
		name string
		got  bool
		want bool
	}{
		// The SSE tier's floor and its two neighbours. "pni" is the kernel's
		// name for SSE3 (Prescott New Instructions), not a typo.
		{"sse2", f.SSE2, has("sse2")},
		{"pni", f.SSE3, has("pni")},
		{"ssse3", f.SSSE3, has("ssse3")},
		{"sse4_1", f.SSE41, has("sse4_1")},
		{"sse4_2", f.SSE42, has("sse4_2")},
		{"popcnt", f.POPCNT, has("popcnt")},
		{"avx2", f.AVX2, has("avx2")},
		{"fma", f.FMA, has("fma")},
		{"f16c", f.F16C, has("f16c")},
		{"avx_vnni", f.AVXVNNI, has("avx_vnni")},
		{"avx512_vnni", f.AVX512VNNI, has("avx512_vnni")},
		{"avx512f", f.AVX512F, has("avx512f")},
		{"avx512bw", f.AVX512BW, has("avx512bw")},
		{"avx512dq", f.AVX512DQ, has("avx512dq")},
		{"avx512vl", f.AVX512VL, has("avx512vl")},
	} {
		if c.got != c.want {
			t.Errorf("%s: probe says %v, /proc/cpuinfo says %v", c.name, c.got, c.want)
		}
	}

	// The OS half has no flag to compare against, so it is checked as an
	// implication: the kernel does not advertise avx2 where it refused to save
	// YMM state, so usable must follow from present. The converse is not
	// asserted.
	if f.AVX2 && !f.UsableAVX2() {
		t.Errorf("avx2 present but UsableAVX2 false: XCR0 said %v/%v", f.OSAVX, f.OSAVX512)
	}
	if f.AVX512F && f.AVX512BW && f.AVX512VL && f.AVX512DQ && !f.UsableAVX512() {
		t.Errorf("the four AVX-512 subsets are present but UsableAVX512 is false")
	}
	if f.UsableAVX512() && !f.OSAVX512 {
		t.Error("UsableAVX512 true with OSAVX512 false: the OS check is not being applied")
	}
	// The tier follows from the flags, checked against the same independent
	// source.
	wantTier := TierNone
	switch {
	case has("avx2") && has("fma") && has("f16c"):
		wantTier = TierAVX2
	case has("sse2") && has("pni") && has("ssse3") && has("sse4_1"):
		wantTier = TierSSE
	}
	if got := HostTier(); got != wantTier {
		t.Errorf("HostTier() = %v, /proc/cpuinfo says %v", got, wantTier)
	}
	t.Logf("tier=%v isa=[%v] sse2=%v sse3=%v ssse3=%v sse4.1=%v sse4.2=%v popcnt=%v",
		HostTier(), f.ISA(), f.SSE2, f.SSE3, f.SSSE3, f.SSE41, f.SSE42, f.POPCNT)
	t.Logf("avx2=%v fma=%v f16c=%v | avx-vnni=%v avx512-vnni=%v | avx512 f=%v bw=%v dq=%v vl=%v | os avx=%v avx512=%v | usable: avx2=%v avx512=%v",
		f.AVX2, f.FMA, f.F16C, f.AVXVNNI, f.AVX512VNNI,
		f.AVX512F, f.AVX512BW, f.AVX512DQ, f.AVX512VL,
		f.OSAVX, f.OSAVX512, f.UsableAVX2(), f.UsableAVX512())
}
