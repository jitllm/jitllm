//go:build amd64 || arm64

package cpu

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// hostTable is the emitter table of the tier this host runs, the one nn emits
// through. A gate that takes its kernel from here runs the AVX2 form on an
// AVX2 host and the SSE form on an SSE-only one, against the same oracle, so
// the gate covers whichever kernel a model on this host executes (RULE 11d).
func hostTable() *Emitters { return EmittersFor(HostTier()) }

// hostBytes takes one field of hostTable and fails the gate on its error.
// The host tier serves every op these gates call: an error is a missing
// kernel, never a reason to skip.
func hostBytes(t testing.TB) func([]byte, error) []byte {
	return func(b []byte, err error) []byte {
		t.Helper()
		if err != nil {
			t.Fatalf("the %v tier's emitter: %v", HostTier(), err)
		}
		return b
	}
}

// onHost is hostBytes then Map. A host-tier kernel this host refuses to map is
// a failure, not a skip: the table and the host disagree about the tier.
func onHost(t testing.TB) func([]byte, error) *Code {
	return func(b []byte, err error) *Code {
		t.Helper()
		c, merr := Map(hostBytes(t)(b, err))
		if merr != nil {
			t.Fatalf("the %v tier's kernel does not map on this host: %v", HostTier(), merr)
		}
		return c
	}
}

// ggufDeclined reports whether this host's tier declines the GGUF row-major
// machinery for qt -- the quantized row-major matvec, its interleaved
// pack-width kernels, the token-tiled GEMM and the activation packer that
// feeds it -- and asserts the decline where it does. The SSE tier declines
// them by design (Emitters.RowMajorGGUF): a .jlm container decodes through the
// packed family, which the SSE tier serves, and never through these. So on an
// SSE host a gate over them asserts the decline by name, and the packed
// family's SSE gates (ssepacked_test.go) are what run there.
func ggufDeclined(t testing.TB, qt quant.Type) bool {
	t.Helper()
	em := hostTable()
	if em.RowMajorGGUF {
		return false
	}
	if em.RowMajorSupported(qt) {
		t.Fatalf("the %v tier declines the GGUF row-major machinery but claims a %v row-major kernel", em.Tier, qt)
	}
	if _, err := em.RowMajor(Spec{W: qt, Rows: 1, Accs: 1, Cols: 1}); err == nil {
		t.Fatalf("the %v tier emitted a %v row-major kernel it says it does not serve", em.Tier, qt)
	}
	t.Logf("DECLINED BY DESIGN on the %v tier: the GGUF row-major %v kernels "+
		"(RowMajorGGUF is false; a container decodes through the packed family, "+
		"which TestSSEPackedMatchesTheOracle gates here)", em.Tier, qt)
	return true
}

// avx2Primitive maps a gate's hand-assembled AVX2 sequence -- a building block
// rather than a kernel a table serves, so there is no host-tier form of the
// same bytes to take instead. Where the host runs the AVX2 tier it returns the
// code. On an SSE-only host the refusal is the assertion: Map must decline the
// bytes by ISA (an acceptance there is the SIGILL), and twin names the gate
// that runs the SSE tier's form of the same sequence against the same
// reference. It returns nil then, and the caller stops.
func avx2Primitive(t testing.TB, code []byte, twin string) *Code {
	t.Helper()
	c, err := Map(code)
	if HostTier() != TierSSE {
		if err != nil {
			t.Fatalf("Map on a %v host: %v", HostTier(), err)
		}
		return c
	}
	if err == nil {
		c.Close()
		t.Fatal("an SSE-only host mapped an AVX2 sequence: Map's ISA gate is not wired")
	}
	if !errors.Is(err, ErrISA) {
		t.Fatalf("Map refused the AVX2 sequence for the wrong reason: %v", err)
	}
	t.Logf("SSE HOST: the AVX2 sequence is refused by ISA, as it must be (%v); "+
		"%s runs the SSE tier's form against the same reference", err, twin)
	return nil
}
