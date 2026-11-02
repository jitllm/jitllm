//go:build amd64

package cpu

import "github.com/jitllm/jitllm/format/quant"

// The native emitter for this GOARCH. Both emitters are host-pure, so either
// can be generated and disassembled anywhere; only execution needs the
// matching CPU. These functions are the only place nn has to care which runs.
//
// SupportedNative is "can this host run a generated row-major kernel for t":
// Supported and the CPU having the instructions. The quantized row-major
// kernels emit VPDPBUSD, which is AVX-VNNI, not baseline. This is the
// row-major family's answer alone: the packed family (the only weight path
// model.Open admits) asks SupportedPackedNative, which has a pre-VNNI form
// (prevnni.go). It asks the host's tier first, since every kernel admitted
// below is VEX-encoded and an SSE host must get the SSE tier's answer.
func SupportedNative(t quant.Type) bool { return Native().RowMajorSupported(t) }

// primaryTier is the tier the primary emitter table generates for on amd64.
const primaryTier = TierAVX2

// primaryRowMajorSupported is SupportedNative's AVX2-tier answer, verbatim.
func primaryRowMajorSupported(t quant.Type) bool {
	if !Supported(t) {
		return false
	}
	if t == quant.F32 || t == quant.F16 || t == quant.BF16 {
		// AVX2 only: VMOVDQU/VCVTPH2PS/VPMOVZXWD + VFMADD231PS. F16C ships
		// with AVX2 on every part this tier admits, so it needs no probe.
		return true
	}
	return CPU().AVXVNNI
}
func EmitNative(s Spec) ([]byte, error) { return Emit(s) }

// rowMajorPF is EmitNative with EmitOpts.Prefetch.
func rowMajorPF(s Spec, pf int) ([]byte, error) { return emitPF(s, pf) }
func BestPackNative(t quant.Type) int           { return BestPack(t) }

// PackWidthNative is BestPackNative with the pin passed in rather than read
// from the package Config, so a JIT's width is its own.
func PackWidthNative(pin int, t quant.Type) int { return PackWidth(pin, t) }
func NativeArch() string                        { return "amd64" }
