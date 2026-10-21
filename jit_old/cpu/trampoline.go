//go:build amd64 || arm64

package cpu

import "unsafe"

// Args is what a generated kernel reads. The kernel receives one pointer — in
// RDI on amd64, x0 on arm64 — and reaches everything through this struct at
// fixed offsets. The struct itself is architecture-neutral: every field is
// 8-byte aligned, so the same layout serves both.
//
// Every field that is an address is a real Go pointer, never a uintptr. That is
// not style: Go's garbage collector scans this struct through the callKernel
// signature, and a uintptr is invisible to it — the slice it refers to could be
// collected while the kernel is mid-loop. Go's GC does not move heap objects, so
// a scanned pointer stays valid for the duration of the call.
//
// Field order is ABI. The generated code addresses these by byte offset, so
// inserting a field in the middle silently repoints every load in every kernel.
// Append only, and the offsets are asserted in a test.
type Args struct {
	Out    *float32 // [rows]      accumulated output, written not added
	W      *byte    // packed GGUF weight bytes, straight from the mapping
	A      *int8    // quantized activations
	AScale *float32 // per-group activation scales
	Rows   int64
	K      int64
	RowStr int64 // bytes between consecutive weight rows
	Scr    *byte // scratch, kernel-defined
	// AHalf is per-16-element activation sums, which the 16-wide k-quants
	// (Q6_K, Q3_K) need for their bias correction.
	AHalf *float32
	// Scratch is per-worker spill space, needed by the interleaved k-quant
	// kernels: their sub-block scales are PER ROW, so eight rows need 512 bytes
	// where the 128-byte red zone only serves one. Each pool worker gets its own
	// slice, so there is no sharing and no false sharing.
	Scratch *byte
	// Cols, OutStr and ASum are the prefill GEMM's additions. Cols is the
	// batch width in tokens. OutStr is the byte stride between output rows,
	// which is 4*Cols. ASum is 128*sum(activations) per (block, token), the
	// bias correction for reading signed weights as unsigned, laid out
	// [block][token] so eight tokens' corrections are one vector load.
	Cols   int64
	OutStr int64
	// ASum is INT32, not float32: the kernel subtracts it from the integer
	// accumulator before converting, because the bias dominates that
	// accumulator and removing it in float would cancel away most of the
	// mantissa.
	ASum *int32
	// AHalfSum is BiasC(t)*sum(q) over each sixteen-element half-block, laid
	// out [half][token]. The 16-wide k-quants carry one scale per sixteen
	// elements, so ASum's 32-element sums would fold two scales together.
	AHalfSum *int32
	// Q32 is the attention query as float32: A is int8 activations and AScale
	// is reused for the attention weights, so no existing field could carry it.
	Q32 *float32
	// Out2, Q2 and AScale2 are the paired attention kernels' second head.
	// EmitAttnScores2 dots one K vector against two queries (Q32 and Q2) into
	// Out and Out2; EmitAttnAcc2 weights one V vector by AScale and AScale2
	// into two outputs.
	Out2    *float32
	Q2      *float32
	AScale2 *float32
	// PD and PSC are the packed layout's scale spans: a jlm container stores
	// a weight as three regions and W carries the first. PD is at 144 and PSC
	// at 152.
	PD  *byte
	PSC *byte
	// DStr is the byte stride between consecutive super-blocks of the d plane.
	// It is not RowStr: a narrow format packs two rows into one word (see
	// kernels.DIndex), so its super-block stride is half the payload's. DStr
	// is at 160.
	DStr int64
}

// callKernel loads args into the platform's first argument register — RDI on
// amd64, x0 on arm64 — and jumps to code. Args itself is architecture-neutral;
// duplicating it per architecture would duplicate the ABI, which is exactly
// what the field-order warning above is about.
//
// amd64: the kernel must obey the System V callee-saved contract for RBX, RBP
// and R12-R15, EXCEPT that RBX, R12, R13 and R15 are saved by the trampoline
// itself so kernels may use them freely. R14 is not offered on any terms: Go's
// register ABI pins it to the current goroutine, and handing the runtime a
// corrupted g is not a bug that reproduces near its cause. An AVX2-tier kernel
// must also VZEROUPPER before returning, or every later SSE instruction in the
// process is taxed — a symptom that appears in code which never called a
// kernel. An SSE-tier kernel must NOT: VZEROUPPER is itself VEX-encoded and
// faults on a host with no AVX, and legacy SSE code leaves no dirty upper half
// to clear (sse_helpers.go).
//
// arm64: nothing is saved, because Go's arm64 ABIInternal has no callee-saved
// registers. Kernels may use x0-x17, x19-x26 and every V register with no
// prologue. The register that must never be written is x28 (g), not R14. x18
// (reserved by macOS), x27 (REGTMP), x29, x30 and sp are also off limits.
// NEON has no AVX-SSE transition penalty, so no VZEROUPPER analogue.
//
//go:noescape
func callKernel(code *byte, args *Args)

// f32bytes reinterprets a float32 slice as the *byte the Args weight pointer
// expects. The attention kernels read the KV cache, which is float32, through
// the same field the quantized kernels use for packed bytes.
//
// It lives here rather than beside one architecture's kernels so tests that
// use it compile on both.
func f32bytes(s []float32) *byte { return (*byte)(unsafe.Pointer(&s[0])) }
