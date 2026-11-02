package composer

// What a backend's lowering of the IR does, by its API's name (backend.API).

// ExactDivOn reports whether ir.OpDiv is the IEEE quotient on the backend of
// api: PTX's div.rn is. SPIR-V's OpFDiv is allowed 2.5 ulp, and Metal's
// division is the library's math mode's.
func ExactDivOn(api string) bool { return api == "ptx" }

// ExactArithOn reports whether a backend's lowering rounds every float
// operation alone (PTX's .rn arithmetic), so two kernels computing a value
// by the same operations write its same bits. SPIR-V lowers an Fma as an
// unfused pair with no NoContraction decoration and MSL compiles with fast
// math, so a driver may contract one kernel's sums differently from
// another's.
func ExactArithOn(api string) bool { return api == "ptx" }

// HalfPairsOn reports whether a backend's lowering runs packed binary16
// arithmetic (ir.OpSubF16x2, ir.OpFmaF16x2): PTX's sub.rn.f16x2 and
// fma.rn.f16x2.
func HalfPairsOn(api string) bool { return api == "ptx" }

// VectorLoadsOn reports whether a backend's lowering reads several words of
// a buffer in one access (ir.OpLoadV): PTX's ld.global.v4.
func VectorLoadsOn(api string) bool { return api == "ptx" }
