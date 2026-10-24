package cpu

// The activation side of the prefill GEMM. It is architecture-neutral: the
// packed layout (eight tokens by four k values per 32-byte lane group) is the
// same shape for AVX-VNNI's VPDPBUSD and for arm64's SDOT.

// GEMMTokens is how many tokens a tile of width nr covers.
func GEMMTokens(nr int) int { return 8 * nr }

// GEMMScratchMax is how many float32 of Args.Scratch the widest tile EmitGEMM
// allows needs, for a caller that sizes one buffer and reuses it for every
// shape. The terms are the accumulator tile (mr*nr of eight floats), the
// per-row scale staging (16 floats), the shuffle staging (8), the unpacked
// super-block (64 floats = 256 bytes per row), and a second accumulator tile
// for the int32 fold (see emitGEMMQ3K). The int tile is sized unconditionally
// rather than per-window: a window-dependent layout would add an invariant
// between nn/ and the emitter. It lives in an untagged file because both
// architectures use it; the arm64 k-quant GEMM's scale staging (256 floats at
// the widest tile) fits within the amd64 figure.
const GEMMScratchMax = 2*8*12 + 8*16 + 8 + 8*64
