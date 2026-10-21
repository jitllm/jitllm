//go:build !amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The non-amd64 face of the packed family: which of these names forward to a
// real emitter and which decline. arm64 has the 64-row tile, the 8-row tail,
// the fused matvec and the token-tiled matmul, all through SDOT; the wide
// kernel declines, which is a supported configuration.
//
// The stub must mirror the amd64 surface exactly: a name that exists on one
// architecture and not the other breaks the darwin cross-build of nn/.
const (
	PackedRows         = 64
	PackedTail         = 8
	PackedWideGroup    = 32
	PackedScratchBytes = 416
)

// PackedSupported is what a caller asks before relying on a generated packed
// matvec. On arm64 it is the arm64 list -- returning false here for a format
// that has a kernel would make a model refuse a tensor it can run.
func PackedSupported(t quant.Type) bool { return SupportedA64Packed(t) }

// PackedFusedSupported is whether EmitPackedMatVecFused emits for t here: the
// arm64 fused kernel, at PackedFusedGroupOf(t) rows a group.
func PackedFusedSupported(t quant.Type) bool {
	_, err := EmitPackedMatVecFused(t, DotVNNI)
	return err == nil
}

// PackedScratch fills the one constant the arm64 packed kernels read from
// memory rather than build in a register: a code table (MXFP4's), which TBL
// needs whole.
func PackedScratch(b []byte) error {
	if len(b) < PackedScratchBytes {
		return fmt.Errorf("jit: PackedScratch: %d bytes, need %d", len(b), PackedScratchBytes)
	}
	putCodes(b)
	return nil
}

// DotKind mirrors the amd64 surface, because nn asks for it before it asks
// whether an emitter exists. It is inert here: SDOT is signed by signed and
// needs neither centring nor a pre-VNNI alternative, so every emitter below
// ignores it.
type DotKind int

const (
	DotVNNI DotKind = iota
	DotVEX
)

func (d DotKind) String() string {
	if d == DotVEX {
		return "vex"
	}
	return "vnni"
}

// HostDotKind is meaningless off amd64 and answers for the only sequence this
// architecture has.
func HostDotKind() DotKind { return DotVNNI }

// PreVNNIPacked has no meaning here: there is no VPMADDUBSW to saturate.
func PreVNNIPacked(quant.Type) error {
	return fmt.Errorf("jit: pre-VNNI: no x86 packed emitter on this architecture")
}

// primaryPackedSupported is SupportedPackedNative itself off amd64: there is
// one tier here.
func primaryPackedSupported(t quant.Type) bool { return SupportedPackedNative(t) }

// SupportedPackedNative is "can this host run the generated packed kernel for
// t": the packed emitter's list (SupportedA64Packed). FEAT_DotProd is not
// asked: without it SDOT is widened (sdotemu.go). It must not ask
// SupportedNative (the row-major list, which lacks MXFP4): a format it says no
// to is a tensor a model refuses to load.
func SupportedPackedNative(t quant.Type) bool { return PackedSupported(t) }

func EmitPackedMatVec(t quant.Type, rows int, _ DotKind) ([]byte, error) {
	return packedMatVecPF(t, rows, 0)
}

// packedMatVecPF is EmitPackedMatVec with EmitOpts.A64Prefetch.
func packedMatVecPF(t quant.Type, rows, pf int) ([]byte, error) {
	// arm64 has its own packed matvec: the layout suits SDOT's four int32
	// lanes, and SDOT is signed by signed so Q8_0 needs no centring or
	// correction term.
	if b, err := EmitA64PackedMatVecPF(t, rows, pf); err == nil {
		return b, nil
	}
	return nil, fmt.Errorf("jit: EmitPackedMatVec: no emitter for this architecture")
}

func EmitPackedMatVecWide(t quant.Type, _ DotKind) ([]byte, error) {
	return nil, fmt.Errorf("jit: EmitPackedMatVecWide: no emitter for this architecture")
}

// EmitPackedMatVecFused carries a two-row-group prefetch on arm64, where
// amd64's base form has none: without it the fused k-quants read about half
// the tiled kernel's rate from DRAM on arm64 (nn.TestPerByteCost), and with
// it every format matches or passes the tiled kernel. nn's duel still tries
// other distances per model.
func EmitPackedMatVecFused(t quant.Type, _ DotKind) ([]byte, error) {
	return EmitA64PackedMatVecFused(t, PackedFusedGroupOf(t), a64FusedAhead)
}

// a64FusedAhead is the arm64 fused kernel's default prefetch, in row groups.
const a64FusedAhead = 2

// PackedOuterElems stays exact even with no emitter: nn asks it before it asks
// whether a kernel exists, and a wrong answer here would be a wrong answer
// about the layout rather than about this architecture.
func PackedOuterElems(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return 0
	}
	sub, _, _, _ := kernels.Layout(q)
	perSuper, _ := kernels.ScaleLayout(q)
	return sub * perSuper
}

// PackedTiledTokens and the functions below are the token-tiled packed matmul's
// surface off amd64; they forward to the arm64 emitter (MaxTiledTokensA64).
const PackedTiledTokens = 4

// MaxTiledTokensNative is the arm64 twin. There is no VNNI question here --
// SDOT is baseline NEON -- so what the emitter produces is what this host runs.
func MaxTiledTokensNative(t quant.Type) int { return MaxTiledTokensA64(t) }

func MaxTiledTokens(t quant.Type) int { return MaxTiledTokensA64(t) }

func EmitPackedMatMulTiled(t quant.Type, k, nrows, tok int, _ DotKind) ([]byte, error) {
	return EmitA64PackedMatMulTiled(t, k, nrows, tok)
}

// GEMMPad is the amd64 GEMM's per-token padding; see packedgemm.go.
const GEMMPad = 64

// StationaryIntAcc reports whether the arm64 stationary GEMM sums t's
// super-blocks in integers; see StationaryIntAccA64.
func StationaryIntAcc(t quant.Type, win int, _ DotKind) bool { return StationaryIntAccA64(t, win) }

// EmitPackedMatMulStationary is the arm64 weight-stationary GEMM; see
// EmitA64PackedMatMulStationary.
func EmitPackedMatMulStationary(t quant.Type, k, nrows, win int, _ DotKind) ([]byte, error) {
	return EmitA64PackedMatMulStationary(t, k, nrows, win)
}

// EmitPackedMatVecFusedWin is the fused kernel with the integer fold where
// win allows it; see EmitA64PackedMatVecFusedWin.
func EmitPackedMatVecFusedWin(t quant.Type, win int) ([]byte, error) {
	return EmitA64PackedMatVecFusedWin(t, PackedFusedGroupOf(t), a64FusedAhead, win)
}

func EmitPackedMatVecFusedAhead(t quant.Type, _ DotKind, ahead int) ([]byte, error) {
	return EmitA64PackedMatVecFused(t, PackedFusedGroupOf(t), ahead)
}

// FusedAheadWords selects the word-ahead prefetch, as on amd64 (packedfused.go).
const FusedAheadWords = 64
