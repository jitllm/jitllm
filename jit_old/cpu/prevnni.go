//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The pre-VNNI packed path: every packed kernel's integer dot product, written
// once, in the two instruction sequences a host can run. VPDPBUSD is AVX-VNNI
// (Alder Lake, Zen 4); older AVX2 hosts get the three-instruction sequence
// instead of declining every quantized type.
//
// dotEmitter is the only place either sequence is written, so a kernel is
// wholly one or wholly the other: one site left on VPDPBUSD would be a SIGILL
// on the first token. The kernels with no pre-VNNI form refuse rather than
// emit a mixture (packedwide.go, packedtiled.go), and
// TestEveryVPDPBUSDSiteIsCoveredOrRefused greps the package to hold that.

// DotKind selects the instruction sequence a packed kernel uses to accumulate
// four-byte integer dot products into int32 lanes.
//
// It is a parameter, not a probe: the emitters must still produce code for a
// target this machine cannot run, or the cross-arch disassembly gates break.
// HostDotKind is what a caller about to execute uses (engine/nn/jit.go).
type DotKind int

const (
	// DotVNNI is VPDPBUSD: one instruction, AVX-VNNI.
	DotVNNI DotKind = iota
	// DotVEX is VPMADDUBSW + VPMADDWD + VPADDD: three instructions, AVX2
	// baseline, and bit-identical to DotVNNI wherever VPMADDUBSW's int16
	DotVEX
)

func (d DotKind) String() string {
	if d == DotVEX {
		return "vex"
	}
	return "vnni"
}

// HostDotKind is the sequence this cpu can execute within the AVX2 tier: it
// answers DotVEX on an SSE host too, and no SSE-tier emitter takes a DotKind.
// It asks cpu.CPU(), the package's single capability probe; tests force the
// pre-VNNI answer with ForceNoVNNIForTest under the jitllmtest tag.
func HostDotKind() DotKind {
	if CPU().AVXVNNI {
		return DotVNNI
	}
	return DotVEX
}

// dotEmitter emits "acc += the four-byte dot products of an unsigned payload
// against a signed activation", in whichever sequence the DotKind names.
//
// ones must hold an i16 all-ones vector for DotVEX and is ignored for DotVNNI;
// flip must hold a 0x80 byte vector for DotVNNI's signed payloads and is
// ignored for DotVEX.
type dotEmitter struct {
	kind DotKind
	ones Reg
	flip Reg
}

// vex reports whether this emitter is the pre-VNNI one.
func (d dotEmitter) vex() bool { return d.kind == DotVEX }

// unsigned accumulates dot(u, act) into acc, where u is an unsigned byte plane
// already masked or shifted into range by the caller. The DotVEX form clobbers
// u (its int16 partials have nowhere else to go), so callers hand it a dead
// temporary or reload the register per word.
func (d dotEmitter) unsigned(a *Buf, acc, u, act Reg) {
	if !d.vex() {
		a.VPDPBUSD(acc, u, act)
		return
	}
	a.VPMADDUBSW(u, u, act)
	a.VPMADDWD(u, u, d.ones)
	a.VPADDD(acc, acc, u)
}

// signed accumulates the same dot over a payload that is a full signed byte
// (Q8_0 only).
//
// The two sequences accumulate different integers here; signedFix makes them
// agree. The VNNI kernel XORs 0x80 into the payload and accumulates dot(w+128,
// a), and the epilogue subtracts 128*sum(a). VPMADDUBSW would saturate on that
// (2*255*127 = 64,770 against int16, whenever w0+w1 > 2 beside a block
// maximum), so the pre-VNNI kernel does not centre: VPSIGNB gives |w| <= 128
// and moves w's sign onto the activation, 2*128*127 = 32,512 fits, and the
// accumulator holds dot(w, a).
//
// It clobbers pay and t.
func (d dotEmitter) signed(a *Buf, acc, pay, act, t Reg) {
	if !d.vex() {
		a.VPXOR(t, pay, d.flip)
		a.VPDPBUSD(acc, t, act)
		return
	}
	if d.naiveSigned(a, acc, pay, act) {
		return
	}
	a.VPSIGNB(t, act, pay) // a * sign(w), still int8
	a.VPSIGNB(pay, pay, pay)
	a.VPMADDUBSW(pay, pay, t) // |w| x (a*sign(w)), pair <= 2*128*127
	a.VPMADDWD(pay, pay, d.ones)
	a.VPADDD(acc, acc, pay)
}

// signedFix puts the accumulators back on VPDPBUSD's scale, so the epilogue's
// f32 chain and its -128*sum(a) correction are unchanged and the two kernels
// are bit-identical. That costs an instruction, and buys the strongest gate:
// emit both kernels on a VNNI box and assert ==.
//
// negSum addresses pairs[2b+1] = -sum(a)*128/8 for this sub-block, so
// int32(that) << 3 is -128*sum(a) and subtracting it adds the 128*sum(a) the
// XOR would have contributed. |sum(a)| <= 127*32, so -16*sum(a) is exact in f32
// and the convert and shift are exact too. It clobbers t.
func (d dotEmitter) signedFix(a *Buf, acc []Reg, negSum Mem, t Reg) {
	if !d.vex() || d.naiveActive() {
		// The naive form already accumulates the centred dot, so the violation
		// is the centring and the saturation and nothing else.
		return
	}
	a.VBROADCASTSS(t, negSum)
	a.VCVTPS2DQ(t, t)
	a.VPSLLD(t, t, 3)
	for _, r := range acc {
		a.VPSUBD(r, r, t)
	}
}

// ones16 materialises the i16 all-ones vector VPMADDWD needs, in two
// instructions and no memory. It is built rather than loaded from
// PackedScratch because loading needs a GPR holding Scr, and inside
// EmitPackedMatVecFused's row loop there is none (RBX is the row counter).
func ones16(a *Buf, dst Reg) {
	a.VPCMPEQW(dst, dst, dst)
	a.VPSRLW(dst, dst, 15)
}

// PreVNNIPacked reports whether the AVX2-only packed kernels exist for t, and
// says why when they do not.
//
// The bound is re-derived from the layout: VPMADDUBSW sums adjacent byte
// products into one int16, so a format is served only while 127*(u0+u1) fits
// in 32767 for every plane. 127 because QuantizeQ8Window scales by
// inv = 127/amax, so |a| <= 127 by construction (internal/oracle/quantize.go),
// and the activation-quantizer kernels are held to it bit for bit
// (TestQuantActMatchesTheGoLoopExactly). A saturating kernel does not crash,
// it gives slightly wrong dots, so a wider payload is refused here.
func PreVNNIPacked(t quant.Type) error {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return fmt.Errorf("jit: pre-VNNI: %s has no device layout", t)
	}
	_, bits, _, _ := kernels.Layout(q)
	hi := kernels.HiPlane(q)
	const actMax = 127
	const i16Max = 32767
	check := func(plane string, umax int) error {
		if pair := actMax * 2 * umax; pair > i16Max {
			return fmt.Errorf("jit: pre-VNNI: %s's %s plane reaches %d, so VPMADDUBSW's "+
				"adjacent pair reaches %d of %d and saturates", t, plane, umax, pair, i16Max)
		}
		return nil
	}
	switch {
	case bits == 4 && kernels.Codes(q) != nil:
		// A table code reaches the dot as its table ENTRY, so the entry bounds
		// it: 24 for MXFP4.
		umax := 0
		for _, c := range kernels.Codes(q) {
			umax = max(umax, int(c))
		}
		if err := check("code table", umax); err != nil {
			return err
		}
	case bits < 8:
		// The field mask bounds the primary plane at 15 (a nibble) or 3 (a
		// 2-bit field) whatever the file holds.
		if err := check("field", 1<<bits-1); err != nil {
			return err
		}
	case kernels.SignedPayload(q):
		// VPSIGNB, not the +128 centring: |w| <= 128 for every input.
		if err := check("signed", 128); err != nil {
			return err
		}
	default:
		// An unsigned 8-bit payload is refused, and no format is one
		// (Q5_K is two-plane). An unmasked byte reaches
		// 255 and the pair saturates; a future such format must be masked or
		// have its bound derived from its packer.
		if err := check("unmasked 8-bit", 0xFF); err != nil {
			return err
		}
	}
	if hi != 0 {
		// The secondary plane's mask is pre-shifted by 4, so its byte is
		// (2^hi-1)<<4: 16 for Q5_K and 48 for Q6_K.
		if err := check("secondary", ((1<<hi)-1)<<4); err != nil {
			return err
		}
	}
	return nil
}

// SupportedPackedNative is "can this host run the generated packed kernel for
// t" -- the container decode path, which is the only weight path model.Open
// admits. It dispatches on the host's tier.
//
// It is separate from SupportedNative because the GGUF row-major emitters
// (matvec.go, gemm.go, gemmk.go) emit VPDPBUSD with no pre-VNNI form; a
// container never decodes through them, so they stay declined on a pre-VNNI
// host. UsableAVX2 is asked because the VEX sequence is YMM code and the
// CPUID bit alone does not say the OS saves YMM state (XCR0).
func SupportedPackedNative(t quant.Type) bool { return Native().PackedSupported(t) }

// primaryPackedSupported is SupportedPackedNative's AVX2-tier answer, verbatim.
func primaryPackedSupported(t quant.Type) bool {
	if !PackedSupported(t) {
		return false
	}
	f := CPU()
	if !f.UsableAVX2() {
		return false
	}
	if f.AVXVNNI {
		return true
	}
	return PreVNNIPacked(t) == nil
}
