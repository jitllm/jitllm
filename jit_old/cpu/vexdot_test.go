//go:build amd64

package cpu

import (
	"encoding/binary"
	"testing"
)

// TestExecuteVEXDotMatchesVPDPBUSD runs the pre-VNNI sequence against a scalar
// reference (prevnni_test.go only disassembles). On a VNNI host it also
// asserts prevnni.go's claim that DotVEX is bit-identical to DotVNNI wherever
// VPMADDUBSW's int16 saturation cannot fire.
func TestExecuteVEXDotMatchesVPDPBUSD(t *testing.T) {
	// dot(unsigned u, signed act), one 32-byte vector, into 8 int32 lanes.
	build := func(kind DotKind) []byte {
		var a Buf
		a.MOVLoad(RSI, At(RDI, 16)) // args.A -- the SIGNED activation
		a.MOVLoad(RDX, At(RDI, 8))  // args.W -- the UNSIGNED payload
		a.MOVLoad(RCX, At(RDI, 56)) // args.Scr
		a.VPXOR(Y0, Y0, Y0)
		d := dotEmitter{kind: kind, ones: Y3}
		if d.vex() {
			ones16(&a, Y3)
		}
		a.VMOVDQULoad(Y1, At(RDX, 0)) // u   (clobbered by unsigned())
		a.VMOVDQULoad(Y2, At(RSI, 0)) // act
		d.unsigned(&a, Y0, Y1, Y2)
		a.VMOVDQUStore(At(RCX, 0), Y0)
		a.VZEROUPPER()
		a.RET()
		return a.Bytes()
	}

	// The inputs stay inside the non-saturating range, the claim's
	// precondition: VPMADDUBSW sums adjacent pairs at int16, and a payload of
	// 0..63 (the widest two-plane format) gives 2*63*127 = 16,002.
	act := make([]int8, 32)
	u := make([]byte, 32)
	for i := range act {
		act[i] = int8(i*37%255 - 127)
		u[i] = byte(i * 11 % 64)
	}

	run := func(code []byte) []int32 {
		c := mustMap(t, code)
		defer c.Close()
		scr := make([]byte, 32)
		args := Args{A: &act[0], W: &u[0], Scr: &scr[0]}
		c.Call(&args)
		out := make([]int32, 8)
		for lane := range out {
			out[lane] = int32(binary.LittleEndian.Uint32(scr[lane*4:]))
		}
		return out
	}

	want := make([]int32, 8)
	for lane := range want {
		for i := 0; i < 4; i++ {
			k := lane*4 + i
			want[lane] += int32(u[k]) * int32(act[k])
		}
	}

	// The VEX arm runs on every AVX2 host; an SSE-only host must refuse it.
	probe := avx2Primitive(t, build(DotVEX), "TestSSEDotMatchesVPDPBUSD")
	if probe == nil {
		return
	}
	probe.Close()
	vex := run(build(DotVEX))
	for lane := range want {
		if vex[lane] != want[lane] {
			t.Errorf("DotVEX lane %d: kernel %d, reference %d", lane, vex[lane], want[lane])
		}
	}

	if !CPU().AVXVNNI {
		t.Log("no AVX-VNNI here, so the bit-identical half did not run; it is " +
			"covered on any Alder Lake or Zen 4. The arm that matters ON THIS " +
			"HOST -- the AVX2 sequence against the reference -- did run.")
		return
	}
	vnni := run(build(DotVNNI))
	for lane := range want {
		if vnni[lane] != want[lane] {
			t.Errorf("DotVNNI lane %d: kernel %d, reference %d", lane, vnni[lane], want[lane])
		}
		if vex[lane] != vnni[lane] {
			t.Errorf("lane %d: DotVEX %d != DotVNNI %d -- prevnni.go's bit-identical "+
				"claim is false, and signedFix spends an instruction to keep it",
				lane, vex[lane], vnni[lane])
		}
	}
}
