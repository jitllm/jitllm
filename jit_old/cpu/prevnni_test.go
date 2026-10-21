//go:build amd64

package cpu

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// prevnniFormats is the packed family, named here rather than derived so that
// a format silently dropping out of the list is a visible edit.
var prevnniFormats = []quant.Type{
	quant.Q4_0, quant.Q8_0, quant.Q4_K, quant.Q5_K, quant.Q3_K, quant.Q6_K,
}

// TestEveryPackedFormatHasAPreVNNIKernel is the capability half: a pre-VNNI x86
// must generate for all six formats. It asks the emitter directly, since a
// missing kernel is invisible to correctness gates.
func TestEveryPackedFormatHasAPreVNNIKernel(t *testing.T) {
	for _, q := range prevnniFormats {
		if err := PreVNNIPacked(q); err != nil {
			t.Errorf("%s: refused pre-VNNI: %v", q, err)
			continue
		}
		if _, err := EmitPackedMatVec(q, PackedRows, DotVEX); err != nil {
			t.Errorf("%s: 64-row tile: %v", q, err)
		}
		if _, err := EmitPackedMatVec(q, PackedTail, DotVEX); err != nil {
			t.Errorf("%s: 8-row tail: %v", q, err)
		}
		if _, err := EmitPackedMatVecFused(q, DotVEX); err != nil {
			t.Errorf("%s: fused: %v", q, err)
		}
	}
}

// TestPreVNNIBoundIsDerivedNotAsserted runs PreVNNIPacked's saturation
// arithmetic and fails if it drifts: a saturating kernel does not crash, it
// returns slightly wrong dot products. No format reaches the unmasked-8-bit arm,
// so that arm must refuse on the widest byte an unmasked payload could carry.
func TestPreVNNIBoundIsDerivedNotAsserted(t *testing.T) {
	// Worst adjacent pair = 127*(u0+u1) against int16's 32767, per plane.
	want := map[quant.Type]int{
		quant.Q4_0: 3810,  // nibble 15
		quant.Q4_K: 3810,  // nibble 15
		quant.Q3_K: 3810,  // nibble 15
		quant.Q5_K: 4064,  // nibble 15, then the 0x10 secondary -> 16
		quant.Q6_K: 12192, // the 0x30 secondary -> 48, the tightest of the six
		quant.Q8_0: 32512, // |w| <= 128 through VPSIGNB, 0.78% of headroom
	}
	for q, pair := range want {
		if pair > 32767 {
			t.Fatalf("%s: the table itself claims %d, over int16", q, pair)
		}
		if err := PreVNNIPacked(q); err != nil {
			t.Errorf("%s: worst pair %d fits and it still refused: %v", q, pair, err)
		}
	}
	// The refusal must fire. quant.F32 has no device layout, which is the one
	// reachable "no"; the unmasked-8-bit arm is checked by construction
	// in the message below, since 127*2*255 = 64770 is what it computes.
	if err := PreVNNIPacked(quant.F32); err == nil {
		t.Error("PreVNNIPacked accepted a type with no device layout")
	}
}

// TestWideAndTiledAreRefusedPreVNNI: the two packed kernels that were not
// ported must refuse rather than emit VPDPBUSD a pre-VNNI host cannot execute
// (a SIGILL on the first token).
func TestWideAndTiledAreRefusedPreVNNI(t *testing.T) {
	for _, q := range prevnniFormats {
		if _, err := EmitPackedMatVecWide(q, DotVEX); err == nil {
			t.Errorf("%s: the WIDE kernel emitted under DotVEX", q)
		}
		// The tile width is narrowed until it fits L1i, as nn.tiledFor does,
		// since a refusal at the widest tok does not mean the format has no
		// tiled kernel.
		emitted := false
		for tok := MaxTiledTokens(q); tok >= 2; tok-- {
			if _, err := EmitPackedMatMulTiled(q, 256, 64, tok, DotVNNI); err != nil {
				continue
			}
			emitted = true
			if _, err := EmitPackedMatMulTiled(q, 256, 64, tok, DotVEX); err == nil {
				t.Errorf("%s: the TILED matmul emitted under DotVEX at tok=%d", q, tok)
			}
			break
		}
		if !emitted {
			t.Logf("%s: no tiled matmul under DotVNNI either (L1i budget)", q)
		}
	}
}

// TestNoVPDPBUSDSurvivesDotVEX disassembles every kernel a pre-VNNI host can
// build and asserts the instruction is not in it. A per-call-site flag can be
// half applied (a #UD on whichever branch was missed); reading the bytes
// catches it. It also asserts the VNNI kernel does contain it, so a
// disassembler that matched nothing would fail.
func TestNoVPDPBUSDSurvivesDotVEX(t *testing.T) {
	if !gnuObjdump(t) {
		// The tool is expected on development hosts; a skip is a real gap.
		t.Skip("GNU objdump is not available to disassemble the emitted stream")
	}
	ran := 0
	for _, q := range prevnniFormats {
		kinds := []struct {
			name string
			emit func(DotKind) ([]byte, error)
		}{
			{"tile", func(d DotKind) ([]byte, error) { return EmitPackedMatVec(q, PackedRows, d) }},
			{"tail", func(d DotKind) ([]byte, error) { return EmitPackedMatVec(q, PackedTail, d) }},
			{"fused", func(d DotKind) ([]byte, error) { return EmitPackedMatVecFused(q, d) }},
		}
		for _, kk := range kinds {
			vnni, err := kk.emit(DotVNNI)
			if err != nil {
				t.Fatalf("%s/%s: DotVNNI: %v", q, kk.name, err)
			}
			vex, err := kk.emit(DotVEX)
			if err != nil {
				t.Fatalf("%s/%s: DotVEX: %v", q, kk.name, err)
			}
			dv, dx := disasm(t, vnni), disasm(t, vex)
			// The control: without this the "absent" assertion below is
			// satisfied by a disassembly that says nothing at all.
			if !strings.Contains(dv, "vpdpbusd") {
				t.Fatalf("%s/%s: the VNNI kernel has no vpdpbusd, so this gate "+
					"cannot tell an absence from a broken disassembly", q, kk.name)
			}
			if strings.Contains(dx, "vpdpbusd") {
				n := strings.Count(dx, "vpdpbusd")
				t.Errorf("%s/%s: %d vpdpbusd survived DotVEX -- a #UD on the "+
					"first token of a pre-VNNI host", q, kk.name, n)
			}
			if !strings.Contains(dx, "vpmaddubsw") {
				t.Errorf("%s/%s: DotVEX emitted no vpmaddubsw, so it is not the "+
					"pre-VNNI sequence at all", q, kk.name)
			}
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no kernel was disassembled -- this gate proved nothing")
	}
	t.Logf("%d kernels disassembled, zero vpdpbusd under DotVEX", ran)
}

func gnuObjdump(t *testing.T) bool {
	t.Helper()
	if _, err := exec.LookPath("objdump"); err != nil {
		return false
	}
	v, err := exec.Command("objdump", "--version").Output()
	return err == nil && strings.Contains(string(v), "GNU objdump")
}

// disasm runs GNU objdump over a raw instruction stream, exactly as
// TestObjdumpRoundTrip does.
func disasm(t *testing.T, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "k.bin")
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", p).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump: %v\n%s", err, out)
	}
	return string(out)
}
