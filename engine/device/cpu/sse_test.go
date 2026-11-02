//go:build amd64

// amd64 only: these compare emitted x86 bytes against objdump, so they are
// about an encoding, not a tier.

package cpu

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestSSEEncodingMatchesObjdump pins the legacy-SSE encoder against a
// disassembler: a missing 0x66 turns PXOR into a valid MMX instruction, and a
// wrong REX bit silently picks a different register.
func TestSSEEncodingMatchesObjdump(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed")
	}
	cases := []struct {
		name string
		emit func(a *Buf)
		want string // objdump -M intel text, normalised
	}{
		{"pxor", func(a *Buf) { a.sseOp(0xEF, p66, m0F, 0, Y0, Y0) }, "pxor xmm0,xmm0"},
		{"pxor hi", func(a *Buf) { a.sseOp(0xEF, p66, m0F, 0, Y9, Y12) }, "pxor xmm9,xmm12"},
		{"paddd", func(a *Buf) { a.sseOp(0xFE, p66, m0F, 0, Y1, Y2) }, "paddd xmm1,xmm2"},
		{"pmaddwd", func(a *Buf) { a.sseOp(0xF5, p66, m0F, 0, Y3, Y4) }, "pmaddwd xmm3,xmm4"},
		// SSSE3, 0F38 map -- the int8 dot's first half.
		{"pmaddubsw", func(a *Buf) { a.sseOp(0x04, p66, m0F38, 0, Y5, Y6) }, "pmaddubsw xmm5,xmm6"},
		// No prefix at all: the .PS forms.
		{"mulps", func(a *Buf) { a.sseOp(0x59, 0, m0F, 0, Y7, Y8) }, "mulps xmm7,xmm8"},
		{"addps", func(a *Buf) { a.sseOp(0x58, 0, m0F, 0, Y10, Y11) }, "addps xmm10,xmm11"},
		{"movaps", func(a *Buf) { a.MOVAPS(Y13, Y14) }, "movaps xmm13,xmm14"},
	}

	dir := t.TempDir()
	ran := 0
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var a Buf
			c.emit(&a)
			b := a.Bytes()
			if len(b) == 0 {
				t.Fatal("emitted nothing")
			}
			got := disasmIntel(t, dir, c.name, b)
			if got != c.want {
				t.Fatalf("emitted % x\n  objdump: %q\n  want:    %q", b, got, c.want)
			}
			ran++
		})
	}
	if ran != len(cases) {
		t.Fatalf("%d of %d cases ran -- this gate proved less than it claims", ran, len(cases))
	}
	t.Logf("%d legacy-SSE encodings pinned against objdump", ran)
}

// TestSSEThreeOperandLowering covers the arity gap: the emitters speak
// three-operand and legacy SSE is two-operand and destructive.
func TestSSEThreeOperandLowering(t *testing.T) {
	// Assert the instructions, not the byte count.
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed")
	}
	dir := t.TempDir()
	lower := func(name string, dst, src1, src2 Reg, commutes bool) []string {
		var a Buf
		a.sse3(0x58, 0, m0F, 0, commutes, dst, src1, src2) // addps
		return disasmAll(t, dir, name, a.Bytes())
	}
	// dst == src1: no copy at all, which is the common accumulator shape.
	if got := lower("same", Y1, Y1, Y2, true); len(got) != 1 || got[0] != "addps xmm1,xmm2" {
		t.Fatalf("dst==src1 lowered to %q, want one bare addps", got)
	}
	// dst == src2 on a commutative op: swap the operands, still no copy.
	if got := lower("swap", Y1, Y2, Y1, true); len(got) != 1 || got[0] != "addps xmm1,xmm2" {
		t.Fatalf("dst==src2 commutative lowered to %q, want one swapped addps", got)
	}
	// Disjoint: one MOVAPS then the op.
	want := []string{"movaps xmm1,xmm2", "addps xmm1,xmm3"}
	if got := lower("copy", Y1, Y2, Y3, true); len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("disjoint lowered to %q, want %q", got, want)
	}
	// dst aliases src2 on a non-commutative op: a copy would destroy src2 and
	// the op would read src1 twice, so it must panic.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("dst==src2 on a non-commutative op did NOT panic; it " +
					"would emit src1 op src1, which is wrong and fast")
			}
		}()
		var d Buf
		d.sse3(0x5C, 0, m0F, 0, false, Y1, Y2, Y1) // subps: a-b != b-a
	}()
}

func disasmIntel(t *testing.T, dir, name string, code []byte) string {
	t.Helper()
	// A raw binary objdump can be told to treat as x86-64 code.
	p := filepath.Join(dir, name+".bin")
	if err := os.WriteFile(p, code, 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", p).Output()
	if err != nil {
		t.Fatalf("objdump: %v", err)
	}
	for _, ln := range strings.Split(string(out), "\n") {
		i := strings.Index(ln, "\t")
		if i < 0 {
			continue
		}
		rest := ln[i+1:]
		j := strings.Index(rest, "\t")
		if j < 0 {
			continue
		}
		insn := strings.TrimSpace(rest[j+1:])
		if insn == "" || strings.HasPrefix(insn, "(bad)") {
			continue
		}
		return strings.Join(strings.Fields(insn), " ")
	}
	t.Fatalf("objdump produced no instruction for % x:\n%s", code, out)
	return ""
}

// disasmAll returns every instruction objdump finds, in order.
func disasmAll(t *testing.T, dir, name string, code []byte) []string {
	t.Helper()
	p := filepath.Join(dir, name+".bin")
	if err := os.WriteFile(p, code, 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", p).Output()
	if err != nil {
		t.Fatalf("objdump: %v", err)
	}
	var got []string
	for _, ln := range strings.Split(string(out), "\n") {
		i := strings.Index(ln, "\t")
		if i < 0 {
			continue
		}
		rest := ln[i+1:]
		j := strings.Index(rest, "\t")
		if j < 0 {
			continue
		}
		insn := strings.TrimSpace(rest[j+1:])
		if insn == "" || strings.HasPrefix(insn, "(bad)") {
			continue
		}
		got = append(got, strings.Join(strings.Fields(insn), " "))
	}
	return got
}
