// Package ssegate is the SSE tier's VEX-leak gate, shared by every x86
// assembler's tests: main's SSE kernels (engine/device/cpu) and the composer's
// (composer/x86 through jit's CPU lowering).
//
// It reads a kernel's bytes back through objdump, which shares no code with
// either encoder, and fails on anything a pre-AVX host would fault on: a VEX or
// EVEX prefix (C4, C5, 62 in the opcode position, including VEX-encoded BMI
// ops), a ymm or zmm register, an instruction from an undeclared extension, and
// "(bad)". It decodes rather than byte-scans (C5 can be a ModRM byte, C4 is
// PINSRW's opcode), so the VEX test is exact. It also checks the trampoline's
// rules: no push, pop or call (RSP must be identical on every path), and no
// write to R14 (Go's g), RBP (callee-saved and not saved by callKernel) or RSP.
package ssegate

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer/x86"
)

// mnemonicISA classifies every mnemonic an SSE-tier kernel may contain by
// the extension that carries it; 0 is base x86-64 (the GP instructions). An
// unclassified mnemonic is a gate failure, so an instruction nobody thought
// about cannot pass by default -- add it here with its extension.
func mnemonicISA() map[string]x86.ISA {
	return func() map[string]x86.ISA {
		m := map[string]x86.ISA{}
		add := func(isa x86.ISA, names string) {
			for _, n := range strings.Fields(names) {
				m[n] = isa
			}
		}
		add(0, `mov movabs movzx movsx movsxd add sub and or xor inc dec neg not test cmp
		lea imul div shl shr sar sal rol ror jmp je jne jz jnz ja jae jb jbe jg jge jl jle
		js jns ret nop prefetcht0 prefetcht1 prefetcht2 prefetchnta
		cmove cmovne cmovz cmovnz cmova cmovae cmovb cmovbe cmovg cmovge cmovl cmovle`)
		add(x86.ISASSE2, `movaps movups movss movd movq movdqu movdqa movhlps movlhps movmskps pmovmskb
		addps subps mulps divps maxps minps andps andnps orps xorps unpcklps unpckhps shufps
		sqrtps rcpps rsqrtps cmpeqps cmpltps cmpleps cmpunordps cmpneqps cmpnltps cmpnleps cmpordps
		addss subss mulss divss maxss minss sqrtss cvtss2sd cvtsd2ss ucomiss comiss
		addpd subpd mulpd divpd
		cvtdq2ps cvtps2dq cvttps2dq cvtps2pd cvtpd2ps cvtsi2ss cvttss2si cvtss2si
		paddb paddw paddd paddq psubb psubw psubd psubq pmullw pmulhw pmulhuw pmuludq pmaddwd
		pand pandn por pxor pcmpeqb pcmpeqw pcmpeqd pcmpgtb pcmpgtw pcmpgtd
		packsswb packssdw packuswb punpcklbw punpcklwd punpckldq punpcklqdq
		punpckhbw punpckhwd punpckhdq punpckhqdq pminub pmaxub pminsw pmaxsw pavgb pavgw psadbw
		pshufd pshuflw pshufhw psrlw psrld psrlq psraw psrad psllw pslld psllq psrldq pslldq
		pinsrw pextrw lfence sfence mfence`)
		add(x86.ISASSE3, `haddps hsubps addsubps movsldup movshdup movddup lddqu`)
		add(x86.ISASSSE3, `pshufb phaddw phaddd phsubw phsubd pmaddubsw psignb psignw psignd pmulhrsw
		pabsb pabsw pabsd palignr`)
		add(x86.ISASSE41, `pmovsxbw pmovsxbd pmovsxbq pmovsxwd pmovsxwq pmovsxdq
		pmovzxbw pmovzxbd pmovzxbq pmovzxwd pmovzxwq pmovzxdq
		pmulld pmuldq pcmpeqq packusdw pminsb pminsd pminuw pminud pmaxsb pmaxsd pmaxuw pmaxud
		ptest pblendvb blendvps blendps pblendw insertps extractps pinsrb pinsrd pinsrq
		pextrb pextrd pextrq roundps roundss roundpd dpps`)
		add(x86.ISASSE42, `pcmpgtq crc32`)
		add(x86.ISAPOPCNT, `popcnt`)
		return m
	}()
}

// Insn is one disassembled instruction: its bytes and its normalised text.
type Insn struct {
	Bytes []byte
	Text  string
}

// Disasm disassembles code with objdump, one row per instruction and the
// whole encoding on that row (--insn-width=16, so a long instruction's bytes
// are not wrapped onto a continuation line). It keeps "(bad)" rows, because a
// gate must see them.
func Disasm(t testing.TB, dir, name string, code []byte) []Insn {
	t.Helper()
	p := filepath.Join(dir, name+".bin")
	if err := os.WriteFile(p, code, 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", "--insn-width=16", p).Output()
	if err != nil {
		t.Fatalf("objdump: %v", err)
	}
	var got []Insn
	for _, ln := range strings.Split(string(out), "\n") {
		f := strings.SplitN(ln, "\t", 3)
		if len(f) < 3 || !strings.HasSuffix(strings.TrimSpace(f[0]), ":") {
			continue
		}
		var bs []byte
		for _, h := range strings.Fields(f[1]) {
			var v byte
			if _, err := fmt.Sscanf(h, "%02x", &v); err != nil {
				t.Fatalf("objdump byte %q in %q: %v", h, ln, err)
			}
			bs = append(bs, v)
		}
		txt := strings.Join(strings.Fields(f[2]), " ")
		if i := strings.Index(txt, " #"); i >= 0 {
			txt = strings.TrimSpace(txt[:i])
		}
		got = append(got, Insn{Bytes: bs, Text: txt})
	}
	return got
}

// Problems returns every way code fails to be a legal SSE-tier kernel. It
// returns rather than failing, so the gate's own self-test can run it against
// violations.
func Problems(t testing.TB, name string, code []byte) []string {
	t.Helper()
	var bad []string
	add := func(f string, a ...any) { bad = append(bad, fmt.Sprintf(f, a...)) }

	declared, ok := x86.KernelISA(code)
	if !ok {
		add("no ISA declaration at code[0] (DeclareISA): Map would treat it as an AVX2 kernel")
	}
	if declared&(x86.ISAAVX2|x86.ISAFMA|x86.ISAF16C|x86.ISAVNNI) != 0 {
		add("declares %v: an SSE-tier kernel may not need any VEX extension", declared)
	}
	ins := Disasm(t, t.TempDir(), name, code)
	isas := mnemonicISA()
	if len(ins) == 0 {
		add("objdump found no instructions")
	}
	for i, in := range ins {
		where := fmt.Sprintf("insn %d (% x) %q", i, in.Bytes, in.Text)
		if strings.HasPrefix(in.Text, "(bad)") {
			add("%s: objdump cannot decode it", where)
			continue
		}
		// The opcode's first byte, past legacy prefixes and REX.
		j := 0
		for j < len(in.Bytes) && isLegacyPrefix(in.Bytes[j]) {
			j++
		}
		if j < len(in.Bytes) && in.Bytes[j]&0xF0 == 0x40 {
			j++
		}
		if j < len(in.Bytes) {
			switch in.Bytes[j] {
			case 0xC4, 0xC5:
				add("%s: VEX-encoded -- faults on a host with no AVX", where)
			case 0x62:
				add("%s: EVEX-encoded -- faults on a host with no AVX-512", where)
			}
		}
		fields := strings.Fields(in.Text)
		mn := fields[0]
		ops := strings.Join(fields[1:], " ")
		if strings.HasPrefix(mn, "v") {
			add("%s: a VEX mnemonic", where)
		}
		if strings.Contains(ops, "ymm") || strings.Contains(ops, "zmm") {
			add("%s: a 256/512-bit register", where)
		}
		switch mn {
		case "push", "pop", "call", "enter", "leave", "pushf", "popf", "pushfq", "popfq":
			add("%s: moves RSP; the trampoline requires RSP identical on every path", where)
			continue
		}
		if strings.HasPrefix(mn, "rex") || mn == "data16" || mn == "addr32" || mn == "lock" || mn == "rep" {
			add("%s: a stray prefix objdump could not attach to an instruction", where)
			continue
		}
		isa, known := isas[mn]
		if mn == "pextrw" && strings.Contains(ops, "PTR") {
			isa = x86.ISASSE41 // the memory-destination form is SSE4.1's
		}
		if !known {
			add("%s: unclassified mnemonic %q -- add it to mnemonicISA with its extension", where, mn)
		} else if isa&^declared != 0 {
			add("%s: needs %v, which the kernel did not declare (%v)", where, isa, declared)
		}
		if dst := strings.SplitN(ops, ",", 2)[0]; WritesFirstOperand(mn) && !strings.Contains(dst, "[") {
			switch dst {
			case "r14", "r14d", "r14w", "r14b", "rbp", "ebp", "bp", "bpl", "rsp", "esp", "sp", "spl":
				add("%s: writes %s (R14 is Go's g; RBP is not saved by callKernel; RSP must not move)", where, dst)
			}
		}
	}
	if len(ins) > 0 && ok {
		if first := ins[0]; len(first.Bytes) != x86.ISAMarkerLen || !strings.HasPrefix(first.Text, "nop") {
			add("the first instruction is %q (% x), not the ISA declaration's NOP", first.Text, first.Bytes)
		}
	}
	return bad
}

// isLegacyPrefix reports the one-byte prefixes that may precede an opcode.
func isLegacyPrefix(b byte) bool {
	switch b {
	case 0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65:
		return true
	}
	return false
}

// WritesFirstOperand is false for the instructions whose first operand is
// only read.
func WritesFirstOperand(mn string) bool {
	switch mn {
	case "cmp", "test", "ptest", "ucomiss", "comiss", "prefetcht0", "prefetcht1",
		"prefetcht2", "prefetchnta", "jmp", "ret", "nop":
		return false
	}
	return !strings.HasPrefix(mn, "j")
}

// Require fails t for every problem Problems finds in code. Where objdump is
// missing it logs and returns rather than skipping, so the calling gate's
// execution half still runs on a host without objdump.
func Require(t testing.TB, name string, code []byte) {
	t.Helper()
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Logf("%s: objdump not installed -- the VEX-leak half of this gate did not run here", name)
		return
	}
	for _, p := range Problems(t, name, code) {
		t.Errorf("%s: %s", name, p)
	}
}
