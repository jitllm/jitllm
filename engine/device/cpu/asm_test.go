package cpu

import (
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Every expected byte string below came out of the system assembler
// (as --64 + objdump -d -M intel). Regenerate with scripts/vexref.sh if an
// instruction is added.
//
// This is T2: a VEX field taken from the wrong bit does not fault, it silently
// names a different register (VEX.R from bit 0 instead of bit 3 turns ymm3
// into ymm11).
//
// The vpdpbusd entries start with c4 (VEX). GNU as and Go's assembler both
// default to 62 (EVEX, AVX-512-VNNI), which SIGILLs on a CPU without AVX-512.
var encodings = []struct {
	name string
	want string
	emit func(*Buf)
}{
	// AVX-VNNI.
	// Prefetch: 0F 18 /1 for T0, /0 for NTA. No REX for a base under r8, and no
	// operand-size prefix ever -- there is no 64-bit form.
	{"prefetcht0 [rsi]", "0f180e", func(a *Buf) { a.PREFETCHT0(At(RSI, 0)) }},
	{"prefetcht0 [rsi+0x100]", "0f188e00010000", func(a *Buf) { a.PREFETCHT0(At(RSI, 256)) }},
	{"prefetcht0 [r11+0x40]", "410f184b40", func(a *Buf) { a.PREFETCHT0(At(R11, 64)) }},
	{"prefetchnta [rdx]", "0f1802", func(a *Buf) { a.PREFETCHNTA(At(RDX, 0)) }},
	{"vpdpbusd ymm1,ymm2,ymm3", "c4e26d50cb", func(a *Buf) { a.VPDPBUSD(Y1, Y2, Y3) }},
	{"vpdpbusd ymm15,ymm14,ymm13", "c4420d50fd", func(a *Buf) { a.VPDPBUSD(Y15, Y14, Y13) }},
	{"vpdpbusd ymm8,ymm0,ymm1", "c4627d50c1", func(a *Buf) { a.VPDPBUSD(Y8, Y0, Y1) }},
	{"vpdpbusd ymm1,ymm2,[rdi+0x40]", "c4e26d504f40", func(a *Buf) { a.VPDPBUSDMem(Y1, Y2, At(RDI, 0x40)) }},
	{"vpdpbusd ymm9,ymm2,[r12+rsi*4+0x100]", "c4426d508cb400010000", func(a *Buf) { a.VPDPBUSDMem(Y9, Y2, Idx(R12, RSI, 4, 0x100)) }},

	{"vpmaddubsw ymm1,ymm2,ymm3", "c4e26d04cb", func(a *Buf) { a.VPMADDUBSW(Y1, Y2, Y3) }},
	{"vpmaddubsw ymm15,ymm14,ymm13", "c4420d04fd", func(a *Buf) { a.VPMADDUBSW(Y15, Y14, Y13) }},
	{"vpmaddwd ymm1,ymm2,ymm3", "c5edf5cb", func(a *Buf) { a.VPMADDWD(Y1, Y2, Y3) }},
	{"vpmaddwd ymm15,ymm14,ymm13", "c4410df5fd", func(a *Buf) { a.VPMADDWD(Y15, Y14, Y13) }},
	{"vpmaddwd ymm1,ymm2,[rsi+0x20]", "c5edf54e20", func(a *Buf) { a.VPMADDWDMem(Y1, Y2, At(RSI, 0x20)) }},
	{"vpmaddwd ymm9,ymm14,[rsi+0x20]", "c50df54e20", func(a *Buf) { a.VPMADDWDMem(Y9, Y14, At(RSI, 0x20)) }},
	{"vpabsb ymm1,ymm2", "c4e27d1cca", func(a *Buf) { a.VPABSB(Y1, Y2) }},
	{"vpabsb ymm12,ymm13", "c4427d1ce5", func(a *Buf) { a.VPABSB(Y12, Y13) }},
	{"dec dword [rdi+0x58]", "ff4f58", func(a *Buf) { a.DECmem(At(RDI, 0x58)) }},

	{"vpaddd ymm1,ymm2,ymm3", "c5edfecb", func(a *Buf) { a.VPADDD(Y1, Y2, Y3) }},
	{"vpaddd ymm15,ymm14,ymm13", "c4410dfefd", func(a *Buf) { a.VPADDD(Y15, Y14, Y13) }},
	{"vpaddd ymm1,ymm2,[rdi+0x40]", "c5edfe4f40", func(a *Buf) { a.VPADDDMem(Y1, Y2, At(RDI, 0x40)) }},
	{"vpaddw ymm1,ymm2,ymm3", "c5edfdcb", func(a *Buf) { a.VPADDW(Y1, Y2, Y3) }},
	{"vpsubb ymm15,ymm14,ymm13", "c4410df8fd", func(a *Buf) { a.VPSUBB(Y15, Y14, Y13) }},
	{"vpand ymm1,ymm2,ymm3", "c5eddbcb", func(a *Buf) { a.VPAND(Y1, Y2, Y3) }},
	{"vpor ymm1,ymm2,ymm3", "c5edebcb", func(a *Buf) { a.VPOR(Y1, Y2, Y3) }},
	{"vpxor ymm15,ymm15,ymm15", "c44105efff", func(a *Buf) { a.VPXOR(Y15, Y15, Y15) }},

	{"vaddps ymm1,ymm2,ymm3", "c5ec58cb", func(a *Buf) { a.VADDPS(Y1, Y2, Y3) }},
	{"vmulps ymm1,ymm2,ymm3", "c5ec59cb", func(a *Buf) { a.VMULPS(Y1, Y2, Y3) }},
	{"vphaddd ymm1,ymm2,ymm3", "c4e26d02cb", func(a *Buf) { a.VPHADDD(Y1, Y2, Y3) }},
	{"vfmadd231ps ymm1,ymm2,ymm3", "c4e26db8cb", func(a *Buf) { a.VFMADD231PS(Y1, Y2, Y3) }},
	{"vfmadd231ps ymm15,ymm14,ymm13", "c4420db8fd", func(a *Buf) { a.VFMADD231PS(Y15, Y14, Y13) }},

	// Shift-by-immediate: ModRM.reg is an opcode extension and the destination
	// travels in vvvv. The one family whose operands are not where they look.
	{"vpsrlw ymm1,ymm2,4", "c5f571d204", func(a *Buf) { a.VPSRLW(Y1, Y2, 4) }},
	{"vpsrlw ymm15,ymm14,4", "c4c10571d604", func(a *Buf) { a.VPSRLW(Y15, Y14, 4) }},
	{"vpsrld ymm1,ymm2,3", "c5f572d203", func(a *Buf) { a.VPSRLD(Y1, Y2, 3) }},
	{"vpsllw ymm1,ymm2,2", "c5f571f202", func(a *Buf) { a.VPSLLW(Y1, Y2, 2) }},
	{"vpslld ymm1,ymm2,5", "c5f572f205", func(a *Buf) { a.VPSLLD(Y1, Y2, 5) }},

	// Memory forms, chosen to exercise every ModRM/SIB corner case.
	{"vmovdqu ymm1,[rdi+0x40]", "c5fe6f4f40", func(a *Buf) { a.VMOVDQULoad(Y1, At(RDI, 0x40)) }},
	{"vmovdqu ymm15,[r13+0]", "c4417e6f7d00", func(a *Buf) { a.VMOVDQULoad(Y15, At(R13, 0)) }},
	{"vmovdqu ymm9,[r12+r14*8+0x1234]", "c4017e6f8cf434120000", func(a *Buf) { a.VMOVDQULoad(Y9, Idx(R12, R14, 8, 0x1234)) }},
	{"vmovdqu [rsp+0x20],ymm3", "c5fe7f5c2420", func(a *Buf) { a.VMOVDQUStore(At(RSP, 0x20), Y3) }},

	{"vpbroadcastd ymm1,[rdi]", "c4e27d580f", func(a *Buf) { a.VPBROADCASTD(Y1, At(RDI, 0)) }},
	{"vpbroadcastd ymm15,[rbp+0]", "c4627d587d00", func(a *Buf) { a.VPBROADCASTD(Y15, At(RBP, 0)) }},
	{"vbroadcastss ymm1,[rdi+0x10]", "c4e27d184f10", func(a *Buf) { a.VBROADCASTSS(Y1, At(RDI, 0x10)) }},

	{"vcvtdq2ps ymm1,ymm2", "c5fc5bca", func(a *Buf) { a.VCVTDQ2PS(Y1, Y2) }},
	{"vcvtdq2ps ymm15,ymm14", "c4417c5bfe", func(a *Buf) { a.VCVTDQ2PS(Y15, Y14) }},
	{"vcvtph2ps ymm1,[rdi]", "c4e27d130f", func(a *Buf) { a.VCVTPH2PS(Y1, At(RDI, 0)) }},
	{"vpinsrb xmm1,xmm1,byte [rdi],0", "c4e371200f00", func(a *Buf) { a.VPINSRBLoad(Y1, Y1, At(RDI, 0), 0) }},
	{"vpinsrb xmm3,xmm2,byte [r13+8],1", "c4c369205d0801", func(a *Buf) { a.VPINSRBLoad(Y3, Y2, At(R13, 8), 1) }},
	{"vextracti128 xmm1,ymm2,1", "c4e37d39d101", func(a *Buf) { a.VEXTRACTI128(Y1, Y2, 1) }},
	{"vzeroupper", "c5f877", func(a *Buf) { a.VZEROUPPER() }},
	{"vpshufb ymm1,ymm2,ymm3", "c4e26d00cb", func(a *Buf) { a.VPSHUFB(Y1, Y2, Y3) }},
	{"vpshufb ymm15,ymm14,ymm13", "c4420d00fd", func(a *Buf) { a.VPSHUFB(Y15, Y14, Y13) }},
	{"vpmovzxbd ymm1,[rdi]", "c4e27d310f", func(a *Buf) { a.VPMOVZXBD(Y1, At(RDI, 0)) }},
	{"vpmovzxbd ymm15,[r13+8]", "c4427d317d08", func(a *Buf) { a.VPMOVZXBD(Y15, At(R13, 8)) }},
	{"vpmovzxbd ymm1,xmm2", "c4e27d31ca", func(a *Buf) { a.VPMOVZXBDReg(Y1, Y2) }},
	{"vpmovsxbd ymm1,[rdi]", "c4e27d210f", func(a *Buf) { a.VPMOVSXBD(Y1, At(RDI, 0)) }},
	{"vshufps ymm1,ymm2,ymm3,0x88", "c5ecc6cb88", func(a *Buf) { a.VSHUFPS(Y1, Y2, Y3, 0x88) }},
	{"vpermq ymm1,ymm2,0xd8", "c4e3fd00cad8", func(a *Buf) { a.VPERMQ(Y1, Y2, 0xD8) }},
	{"vpermq ymm15,ymm14,0xd8", "c443fd00fed8", func(a *Buf) { a.VPERMQ(Y15, Y14, 0xD8) }},
	{"vpmovsxbd ymm15,[rbx+8]", "c4627d217b08", func(a *Buf) { a.VPMOVSXBD(Y15, At(RBX, 8)) }},
	{"vpsllw ymm1,ymm2,4", "c5f571f204", func(a *Buf) { a.VPSLLW(Y1, Y2, 4) }},
	{"vmovdqu xmm1,[rdi+4]", "c5fa6f4f04", func(a *Buf) { a.VMOVDQULoadx(Y1, At(RDI, 4)) }},
	{"vpsrlw ymm1,ymm2,2", "c5f571d202", func(a *Buf) { a.VPSRLW(Y1, Y2, 2) }},
	{"vbroadcastss ymm1,[rsp-0x40]", "c4e27d184c24c0", func(a *Buf) { a.VBROADCASTSS(Y1, At(RSP, -0x40)) }},

	// Kernel instructions: nibble unpack, scale conversion, horizontal reduction.
	{"vbroadcasti128 ymm1,[rdi]", "c4e27d5a0f", func(a *Buf) { a.VBROADCASTI128(Y1, At(RDI, 0)) }},
	{"vpblendd ymm1,ymm2,ymm3,0xf0", "c4e36d02cbf0", func(a *Buf) { a.VPBLENDD(Y1, Y2, Y3, 0xF0) }},
	{"vpblendd ymm15,ymm14,ymm13,0xf0", "c4430d02fdf0", func(a *Buf) { a.VPBLENDD(Y15, Y14, Y13, 0xF0) }},
	{"vcvtph2ps xmm1,[rdi]", "c4e279130f", func(a *Buf) { a.VCVTPH2PSx(Y1, At(RDI, 0)) }},
	{"vbroadcastss ymm1,xmm2", "c4e27d18ca", func(a *Buf) { a.VBROADCASTSSReg(Y1, Y2) }},
	{"vbroadcastss ymm15,xmm14", "c4427d18fe", func(a *Buf) { a.VBROADCASTSSReg(Y15, Y14) }},
	{"vextractf128 xmm1,ymm2,1", "c4e37d19d101", func(a *Buf) { a.VEXTRACTF128(Y1, Y2, 1) }},
	{"vaddps xmm1,xmm2,xmm3", "c5e858cb", func(a *Buf) { a.VADDPSx(Y1, Y2, Y3) }},
	{"vhaddps xmm1,xmm2,xmm3", "c5eb7ccb", func(a *Buf) { a.VHADDPSx(Y1, Y2, Y3) }},
	{"vmovss [rdi+4],xmm1", "c5fa114f04", func(a *Buf) { a.VMOVSSStore(At(RDI, 4), Y1) }},

	// Scalar, enough to drive a loop.
	{"mov rax,[rdi+0x8]", "488b4708", func(a *Buf) { a.MOVLoad(RAX, At(RDI, 8)) }},
	{"mov r15,[r13+0]", "4d8b7d00", func(a *Buf) { a.MOVLoad(R15, At(R13, 0)) }},
	{"mov [rsp+0x10],rbx", "48895c2410", func(a *Buf) { a.MOVStore(At(RSP, 0x10), RBX) }},
	{"mov rcx,0x1234", "48c7c134120000", func(a *Buf) { a.MOVimm32(RCX, 0x1234) }},
	{"add rdi,0x40", "4883c740", func(a *Buf) { a.ADDimm(RDI, 0x40) }},
	{"add rdi,0x4000", "4881c700400000", func(a *Buf) { a.ADDimm(RDI, 0x4000) }},
	{"sub rsi,0x8", "4883ee08", func(a *Buf) { a.SUBimm(RSI, 8) }},
	{"mov r11,rsi", "4989f3", func(a *Buf) { a.MOVQ(R11, RSI) }},
	{"mov rdi,r8", "4c89c7", func(a *Buf) { a.MOVQ(RDI, R8) }},
	{"mov r10,r9", "4d89ca", func(a *Buf) { a.MOVQ(R10, R9) }},
	{"mov rax,rbx", "4889d8", func(a *Buf) { a.MOVQ(RAX, RBX) }},
	{"add rdx,r8", "4c01c2", func(a *Buf) { a.ADDQ(RDX, R8) }},
	{"add r10,rdx", "4901d2", func(a *Buf) { a.ADDQ(R10, RDX) }},
	{"sub rbx,rdx", "4829d3", func(a *Buf) { a.SUBQ(RBX, RDX) }},
	{"sub r9,r11", "4d29d9", func(a *Buf) { a.SUBQ(R9, R11) }},
	{"dec ecx", "ffc9", func(a *Buf) { a.DEC(RCX) }},
	{"ret", "c3", func(a *Buf) { a.RET() }},
}

// normEncodings also came from `as --64` + objdump.
var normEncodings = []struct {
	name string
	emit func(*Buf)
	want string
}{
	{"vsqrtps ymm1,ymm3", func(a *Buf) { a.VSQRTPS(Y1, Y3) }, "c5fc51cb"},
	{"vdivps ymm1,ymm3,ymm4", func(a *Buf) { a.VDIVPS(Y1, Y3, Y4) }, "c5e45ecc"},
	{"vmovaps ymm1,ymm3", func(a *Buf) { a.VMOVAPSReg(Y1, Y3) }, "c5fc28cb"},
	{"vmaxps ymm1,ymm3,ymm4", func(a *Buf) { a.VMAXPS(Y1, Y3, Y4) }, "c5e45fcc"},
	{"vminps ymm1,ymm3,ymm4", func(a *Buf) { a.VMINPS(Y1, Y3, Y4) }, "c5e45dcc"},
	{"vcvtps2dq ymm1,ymm3", func(a *Buf) { a.VCVTPS2DQ(Y1, Y3) }, "c5fd5bcb"},
	{"vfmadd213ps ymm1,ymm3,ymm4", func(a *Buf) { a.VFMADD213PS(Y1, Y3, Y4) }, "c4e265a8cc"},
}

func TestNormEncodings(t *testing.T) {
	for _, tc := range normEncodings {
		var a Buf
		tc.emit(&a)
		if got := hex.EncodeToString(a.Bytes()); got != tc.want {
			t.Errorf("%s\n got  %s\n want %s", tc.name, got, tc.want)
		}
	}
}

func TestEncodings(t *testing.T) {
	for _, tc := range encodings {
		var a Buf
		tc.emit(&a)
		if got := hex.EncodeToString(a.Bytes()); got != tc.want {
			t.Errorf("%s\n got  %s\n want %s", tc.name, got, tc.want)
		}
	}
	t.Logf("%d encodings byte-exact against the system assembler", len(encodings))
}

// TestNoEVEX: a vpdpbusd starting with 0x62 is AVX-512-VNNI, which SIGILLs in
// generated code with no Go stack trace on a CPU without AVX-512.
func TestNoEVEX(t *testing.T) {
	var a Buf
	a.VPDPBUSD(Y1, Y2, Y3)
	a.VPDPBUSD(Y15, Y14, Y13)
	a.VPDPBUSDMem(Y9, Y2, Idx(R12, RSI, 4, 0x100))
	b := a.Bytes()
	if b[0] == 0x62 {
		t.Fatalf("emitted EVEX (0x62); this CPU has AVX-VNNI without AVX-512 and will SIGILL")
	}
	for i, v := range b {
		if v == 0x62 && (i == 0 || b[i-1] != 0x00) {
			t.Logf("note: 0x62 at offset %d (may be a displacement byte, not a prefix)", i)
		}
	}
	if b[0] != 0xC4 {
		t.Errorf("vpdpbusd starts with %#02x, want 0xc4 (VEX3)", b[0])
	}
}

// TestObjdumpRoundTrip disassembles what we emitted and checks it says what we
// meant, catching a valid-but-wrong encoding.
func TestObjdumpRoundTrip(t *testing.T) {
	// It must be GNU objdump that can target x86-64: -b binary / -m
	// i386:x86-64 are GNU options LLVM's objdump (macOS's `objdump`) lacks,
	// and a native arm64 binutils has no x86 target, so the cross one comes
	// first.
	//
	// The target is probed by disassembling a real byte with it: `objdump -i`
	// lists binutils 2.42's x86 architectures as plain "i386", though
	// -m i386:x86-64 works there, so a search of that list skipped this gate
	// on every host.
	probe := filepath.Join(t.TempDir(), "nop.bin")
	if err := os.WriteFile(probe, []byte{0x90}, 0o644); err != nil {
		t.Fatal(err)
	}
	objdump := ""
	for _, name := range []string{"x86_64-linux-gnu-objdump", "objdump"} {
		v, err := exec.Command(name, "--version").Output()
		if err != nil || !strings.Contains(string(v), "GNU objdump") {
			continue
		}
		if d, err := exec.Command(name, "-D", "-b", "binary", "-m", "i386:x86-64", probe).Output(); err == nil &&
			strings.Contains(string(d), "nop") {
			objdump = name
			break
		}
	}
	if objdump == "" {
		t.Skip("no GNU objdump with an x86-64 target (install binutils, or binutils-x86-64-linux-gnu on arm64)")
	}
	var a Buf
	for _, tc := range encodings {
		tc.emit(&a)
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "code.bin")
	if err := os.WriteFile(bin, a.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", bin).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump: %v\n%s", err, out)
	}
	text := string(out)
	if strings.Contains(text, "(bad)") {
		t.Errorf("objdump found invalid instructions:\n%s", text)
	}
	// Every mnemonic we emitted must appear in the disassembly.
	for _, want := range []string{
		"vpdpbusd", "vpmaddubsw", "vpmaddwd", "vpaddd", "vpsubb", "vpand", "vpxor",
		"vfmadd231ps", "vpsrlw", "vmovdqu", "vpbroadcastd", "vbroadcastss",
		"vcvtdq2ps", "vcvtph2ps", "vextracti128", "vzeroupper", "vphaddd",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("objdump never saw %q in the emitted stream", want)
		}
	}
	t.Logf("objdump round-trip clean over %d instructions", len(encodings))
}

// TestLabels: a backward loop branch must land exactly on its target, and a
// dangling label must be impossible to get bytes out of.
func TestLabels(t *testing.T) {
	var a Buf
	top := a.Label()
	a.Bind(top)
	a.VPDPBUSD(Y1, Y2, Y3) // 5 bytes
	a.DEC(RCX)             // 2 bytes
	a.JNZ(top)             // short form: 2 bytes, disp -9
	a.RET()
	b := a.Bytes()
	if got := hex.EncodeToString(b); got != "c4e26d50cbffc975f7c3" {
		t.Errorf("loop encoded as %s", got)
	}
	// disp is -9: from the byte after the jump (offset 9) back to 0.
	if int8(b[8]) != -9 {
		t.Errorf("backward jump displacement is %d, want -9", int8(b[8]))
	}

	var d Buf
	d.JNZ(d.Label()) // never bound
	func() {
		defer func() {
			if recover() == nil {
				t.Error("Bytes() returned code with an unbound label")
			}
		}()
		d.Bytes()
	}()
}

// TestForwardJump exercises the fixup path a loop prologue needs.
func TestForwardJump(t *testing.T) {
	var a Buf
	done := a.Label()
	a.JNZ(done)
	a.VZEROUPPER()
	a.Bind(done)
	a.RET()
	b := a.Bytes()
	// near jcc is 6 bytes; it must skip the 3-byte vzeroupper.
	if len(b) != 6+3+1 {
		t.Fatalf("got %d bytes, want 10", len(b))
	}
	if d := int32(uint32(b[2]) | uint32(b[3])<<8 | uint32(b[4])<<16 | uint32(b[5])<<24); d != 3 {
		t.Errorf("forward jump displacement is %d, want 3", d)
	}
}
