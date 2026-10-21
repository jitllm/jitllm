package cpu

import (
	"bytes"
	"testing"
)

// TestSHLimmEncoding pins SHL's encoding against hand-decoded x86. It shares
// SHRimm's C1 form and differs only in the /digit (/4 against /5); a wrong
// digit is still a valid instruction, shifting the other way.
func TestSHLimmEncoding(t *testing.T) {
	for _, c := range []struct {
		name string
		fn   func(*Buf)
		want []byte
	}{
		{"shr rsi,1", func(a *Buf) { a.SHRimm(RSI, 1) }, []byte{0x48, 0xC1, 0xEE, 0x01}},
		{"shl rsi,1", func(a *Buf) { a.SHLimm(RSI, 1) }, []byte{0x48, 0xC1, 0xE6, 0x01}},
		{"shl r13,1", func(a *Buf) { a.SHLimm(R13, 1) }, []byte{0x49, 0xC1, 0xE5, 0x01}},
		{"shr r13,1", func(a *Buf) { a.SHRimm(R13, 1) }, []byte{0x49, 0xC1, 0xED, 0x01}},
		{"shl r15,1", func(a *Buf) { a.SHLimm(R15, 1) }, []byte{0x49, 0xC1, 0xE7, 0x01}},
	} {
		var a Buf
		c.fn(&a)
		got := a.Bytes()
		if len(got) != len(c.want) {
			t.Errorf("%s: %d bytes (%x), want %d", c.name, len(got), got, len(c.want))
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%s: %x, want %x", c.name, got, c.want)
				break
			}
		}
	}
}

// TestADDQmemEncoding pins add r64, [mem], a hand-added encoding with one
// caller, so nothing else would notice a wrong opcode byte.
//
//	48 03 87 a0 00 00 00    add rax, [rdi+0xa0]
//	4c 03 87 a0 00 00 00    add r8,  [rdi+0xa0]
func TestADDQmemEncoding(t *testing.T) {
	for _, tc := range []struct {
		dst  Reg
		want []byte
	}{
		{RAX, []byte{0x48, 0x03, 0x87, 0xa0, 0x00, 0x00, 0x00}},
		{R8, []byte{0x4c, 0x03, 0x87, 0xa0, 0x00, 0x00, 0x00}},
	} {
		var a Buf
		a.ADDQmem(tc.dst, At(RDI, 160))
		if got := a.Bytes(); !bytes.Equal(got, tc.want) {
			t.Errorf("ADDQmem(%v, [rdi+160]) = % x, want % x", tc.dst, got, tc.want)
		}
	}
}
