//go:build amd64 && linux

package cpu

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"
)

// TestArgsLayout pins the struct offsets the generated code addresses by.
// Inserting a field in the middle of Args silently repoints every load in every
// kernel, and the symptom is wrong numbers, not a crash.
func TestArgsLayout(t *testing.T) {
	var a Args
	for _, tc := range []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"Out", unsafe.Offsetof(a.Out), 0},
		{"Out2", unsafe.Offsetof(a.Out2), 120},
		{"Q2", unsafe.Offsetof(a.Q2), 128},
		{"AScale2", unsafe.Offsetof(a.AScale2), 136},
		{"W", unsafe.Offsetof(a.W), 8},
		{"A", unsafe.Offsetof(a.A), 16},
		{"AScale", unsafe.Offsetof(a.AScale), 24},
		{"Rows", unsafe.Offsetof(a.Rows), 32},
		{"K", unsafe.Offsetof(a.K), 40},
		{"RowStr", unsafe.Offsetof(a.RowStr), 48},
		{"Scr", unsafe.Offsetof(a.Scr), 56},
		{"AHalf", unsafe.Offsetof(a.AHalf), 64},
		{"Scratch", unsafe.Offsetof(a.Scratch), 72},
	} {
		if tc.got != tc.want {
			t.Errorf("Args.%s is at offset %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

// TestExecuteScalar is the smallest end-to-end proof: emit, map, call, observe.
// It reads a field out of Args and writes it through another, so it exercises
// argument passing, a load, a store and the return path.
func TestExecuteScalar(t *testing.T) {
	var a Buf
	// Scalar moves only: declared at the SSE2 floor, every x86-64 host runs it.
	a.DeclareISA(ISASSE2)
	a.MOVLoad(RAX, At(RDI, 32)) // args.Rows
	a.MOVLoad(RCX, At(RDI, 56)) // args.Scr
	a.MOVStore(At(RCX, 0), RAX)
	a.RET()

	code := mustMap(t, a.Bytes())
	defer code.Close()

	scr := make([]byte, 8)
	args := Args{Rows: 0x1234_5678_9abc, Scr: &scr[0]}
	code.Call(&args)

	if got := binary.LittleEndian.Uint64(scr); got != 0x1234_5678_9abc {
		t.Fatalf("kernel wrote %#x, want %#x", got, 0x1234_5678_9abc)
	}
}

// TestExecuteVPDPBUSD runs the instruction the whole project is built around, on
// real hardware, and checks its arithmetic against a scalar reference.
//
// VPDPBUSD treats the first source as unsigned bytes and the second as signed,
// accumulating four products into each 32-bit lane. That asymmetry decides the
// weight layout, so it is worth asserting rather than assuming.
func TestExecuteVPDPBUSD(t *testing.T) {
	// This one executes VPDPBUSD itself; the AVX2 sequence that replaces it
	// on a non-VNNI host is gated by TestExecuteVEXDotMatchesVPDPBUSD.
	requireVNNIHost(t)
	var a Buf
	a.MOVLoad(RSI, At(RDI, 16)) // args.A  — unsigned side
	a.MOVLoad(RDX, At(RDI, 8))  // args.W  — signed side
	a.MOVLoad(RCX, At(RDI, 56)) // args.Scr
	a.VPXOR(Y0, Y0, Y0)
	a.VMOVDQULoad(Y1, At(RSI, 0))
	a.VPDPBUSDMem(Y0, Y1, At(RDX, 0))
	a.VMOVDQUStore(At(RCX, 0), Y0)
	a.VZEROUPPER()
	a.RET()

	code, err := Map(a.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer code.Close()

	act := make([]int8, 32) // read as unsigned by the instruction
	w := make([]byte, 32)   // read as signed
	for i := range act {
		act[i] = int8(i * 7 % 251)
		w[i] = byte(int8(i*13%127 - 63))
	}
	scr := make([]byte, 32)
	args := Args{A: &act[0], W: &w[0], Scr: &scr[0]}
	code.Call(&args)

	for lane := 0; lane < 8; lane++ {
		var want int32
		for i := 0; i < 4; i++ {
			k := lane*4 + i
			want += int32(uint8(act[k])) * int32(int8(w[k]))
		}
		got := int32(binary.LittleEndian.Uint32(scr[lane*4:]))
		if got != want {
			t.Errorf("lane %d: kernel got %d, reference %d", lane, got, want)
		}
	}
}

// TestExecuteLoop exercises a backward branch, which every kernel inner loop is.
func TestExecuteLoop(t *testing.T) {
	// Executes VPDPBUSD (see TestExecuteVPDPBUSD).
	requireVNNIHost(t)
	var a Buf
	a.MOVLoad(RSI, At(RDI, 16)) // A
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(RCX, At(RDI, 56)) // Scr
	a.MOVLoad(RAX, At(RDI, 40)) // K = number of 32-byte blocks
	a.VPXOR(Y0, Y0, Y0)
	top := a.Label()
	a.Bind(top)
	a.VMOVDQULoad(Y1, At(RSI, 0))
	a.VPDPBUSDMem(Y0, Y1, At(RDX, 0))
	a.ADDimm(RSI, 32)
	a.ADDimm(RDX, 32)
	a.DEC(RAX)
	a.JNZ(top)
	a.VMOVDQUStore(At(RCX, 0), Y0)
	a.VZEROUPPER()
	a.RET()

	code, err := Map(a.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	defer code.Close()

	const blocks = 64
	act := make([]int8, blocks*32)
	w := make([]byte, blocks*32)
	for i := range act {
		act[i] = int8(i % 127)
		w[i] = byte(int8(i%97 - 48))
	}
	scr := make([]byte, 32)
	args := Args{A: &act[0], W: &w[0], K: blocks, Scr: &scr[0]}
	code.Call(&args)

	var want [8]int32
	for b := 0; b < blocks; b++ {
		for lane := 0; lane < 8; lane++ {
			for i := 0; i < 4; i++ {
				k := b*32 + lane*4 + i
				want[lane] += int32(uint8(act[k])) * int32(int8(w[k]))
			}
		}
	}
	for lane := 0; lane < 8; lane++ {
		got := int32(binary.LittleEndian.Uint32(scr[lane*4:]))
		if got != want[lane] {
			t.Errorf("lane %d: kernel got %d, reference %d", lane, got, want[lane])
		}
	}
}

// TestMapRejects: an empty program must not become an executable page.
func TestMapRejects(t *testing.T) {
	if _, err := Map(nil); err == nil {
		t.Error("Map accepted empty code")
	}
	var ret Buf
	ret.DeclareISA(ISASSE2) // a bare ret runs on every x86-64 host
	ret.RET()
	c, err := Map(ret.Bytes())
	if err != nil {
		t.Fatalf("Map: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Error("double Close should be a no-op")
	}
	defer func() {
		if recover() == nil {
			t.Error("Call on closed code should panic, not segfault")
		}
	}()
	c.Call(&Args{})
}

var _ = math.Float32bits
