package cpu

import (
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// The arm64 kernels cannot run on an amd64 box, so these gates check what can
// be checked without arm64 hardware: that they generate, that they contain the
// instructions they claim to, and that they never write a register the Go
// runtime owns. The arithmetic gate lives in exec_a64_test.go.

func TestA64EmitRejects(t *testing.T) {
	if _, err := EmitA64(Spec{W: quant.Q2_K, Rows: 1, Cols: 1}); err == nil {
		t.Error("EmitA64 accepted a quant type with no arm64 kernel")
	}
	if _, err := EmitA64(Spec{W: quant.Q4_0, Rows: 4, Cols: 1}); err == nil {
		t.Error("EmitA64 accepted an interleave width it does not implement")
	}
	if _, err := EmitA64(Spec{W: quant.Q4_0, Rows: 1, Cols: 4}); err == nil {
		t.Error("EmitA64 accepted a batched shape; only decode matvec exists")
	}
	for _, wt := range a64KernelTypes {
		code, err := EmitA64(Spec{W: wt, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatalf("EmitA64(%s): %v", wt, err)
		}
		if len(code) == 0 || len(code)%4 != 0 {
			t.Errorf("%s kernel is %d bytes; A64 code is always a multiple of 4", wt, len(code))
		}
		t.Logf("%s arm64 matvec kernel: %d bytes, %d instructions", wt, len(code), len(code)/4)
	}
}

// TestA64KernelRegisterDiscipline is the arm64 counterpart of "R14 is not
// offered on any terms".
//
// x28 is Go's goroutine pointer on arm64 (not R14 as on amd64); clobbering it
// fails far from its cause. x18 is reserved by macOS, x27 is the Go
// assembler's REGTMP, x29/x30/sp are the frame pointer, return address and
// stack.
//
// The check is textual: decoding Rd needs the instruction class, and these
// kernels never mention those registers in any position.
func TestA64KernelRegisterDiscipline(t *testing.T) {
	mc := lookAny("llvm-mc-18", "llvm-mc", "llvm-mc-17", "llvm-mc-19")
	if mc == "" {
		t.Skip("no llvm-mc to disassemble aarch64")
	}
	for _, wt := range a64KernelTypes {
		code, err := EmitA64(Spec{W: wt, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatal(err)
		}
		text := a64Disasm(t, mc, code)
		for _, reg := range []string{"x18", "x27", "x28", "x29", "x30", "sp", "xzr", "wzr"} {
			if regexp.MustCompile(`\b` + reg + `\b`).MatchString(text) {
				t.Errorf("%s kernel touches %s, which the runtime or the ABI owns:\n%s",
					wt, reg, text)
			}
		}
	}
}

// TestA64KernelShape pins the instruction sequence each kernel is claimed to
// be, in the disassembler's words.
//
// SDOT consumes 16 bytes and a quant block is 32 elements, so every block takes
// exactly two. There must be no EOR: SDOT is signed x signed, so the amd64
// sign-bias would double-bias every weight.
func TestA64KernelShape(t *testing.T) {
	mc := lookAny("llvm-mc-18", "llvm-mc", "llvm-mc-17", "llvm-mc-19")
	if mc == "" {
		t.Skip("no llvm-mc to disassemble aarch64")
	}
	// The golden shape is the SDOT kernel's; on a chip without FEAT_DotProd
	// the widened form is sdotemu_test.go's.
	old := forceDotEncoding
	forceDotEncoding = true
	defer func() { forceDotEncoding = old }()
	for _, tc := range []struct {
		wt      quant.Type
		want    map[string]int
		forbid  []string
		advance string
	}{
		{
			wt: quant.Q8_0,
			want: map[string]int{
				"sdot": 2, "ldur": 2, "ldr": 10, "movi": 2, "scvtf": 1,
				"fcvt": 1, "fmul": 1, "fmla": 1, "faddp": 2, "cbnz": 2,
				"ret": 1, "and": 0, "ushr": 0, "sub\tv": 0,
			},
			forbid:  []string{"eor", "xor", "usdot", "udot", "addv"},
			advance: "add\tx2, x2, #34",
		},
		{
			wt: quant.Q4_0,
			want: map[string]int{
				"sdot": 2, "ldur": 1, "ldr": 10, "movi": 4, "scvtf": 1,
				"fcvt": 1, "fmul": 1, "fmla": 1, "faddp": 2, "cbnz": 2,
				"ret": 1, "and": 1, "ushr": 1, "sub\tv": 2,
			},
			forbid:  []string{"eor", "xor", "usdot", "udot", "addv"},
			advance: "add\tx2, x2, #18",
		},
	} {
		t.Run(tc.wt.String(), func(t *testing.T) {
			code, err := EmitA64(Spec{W: tc.wt, Rows: 1, Accs: 1, Cols: 1})
			if err != nil {
				t.Fatal(err)
			}
			text := a64Disasm(t, mc, code)
			for m, n := range tc.want {
				if got := strings.Count(text, "\t"+m); got != n {
					t.Errorf("%s: %d x %q, want %d\n%s", tc.wt, got, m, n, text)
				}
			}
			for _, f := range tc.forbid {
				if strings.Contains(text, f) {
					t.Errorf("%s kernel contains %q; SDOT is signed x signed and needs "+
						"none of the amd64 bias machinery\n%s", tc.wt, f, text)
				}
			}
			// The weight cursor must advance by exactly one block per iteration.
			// An 18 where a 34 belongs reads every row misaligned and produces a
			// plausible number, not a crash.
			if !strings.Contains(text, tc.advance) {
				t.Errorf("%s kernel never advances the weight cursor by one block (%q)\n%s",
					tc.wt, tc.advance, text)
			}
		})
	}
}

// TestA64UnencodableStride records why the amd64 pack8 trick is not ported.
// pack8 bakes the row stride as a disp32 so eight rows come off one base
// pointer, a workaround for x86's register pressure; arm64 has registers to
// spare. It is also unencodable: [base + r*stride + 2] is 2 mod 16 for both
// formats, so LDR Q cannot address it at any row, and LDUR runs out of window
// at +/-256.
func TestA64UnencodableStride(t *testing.T) {
	const k, rows = 2048, Pack8
	panics := func(f func()) (did bool) {
		defer func() { did = recover() != nil }()
		f()
		return
	}
	for _, wt := range a64KernelTypes {
		stride := RowBytes(wt, k)
		off := int32((rows-1)*stride) + 2 // row 7's weight payload
		var a A64
		if !panics(func() { a.LDRq(V0, X2, off) }) {
			t.Errorf("%s: LDR Q accepted +%d; it must be a multiple of 16", wt, off)
		}
		if !panics(func() { a.LDURq(V0, X2, off) }) {
			t.Errorf("%s: LDUR accepted +%d; its window is only +/-256", wt, off)
		}
		// The f16 scale is a second, independent squeeze: LDR H's scaled
		// maximum is 8190, which Q4_0's row 7 (+8064) just fits under and
		// Q8_0's (+15232) does not.
		hOK := !panics(func() { a.LDRh(V1, X2, off-2) })
		t.Logf("%s k=%d: row %d weights at +%d (mod 16 = %d) — LDR Q no, LDUR no; "+
			"scale at +%d reachable by LDR H: %v",
			wt, k, rows-1, off, off%16, off-2, hOK)
		if wt == quant.Q8_0 && hOK {
			t.Errorf("Q8_0 row %d scale at +%d should exceed LDR H's 8190 maximum", rows-1, off-2)
		}
	}
}

func a64Disasm(t *testing.T, mc string, code []byte) string {
	t.Helper()
	var in strings.Builder
	for _, b := range code {
		fmt.Fprintf(&in, "0x%02x ", b)
	}
	cmd := exec.Command(mc, "--disassemble", "-triple=aarch64", "-mattr=+dotprod,+fullfp16")
	cmd.Stdin = strings.NewReader(in.String())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", mc, err, out)
	}
	return string(out)
}
