package ssegate

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer/x86"
)

// TestProblemsFindsEveryViolation holds the gate to kernels it must pass and
// kernels it must fail: R14 and X15 are a kernel's (callKernel's callers
// reload g and zero X15 after it), while a write to RBP or RSP, a push and a
// VEX encoding each fail.
func TestProblemsFindsEveryViolation(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed")
	}
	kernel := func(emit func(a *x86.Buf)) []byte {
		var a x86.Buf
		a.DeclareISA(x86.ISATierSSE)
		emit(&a)
		a.RET()
		b, err := a.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	legal := kernel(func(a *x86.Buf) {
		a.MOVQ(x86.R14, x86.RAX)
		a.ADDQ(x86.R14, x86.RDI)
		a.XORPS(x86.XMM15, x86.XMM15, x86.XMM15)
	})
	if p := Problems(t, "legal", legal); len(p) > 0 {
		t.Errorf("a kernel using R14 and X15 fails the gate: %v", p)
	}
	for _, c := range []struct {
		name, want string
		code       []byte
	}{
		{"rbp", "writes rbp", kernel(func(a *x86.Buf) { a.MOVQ(x86.RBP, x86.RAX) })},
		{"rsp", "writes rsp", kernel(func(a *x86.Buf) { a.ADDQ(x86.RSP, x86.RAX) })},
		{"push", "moves RSP", append(kernel(func(*x86.Buf) {}), 0x50, 0xC3)},
		{"vex", "VEX-encoded", append(kernel(func(*x86.Buf) {}), 0xC5, 0xF8, 0x57, 0xC0, 0xC3)},
	} {
		p := Problems(t, c.name, c.code)
		found := false
		for _, s := range p {
			found = found || strings.Contains(s, c.want)
		}
		if !found {
			t.Errorf("%s: the gate missed it (%q not among %v)", c.name, c.want, p)
		}
	}
}
