// Package objdump reads x86-64 machine code back through GNU objdump, which
// shares no code with any encoder here: the assemblers' tests and the SSE
// gate hold their bytes to it.
package objdump

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

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
