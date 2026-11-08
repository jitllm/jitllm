// Package testbin hands a test one of the tree's commands, built: the binary
// JITLLM_TEST_BINDIR holds under the command's name where that is set (a test
// binary shipped to a box with no Go toolchain, the commands built beside
// it), else one go build of pkg in dir into the test's own directory.
package testbin

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// Build is command name, pkg built in dir.
func Build(t testing.TB, name, dir, pkg string) string {
	t.Helper()
	exe := name
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	if d := os.Getenv("JITLLM_TEST_BINDIR"); d != "" {
		bin := filepath.Join(d, exe)
		if _, err := os.Stat(bin); err != nil {
			t.Fatalf("JITLLM_TEST_BINDIR holds no %s: %v", name, err)
		}
		return bin
	}
	bin := filepath.Join(t.TempDir(), exe)
	cmd := exec.Command("go", "build", "-o", bin, pkg)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building %s: %v\n%s", name, err, out)
	}
	return bin
}
