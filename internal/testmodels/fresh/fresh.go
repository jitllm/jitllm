// Package fresh makes a test's container one this build reads.
//
// The engine reads nothing but containers and refuses one of another container
// version, so a container an earlier build left in the model directory fails
// every gate that names it until someone converts it by hand. The suite
// converts instead (AGENTS.md, Scope: "the test suite converts"): a container
// that does not open is reconverted from the GGUF beside it.
//
// It is its own package rather than part of testmodels because it imports the
// converter, whose own tests import testmodels. Tests only, as testmodels is
// (TestNoProductionCodeImportsTestModels covers both).
package fresh

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

var done sync.Map

type result struct {
	once sync.Once
	err  error
}

// Container makes the container at path one this build reads, once per
// process: a container that does not open and has X.gguf beside it is
// reconverted from that. A container that opens, a path that is not a
// container, an absent one and one with no single GGUF beside it (a merged
// vision container) are left alone, and the caller's load says what is wrong.
func Container(path string) error {
	if !strings.HasSuffix(path, jlm.Ext) {
		return nil
	}
	v, _ := done.LoadOrStore(path, &result{})
	r := v.(*result)
	r.once.Do(func() { r.err = reconvert(path) })
	return r.err
}

// Must is Container failing t.
func Must(t testing.TB, path string) {
	t.Helper()
	if err := Container(path); err != nil {
		t.Fatal(err)
	}
}

func reconvert(path string) error {
	c, openErr := jlm.Open(path)
	if openErr == nil {
		c.Close()
		return nil
	}
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	src := strings.TrimSuffix(path, jlm.Ext) + ".gguf"
	if _, err := os.Stat(src); err != nil {
		return nil
	}
	// A private name and a rename: another test binary must not open a
	// half-written container. The private name keeps the extension, as the
	// converter writes nothing else.
	tmp := filepath.Join(filepath.Dir(path),
		fmt.Sprintf(".%s.%d%s", strings.TrimSuffix(filepath.Base(path), jlm.Ext), os.Getpid(), jlm.Ext))
	if _, err := convert.FromGGUFs(src, "", tmp, jlm.Fingerprint{Host: "test"}); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("%s does not open (%v), and reconverting it from %s failed: %w",
			filepath.Base(path), openErr, filepath.Base(src), err)
	}
	return os.Rename(tmp, path)
}
