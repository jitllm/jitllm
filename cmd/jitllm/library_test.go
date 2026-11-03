package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/library"
)

// A library name must become the model's pinned references; anything else
// must pass through untouched.
func TestLibraryArgsExpandsANameAndNothingElse(t *testing.T) {
	m := library.Models()[0]
	got := libraryArgs([]string{m.Name, "out.jlm"})
	if len(got) < 2 || got[0] != m.Ref().String() || got[len(got)-1] != "out.jlm" {
		t.Fatalf("libraryArgs(%q) = %q", m.Name, got)
	}
	for _, v := range library.Models() {
		if !v.Vision() {
			continue
		}
		got := libraryArgs([]string{v.Name})
		if tr, _ := v.TowerRef(); len(got) != 2 || got[1] != tr.String() {
			t.Errorf("a vision model's tower was not passed: %q", got)
		}
		break
	}
	// A file that happens to have a library name stays a file.
	dir := t.TempDir()
	local := filepath.Join(dir, m.Name)
	os.WriteFile(local, []byte("x"), 0o644)
	wd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(wd)
	if got := libraryArgs([]string{m.Name}); got[0] != m.Name {
		t.Errorf("an existing file named %q was taken for the library entry: %q", m.Name, got)
	}
	for _, keep := range [][]string{{"hf://o/r/x.gguf"}, {"./x.gguf"}, {"no-such-model"}} {
		if got := libraryArgs(keep); strings.Join(got, " ") != strings.Join(keep, " ") {
			t.Errorf("libraryArgs(%q) changed it to %q", keep, got)
		}
	}
}

// A conversion that fails must leave the container it was replacing.
//
// convertCmd must not remove dst on error: jlm.Write never writes dst until it
// is complete, so removing it would delete the working model on a failed
// re-convert.
func TestAFailedConvertKeepsTheExistingContainer(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "broken.gguf")
	dst := filepath.Join(dir, "broken.jlm")
	os.WriteFile(src, []byte("not a gguf"), 0o644)
	os.WriteFile(dst, []byte("the user's container"), 0o644)
	if err := convertCmd([]string{src, dst}); err == nil {
		t.Fatal("setup: a garbage GGUF converted")
	}
	if b, err := os.ReadFile(dst); err != nil || string(b) != "the user's container" {
		t.Fatalf("the existing container was lost: %v %q", err, b)
	}
}
