package cpu

import (
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoAssemblyBesidesTheTrampolines: this project contains no pregenerated
// kernel -- no assembly, and no source in a language a kernel is written in
// (PTX, CUDA, HIP, OpenCL, Metal, GLSL, HLSL, WGSL, SPIR-V, C); every kernel is
// emitted at run time for the device in front of it. The two trampolines are
// the Go-to-generated-code call boundary, which Go can only express in its own
// assembler. The C allowed is llama.cpp's harness: it measures or drives the
// reference, and no jitllm token runs it.
func TestNoAssemblyBesidesTheTrampolines(t *testing.T) {
	testmodels.SourceTree(t)
	allowed := map[string]bool{
		"engine/device/cpu/trampoline_amd64.s": true,
		"engine/device/cpu/trampoline_arm64.s": true,
		"scripts/gshim.c":                      true,
		"scripts/kshim.c":                      true,
		"scripts/lcpplogits/lcpplogits.c":      true,
		"scripts/ref/q3kbench.c":               true,
		"scripts/ref/q6kbench.c":               true,
	}
	kernel := map[string]bool{
		".s": true, ".asm": true, ".ptx": true, ".cu": true, ".cuh": true, ".hip": true,
		".cl": true, ".metal": true, ".glsl": true, ".comp": true, ".hlsl": true,
		".wgsl": true, ".spv": true, ".spvasm": true,
		".c": true, ".h": true, ".cc": true, ".cpp": true, ".hpp": true,
	}
	root := filepath.Join("..", "..", "..")
	walked, seen := 0, 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// A dot-directory (.git, a local tool's worktrees) is not this
			// tree's source.
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			switch d.Name() {
			case "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		walked++
		if !kernel[strings.ToLower(filepath.Ext(p))] {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		rel = filepath.ToSlash(rel)
		if !allowed[rel] {
			t.Errorf("%s: no pregenerated kernel besides the trampolines -- emit it at run time", rel)
		}
		seen++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if walked < 500 || seen < len(allowed) {
		t.Fatalf("walked %d files and saw %d kernel-language files from %s -- the walk is not seeing the module", walked, seen, root)
	}
}
