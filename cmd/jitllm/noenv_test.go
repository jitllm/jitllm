package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOnlyTheCommandReadsTheEnvironment is the gate behind AGENTS.md's "the
// libraries read no environment at all".
//
// It parses the tree and looks for calls to os.Getenv rather than grepping for
// JITLLM_ literals, since a name can reach the call through an argument or a
// table.
//
// What is allowed, and why, in the two exemptions below.
func TestOnlyTheCommandReadsTheEnvironment(t *testing.T) {
	testmodels.SourceTree(t)
	// Absolute: with root "../..", the root's Name() is "..", the dot-prefix
	// skip below would match it, and the walk would scan nothing.
	root, aerr := filepath.Abs("../..")
	if aerr != nil {
		t.Fatal(aerr)
	}
	// cmd/ is where the environment belongs: a knob in the process environment
	// cannot be set by an embedded caller, so the commands read the names and
	// pass options down. Tests may read their own selectors (JITLLM_MODEL,
	// JITLLM_SLOW) -- those choose whether a gate runs and reach no engine code.
	// dev/ holds main packages too; the rule is about a library reading the
	// environment behind a caller's back.
	allowDir := func(p string) bool {
		return strings.HasPrefix(p, "cmd/") ||
			strings.Contains(p, "/cmd/") ||
			strings.HasPrefix(p, "dev/") ||
			strings.HasPrefix(p, "scripts/")
	}
	// The one library file left, and it is not in a release build: fault
	// injection is compiled only under -tags jitllmfault, because an
	// environment variable that makes a shipped engine pretend its GPU broke is
	// a footgun with no upside. See the file's own argument.
	allowFile := map[string]bool{
		"jit_old/gpu/tier/fault.go": true,
		// The same file in main's tier, under the same tag; paritytest is
		// imported by tests alone.
		"paritytest/maintier/fault.go": true,
		// JITLLM_MODELS, read by the test suites' model finder. Only _test.go
		// files import it, so no engine build reads that name.
		"internal/testmodels/testmodels.go": true,
		// JITLLM_TEST_DEVICES, the test binaries' device selection. Only
		// _test.go files and internal/testlock import it
		// (internal/testdevices' guard).
		"internal/testdevices/testdevices.go": true,
	}

	var bad []string
	checked := 0
	fset := token.NewFileSet()
	err := filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return rerr
		}
		// The exemptions are written with '/', which Windows' Rel does not use.
		rel = filepath.ToSlash(rel)
		if fi.IsDir() {
			if p == root {
				return nil
			}
			// Dot-directories are not this repo's source: a local tool's
			// directory may hold whole worktrees at other commits, and scanning
			// one reports old code as a live breach.
			if n := fi.Name(); strings.HasPrefix(n, ".") ||
				n == "docs" || n == "testdata" || n == "models" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		if allowDir(rel) || allowFile[rel] {
			return nil
		}
		// Not swallowed: a build tag is a comment, so every file in the tree
		// parses whatever its GOOS/GOARCH. A file that does not parse is a file
		// this gate did not read, and an unread file is how it goes quiet.
		f, perr := parser.ParseFile(fset, p, nil, 0)
		if perr != nil {
			t.Errorf("%s: %v", rel, perr)
			return nil
		}
		checked++
		// A main package is a command wherever it sits -- the desktop app's
		// is ui/main.go -- and a command is where the environment belongs.
		if f.Name.Name == "main" {
			return nil
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkg, ok := sel.X.(*ast.Ident)
			if !ok || pkg.Name != "os" {
				return true
			}
			switch sel.Sel.Name {
			case "Getenv", "LookupEnv", "Environ", "ExpandEnv":
				bad = append(bad, rel+":"+
					fset.Position(call.Pos()).String()[len(fset.Position(call.Pos()).Filename)+1:]+
					" os."+sel.Sel.Name)
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	// A tree with no source in it (a test binary run where only testdata was
	// copied) walks nothing and finds no breach: that is not a pass.
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("%s is not the module root: %v", root, err)
	}
	if checked < 100 {
		t.Fatalf("read only %d library files under %s -- the walk is not seeing the module", checked, root)
	}
	if len(bad) > 0 {
		t.Fatalf("a library package reads the environment; make it a With...() option "+
			"and read the name in cmd/:\n  %s", strings.Join(bad, "\n  "))
	}
}
