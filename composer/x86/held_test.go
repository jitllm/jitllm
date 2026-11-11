package x86

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// TestReloadsAreElidedOnlyWhileHeld holds MOVLoad's elision to a register
// still holding the same load: across VEX instructions that name neither it
// nor write memory, and never across a store, an instruction naming the
// register, a load through it changed, a bound label or a general
// instruction. Each arm is an elided reload made into an emitted one.
func TestReloadsAreElidedOnlyWhileHeld(t *testing.T) {
	slot, other := At(RSP, -0x148), At(RSP, -0x150)
	emit := func(between func(*Buf)) (first, second []byte) {
		var a Buf
		a.MOVLoad(R15, slot)
		a.VBROADCASTSS(Y8, At(R15, 0xc0))
		a.VPAND(Y6, Y13, Y6)
		a.VPMADDUBSW(Y9, Y6, Y8)
		if between != nil {
			between(&a)
		}
		n := a.Len()
		a.MOVLoad(R15, slot)
		b, err := a.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		return b[:n], b[n:]
	}
	if _, again := emit(nil); len(again) != 0 {
		t.Fatalf("a reload of a register holding the load emitted % x", again)
	}
	for name, between := range map[string]func(*Buf){
		"a general store":         func(a *Buf) { a.MOVStore(slot, RAX) },
		"a vector store":          func(a *Buf) { a.VMOVDQUStore(At(RSP, -0x160), Y1) },
		"a VEX op naming it":      func(a *Buf) { a.VPAND(Y15, Y1, Y2) },
		"a shift naming it":       func(a *Buf) { a.VPSRLD(Y15, Y1, 4) },
		"a load into it":          func(a *Buf) { a.MOVLoad(R15, other) },
		"a bound label":           func(a *Buf) { a.Bind(a.Label()) },
		"a general instruction":   func(a *Buf) { a.TESTQ(RAX, RCX) },
		"its base loaded from it": func(a *Buf) { a.MOVLoad(RSP, other) },
	} {
		if _, again := emit(between); len(again) == 0 {
			t.Errorf("after %s the reload was elided", name)
		}
	}
	for name, c := range map[string]struct {
		held     Mem
		between  func(*Buf)
		keepHeld bool
	}{
		"a move into another register":     {slot, func(a *Buf) { a.MOVQ(RBX, R9) }, true},
		"index arithmetic":                 {slot, func(a *Buf) { a.ADDQ(R9, R12); a.MOVL(R9, R9); a.ADDimm(R9, 0xc00) }, true},
		"a move into it":                   {slot, func(a *Buf) { a.MOVQ(R15, R9) }, false},
		"a write to its base":              {slot, func(a *Buf) { a.ADDimm(RSP, 8) }, false},
		"a lea into it":                    {slot, func(a *Buf) { a.LEAQ(R15, At(R9, 4)) }, false},
		"a store to another slot":          {slot, func(a *Buf) { a.MOVStore(At(RSP, -0x50), R9) }, true},
		"a store to its slot":              {slot, func(a *Buf) { a.MOVStore(At(RSP, -0x14c), R9) }, false},
		"a store through a heap base":      {slot, func(a *Buf) { a.MOVStore(At(R8, 0), R9) }, true},
		"a store through an indexed stack": {slot, func(a *Buf) { a.MOVStore(Idx(RSP, R9, 1, 0), R9) }, false},
		"a heap load, a heap store":        {At(RDI, 0x88), func(a *Buf) { a.MOVStore(At(R8, 0), R9) }, false},
		"an argument word, a stack store":  {At(RDI, 0x88), func(a *Buf) { a.MOVStore(At(RSP, -0x50), R9) }, true},
		"an add into its slot":             {slot, func(a *Buf) { a.ADDMemImm8(slot, 0x20) }, false},
		"an add into another slot":         {slot, func(a *Buf) { a.ADDMemImm8(At(RSP, -0x158), 0x20) }, true},
	} {
		var a Buf
		a.MOVLoad(R15, c.held)
		c.between(&a)
		n := a.Len()
		a.MOVLoad(R15, c.held)
		if elided := a.Len() == n; elided != c.keepHeld {
			t.Errorf("after %s the reload elided %v, want %v", name, elided, c.keepHeld)
		}
	}
	var a Buf
	a.MOVLoad(R13, At(R15, 8))
	a.MOVLoad(R15, other)
	n := a.Len()
	a.MOVLoad(R13, At(R15, 8))
	if a.Len() == n {
		t.Error("a load through a register loaded since was elided")
	}
	var b Buf
	b.MOVLoad(R15, slot)
	b.MOVLoad(R13, other)
	n = b.Len()
	b.MOVLoad(R15, slot)
	if b.Len() != n {
		t.Error("a load into another register forgot what R15 held")
	}
	var c, d Buf
	c.MOVLoad(R15, slot)
	d.MOVLoad(R15, slot)
	d.MOVLoad(R15, slot)
	cb, _ := c.Bytes()
	db, _ := d.Bytes()
	if !bytes.Equal(cb, db) {
		t.Errorf("one load and an elided second encode % x and % x", cb, db)
	}
}

// TestNoStoreTakesTheLoadPath holds every caller of vopMem, which keeps
// the loads held, to a load or a computation: a method named as a store
// goes through vopStore.
func TestNoStoreTakesTheLoadPath(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	calls := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		af, err := parser.ParseFile(fs, f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range af.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				ce, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				if se, ok := ce.Fun.(*ast.SelectorExpr); ok && se.Sel.Name == "vopMem" {
					calls++
					if (strings.Contains(fd.Name.Name, "Store") || strings.Contains(fd.Name.Name, "Extract")) && fd.Name.Name != "vopStore" {
						t.Errorf("%s calls vopMem, which keeps the loads held: a store goes through vopStore", fd.Name.Name)
					}
				}
				return true
			})
		}
	}
	if calls < 10 {
		t.Fatalf("read %d calls of vopMem: the gate read nothing", calls)
	}
}
