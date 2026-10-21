// Package ptx is an assembler for PTX text: registers, labels and a buffer.
//
// It is the same shape as the CPU-side assembler: a Buf to append to, labels
// to bind and branch to, and no notion of what the code means. It adds two
// structural guarantees over hand-written text: registers cannot be numbered
// by hand (see Reg), and .reg declarations are derived from use (see Text).
package ptx

import (
	"fmt"
	"strings"
)

// Class is a PTX register class. PTX has unbounded virtual registers per class
// and ptxas does the real allocation, so the assembler's only job is to hand
// out distinct names and declare exactly as many as were used.
type Class int

const (
	Pred Class = iota
	B16
	B32
	B64
	F32
	F64
	nClass
)

var className = [nClass]string{"p", "rs", "r", "rd", "f", "fd"}
var classDecl = [nClass]string{".pred", ".b16", ".b32", ".b64", ".f32", ".f64"}

// Reg is a virtual register.
//
// There is no exported way to construct one with a chosen number: hand-picked
// register numbers with per-role bases collided as a tile grew and produced a
// fast, wrong kernel reading its own operands. A register you cannot name
// cannot alias.
type Reg struct {
	class Class
	n     int
	ok    bool
}

func (r Reg) String() string {
	if !r.ok {
		panic("ptx: use of a zero Reg -- registers come from Buf.Reg")
	}
	return fmt.Sprintf("%%%s%d", className[r.class], r.n)
}

// Buf accumulates PTX text and the register counters it implies.
type Buf struct {
	body strings.Builder
	// decls holds kernel-scope declarations that must precede the body: only
	// `.shared` arrays.
	decls  strings.Builder
	next   [nClass]int
	labels int
}

// Decl adds a kernel-scope declaration, emitted after the .reg block and before
// the body.
func (b *Buf) Decl(format string, args ...any) {
	fmt.Fprintf(&b.decls, " "+format+"\n", args...)
}

// Reg allocates a fresh virtual register of the given class.
func (b *Buf) Reg(c Class) Reg {
	r := Reg{class: c, n: b.next[c], ok: true}
	b.next[c]++
	return r
}

// Label returns a fresh, unique label name.
func (b *Buf) Label() string {
	b.labels++
	return fmt.Sprintf("L%d", b.labels)
}

// Op appends one instruction. Arguments are formatted with %v, so a Reg prints
// as its PTX name and an int as an immediate.
func (b *Buf) Op(format string, args ...any) {
	fmt.Fprintf(&b.body, " "+format+"\n", args...)
}

// Bind places a label.
func (b *Buf) Bind(l string) { fmt.Fprintf(&b.body, "%s:\n", l) }

// Text renders the complete module.
//
// The .reg declarations are written last, from the counters. Declared up front
// from an estimate, a too-small count made ptxas reject the module in a way
// that looked like a hardware limit.
func (b *Buf) Text(target, entry string, params []Param) string {
	var s strings.Builder
	fmt.Fprintf(&s, ".version 7.8\n.target %s\n.address_size 64\n", target)
	fmt.Fprintf(&s, ".visible .entry %s(\n", entry)
	for i, p := range params {
		sep := ","
		if i == len(params)-1 {
			sep = ""
		}
		fmt.Fprintf(&s, "  .param .%s %s%s\n", p.Type, p.Name, sep)
	}
	s.WriteString(")\n{\n")
	for c := Class(0); c < nClass; c++ {
		if b.next[c] > 0 {
			fmt.Fprintf(&s, " .reg %s %%%s<%d>;\n", classDecl[c], className[c], b.next[c])
		}
	}
	s.WriteString(b.decls.String())
	s.WriteString(b.body.String())
	s.WriteString("}\n")
	return s.String()
}

// Param is one kernel parameter.
type Param struct {
	Name string
	Type string // "u64" for a buffer pointer, "u32" for a scalar
}
