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

func className(c Class) string {
	switch c {
	case 0:
		return "p"
	case 1:
		return "rs"
	case 2:
		return "r"
	case 3:
		return "rd"
	case 4:
		return "f"
	}
	return "fd"
}

func classDecl(c Class) string {
	switch c {
	case 0:
		return ".pred"
	case 1:
		return ".b16"
	case 2:
		return ".b32"
	case 3:
		return ".b64"
	case 4:
		return ".f32"
	}
	return ".f64"
}

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
	return fmt.Sprintf("%%%s%d", className(r.class), r.n)
}

// Buf accumulates PTX text and the register counters it implies.
type Buf struct {
	body strings.Builder
	// decls holds kernel-scope declarations that must precede the body: only
	// `.shared` arrays.
	decls strings.Builder
	// module holds module-scope declarations, before the entry: only the
	// extern array of dynamic shared memory.
	module strings.Builder
	next   [nClass]int
	labels int
	// gridBroken and depBroken are the options' violations: waits that wait
	// for nothing.
	gridBroken bool
	depBroken  bool
}

// Option changes how Lower emits.
type Option func(*Buf)

// BrokenGridBarrier makes every grid barrier count its arrival and leave
// without waiting: the violation the grid barrier's gate must fail under.
func BrokenGridBarrier() Option { return func(b *Buf) { b.gridBroken = true } }

// BrokenDepWait makes every dependency-counter wait return without waiting:
// the violation the dep block's gate must fail under.
func BrokenDepWait() Option { return func(b *Buf) { b.depBroken = true } }

// Module adds a module-scope declaration, emitted before the entry.
func (b *Buf) Module(format string, args ...any) {
	fmt.Fprintf(&b.module, format+"\n", args...)
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
func (b *Buf) Text(target, entry string, params []Param, group [3]int, resident int) string {
	var s strings.Builder
	fmt.Fprintf(&s, ".version 7.8\n.target %s\n.address_size 64\n", target)
	s.WriteString(b.module.String())
	fmt.Fprintf(&s, ".visible .entry %s(\n", entry)
	for i, p := range params {
		sep := ","
		if i == len(params)-1 {
			sep = ""
		}
		fmt.Fprintf(&s, "  .param .%s %s%s\n", p.Type, p.Name, sep)
	}
	s.WriteString(")\n")
	if resident > 0 {
		fmt.Fprintf(&s, ".maxntid %d, %d, %d\n.minnctapersm %d\n", max(group[0], 1), max(group[1], 1), max(group[2], 1), resident)
	}
	s.WriteString("{\n")
	for c := Class(0); c < nClass; c++ {
		if b.next[c] > 0 {
			fmt.Fprintf(&s, " .reg %s %%%s<%d>;\n", classDecl(c), className(c), b.next[c])
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
