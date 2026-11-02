package cpu

import (
	"encoding/binary"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// The host's register report. The emitters allocate registers by hand, so
// there is no compiler to ask: what a kernel asked of the register file and
// the stack is counted as it is emitted, and recorded per emitter when its
// bytes are taken (Buf.Bytes, A64.Bytes).

// regUse is one amd64 kernel's demand: the highest vector register it named
// and how many memory operands it addressed off RSP -- the red-zone staging a
// kernel does when its live values outgrow the register file, i.e. its spills.
type regUse struct {
	vecHigh int // highest vector register index named, -1 for none
	stack   int
	vex     bool
	touched bool
}

func (r *regUse) vec(rs ...Reg) {
	if !r.touched {
		r.vecHigh, r.touched = -1, true
	}
	for _, x := range rs {
		if int(x) > r.vecHigh {
			r.vecHigh = int(x)
		}
	}
}

// KernelRegs is one emitter's register facts on this host.
type KernelRegs struct {
	// Emitter is the function that took the kernel's bytes.
	Emitter string `json:"emitter"`
	// ISA is "avx2", "sse" or "neon"; File is that ISA's vector register
	// file: 16 YMM, 16 XMM, 32 V.
	ISA  string `json:"isa"`
	File int    `json:"file"`
	// VecUsed is how many of File the kernel names (highest index + 1), the
	// worst over every shape it was emitted for; -1 where it is not counted
	// (arm64, whose emitters write instruction words directly).
	VecUsed int `json:"vec_used"`
	// StackRefs is memory operands addressed off the stack pointer, worst
	// shape: register spills and red-zone staging. Zero is the bar for a hot
	// kernel.
	StackRefs int `json:"stack_refs"`
	// Bytes is the largest code size emitted, against a 32 KB L1i.
	Bytes int `json:"bytes"`
	// Emits counts the kernels this emitter produced.
	Emits int `json:"emits"`
}

var regLedger struct {
	mu sync.Mutex
	by map[string]*KernelRegs
}

func emitterName(skip int) string {
	pc, _, _, ok := runtime.Caller(skip)
	if !ok {
		return "?"
	}
	f := runtime.FuncForPC(pc)
	if f == nil {
		return "?"
	}
	n := f.Name()
	if i := strings.LastIndexByte(n, '/'); i >= 0 {
		n = n[i+1:]
	}
	return n
}

func noteKernel(name, isa string, file, vecUsed, stack, size int) {
	regLedger.mu.Lock()
	defer regLedger.mu.Unlock()
	if regLedger.by == nil {
		regLedger.by = map[string]*KernelRegs{}
	}
	k := regLedger.by[name]
	if k == nil {
		k = &KernelRegs{Emitter: name, ISA: isa, File: file, VecUsed: vecUsed}
		regLedger.by[name] = k
	}
	k.Emits++
	k.VecUsed = max(k.VecUsed, vecUsed)
	k.StackRefs = max(k.StackRefs, stack)
	k.Bytes = max(k.Bytes, size)
}

// noteEmit records an amd64 kernel; it is called from Buf.Bytes, so the
// emitter is two frames up.
func noteEmit(r regUse, size int, vex bool) {
	isa := "sse"
	if vex {
		isa = "avx2"
	}
	used := 0
	if r.touched {
		used = r.vecHigh + 1
	}
	noteKernel(emitterName(3), isa, 16, used, r.stack, size)
}

// noteEmitA64 records an arm64 kernel from its instruction words: a load or
// store (op0 = x1x0) whose base register Rn is 31 addresses off SP.
func noteEmitA64(code []byte) {
	stack := 0
	for i := 0; i+4 <= len(code); i += 4 {
		w := binary.LittleEndian.Uint32(code[i:])
		literal := (w>>27)&7 == 3 && (w>>24)&1 == 0 // LDR (literal): no Rn
		if (w>>25)&5 == 4 && !literal && (w>>5)&31 == 31 {
			stack++
		}
	}
	noteKernel(emitterName(3), "neon", 32, -1, stack, len(code))
}

// KernelRegisters is every host kernel emitter that has produced code in this
// process, with what it asked of the register file and the stack, sorted by
// name.
func KernelRegisters() []KernelRegs {
	regLedger.mu.Lock()
	defer regLedger.mu.Unlock()
	out := make([]KernelRegs, 0, len(regLedger.by))
	for _, k := range regLedger.by {
		out = append(out, *k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Emitter < out[j].Emitter })
	return out
}
