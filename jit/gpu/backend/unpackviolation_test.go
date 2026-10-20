package backend_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// Violations for TestUnpackMatchesHostPacker (RULE 10): each is a one-field
// edit to the generated IR of the kernel the gate uses, and the gate must
// report it. The edits are on the IR rather than on unpack.go because a source
// mutation cannot be committed, and a Load/Store Imm is the sub-block offset
// packRows and packSub compute.

// storeOps returns the indices of the ops that store into parameter p.
func storeOps(k *ir.Kernel, p int, kind ir.Kind) []int {
	var pv ir.Value
	for i, o := range k.Ops {
		if o.Kind == ir.OpParam && int(o.Imm) == p {
			pv = ir.Value(i + 1)
		}
	}
	var out []int
	for i, o := range k.Ops {
		if o.Kind == kind && o.Args[0] == pv {
			out = append(out, i)
		}
	}
	return out
}

// bend copies a kernel and adds delta to the Imm of the nth Load/Store on
// parameter p. It returns the copy and a description of what it broke.
func bend(k *ir.Kernel, kind ir.Kind, p, nth int, delta int64) (*ir.Kernel, string) {
	c := *k
	c.Ops = append([]ir.Op(nil), k.Ops...)
	ix := storeOps(&c, p, kind)
	if nth >= len(ix) {
		panic(fmt.Sprintf("param %d has %d %s ops, wanted #%d", p, len(ix), kind, nth))
	}
	i := ix[nth]
	was := c.Ops[i].Imm
	c.Ops[i].Imm += delta
	return &c, fmt.Sprintf("op %d (%s on param %d): element offset %d -> %d",
		i, kind, p, was, c.Ops[i].Imm)
}

// TestUnpackViolationsAreCaught bends the kernel five ways and requires the byte
// gate to reject each one.
func TestUnpackViolationsAreCaught(t *testing.T) {
	gpuLock(t)
	devs := importDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	cases := q4kCases(t)
	c := cases[0] // the smallest real tensor; a layout error does not need size
	rows, k := c.rows, c.k
	want := struct{ qs, d, sc []uint32 }{}
	var err error
	want.qs, want.d, want.sc, err = kernels.PackWeights(kernels.Q4_K, c.src, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	good, err := oldkernels.Unpack(kernels.Q4_K, rows, k)
	if err != nil {
		t.Fatal(err)
	}

	// Parameters are declared in the order (pSrc, pQS, pD, pSC).
	const (
		pSrc = iota
		pQS
		pD
		pSC
	)
	type viol struct {
		name string
		make func() (*ir.Kernel, string)
	}
	viols := []viol{
		{"payload sub-block offset off by one row stride", func() (*ir.Kernel, string) {
			// packSub writes sub-block si's word w at (si*words+w)*nrows+r.
			// One nrows is one word of one sub-block landing in the next.
			return bend(good, ir.OpStore, pQS, 5, int64(rows))
		}},
		{"payload sub-block offset off by a whole sub-block", func() (*ir.Kernel, string) {
			return bend(good, ir.OpStore, pQS, 0, int64(4*rows))
		}},
		{"payload read from the next source word", func() (*ir.Kernel, string) {
			// The read side of the same mistake: extract's grp advances 32 bytes
			// per 64-element group, and this takes one word too many.
			return bend(good, ir.OpLoad, pSrc, 7, 1)
		}},
		{"scale pair written to the next super-block", func() (*ir.Kernel, string) {
			return bend(good, ir.OpStore, pSC, 2, int64(rows))
		}},
		{"d taken from the wrong source word", func() (*ir.Kernel, string) {
			// This one bends the read: the single d store is at offset 0, and
			// moving it would write out of bounds (which crashes on llvmpipe)
			// rather than produce a wrong layout. Word 0 of a Q4_K block is
			// d|dmin and word 1 is the first four scale bytes.
			return bend(good, ir.OpLoad, pSrc, 0, 1)
		}},
	}
	for _, d := range devs {
		d := d
		t.Run(devTag(d), func(t *testing.T) {
			defer d.Close()
			t.Logf("device: %s", d.Name())
			for _, v := range viols {
				v := v
				t.Run(v.name, func(t *testing.T) {
					kk, what := v.make()
					if err := kk.Validate(); err != nil {
						t.Fatalf("the bent kernel does not even validate (%v); the violation "+
							"must be one a real mistake could produce", err)
					}
					gq, gd, gs, over := runUnpack(t, d, kk, c, len(want.qs), len(want.d), len(want.sc))
					n := diffWords(want.qs, gq) + diffWords(want.d, gd) + diffWords(want.sc, gs) + over
					if n == 0 {
						t.Fatalf("BENT AND STILL EQUAL: %s -- %s produced byte-identical "+
							"output, so TestUnpackMatchesHostPacker is not checking this",
							v.name, what)
					}
					t.Logf("caught: %s -- %s; %d words differ, %d of them past an end", v.name, what, n, over)
				})
			}
		})
	}
}

func diffWords(want, got []uint32) int {
	n := 0
	for i := range want {
		if want[i] != got[i] {
			n++
		}
	}
	return n
}

// TestUnpackRefusesAnAliasedBuffer is RULE 13, read back from ir.Validate rather
// than asserted in a comment.
//
// The unpack is legal because it reads one buffer and writes three others:
// surplus threads clamp to the last item and recompute it from read-only input.
// Unpacking in place would apply the transform twice to the last item, and
// Validate sees it.
func TestUnpackRefusesAnAliasedBuffer(t *testing.T) {
	good, err := oldkernels.Unpack(kernels.Q4_K, 256, 2048)
	if err != nil {
		t.Fatal(err)
	}
	if err := good.Validate(); err != nil {
		t.Fatalf("the shipping kernel does not validate: %v", err)
	}

	// The violation: the same buffer as source and destination.
	b := ir.New("unpack_inplace", [3]int{128, 1, 1})
	pBuf := b.Param("pBuf", ir.U32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, 255))
	lo := b.And(ir.U32, b.Load(ir.U32, pBuf, i, 0), b.Const(ir.U32, 0x0F0F0F0F))
	b.Store(pBuf, i, lo, 0)
	err = b.Done().Validate()
	if err == nil {
		t.Fatal("ir.Validate accepted a kernel that reads and writes one buffer; RULE 13 is " +
			"then a comment rather than a check, and an in-place unpack would apply the " +
			"nibble mask twice to the last item under the thread clamp")
	}
	if !strings.Contains(err.Error(), "both read and written") {
		t.Fatalf("Validate refused for the wrong reason: %v", err)
	}
	t.Logf("ir.Validate refuses the in-place form: %v", err)

	// And the shipping kernel's buffers are what that check is reading: one
	// loaded, three stored, no overlap.
	var loaded, stored [4]bool
	pv := map[ir.Value]int{}
	for i, o := range good.Ops {
		if o.Kind == ir.OpParam {
			pv[ir.Value(i+1)] = int(o.Imm)
		}
		switch o.Kind {
		case ir.OpLoad:
			loaded[pv[o.Args[0]]] = true
		case ir.OpStore:
			stored[pv[o.Args[0]]] = true
		}
	}
	for p, name := range []string{"pSrc", "pQS", "pD", "pSC"} {
		want := p != 0
		if loaded[p] == want || stored[p] != want {
			t.Errorf("%s: loaded=%v stored=%v; pSrc must be read-only and the three "+
				"destinations write-only", name, loaded[p], stored[p])
		}
	}
}
