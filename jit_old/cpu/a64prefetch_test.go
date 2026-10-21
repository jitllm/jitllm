package cpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestA64PackedMatVecPrefetchesOneTileAhead pins the PRFM EmitA64PackedMatVec
// issues beside each payload line: one per 128-byte line, rows*4 bytes (one
// tile) past it. The kernel's answer does not depend on it -- a hint never
// changes a result -- so no numeric gate can see it go missing, and without it
// CPU decode slows sharply (a row count that is not a multiple of 2048
// defeats the hardware's own prefetch).
// A64Prefetch=-1 is the violation and must read zero hints.
func TestA64PackedMatVecPrefetchesOneTileAhead(t *testing.T) {
	count := func(t *testing.T, pf int) (n int, offs map[uint32]bool) {
		code, err := EmitA64PackedMatVecPF(quant.Q4_K, PackedRows, pf)
		if err != nil {
			t.Fatal(err)
		}
		offs = map[uint32]bool{}
		for i := 0; i+4 <= len(code); i += 4 {
			w := uint32(code[i]) | uint32(code[i+1])<<8 | uint32(code[i+2])<<16 | uint32(code[i+3])<<24
			if w&0xFFC0001F == 0xF9800000 { // prfm pldl1keep, [xn, #imm]
				n++
				offs[(w>>10&0xFFF)*8] = true
			}
		}
		return n, offs
	}
	n, offs := count(t, 0)
	// Q4_K: 8 sub-blocks x 4 payload words, each 64 rows = 256 bytes = two lines.
	if want := 8 * 4 * 2; n != want {
		t.Fatalf("%d PRFMs in the Q4_K kernel, want %d (one per payload line)", n, want)
	}
	for _, o := range []uint32{PackedRows * 4, PackedRows*4 + 128} {
		if !offs[o] {
			t.Fatalf("no PRFM at +%d; saw offsets %v -- the hint is one tile (rows*4 bytes) ahead", o, offs)
		}
	}
	if n, _ := count(t, -1); n != 0 {
		t.Fatalf("A64Prefetch=-1 emitted %d PRFMs, want none", n)
	}
}
