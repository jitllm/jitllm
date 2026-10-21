package cpu

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestGEMMEmissionIsDeterministic: one (format, tile, window) must produce one
// byte sequence.
//
// byte sequence. Ranging over a Go map (e.g. for hoisted constants) would make
// every emission correct yet different, breaking everything that compares
// emitted code. Sixteen emissions: with two hoisted constants in a map, this
// fails with probability 1 - 2^-15.
func TestGEMMEmissionIsDeterministic(t *testing.T) {
	for _, q := range []quant.Type{quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K} {
		for _, tl := range [][2]int{{2, 2}, {4, 1}, {2, 1}, {1, 4}} {
			for _, win := range []int{Q8Block, 256} {
				first, err := emitGEMMA64K(q, tl[0], tl[1], 8, win)
				if err != nil {
					continue
				}
				for i := 0; i < 15; i++ {
					again, err := emitGEMMA64K(q, tl[0], tl[1], 8, win)
					if err != nil {
						t.Fatalf("%v %dx%d/%d: emission %d failed: %v", q, tl[0], tl[1], win, i, err)
					}
					if !bytes.Equal(first, again) {
						t.Fatalf("%v %dx%d/%d: emission %d differs from the first (%d vs %d bytes)",
							q, tl[0], tl[1], win, i, len(first), len(again))
					}
				}
			}
		}
	}
}

// TestGEMMUnpackKonstsAreHoisted: no unpack immediate may be rebuilt inside the
// sub-block loop at a tile that ships.
//
// The sub-block loop is fully unrolled, so an immediate materialised at its
// point of use appears once per sub-block per row, while one in the constant
// pool appears once. The bound is the pool plus emitA64KScales' three masks
// (once per row per super-block, a separate open hoist).
func TestGEMMUnpackKonstsAreHoisted(t *testing.T) {
	for _, q := range []quant.Type{quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K} {
		var zero byte
		switch q {
		case quant.Q6_K:
			zero = 32
		case quant.Q3_K:
			zero = 4
		}
		distinct := map[byte]bool{}
		for _, imm := range a64KUnpackKonsts(q, zero) {
			distinct[imm] = true
		}
		for _, tl := range [][2]int{{2, 2}, {4, 1}, {2, 1}} {
			b, err := emitGEMMA64K(q, tl[0], tl[1], 8, 256)
			if err != nil {
				continue
			}
			n := 0
			for i := 0; i+4 <= len(b); i += 4 {
				// MOVI Vd.16B, #imm8 -- imm8 is split across bits 18:16 and 9:5.
				if binary.LittleEndian.Uint32(b[i:])&^uint32(0x000703FF) == 0x4F00E400 {
					n++
				}
			}
			bound := len(distinct) + 3*tl[0]
			if n > bound {
				t.Errorf("%v %dx%d/256: %d MOVI16b against a bound of %d -- an unpack immediate is being rebuilt inside the sub-block loop",
					q, tl[0], tl[1], n, bound)
			}
			t.Logf("%v %dx%d/256: %d MOVI16b, bound %d", q, tl[0], tl[1], n, bound)
		}
	}
}
