package tier

import (
	"testing"
)

// groupPairs must give every (token, slot) pair its own column inside its
// expert's run, pad each run to a whole unit, and tselOf must name each
// block's expert -- at the dp4a matvec's groupTok and at a tensor-core token
// block, and at groupTok over runs padded to the block (the dp4a fallback on a
// card whose runs are laid out for GemmVolta).
func TestGroupPairs(t *testing.T) {
	const rows, k, nExp = 5, 2, 4
	// ExpertRank's layout: k ids and a bin a row.
	sel := []uint32{3, 1, 9, 1, 0, 9, 3, 2, 9, 1, 3, 9, 0, 1, 9}
	for _, c := range []struct{ unit, block, used int }{
		// Experts 0,1,2,3 hold 2,4,1,3 pairs: one block each either way.
		{groupTok, groupTok, 4}, {32, 32, 4}, {32, groupTok, 32},
	} {
		np := (rows*k + nExp*(c.unit-1) + c.unit - 1) / c.unit * c.unit
		perm, colE, tsel, pos := make([]uint32, np), make([]uint32, np), make([]uint32, np/c.block), make([]uint32, rows*k)
		cols := groupPairs(sel, rows, k, nExp, c.unit, perm, colE, pos)
		tselOf(colE, c.block, tsel)
		seen := map[uint32]bool{}
		for r := 0; r < rows; r++ {
			for s := 0; s < k; s++ {
				j, e := pos[r*k+s], sel[r*(k+1)+s]
				if seen[j] || perm[j] != uint32(r) || colE[j] != e || tsel[j/uint32(c.block)] != e {
					t.Fatalf("unit %d block %d: pair (%d,%d) expert %d: column %d holds token %d expert %d, block expert %d",
						c.unit, c.block, r, s, e, j, perm[j], colE[j], tsel[j/uint32(c.block)])
				}
				seen[j] = true
			}
		}
		if cols%c.unit != 0 || cols/c.block != c.used {
			t.Fatalf("unit %d block %d: %d columns in use, %d blocks, want %d", c.unit, c.block, cols, cols/c.block, c.used)
		}
		for j := range perm {
			if !seen[uint32(j)] && perm[j] != rows {
				t.Fatalf("unit %d: padding column %d reads token %d, want the zero row %d", c.unit, j, perm[j], rows)
			}
		}
		// Every block in use lies inside ONE expert's run.
		for b := 0; b < cols/c.block; b++ {
			for j := b * c.block; j < (b+1)*c.block; j++ {
				if colE[j] != tsel[b] {
					t.Fatalf("unit %d block %d: block %d mixes experts %d and %d", c.unit, c.block, b, tsel[b], colE[j])
				}
			}
		}
	}
}

// groupPairs sorts the rows*k routed (token, slot) pairs by expert. sel holds
// k+1 ids a row (ExpertRank's layout; slot k is its bin). Each expert's run is
// padded to a whole unit of columns. It fills perm (the token each sorted
// column reads; rows for a padding column, which GatherRows turns into zeros),
// colE (each column's expert) and pos (each pair's column), and returns how
// many columns are in use -- a multiple of unit. Experts in ascending order,
// pairs in (token, slot) order. tselOf names each block's expert.
func groupPairs(sel []uint32, rows, k, nExp, unit int, perm, colE, pos []uint32) int {
	count := make([]int, nExp)
	for r := 0; r < rows; r++ {
		for s := 0; s < k; s++ {
			if e := sel[r*(k+1)+s]; int(e) < nExp {
				count[e]++
			}
		}
	}
	start := make([]int, nExp)
	c := 0
	for e, n := range count {
		start[e] = c
		c += (n + unit - 1) / unit * unit
	}
	for i := range perm {
		perm[i], colE[i] = uint32(rows), 0
	}
	for e, n := range count {
		for j := start[e]; j < start[e]+(n+unit-1)/unit*unit; j++ {
			colE[j] = uint32(e)
		}
	}
	next := start
	for r := 0; r < rows; r++ {
		for s := 0; s < k; s++ {
			e := sel[r*(k+1)+s]
			if int(e) >= nExp {
				// Names no expert: the spare column past the last (np).
				pos[r*k+s] = uint32(len(perm))
				continue
			}
			j := next[e]
			next[e]++
			perm[j], pos[r*k+s] = uint32(r), uint32(j)
		}
	}
	return c
}

// tselOf names the expert of each block of unit sorted columns. A unit that
// divides the padding unit groupPairs used reads each block inside one run.
func tselOf(colE []uint32, unit int, tsel []uint32) {
	for t := range tsel {
		tsel[t] = colE[t*unit]
	}
}

type grouping struct {
	perm, colE, pos, tsel, tselU, pair []uint32
	cols                               int
}
