package tier

import "testing"

// TestVisionAttentionAlwaysChunksOverBudget: a vision block whose score planes
// exceed the budget takes its attention in query chunks whatever its row
// count factors into -- the chunk a multiple of 16 and of both tiles, its
// planes inside the budget, and the scratch's rows whole chunks. Kimi-VL's
// capacity, 5108 rows (four times a prime), is the case that had no chunk: the
// earlier rule searched only halvings that divide the rows, found none and
// allocated the planes whole. That rule runs here as the violation.
func TestVisionAttentionAlwaysChunksOverBudget(t *testing.T) {
	old := func(rows, heads, sstride, qtile, atile int) int {
		planes := func(r int) int { return 2 * r * heads * sstride * 4 }
		if planes(rows) <= visionPlaneBudget {
			return rows
		}
		for r := rows / 2; r >= 16; r /= 2 {
			if rows%r == 0 && r%16 == 0 && r%qtile == 0 && r%atile == 0 && planes(r) <= visionPlaneBudget {
				return r
			}
		}
		return rows
	}
	check := func(arows, capRows, rows, heads, sstride, qtile, atile int) string {
		per := 2 * heads * sstride * 4
		switch {
		case arows >= capRows:
			return "the attention is not chunked"
		case arows*per > visionPlaneBudget:
			return "a chunk's planes exceed the budget"
		case arows%16 != 0 || arows%qtile != 0 || arows%atile != 0:
			return "a chunk does not keep the tiles whole"
		case capRows%arows != 0 || capRows < rows:
			return "the scratch is not whole chunks covering the rows"
		}
		return ""
	}
	for _, c := range []struct{ rows, heads, qtile, atile int }{
		{5108, 16, 64, 64}, // Kimi-VL's capacity at MoonViT's heads
		{5108, 4, 64, 64},  // and at the fixture's
		{4096, 16, 64, 64}, // a power of two: a divisor, no rounding
		{4997, 16, 16, 32}, // a prime
		{3600, 16, 64, 16},
	} {
		sstride := (c.rows + maxKeyTile + 3) &^ 3
		if c.rows*2*c.heads*sstride*4 <= visionPlaneBudget {
			t.Fatalf("%d rows x %d heads fit the budget: the case proves nothing", c.rows, c.heads)
		}
		arows, capRows := visionChunk(c.rows, c.heads, sstride, c.qtile, c.atile, visionPlaneBudget)
		if why := check(arows, capRows, c.rows, c.heads, sstride, c.qtile, c.atile); why != "" {
			t.Errorf("%d rows, %d heads: %s (chunk %d, scratch %d)", c.rows, c.heads, why, arows, capRows)
		}
		t.Logf("%d rows, %d heads: chunks of %d over %d rows (%d passes)", c.rows, c.heads, arows, capRows,
			capRows/arows)
	}
	// The violation: Kimi-VL's capacity under the earlier rule.
	sstride := (5108 + maxKeyTile + 3) &^ 3
	if ar := old(5108, 16, sstride, 64, 64); check(ar, 5108, 5108, 16, sstride, 64, 64) == "" {
		t.Fatalf("the earlier rule chunks 5108 rows (%d): the gate cannot see the whole-plane fallback", ar)
	}
	// And a tower that fits takes its rows in one pass, unrounded.
	if ar, cr := visionChunk(1024, 4, 1024+maxKeyTile, 64, 64, visionPlaneBudget); ar != 1024 || cr != 1024 {
		t.Fatalf("1024 rows that fit the budget took chunks of %d over %d", ar, cr)
	}
}
