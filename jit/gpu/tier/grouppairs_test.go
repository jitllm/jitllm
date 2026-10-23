package tier

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
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

// voltaGroupTiles must offer only tiles that divide the rows and span the
// padding unit, widest first: gpt-oss's 2880 rows lead with 96-row blocks,
// V2-Lite's 1408 and every power of two with 128, 2848 (89 x 32) has only 32,
// and a unit that is not a whole number of 32-column blocks has none.
func TestVoltaGroupTilesDivideTheShape(t *testing.T) {
	for _, c := range []struct {
		rows, unit int
		bms        []int
	}{{2880, 64, []int{96, 64, 32}}, {1408, 64, []int{128, 64, 32}}, {768, 32, []int{128, 96, 64, 32}},
		{96, 32, []int{96, 32}}, {2848, 32, []int{32}}, {2880, 48, nil}} {
		tls := voltaGroupTiles(0, c.rows, c.unit)
		var bms []int
		for _, tl := range tls {
			if tl.Toks() != c.unit || c.rows%tl.Rows() != 0 {
				t.Fatalf("rows %d unit %d: tile %+v is %dx%d", c.rows, c.unit, tl, tl.Rows(), tl.Toks())
			}
			bms = append(bms, tl.Rows())
		}
		if fmt.Sprint(bms) != fmt.Sprint(c.bms) {
			t.Fatalf("rows %d unit %d: row blocks %v, want %v", c.rows, c.unit, bms, c.bms)
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
			count[sel[r*(k+1)+s]]++
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

// TestDeviceGroupingMatchesTheReference holds kernels.ExpertGroup* -- the
// batched mixture's grouping, which moved from Go onto the device -- to
// groupPairs and tselOf above, bit for bit, on every real backend: random
// selections, every pair on ONE expert, most experts empty, k of 1, and the
// dp4a and tensor-core padding units.
func TestDeviceGroupingMatchesTheReference(t *testing.T) {
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	type tc struct {
		name                string
		rows, k, nExp, unit int
		pick                func(r, s int) uint32
	}
	rng := rand.New(rand.NewSource(7))
	cases := []tc{
		{"random", 37, 3, 16, groupTok, func(r, s int) uint32 { return uint32(rng.Intn(16)) }},
		{"random-volta", 61, 2, 64, 32, func(r, s int) uint32 { return uint32(rng.Intn(64)) }},
		{"one-expert", 20, 4, 8, groupTok, func(r, s int) uint32 { return 5 }},
		{"sparse", 33, 2, 128, 64, func(r, s int) uint32 { return uint32([]int{0, 127, 64}[(r+s)%3]) }},
		{"k1", 9, 1, 4, groupTok, func(r, s int) uint32 { return uint32(r % 3) }},
		// olmoe's chunk: 4096 pairs over 64 blocks of the two-level scatter.
		{"olmoe", 512, 8, 64, 32, func(r, s int) uint32 { return uint32(rng.Intn(64)) }},
	}
	ran := 0
	for _, d := range devs {
		for _, c := range cases {
			rows, k, nExp, unit := c.rows, c.k, c.nExp, c.unit
			np := (rows*k + nExp*(unit-1) + unit - 1) / unit * unit
			sel := make([]uint32, rows*(k+1))
			for r := 0; r < rows; r++ {
				// A slot's experts are distinct in a real selection; the
				// grouping does not rely on it, so neither does this.
				for s := 0; s < k; s++ {
					sel[r*(k+1)+s] = c.pick(r, s)
				}
				sel[r*(k+1)+k] = 999 // the bin: never read
			}
			perm, colE, pos := make([]uint32, np), make([]uint32, np), make([]uint32, rows*k)
			cols := groupPairs(sel, rows, k, nExp, unit, perm, colE, pos)
			tsel, tselU := make([]uint32, np/groupTok), make([]uint32, np/unit)
			tselOf(colE, groupTok, tsel)
			tselOf(colE, unit, tselU)
			pair := make([]uint32, np)
			for i, j := range pos {
				pair[j] = uint32(i)
			}
			got, err := runGrouping(d, sel, rows, k, nExp, unit, np)
			if err != nil {
				t.Fatalf("%s %s: %v", d.API(), c.name, err)
			}
			for _, cmp := range []struct {
				what      string
				got, want []uint32
			}{{"perm", got.perm, perm}, {"colE", got.colE, colE}, {"pos", got.pos, pos},
				{"tsel", got.tsel, tsel}, {"tselU", got.tselU, tselU}, {"colPair", got.pair, pair}} {
				for i := range cmp.want {
					if cmp.got[i] != cmp.want[i] {
						t.Fatalf("%s %s: %s[%d] = %d, want %d", d.API(), c.name, cmp.what, i, cmp.got[i], cmp.want[i])
					}
				}
			}
			if got.cols != cols {
				t.Fatalf("%s %s: %d used columns, want %d", d.API(), c.name, got.cols, cols)
			}
			ran++
		}
		d.Close()
	}
	t.Logf("%d device case(s) bit-exact", ran)
}

type grouping struct {
	perm, colE, pos, tsel, tselU, pair []uint32
	cols                               int
}

// runGrouping launches the five grouping kernels in the tier's order and reads
// every table back.
func runGrouping(d backend.Device, sel []uint32, rows, k, nExp, unit, np int) (grouping, error) {
	var out grouping
	var err error
	comp := func(mk func() (*ir.Kernel, error)) backend.Kernel {
		if err != nil {
			return nil
		}
		var kk *ir.Kernel
		if kk, err = mk(); err != nil {
			return nil
		}
		var c backend.Kernel
		c, err = d.Compile(kk)
		return c
	}
	bcount := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupBlockCount(rows, k, nExp) })
	count := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupCount(rows, k, nExp) })
	ends := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupEnds(nExp, unit) })
	cols := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupColumns(rows, nExp, np, true) })
	scatter := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupScatter(rows, k, nExp, unit, true) })
	tiles := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupTiles(nExp, np/groupTok, groupTok) })
	tilesU := comp(func() (*ir.Kernel, error) { return kernels.ExpertGroupTiles(nExp, np/unit, unit) })
	if err != nil {
		return out, err
	}
	buf := func(n int) backend.Buf {
		if err != nil {
			return nil
		}
		var b backend.Buf
		if b, err = d.Alloc(n * 4); err == nil {
			// Poisoned (RULE 13): a table entry nothing wrote reads 0xDEAD.
			poison := make([]uint32, n)
			for i := range poison {
				poison[i] = 0xDEAD
			}
			err = b.Write(u32b(poison))
		}
		return b
	}
	nbe := kernels.GroupBlocks(rows, k) * nExp
	bSel, bCnt, bEnds, bBC, bOff := buf(len(sel)), buf(2*nExp), buf(nExp), buf(nbe), buf(nbe)
	bPerm, bColE, bPos, bPair := buf(np), buf(np), buf(rows*k), buf(np)
	bTsel, bTselU := buf(np/groupTok), buf(np/unit)
	if err != nil {
		return out, err
	}
	if err = bSel.Write(u32b(sel)); err != nil {
		return out, err
	}
	d.Session(func(s backend.Session) {
		for _, l := range []struct {
			k    backend.Kernel
			n    int
			bufs []backend.Buf
		}{
			{bcount, nbe, []backend.Buf{bSel, bBC}},
			{count, nbe, []backend.Buf{bBC, bOff, bCnt}},
			{ends, nExp, []backend.Buf{bCnt, bEnds}},
			{cols, np, []backend.Buf{bEnds, bColE, bPerm, bPair}},
			{scatter, rows * k, []backend.Buf{bSel, bCnt, bEnds, bOff, bPerm, bPos, bPair}},
			{tiles, np / groupTok, []backend.Buf{bEnds, bTsel}},
			{tilesU, np / unit, []backend.Buf{bEnds, bTselU}},
		} {
			if err == nil {
				err = s.Launch(l.k, (l.n+127)/128, 128, l.bufs...)
			}
		}
		if err == nil {
			err = s.Sync()
		}
	})
	read := func(b backend.Buf, n int) []uint32 {
		v := make([]uint32, n)
		if err == nil {
			p := make([]byte, n*4)
			if err = b.Read(p); err == nil {
				for i := range v {
					v[i] = uint32(p[4*i]) | uint32(p[4*i+1])<<8 | uint32(p[4*i+2])<<16 | uint32(p[4*i+3])<<24
				}
			}
		}
		return v
	}
	out.perm, out.colE, out.pos, out.pair = read(bPerm, np), read(bColE, np), read(bPos, rows*k), read(bPair, np)
	out.tsel, out.tselU = read(bTsel, np/groupTok), read(bTselU, np/unit)
	if e := read(bEnds, nExp); err == nil {
		out.cols = int(e[nExp-1])
	}
	for _, b := range []backend.Buf{bSel, bCnt, bEnds, bBC, bOff, bPerm, bColE, bPos, bPair, bTsel, bTselU} {
		b.Free()
	}
	for _, c := range []backend.Kernel{bcount, count, ends, cols, scatter, tiles, tilesU} {
		c.Close()
	}
	return out, err
}
