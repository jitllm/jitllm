package kernels

import (
	"fmt"
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// Paged KV addressing, the ABI of docs/design/device-kv-paging.md.
//
// A paged kernel takes two data buffers after its existing parameters and
// before any optional ones (sinks): pTab, the page ids, and pRow, one
// descriptor of PRowWords u32 per query row. Position t of a row is page
// pTab[tabOff + t/P], and within a page of P positions of one layer:
//
//	K   [kvRow][P]  element e of position t at pid*P*kvRow + e*P + t%P
//	V   [P][kvRow]  at (pid*P + t%P)*kvRow + e, in words /2 when packed f16
//
// P is baked (a multiple of 64, so every key tile of 32 or 64 lies in one
// page); the table and the descriptors are data. A parameter the paged form
// supersedes (a count, an offset) stays in the list so the launch keeps its
// shape, and is not read.

// The words of a row descriptor, in order.
const (
	PRowTab   = iota // where the row's table starts in pTab
	PRowStart        // first key position attended (window or chunk start)
	PRowEnd          // one past the last key position attended
	PRowWrite        // the position this row's new K/V is written at
	PRowWords
)

// checkPage refuses a page size the tile geometry cannot use.
func checkPage(what string, p int) error {
	if p < 64 || p%64 != 0 {
		return fmt.Errorf("kernels: %s: page size %d must be a positive multiple of 64", what, p)
	}
	return nil
}

// pagedDesc reads word w of row r's descriptor.
func pagedDesc(b *ir.Builder, pRow, row ir.Value, w int) ir.Value {
	return b.Load(ir.U32, pRow, b.Mul(ir.U32, row, b.Const(ir.U32, PRowWords)), int64(w))
}

// pageDiv and pageRem are t/P and t%P, as a shift and a mask where P is a
// power of two: a division by a constant is a multiply-high and a correction
// on every backend, and the per-key paths do one per key.
func pageDiv(b *ir.Builder, t ir.Value, page int) ir.Value {
	if page&(page-1) == 0 {
		return b.Shr(ir.U32, t, b.Const(ir.U32, int64(bits.TrailingZeros(uint(page)))))
	}
	return b.Div(ir.U32, t, b.Const(ir.U32, int64(page)))
}

func pageRem(b *ir.Builder, t ir.Value, page int) ir.Value {
	if page&(page-1) == 0 {
		return b.And(ir.U32, t, b.Const(ir.U32, int64(page-1)))
	}
	return b.Rem(ir.U32, t, b.Const(ir.U32, int64(page)))
}

// pageOf is the page id holding position t of the row whose table starts at
// tabOff.
func pageOf(b *ir.Builder, pTab, tabOff, t ir.Value, page int) ir.Value {
	d := pageDiv(b, t, page)
	if pagedFault == "(t-1)/P" {
		one := b.Const(ir.U32, 1)
		d = pageDiv(b, b.Sub(ir.U32, b.Max(ir.U32, t, one), one), page)
	}
	return b.Load(ir.U32, pTab, b.Add(ir.U32, tabOff, d), 0)
}

// pagedWrite is where row `row`'s new position lands: its page id and its
// offset in that page.
func pagedWrite(b *ir.Builder, pTab, pRow, row ir.Value, page int) (pid, off ir.Value) {
	tab := pagedDesc(b, pRow, row, PRowTab)
	wp := pagedDesc(b, pRow, row, PRowWrite)
	return pageOf(b, pTab, tab, wp, page), pageRem(b, wp, page)
}

// PagedCopyRowsT is CopyAtRowsT into a paged K pool: element i of source row r
// lands at pid*page*n + i*page + writePos%page, pid and writePos from row r's
// descriptor. n is the whole kvRow (the K region is [kvRow][page]).
// Parameters: pSrc, pDst (the K pool), pOff (not read), pTab, pRow.
func PagedCopyRowsT(n, rows, page int) (*ir.Kernel, error) {
	if err := checkPage("PagedCopyRowsT", page); err != nil {
		return nil, err
	}
	if n < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: PagedCopyRowsT: n=%d rows=%d", n, rows)
	}
	b := ir.New("pagedcopyrowst", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	b.Param("pOff", ir.U32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n*rows-1)))
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(n)))
	row := b.Div(ir.U32, flat, b.Const(ir.U32, int64(n)))
	pid, off := pagedWrite(b, pTab, pRow, row, page)
	dst := b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page*n))),
		b.Add(ir.U32, b.Mul(ir.U32, i, b.Const(ir.U32, int64(page))), off))
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, flat, 0), 0)
	return b.Done()
}

// PagedCopyRows is CopyAtRows (f16: CopyAtF16) into a paged row-major pool:
// source row r's n elements land at columns col..col+n-1 of position
// writePos's row, which is width elements wide. V is width == n, col == 0; an
// MLA latent row writes its two parts at their columns. Packed f16 needs n,
// width and col even, so a position's row is whole words and no word crosses
// a page edge. Parameters: pSrc, pDst (the pool), pOff (not read), pTab, pRow.
func PagedCopyRows(n, rows, page, width, col int, f16 bool) (*ir.Kernel, error) {
	if err := checkPage("PagedCopyRows", page); err != nil {
		return nil, err
	}
	if n < 1 || rows < 1 || col < 0 || col+n > width || (f16 && (n%2 != 0 || width%2 != 0 || col%2 != 0)) {
		return nil, fmt.Errorf("kernels: PagedCopyRows: n=%d rows=%d width=%d col=%d f16=%v", n, rows, width, col, f16)
	}
	per := n
	if f16 {
		per = n / 2
	}
	b := ir.New("pagedcopyrows", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	b.Param("pOff", ir.U32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(per*rows-1)))
	pagedCopyBody(b, n, page, width, col, f16, pSrc, pDst, pTab, pRow, flat)
	return b.Done()
}

// pagedCopyBody is PagedCopyRows after its thread index (clamped by the
// caller): one element, or one packed f16 word, of a source row into its
// position's row of the pool.
func pagedCopyBody(b *ir.Builder, n, page, width, col int, f16 bool, pSrc, pDst, pTab, pRow, flat ir.Value) {
	per := n
	if f16 {
		per = n / 2
	}
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(per)))
	row := b.Div(ir.U32, flat, b.Const(ir.U32, int64(per)))
	pid, off := pagedWrite(b, pTab, pRow, row, page)
	// The destination row's first element, in elements.
	at := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page))), off),
		b.Const(ir.U32, int64(width))), b.Const(ir.U32, int64(col)))
	if !f16 {
		b.Store(pDst, b.Add(ir.U32, at, i), b.Load(ir.F32, pSrc, flat, 0), 0)
		return
	}
	src := b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(n))), b.Mul(ir.U32, i, b.Const(ir.U32, 2)))
	w := b.PackF16(b.Load(ir.F32, pSrc, src, 0), b.Load(ir.F32, pSrc, src, 1))
	b.Store(pDst, b.Add(ir.U32, b.Shr(ir.U32, at, b.Const(ir.U32, 1)), i), b.Bitcast(ir.F32, w), 0)
}

// PagedRoPERowsT is RoPERowsT into a paged K pool: the rotated pair lands at
// its elements of writePos in row r's page, every head of the kvRow
// nHeads*headDim. A partial rotary writes only the nRot dimensions it
// rotates, so PagedCopyRowsT runs first for the tail, as the tier does.
// Parameters: pSrc, pCS, pOff (not read), pDst (the K pool), pTab, pRow.
func PagedRoPERowsT(nHeads, headDim, nRot int, neox bool, rows, page int) (*ir.Kernel, error) {
	return pagedRoPERowsT(nHeads, headDim, nHeads*headDim, nRot, neox, rows, page)
}

// PagedRoPERowsTIn is PagedRoPERowsT turning the first nHeads heads of a
// wider row, rowW floats in the source and the pool alike (a QSA row, whose
// last head is the indexer's raw key and does not turn).
func PagedRoPERowsTIn(nHeads, headDim, rowW, nRot int, neox bool, rows, page int) (*ir.Kernel, error) {
	if rowW < nHeads*headDim {
		return nil, fmt.Errorf("kernels: PagedRoPERowsTIn: a row of %d under %d heads of %d", rowW, nHeads, headDim)
	}
	return pagedRoPERowsT(nHeads, headDim, rowW, nRot, neox, rows, page)
}

func pagedRoPERowsT(nHeads, headDim, rowW, nRot int, neox bool, rows, page int) (*ir.Kernel, error) {
	if err := checkPage("PagedRoPERowsT", page); err != nil {
		return nil, err
	}
	if nRot%2 != 0 || nRot <= 0 || nRot > headDim || nHeads < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: PagedRoPERowsT: nRot=%d headDim=%d heads=%d rows=%d", nRot, headDim, nHeads, rows)
	}
	pairs := nRot / 2
	b := ir.New("pagedroperowst", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pCS := b.Param("pCS", ir.F32)
	b.Param("pOff", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(nHeads*pairs*rows-1))
	pagedRopeBody(b, nHeads, headDim, rowW, nRot, neox, page, pSrc, pCS, pDst, pTab, pRow, flat)
	return b.Done()
}

// pagedRopeBody is PagedRoPERowsT after its thread index (clamped by the
// caller): one rotated pair of k into its position in the transposed pool.
func pagedRopeBody(b *ir.Builder, nHeads, headDim, rowW, nRot int, neox bool, page int,
	pSrc, pCS, pDst, pTab, pRow, flat ir.Value) {
	pairs := nRot / 2
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	t := b.Rem(ir.U32, flat, u(nHeads*pairs))
	row := b.Div(ir.U32, flat, u(nHeads*pairs))
	p := b.Rem(ir.U32, t, u(pairs))
	h := b.Div(ir.U32, t, u(pairs))
	head := b.Mul(ir.U32, h, u(headDim))
	var base ir.Value
	stride := 1
	if neox {
		base, stride = b.Add(ir.U32, head, p), pairs
	} else {
		base = b.Add(ir.U32, head, b.Mul(ir.U32, p, u(2)))
	}
	cs := b.Add(ir.U32, b.Mul(ir.U32, p, u(2)), b.Mul(ir.U32, row, u(nRot)))
	src := b.Add(ir.U32, base, b.Mul(ir.U32, row, u(rowW)))
	c := b.Load(ir.F32, pCS, cs, 0)
	s := b.Load(ir.F32, pCS, cs, 1)
	x0 := b.Load(ir.F32, pSrc, src, 0)
	x1 := b.Load(ir.F32, pSrc, src, int64(stride))
	pid, off := pagedWrite(b, pTab, pRow, row, page)
	out := b.Add(ir.U32, b.Mul(ir.U32, pid, u(page*rowW)),
		b.Add(ir.U32, b.Mul(ir.U32, base, u(page)), off))
	b.Store(pDst, out, b.Sub(ir.F32, b.Mul(ir.F32, x0, c), b.Mul(ir.F32, x1, s)), 0)
	b.Store(pDst, out, b.Add(ir.F32, b.Mul(ir.F32, x0, s), b.Mul(ir.F32, x1, c)), int64(stride*page))
}

// PagedCopyRowsTF16 is PagedCopyRowsT into a binary16 K pool
// (FlashShape.F16K): dimensions 2i and 2i+1 of source row r are one word, at
// pid*page*n/2 + i*page + writePos%page. A thread writes a whole word, so
// rows landing in one page never race on a half. n (the kvRow) is even.
// Parameters: pSrc, pDst (the K pool), pOff (not read), pTab, pRow.
func PagedCopyRowsTF16(n, rows, page int) (*ir.Kernel, error) {
	if err := checkPage("PagedCopyRowsTF16", page); err != nil {
		return nil, err
	}
	if n < 2 || n%2 != 0 || rows < 1 {
		return nil, fmt.Errorf("kernels: PagedCopyRowsTF16: n=%d rows=%d", n, rows)
	}
	per := n / 2
	b := ir.New("pagedcopyrowstf16", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	b.Param("pOff", ir.U32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), u(per*rows-1))
	i := b.Rem(ir.U32, flat, u(per))
	row := b.Div(ir.U32, flat, u(per))
	pid, off := pagedWrite(b, pTab, pRow, row, page)
	src := b.Add(ir.U32, b.Mul(ir.U32, row, u(n)), b.Mul(ir.U32, i, u(2)))
	w := b.PackF16(b.Load(ir.F32, pSrc, src, 0), b.Load(ir.F32, pSrc, src, 1))
	dst := b.Add(ir.U32, b.Mul(ir.U32, pid, u(page*per)), b.Add(ir.U32, b.Mul(ir.U32, i, u(page)), off))
	b.Store(pDst, dst, b.Bitcast(ir.F32, w), 0)
	return b.Done()
}

// PagedRoPERowsTF16 is PagedRoPERowsT into a binary16 K pool: a thread
// rotates the pairs that fill whole words -- one pair (2p, 2p+1) interleaved,
// two adjacent pairs NEOX, whose four dimensions are two words -- so no two
// threads write halves of one word. NEOX needs nRot a multiple of 4. The
// tail of a partial rotary is PagedCopyRowsTF16's, run first.
// Parameters: pSrc, pCS, pOff (not read), pDst (the K pool), pTab, pRow.
func PagedRoPERowsTF16(nHeads, headDim, nRot int, neox bool, rows, page int) (*ir.Kernel, error) {
	if err := checkPage("PagedRoPERowsTF16", page); err != nil {
		return nil, err
	}
	if nRot%2 != 0 || nRot <= 0 || nRot > headDim || headDim%2 != 0 || nHeads < 1 || rows < 1 || (neox && nRot%4 != 0) {
		return nil, fmt.Errorf("kernels: PagedRoPERowsTF16: nRot=%d headDim=%d heads=%d rows=%d neox=%v", nRot, headDim, nHeads, rows, neox)
	}
	pairs := nRot / 2
	items := pairs // one word each
	if neox {
		items = pairs / 2 // two words each
	}
	b := ir.New("pagedroperowstf16", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pCS := b.Param("pCS", ir.F32)
	b.Param("pOff", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	u := func(x int) ir.Value { return b.Const(ir.U32, int64(x)) }
	add := func(x, y ir.Value) ir.Value { return b.Add(ir.U32, x, y) }
	mul := func(x, y ir.Value) ir.Value { return b.Mul(ir.U32, x, y) }
	flat := b.Min(ir.U32, add(mul(b.CTAID(), b.NTID()), b.TID()), u(nHeads*items*rows-1))
	t := b.Rem(ir.U32, flat, u(nHeads*items))
	row := b.Div(ir.U32, flat, u(nHeads*items))
	it := b.Rem(ir.U32, t, u(items))
	head := mul(b.Div(ir.U32, t, u(items)), u(headDim))
	src := mul(row, u(nHeads*headDim))
	pid, off := pagedWrite(b, pTab, pRow, row, page)
	// rot is pair p's rotation of dimensions i0 and i1 (head-relative).
	rot := func(p, i0, i1 ir.Value) (ir.Value, ir.Value) {
		cs := add(mul(p, u(2)), mul(row, u(nRot)))
		c, sn := b.Load(ir.F32, pCS, cs, 0), b.Load(ir.F32, pCS, cs, 1)
		x0 := b.Load(ir.F32, pSrc, add(src, add(head, i0)), 0)
		x1 := b.Load(ir.F32, pSrc, add(src, add(head, i1)), 0)
		return b.Sub(ir.F32, b.Mul(ir.F32, x0, c), b.Mul(ir.F32, x1, sn)), b.Add(ir.F32, b.Mul(ir.F32, x0, sn), b.Mul(ir.F32, x1, c))
	}
	// word stores dimensions e and e+1 (head-relative, e even) packed.
	word := func(e, lo, hi ir.Value) {
		at := add(mul(pid, u(page*nHeads*headDim/2)), add(mul(b.Shr(ir.U32, add(head, e), u(1)), u(page)), off))
		b.Store(pDst, at, b.Bitcast(ir.F32, b.PackF16(lo, hi)), 0)
	}
	if !neox {
		e := mul(it, u(2))
		y0, y1 := rot(it, e, add(e, u(1)))
		word(e, y0, y1)
		return b.Done()
	}
	// Pairs p = 2*it and p+1: dimensions p, p+1 (one word) and p+pairs,
	// p+1+pairs (another, pairs being even).
	p := mul(it, u(2))
	p1 := add(p, u(1))
	a0, a1 := rot(p, p, add(p, u(pairs)))
	c0, c1 := rot(p1, p1, add(p1, u(pairs)))
	if pagedFault == "f16krope" {
		a0, c0 = c0, a0
	}
	word(p, a0, c0)
	word(add(p, u(pairs)), a1, c1)
	return b.Done()
}
