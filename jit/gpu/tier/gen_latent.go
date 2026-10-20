package tier

import "github.com/jitllm/jitllm/jit/gpu/compose"

// genLatentWrite is a latent row and its positional key into this position's
// paged row (OpLatentAppend): the latent at the row's first columns, the key
// after it, turned by the rotary as it is stored where the step says so
// (o.N == 1), each a composed row group addressed through the row's
// descriptor and the page table (compose.PagedPos). The pool's row is
// row-major [page][width] a page, width the block's whole stored row (an
// indexer's key may follow the two). nil where the history is not paged or
// the widths do not lay the two parts in the row.
func (g *devTier) genLatentWrite(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	p := &bs.p
	if bs.pkv == nil || len(o.In) != 2 {
		return nil
	}
	lat, rot, width := prog.vals[o.In[0]].Width, prog.vals[o.In[1]].Width, p.KVRow()
	if lat < 1 || rot < 1 || lat+rot > width || o.N == 1 && rot%2 != 0 {
		return nil
	}
	P := g.kvPage()
	var out []genLaunch
	for part, w := range [2]int{lat, rot} {
		rb := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
		r := &rb.r
		in := rb.buf(compose.BufRow, gaIn0+genArg(part))
		x := -1
		if part == 1 && o.N == 1 {
			x = r.Rotary(in, rb.buf(compose.BufVec, gaCS), rot, rot, p.RopeNeox, false)
		}
		desc, tab := rb.buf(compose.BufU32, gaPRow), rb.buf(compose.BufU32, gaTab)
		pid, off := r.PagedPos(desc, tab, P)
		if x < 0 {
			x = r.Load(in)
		}
		r.StoreAt(rb.buf(compose.BufOut, gaKPool), x, r.UBin(compose.OpAdd, r.PagedRow(pid, off, P, width, part*lat), r.Column()))
		k := g.rowKern(r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		out = append(out, l)
	}
	return out
}
