package tier

import "github.com/jitllm/jitllm/jit_old/gpu/compose"

// tileGemmTiles are the tile GEMM's blockings (MT, NT; two subgroups each
// way), widest first: the narrower ones for a row or token count the wider
// do not divide.
func tileGemmTiles() [][2]int {
	return [][2]int{{4, 4}, {4, 2}, {2, 2}, {1, 2}}
}

// genGemmTile is a prompt chunk's projection of m over rows tokens on the
// collective binary16 tiles where the device lowers them: the row made the
// token-major binary16 operand, the widest blocking the rows divide over
// the leading tokens it divides and the first narrower one that divides the
// rest over the tail. Nil where no blocking serves. Callers hold g.mu.
func (g *devTier) genGemmTile(o *devOp, m mv, rows int) []genLaunch {
	if !g.tilesOK() || m.experts > 1 || m.pad || m.slots > 0 || o.FIn == 0 {
		return nil
	}
	sub := m.q.Info().Sub
	if sub <= 0 || m.k%sub != 0 {
		return nil
	}
	if !g.genInSub && !g.sizeF16(rows*m.k*2) || g.genInSub && g.f16Cap < rows*m.k*2 {
		return nil
	}
	tall := g.tileTall()
	of := func(t [2]int, n, off int) compose.GemmTile {
		return compose.GemmTile{T: m.q, Rows: m.rows, K: m.k, NTok: n, MT: t[0], NT: t[1], WM: 2, WN: 2,
			KB: max(32/sub, 1), Tall: tall && t[0]%2 == 0, TokOff: off}
	}
	var ls []genLaunch
	launch := func(rb *rowBuild) bool {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
		return true
	}
	gemm := func(s compose.GemmTile) bool {
		rb := &rowBuild{r: compose.Row{Width: s.Threads(), Rows: s.Groups()}}
		qs, d, sc := rb.buf(compose.BufU32, gaWQS), rb.buf(compose.BufU32, gaWD), rb.buf(compose.BufU32, gaWSC)
		b, out := rb.buf(compose.BufU32, gaF16), rb.buf(compose.BufTile, gaOut0)
		bias := -1
		if m.bias != nil {
			bias = rb.buf(compose.BufTile, gaMBias)
		}
		return rb.r.GemmTile(s, qs, d, sc, b, out, bias, -1) == nil && launch(rb)
	}
	tiles := tileGemmTiles()
	for wi, t := range tiles {
		s := of(t, rows, 0)
		if m.rows%s.BlockRows() != 0 {
			continue
		}
		lead, rest := rows, 0
		if rows%s.Toks() != 0 {
			if g.NoTileTail || rows < s.Toks() {
				continue
			}
			lead, rest = rows-rows%s.Toks(), rows%s.Toks()
		}
		var tail compose.GemmTile
		if rest > 0 {
			for _, tt := range tiles[wi+1:] {
				ts := of(tt, rest, lead)
				ts.KB = s.KB
				if m.rows%ts.BlockRows() == 0 && rest%ts.Toks() == 0 {
					tail = ts
					break
				}
			}
			if tail.NTok == 0 {
				continue
			}
		}
		ls = ls[:0]
		cv := &rowBuild{r: compose.Row{Width: rows * m.k / sub, Rows: 1}}
		x, f16 := cv.buf(compose.BufRow, gaIn0+genArg(o.FIn-1)), cv.buf(compose.BufOutU32, gaF16)
		if cv.r.ActF16T(m.q, m.k, x, f16) != nil || !launch(cv) {
			return nil
		}
		ls[0].toF16 = true
		s.NTok = lead
		if !gemm(s) || tail.NTok > 0 && !gemm(tail) {
			return nil
		}
		return ls
	}
	return nil
}
