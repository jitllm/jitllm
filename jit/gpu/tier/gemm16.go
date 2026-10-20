package tier

import "github.com/jitllm/jitllm/jit/gpu/compose"

// gemm16Tiles are the m16n8 GEMM's blockings in the order a shape takes
// them: four subgroups a workgroup, the most rows and tokens first.
func gemm16Tiles() [][2]int {
	return [][2]int{{4, 4}, {4, 2}, {3, 4}, {3, 2}, {2, 4}, {2, 2}, {1, 2}}
}

// gemm16Of is a prompt chunk's matvec of m over rows tokens as the binary16
// m16n8 GEMM where the device lowers the instruction: the tile that brings
// the grid to a sixteenth of the threads the device holds resident, else
// the first that divides the shape, its k split to that fill, a tile the
// composer cannot form passed for the next. Callers hold g.mu.
func (g *devTier) gemm16Of(m mv, rows int) (compose.Gemm16, bool) {
	k := g.f16MMA()
	if k == 0 || g.composeTarget().API != compose.APIPTX || m.experts > 1 || m.bias != nil || m.pad {
		return compose.Gemm16{}, false
	}
	sub := m.q.Info().Sub
	if sub <= 0 || m.k%sub != 0 {
		return compose.Gemm16{}, false
	}
	kbN := max(32/sub, 1)
	fill := g.dev.Slots() / 16
	of := func(t [2]int) compose.Gemm16 {
		return compose.Gemm16{T: m.q, Rows: m.rows, K: m.k, NTok: rows, MT: t[0], NT: t[1], WM: 2, WN: 2, KB: kbN, K8: k == 8}
	}
	tiles := gemm16Tiles()
	first := -1
	if fill > 0 {
		most := 0
		for i, t := range tiles {
			s := of(t)
			if m.rows%(32*t[0]) != 0 || rows%(16*t[1]) != 0 {
				continue
			}
			groups := s.Groups()
			if groups*s.Threads() >= fill {
				first = i
				break
			}
			if groups > most {
				first, most = i, groups
			}
		}
	}
	for i := -1; i < len(tiles); i++ {
		if i < 0 && first < 0 || i >= 0 && i == first {
			continue
		}
		idx := i
		if i < 0 {
			idx = first
		}
		s := of(tiles[idx])
		if m.rows%(32*tiles[idx][0]) != 0 || rows%(16*tiles[idx][1]) != 0 {
			continue
		}
		trips := m.k / sub / kbN
		groups := s.Groups()
		split := kb0Split(trips, groups)
		if fill > 0 {
			split = fillSplit(trips, groups*s.Threads(), fill)
		}
		if f := g.kb.batch.Split; f >= 1 {
			split = f
		}
		for ; split >= 1; split /= 2 {
			s.Split = split
			r := compose.Row{Width: s.Threads(), Rows: s.Groups()}
			if r.Gemm16(s, r.Buf(compose.BufU32), r.Buf(compose.BufU32), r.Buf(compose.BufU32), r.Buf(compose.BufU32), r.Buf(compose.BufOut)) == nil {
				return s, true
			}
		}
	}
	return compose.Gemm16{}, false
}

// genGemm16 is op o's launches as the binary16 GEMM s: its float row made
// the token-major binary16 operand (skipped where the last conversion made
// it from the same row), the GEMM into the output or a split's planes, and
// the planes summed.
func (g *devTier) genGemm16(o *devOp, s compose.Gemm16) []genLaunch {
	if o.FIn == 0 || !g.genInSub && !g.sizeF16(s.NTok*s.K*2) || g.genInSub && g.f16Cap < s.NTok*s.K*2 {
		return nil
	}
	sub := s.T.Info().Sub
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
	cv := &rowBuild{r: compose.Row{Width: s.NTok * s.K / sub, Rows: 1}}
	x, b := cv.buf(compose.BufRow, gaIn0+genArg(o.FIn-1)), cv.buf(compose.BufOutU32, gaF16)
	if cv.r.ActF16T(s.T, s.K, x, b) != nil || !launch(cv) {
		return nil
	}
	ls[0].toF16 = true
	out := gaOut0
	if s.Split > 1 {
		out = gaSplit
		need := s.Rows * s.NTok * s.Split
		if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
			return nil
		}
	}
	gm := &rowBuild{r: compose.Row{Width: s.Threads(), Rows: s.Groups()}}
	qs, d, sc := gm.buf(compose.BufU32, gaWQS), gm.buf(compose.BufU32, gaWD), gm.buf(compose.BufU32, gaWSC)
	bf, o16 := gm.buf(compose.BufU32, gaF16), gm.buf(compose.BufOut, out)
	if gm.r.Gemm16(s, qs, d, sc, bf, o16) != nil || !launch(gm) {
		return nil
	}
	if s.Split > 1 {
		rd := &rowBuild{r: compose.Row{Width: s.Rows, Rows: s.NTok}}
		part := rd.buf(compose.BufRow, gaSplit)
		rd.store(rd.r.SumPlanes(part, s.Split), gaOut0)
		if !launch(rd) {
			return nil
		}
	}
	return ls
}
