package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
)

// The tower's projector on the CPU device (towerProjOf): where the composed
// CPU path is on, a picture's patch rows are regrouped and projected by
// what the composer forms for the projector's graphs, each composed once at
// the vision State's placement at the most rows a picture holds, and every
// picture counted served or declined by name (CPUProjectorSteps); main's
// projector runs one it declines.

// cpuProjState is a vision State's composed projector and its census.
type cpuProjState struct {
	group, proj, win cpuHeadForm
	tp               towerProj
	// why and err are why the front end states no projector graphs.
	why      Uncovered
	err      error
	served   int
	declined [jit.NumDeclines]int
	// req is the request's words: the grid's width, then a row's name a
	// row (the regroup's patch, or the put-back's merged row). in, grp and
	// out are a call's rows where a buffer it is handed stops short of the
	// rows its form's units reach.
	req          []uint32
	in, grp, out []float32
	unit         int
	// entry is the rows entering the tower's first block (towerEntryOf),
	// entryWhy and entryErr why the front end states none; pos is its
	// step's positions (the live rows, then a row's), posW the position
	// table as the form reads it, px and x a call's pixels and rows where
	// the buffers handed it stop short of its units.
	entry         cpuHeadForm
	entryWhy      Uncovered
	entryErr      error
	entryServed   int
	entryDeclined [jit.NumDeclines]int
	pos           []uint32
	posW          tensor
	px, x         []float32
	entryUnit     int
	// cards is the entry and the projector as a card runs them, and the
	// pictures each served there; cardGrp the regrouped rows between the
	// projector's two parts there.
	cards                         picRows
	cardEntry, cardProj, cardMiss int
	cardGrp                       []float32
}

// picRows is the tower's entry and projector as a card runs them
// (nn.RowsPart), built at their first use; nil for one the front end does
// not state.
type picRows struct {
	built                   bool
	entry, group, proj, win *nn.RowsPart
}

// CardRowsSteps is how many pictures a card served the composed entry and
// projector of, and how many parts it was asked for and did not run.
func (s *State) CardRowsSteps() (entry, proj, missed int) {
	st := &s.cpu.proj
	return st.cardEntry, st.cardProj, st.cardMiss
}

// picBufs holds the request and position words, each made once for the
// tower's most rows.
func (s *State) picBufs() {
	st := &s.cpu.proj
	if st.req != nil {
		return
	}
	n := s.vis.t.Cfg.MaxSeq()
	st.req, st.pos = make([]uint32, jit.RequestRows+n), make([]uint32, 1+n)
}

// picRowsParts is the State's entry and projector as a card runs them.
func (s *State) picRowsParts() *picRows {
	pr := &s.cpu.proj.cards
	if pr.built {
		return pr
	}
	pr.built = true
	t := s.vis.t
	c := t.Cfg
	E := c.NEmbd
	w := map[jlm.Role]nn.Weight{}
	v := map[jlm.Role][]float32{}
	for _, m := range []struct {
		r jlm.Role
		t *tensor
	}{{jlm.RoleVPatchEmbd, &t.patchW}, {jlm.RoleVProj, &t.projW}, {jlm.RoleVProj2, &t.projW2}, {jlm.RoleVProj3, &t.projW3}} {
		if m.t.e != nil {
			w[m.r] = wt(*m.t)
		}
	}
	if t.posW != nil {
		w[jlm.RoleVPosEmbd] = nn.Weight{T: quant.F32, Data: unsafe.Slice((*byte)(unsafe.Pointer(&t.posW[0])), 4*len(t.posW)),
			Rows: len(t.posW) / E, K: E}
	}
	for _, x := range []towerVec{{jlm.RoleVPatchBias, t.patchB}, {jlm.RoleVClassEmbd, t.clsW}, {jlm.RoleVPreNorm, t.preLnW},
		{jlm.RoleVPreNormBias, t.preLnB}, {jlm.RoleVProjBias, t.projB}, {jlm.RoleVProj2Bias, t.projB2},
		{jlm.RoleVProjNorm, t.projNorm}, {jlm.RoleVProjNormBias, t.projNormB},
		{jlm.RoleVProjInNorm, t.projInNorm}, {jlm.RoleVProjInNormBias, t.projInNormB},
		{jlm.RoleVStdBias, t.stdBias()}, {jlm.RoleVStdScale, t.stdScale()}} {
		if len(x.V) != 0 {
			v[x.Role] = x.V
		}
	}
	part := func(g *graph.Graph, rows int) *nn.RowsPart {
		if g == nil {
			return nil
		}
		return &nn.RowsPart{Graph: g, Rows: rows, Weights: w, Vectors: v}
	}
	fg := s.m.frontGraphs()
	tp, why := fg.towerProj, fg.towerProjWhy
	// A tower under a fault states its graphs again: a fault is a rewrite
	// of them.
	if c.fault != towerFaultNone {
		tp, why, _ = s.m.towerProjOf(c.fault == towerFaultPoolCorner)
	}
	S := max(c.Scale, 1)
	patches := c.MaxSeq()
	merged := max(c.MaxTokens(), (patches+S*S-1)/(S*S))
	eg, ewhy := fg.towerEntry, fg.towerEntryWhy
	if c.fault != towerFaultNone {
		eg, ewhy, _ = s.m.towerEntryOf()
	}
	if ewhy == Covered {
		pr.entry = part(eg, patches)
	}
	if why == Covered {
		pr.group, pr.proj, pr.win = part(tp.group, patches), part(tp.proj, merged), part(tp.win, merged)
	}
	s.picBufs()
	s.cpu.proj.cardGrp = make([]float32, patches*E)
	return pr
}

// cardRows is the card a picture's rows run on where its block li is on
// one, nil otherwise.
func (s *State) cardRows(li int) nn.PictureRowsDevice {
	if s.ld == nil || !s.devAt(li) {
		return nil
	}
	rd, _ := s.ld.(nn.PictureRowsDevice)
	return rd
}

// prepPicRows forms a vision State's entry and projector on the card
// holding its tower's first or last block, at the placement that put it
// there.
func (s *State) prepPicRows() {
	if s.vis == nil {
		return
	}
	rd := s.cardRows(s.lo)
	if rd == nil {
		rd = s.cardRows(s.hi - 1)
	}
	if rd == nil {
		return
	}
	pr := s.picRowsParts()
	for _, p := range [...]*nn.RowsPart{pr.entry, pr.group, pr.proj, pr.win} {
		if p != nil {
			rd.PrepRows(p)
		}
	}
}

// CPUEntrySteps is CPUProjectorSteps for the rows entering the tower.
func (s *State) CPUEntrySteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.proj.entryServed, s.cpu.proj.entryDeclined
}

// CPUEntryUncovered is CPUProjectorUncovered for the rows entering the
// tower.
func (s *State) CPUEntryUncovered() (Uncovered, error) {
	return s.cpu.proj.entryWhy, s.cpu.proj.entryErr
}

// cpuPlaceEntry composes the tower's entry for the CPU at the vision
// State's placement.
func (s *State) cpuPlaceEntry() {
	st := &s.cpu.proj
	t := s.vis.t
	c := t.Cfg
	g, why, err := s.m.towerEntryOf()
	st.entryWhy, st.entryErr = why, err
	if why != Covered {
		st.entry = cpuHeadForm{composed: true, why: jit.DeclineNoGraph}
		return
	}
	var over map[jlm.Role]jlm.Type
	if t.posW != nil {
		over = map[jlm.Role]jlm.Type{jlm.RoleVPosEmbd: jlm.TypeF32}
		st.posW = tensor{data: unsafe.Slice((*byte)(unsafe.Pointer(&t.posW[0])), 4*len(t.posW)), rows: len(t.posW) / c.NEmbd, k: c.NEmbd}
	}
	rows := c.MaxSeq()
	s.cpuComposeHeadAs(&st.entry, g, rows, over)
	u := 1
	if st.entry.f != nil {
		u = max(u, st.entry.f.unit)
	}
	st.entryUnit = u
	up := (rows + u - 1) / u * u
	P := c.PatchSz
	st.px = make([]float32, up*3*P*P)
	st.x = make([]float32, up*c.NEmbd)
}

// cpuEntry writes the n rows entering the tower's first block into s.bx
// from the picture px through the composed entry, the patches from row
// first, or says why not.
func (s *State) cpuEntry(px []float32, n, first int) jit.Decline {
	st := &s.cpu.proj
	v := s.vis
	t, c := v.t, v.t.Cfg
	gh, gw := s.patchGrid()
	tableRows := t.posW == nil || v.ord == nil && (c.PosBuckets == 0 || bucketsAreRows(gh, gw, c.PosBuckets))
	if rd := s.cardRows(s.lo); rd != nil && v.px2 == nil && tableRows {
		if pr := s.picRowsParts(); pr.entry != nil && n <= pr.entry.Rows {
			s.entryWords(n, first)
			if rd.RunRows(pr.entry, n, st.req[:jit.RequestRows+n], st.pos[:1+n], px, s.bx[:n*c.NEmbd]) {
				st.cardEntry++
				return jit.DeclineServes
			}
			st.cardMiss++
		}
	}
	if s.cpu.args == nil {
		return jit.DeclineCaller
	}
	f, why := st.entry.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	switch {
	case why != jit.DeclineServes:
	case n > f.rows || v.px2 != nil:
		why = jit.DeclineNoRoom
	case !tableRows:
		// A table row per position the step names, and a patch per
		// position: the two must name the same row.
		why = jit.DeclineNoTable
	}
	if why != jit.DeclineServes {
		st.entryDeclined[why]++
		return why
	}
	s.entryWords(n, first)
	E, P := c.NEmbd, c.PatchSz
	u := st.entryUnit
	up := (n + u - 1) / u * u
	in := px
	if cap(px) < up*3*P*P {
		in = st.px
		copy(in, px)
	}
	out := s.bx
	if cap(out) < up*E {
		out = st.x
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: n, req: st.req, pos: st.pos}
	b.io[0], b.io[1] = unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0])
	s.cpu.call[0] = uint32(n)
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
	if &out[0] != &s.bx[0] {
		copy(s.bx[:n*E], out)
	}
	st.entryServed++
	return jit.DeclineServes
}

// entryWords writes the entry's request (the grid's width) and its
// positions: the live rows, then each row's -- a patch row's its patch's
// plus first.
func (s *State) entryWords(n, first int) {
	st := &s.cpu.proj
	s.picBufs()
	gh, gw := s.patchGrid()
	np := gh * gw
	st.pos[0] = uint32(n)
	for r := range n {
		// A class row after the patches reads its own row.
		at := r
		if r >= first && r-first < np {
			p := s.patchAt(r - first)
			if s.cpu.fault == cpuFaultEntryRows {
				p = (p + 1) % np
			}
			at = p + first
		}
		st.pos[1+r] = uint32(at)
	}
	st.req[jit.RequestGrid] = uint32(gw)
}

// bucketsAreRows says a gh x gw grid's position buckets (bucketPositions)
// are its rows' own.
func bucketsAreRows(gh, gw, B int) bool {
	return gh == B && gw == B
}

// CPUProjectorSteps is how many pictures the composed projector served on
// the CPU, and how many it declined, by reason; with the composed CPU off
// nothing is counted.
func (s *State) CPUProjectorSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.proj.served, s.cpu.proj.declined
}

// CPUProjectorUncovered is why the front end states no projector graphs
// for the State's tower, Covered where it does.
func (s *State) CPUProjectorUncovered() (Uncovered, error) { return s.cpu.proj.why, s.cpu.proj.err }

// cpuPlaceProjector composes the projector's graphs for the CPU at the
// vision State's placement, with the buffers their calls bind.
func (s *State) cpuPlaceProjector() {
	st := &s.cpu.proj
	t := s.vis.t
	c := t.Cfg
	s.picBufs()
	s.cpuPlaceEntry()
	st.tp, st.why, st.err = s.m.towerProjOf(c.fault == towerFaultPoolCorner)
	if st.why != Covered {
		st.group = cpuHeadForm{composed: true, why: jit.DeclineNoGraph}
		st.proj, st.win = st.group, st.group
		return
	}
	S := max(c.Scale, 1)
	patches := c.MaxSeq()
	merged := max(c.MaxTokens(), (patches+S*S-1)/(S*S))
	if st.tp.group != nil {
		s.cpuComposeHead(&st.group, st.tp.group, jit.DeclineNoGraph, patches)
	}
	s.cpuComposeHead(&st.proj, st.tp.proj, jit.DeclineNoGraph, merged)
	if st.tp.win != nil {
		s.cpuComposeHead(&st.win, st.tp.win, jit.DeclineNoGraph, merged)
	}
	unit := 1
	for _, hf := range []*cpuHeadForm{&st.group, &st.proj, &st.win} {
		if hf.f != nil {
			unit = max(unit, hf.f.unit)
		}
	}
	st.unit = unit
	up := func(n int) int { return (n + unit - 1) / unit * unit }
	E := c.NEmbd
	st.in = make([]float32, up(patches)*E)
	st.grp = make([]float32, up(patches)*E)
	st.out = make([]float32, up(merged)*c.ProjDim)
}

// cpuProject projects the picture's np patch rows (behind the tower's last
// norm, the class row dropped) through the composed projector into the
// State's output rows, or says why not.
func (s *State) cpuProject(patches []float32) ([]float32, jit.Decline) {
	st := &s.cpu.proj
	v := s.vis
	c := v.t.Cfg
	gh, gw := s.patchGrid()
	S := max(c.Scale, 1)
	// The merged grid is each axis's whole groups: a pool over a grid its
	// scale does not divide drops the remainder, as main's does.
	np, nt := gh*gw, (gh/S)*(gw/S)
	// A picture in the window order holds each group in consecutive rows.
	windowed := v.ord != nil
	if out, ok := s.cardProject(patches, np, nt, windowed); ok {
		return out, jit.DeclineServes
	}
	if s.cpu.args == nil {
		return nil, jit.DeclineCaller
	}
	hf, gf := &st.proj, &st.group
	if windowed {
		hf = &st.win
	}
	if windowed || S == 1 {
		gf = nil
	}
	proj, why := hf.formOf()
	var group *cpuBlockForm
	if why == jit.DeclineServes && gf != nil {
		group, why = gf.formOf()
	}
	for _, f := range []*cpuBlockForm{group, proj} {
		if why == jit.DeclineServes && f != nil {
			why = s.cpuWeights(f, -1, nil)
		}
	}
	if why == jit.DeclineServes && (windowed && nt*S*S != np || group != nil && np > group.rows || nt > proj.rows) {
		why = jit.DeclineNoRoom
	}
	if why != jit.DeclineServes {
		st.declined[why]++
		return nil, why
	}
	E := c.NEmbd
	s.projWords(gh, gw, windowed)
	src := patches[:np*E]
	if group != nil {
		s.cpuProjRun(group, s.cpuProjRows(src, st.in, np, E), st.grp, np)
		src = st.grp[:np*E]
	}
	s.cpuProjRun(proj, s.cpuProjRows(src, st.in, nt, E*S*S), st.out, nt)
	st.served++
	out := v.out[:nt*c.ProjDim]
	copy(out, st.out)
	return out, jit.DeclineServes
}

// cpuProjDeclined counts a picture whose projector the composed CPU path
// declines before cpuProject is asked: the reason its forms hold.
func (s *State) cpuProjDeclined() {
	if s.cpu.args == nil {
		return
	}
	_, why := s.cpu.proj.proj.formOf()
	if why != jit.DeclineServes {
		s.cpu.proj.declined[why]++
	}
}

// projWords writes the projector's request: the grid's width, then each
// merged row's patches where the rows are in raster order (merged row g's
// slot dy*S+dx the patch at (gy*S+dy, gx*S+dx)), or each raster merged
// row's window-order row where they are in the window order.
func (s *State) projWords(gh, gw int, windowed bool) {
	st := &s.cpu.proj
	s.picBufs()
	v := s.vis
	c := v.t.Cfg
	S := max(c.Scale, 1)
	nt := (gh / S) * (gw / S)
	st.req[jit.RequestGrid] = uint32(gw)
	if S == 1 {
		return
	}
	if windowed {
		// Merged row u of the raster order is the window order's row whose
		// first patch lies in u.
		mw := gw / S
		for i := range nt {
			p := v.ord[i*S*S]
			u := (p/gw/S)*mw + (p%gw)/S
			if qwenNoUnwindow {
				u = i
			}
			st.req[jit.RequestRows+u] = uint32(i)
		}
		return
	}
	gsw := gw / S
	for g := range nt {
		gy, gx := g/gsw, g%gsw
		for dy := range S {
			for dx := range S {
				slot := dy*S + dx
				if c.fault == towerFaultShuffleSwap {
					slot = dx*S + dy
				}
				st.req[jit.RequestRows+g*S*S+slot] = uint32((gy*S+dy)*gw + gx*S + dx)
			}
		}
	}
	// The patches no whole group holds are regrouped from patch 0 and never
	// read.
	clear(st.req[jit.RequestRows+nt*S*S : jit.RequestRows+gh*gw])
}

// cardProject is cpuProject on the card holding the tower's last block,
// false where none does or it does not run the parts.
func (s *State) cardProject(patches []float32, np, nt int, windowed bool) ([]float32, bool) {
	rd := s.cardRows(s.hi - 1)
	if rd == nil {
		return nil, false
	}
	st := &s.cpu.proj
	pr := s.picRowsParts()
	c := s.vis.t.Cfg
	E, S := c.NEmbd, max(c.Scale, 1)
	proj, group := pr.proj, pr.group
	if windowed {
		proj, group = pr.win, nil
	}
	if S == 1 {
		group = nil
	}
	if proj == nil || S > 1 && !windowed && group == nil || windowed && nt*S*S != np || nt > proj.Rows || np > c.MaxSeq() {
		return nil, false
	}
	gh, gw := s.patchGrid()
	s.projWords(gh, gw, windowed)
	src := patches[:np*E]
	if group != nil {
		if !rd.RunRows(group, np, st.req[:jit.RequestRows+np], nil, src, st.cardGrp[:np*E]) {
			st.cardMiss++
			return nil, false
		}
		src = st.cardGrp[:np*E]
	}
	out := s.vis.out[:nt*c.ProjDim]
	if !rd.RunRows(proj, nt, st.req[:jit.RequestRows+nt], nil, src, out) {
		st.cardMiss++
		return nil, false
	}
	st.cardProj++
	return out, true
}

// cpuProjRows is the first rows rows of x, each width wide, as a call
// binds them: x where it holds the rows the forms' units reach, else its
// rows copied into pad.
func (s *State) cpuProjRows(x, pad []float32, rows, width int) []float32 {
	u := s.cpu.proj.unit
	n := (rows + u - 1) / u * u * width
	if cap(x) >= n {
		return x[:n]
	}
	copy(pad, x[:rows*width])
	return pad
}

// cpuProjRun runs form f over rows rows of in into out, one call.
func (s *State) cpuProjRun(f *cpuBlockForm, in, out []float32, rows int) {
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: rows, req: s.cpu.proj.req}
	b.io[0], b.io[1] = unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0])
	s.cpu.call[0] = uint32(rows)
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
}
