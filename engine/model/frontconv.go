package model

import (
	"fmt"
	"maps"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// A convolutional tower block (a block whose activation is a picture's
// grid, mobilenet.go) stated in the op language: its rows are its input
// grid's positions and a row its channels, the step's rows its input's.
// What it does is read off the tensors it holds -- a full convolution's
// matrix (one over a window of rows, OpUnfold and OpMatVec), depthwise
// filters (OpConv) where it holds them, the pointwise matrices between,
// each followed by its norm -- at the geometry the container states for
// the block (its place and the grid it reads). A stride above one makes
// the grid its own rows (graph.Value.Rows); the residual is added where
// the output's grid and channels are the input's.

// convTensors is block l's matrices by the role each plays.
func convTensors(l *layer) [8]struct {
	r jlm.Role
	t *tensor
} {
	return [...]struct {
		r jlm.Role
		t *tensor
	}{{jlm.RoleVMNConvExp, &l.up}, {jlm.RoleVMNConvPwl, &l.down}, {jlm.RoleVMNPwExp, &l.up}, {jlm.RoleVMNPwProj, &l.down},
		{jlm.RoleVAttnQ, &l.wq}, {jlm.RoleVAttnK, &l.wk}, {jlm.RoleVAttnV, &l.wv}, {jlm.RoleVAttnOut, &l.wo}}
}

// convVec is a convolutional block's vector, the role it plays and, for a
// depthwise filter, its side (Taps x Taps rows of its channels).
type convVec struct {
	Role jlm.Role
	V    []float32
	Taps int
}

// convVectors is block l's norms and depthwise filters by the role each
// plays.
func convVectors(b *mnBlock) [13]convVec {
	return [...]convVec{
		{Role: jlm.RoleVMNNorm1, V: b.norm1}, {Role: jlm.RoleVMNNorm2, V: b.norm2},
		{Role: jlm.RoleVMNPwExpNorm, V: b.norm1}, {Role: jlm.RoleVMNPwProjNorm, V: b.norm2},
		{Role: jlm.RoleVMNDwStart, V: b.dwStart, Taps: b.kStart}, {Role: jlm.RoleVMNDwStartNorm, V: b.dwStartNorm},
		{Role: jlm.RoleVMNDwMid, V: b.dwMid, Taps: b.kMid}, {Role: jlm.RoleVMNDwMidNorm, V: b.dwMidNorm},
		{Role: jlm.RoleVAttnNorm, V: b.norm1},
		{Role: jlm.RoleVMNKDown, V: b.kDown, Taps: b.kvK}, {Role: jlm.RoleVMNKDownNorm, V: b.kDownNorm},
		{Role: jlm.RoleVMNVDown, V: b.vDown, Taps: b.kvK}, {Role: jlm.RoleVMNVDownNorm, V: b.vDownNorm},
	}
}

// convInventory is the roles block li holds in the container and the
// shape each binds at: a matrix's, a norm's width, a depthwise filter's
// Taps*Taps rows of its channels. A role the container holds for the
// block and the loaded block does not read is reported apart.
func (m *Model) convInventory(li int) (graph.Facts, []jlm.Role) {
	t, l := m.tower, &m.layers[li]
	vi := int32(li - t.base)
	inv := map[jlm.Role]graph.Shape{}
	var held []jlm.Role
	for _, e := range m.container.Entries() {
		if e.Role.Vision() && e.Block == vi {
			held = append(held, e.Role)
		}
	}
	has := func(r jlm.Role) bool { return slices.Contains(held, r) }
	for _, x := range convTensors(l) {
		if has(x.r) && x.t.e != nil {
			inv[x.r] = graph.Shape{K: x.t.k, Rows: x.t.rows}
		}
	}
	for _, v := range convVectors(l.mn) {
		switch {
		case !has(v.Role) || len(v.V) == 0:
		case v.Taps > 0:
			inv[v.Role] = graph.Shape{K: len(v.V) / (v.Taps * v.Taps), Rows: v.Taps * v.Taps}
		default:
			inv[v.Role] = graph.Shape{K: len(v.V)}
		}
	}
	var unread []jlm.Role
	for _, r := range held {
		if _, ok := inv[r]; !ok && r != jlm.RoleVMNGeom {
			unread = append(unread, r)
		}
	}
	return graph.Facts{Inventory: inv}, unread
}

// convGraph is convolutional tower block li's graph, or why there is none.
func (m *Model) convGraph(li int) (*graph.Graph, Uncovered, error) {
	l := &m.layers[li]
	bl := l.mn
	f, unread := m.convInventory(li)
	if len(unread) > 0 {
		return nil, UncoveredUnbound, fmt.Errorf("the container holds %v for tower block %d, which the loaded block does not read", unread, li-m.tower.base)
	}
	has := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	if has(jlm.RoleVAttnQ) {
		return nil, UncoveredGridAttend, fmt.Errorf("attends over keys and values of a %dx%d grid from a %dx%d grid's queries",
			bl.hkv, bl.wkv, bl.hin, bl.win)
	}
	act, err := m.container.Vision().Activation(false)
	if err != nil {
		return nil, UncoveredAct, err
	}
	eps := m.tower.Cfg.Eps
	b := graph.NewBlock(li)
	x := b.In("x", bl.cin)
	gelu := func(v graph.ValueID) (graph.ValueID, error) {
		return b.Stated("act", &act, []graph.ValueID{v}, b.G.Values[v].Width, f)
	}
	// grid is a grid's rows where a stride makes them its own.
	grid := func(h, w, s int) int {
		if s > 1 {
			return h * w
		}
		return 0
	}
	// depthwise is a depthwise filter of role r over v, a grid w wide, at
	// stride s into an h x wo grid, and its norm.
	depthwise := func(v graph.ValueID, r, norm jlm.Role, taps, w, s, h, wo int) graph.ValueID {
		c := b.G.Values[v].Width
		d := b.Emit(graph.OpConv, []graph.ValueID{v}, []graph.Result{{Name: r.String(), Width: c, Rows: grid(h, wo, s)}},
			[]graph.Binding{{Role: r, K: c, Rows: taps * taps}}, graph.Attrs{Taps: taps, GridW: w, Stride: s}, nil, nil)[0]
		return b.Norm(d, norm, eps)
	}
	var y graph.ValueID
	switch {
	case has(jlm.RoleVMNConvExp) && has(jlm.RoleVMNConvPwl) && has(jlm.RoleVMNNorm1) && has(jlm.RoleVMNNorm2):
		// A full convolution over a window of rows: the window whose rows
		// of the input's channels the matrix reads, a whole quantization
		// block past them.
		k := f.Inventory[jlm.RoleVMNConvExp].K
		taps := 1
		for pad32(taps*taps*bl.cin) < k {
			taps += 2
		}
		if pad32(taps*taps*bl.cin) != k {
			return nil, UncoveredGeometry, fmt.Errorf("a full convolution's matrix %d wide over %d channels is no square window", k, bl.cin)
		}
		u := b.Emit(graph.OpUnfold, []graph.ValueID{x}, []graph.Result{{Name: "unfold", Width: k, Rows: grid(bl.hout, bl.wout, bl.stride)}},
			nil, graph.Attrs{Taps: taps, GridW: bl.win, Stride: bl.stride}, nil, nil)[0]
		e := b.Norm(b.MatVec(u, jlm.RoleVMNConvExp, bl.cexp), jlm.RoleVMNNorm1, eps)
		if e, err = gelu(e); err != nil {
			return nil, UncoveredAct, err
		}
		y = b.Norm(b.MatVec(e, jlm.RoleVMNConvPwl, bl.cout), jlm.RoleVMNNorm2, eps)
	case has(jlm.RoleVMNPwExp) && has(jlm.RoleVMNPwProj) && has(jlm.RoleVMNPwExpNorm) && has(jlm.RoleVMNPwProjNorm):
		in, w := x, bl.win
		if has(jlm.RoleVMNDwStart) {
			in = depthwise(in, jlm.RoleVMNDwStart, jlm.RoleVMNDwStartNorm, bl.kStart, w, bl.sStart, bl.hmid, bl.wmid)
			w = bl.wmid
		}
		e := b.Norm(b.MatVec(in, jlm.RoleVMNPwExp, bl.cexp), jlm.RoleVMNPwExpNorm, eps)
		if e, err = gelu(e); err != nil {
			return nil, UncoveredAct, err
		}
		if has(jlm.RoleVMNDwMid) {
			e = depthwise(e, jlm.RoleVMNDwMid, jlm.RoleVMNDwMidNorm, bl.kMid, w, bl.sMid, bl.hout, bl.wout)
			if e, err = gelu(e); err != nil {
				return nil, UncoveredAct, err
			}
		}
		y = b.Norm(b.MatVec(e, jlm.RoleVMNPwProj, bl.cout), jlm.RoleVMNPwProjNorm, eps)
	default:
		return nil, UncoveredKind, fmt.Errorf("tower block %d holds %v: no convolution the front end states", li-m.tower.base, slices.Sorted(maps.Keys(f.Inventory)))
	}
	if bl.residual {
		y = b.Add(y, x)
	}
	b.Out(y)
	return verified(b, f)
}
