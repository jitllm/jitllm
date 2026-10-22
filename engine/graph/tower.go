package graph

import "github.com/jitllm/jitllm/format/jlm"

// TowerEntry is a vision tower's input rows: the picture's patches through
// the patch embedding (RoleVPatchEmbd, Patch x Patch x 3 wide, stored K
// wide), its bias where PatchBias, after the class row where Class
// (RoleVClassEmbd, row 0), each row's position's row of the position table
// where Positions (the step's position a row is its table row), and a
// LayerNorm over the sum where PreNorm.
type TowerEntry struct {
	NEmbd, Patch, K, Positions int
	PatchBias, Class, PreNorm  bool
	Eps                        float64
}

func (e *TowerEntry) Build() *Graph {
	g := &Graph{Block: -1}
	px := g.Input("pixels", 3*e.Patch*e.Patch)
	lead := 0
	if e.Class {
		lead = 1
	}
	p := g.row("patches", e.K, Op{Kind: OpPatchify, In: []ValueID{px}, Attr: Attrs{Span: e.Patch, Offset: lead}})
	ws := []Binding{{Role: jlm.RoleVPatchEmbd, K: e.K, Rows: e.NEmbd}}
	if e.PatchBias {
		ws = append(ws, Binding{Role: jlm.RoleVPatchBias, K: e.NEmbd})
	}
	x := g.row("patch_embd", e.NEmbd, Op{Kind: OpMatVec, In: []ValueID{p}, Weight: ws})
	if e.Class {
		x = g.row("x_class", e.NEmbd, Op{Kind: OpLeadRow, In: []ValueID{x},
			Weight: []Binding{{Role: jlm.RoleVClassEmbd, K: e.NEmbd}}, Attr: Attrs{Offset: lead}})
	}
	if e.Positions > 0 {
		pos := g.value("pos_embd", e.NEmbd)
		g.op(Op{Kind: OpEmbed, Out: []ValueID{pos}, Weight: []Binding{{Role: jlm.RoleVPosEmbd, K: e.NEmbd, Rows: e.Positions}},
			Attr: Attrs{Positional: true}})
		x = g.row("x_pos", e.NEmbd, Op{Kind: OpAdd, In: []ValueID{x, pos}})
	}
	if e.PreNorm {
		x = g.row("x_normed", e.NEmbd, Op{Kind: OpLayerNorm, In: []ValueID{x},
			Weight: []Binding{{Role: jlm.RoleVPreNorm, K: e.NEmbd}, {Role: jlm.RoleVPreNormBias, K: e.NEmbd}}, Attr: Attrs{Eps: e.Eps}})
	}
	g.Out = []ValueID{x}
	return g
}

// TowerHead is a vision tower's projection: the LayerNorm after the blocks
// where PostNorm, the class row dropped where Class, the grid regrouped
// Scale x Scale into rows where Scale is above one, and the projector's
// matrix (RoleVProj, its bias where ProjBias) to the text model's width --
// or where Hidden, that matrix to Hidden wide, Act, and a second matrix
// (RoleVProj2, its bias where ProjBias) to the text model's.
type TowerHead struct {
	NEmbd, Scale, ProjDim, Hidden int
	PostNorm, Class, ProjBias     bool
	Act                           Act
	Eps                           float64
}

func (h *TowerHead) Build() *Graph {
	g := &Graph{Block: -1}
	x := g.Input("x", h.NEmbd)
	if h.PostNorm {
		x = g.row("post_norm", h.NEmbd, Op{Kind: OpLayerNorm, In: []ValueID{x},
			Weight: []Binding{{Role: jlm.RoleVPostNorm, K: h.NEmbd}, {Role: jlm.RoleVPostNormBias, K: h.NEmbd}}, Attr: Attrs{Eps: h.Eps}})
	}
	if h.Class {
		x = g.row("patch_rows", h.NEmbd, Op{Kind: OpDropRows, In: []ValueID{x}, Attr: Attrs{Offset: 1}})
	}
	w := h.NEmbd
	if h.Scale > 1 {
		w *= h.Scale * h.Scale
		x = g.row("regrouped", w, Op{Kind: OpRegroup, In: []ValueID{x}, Attr: Attrs{Span: h.Scale}})
	}
	mv := func(name string, x ValueID, k, rows int, role, bias jlm.Role) ValueID {
		ws := []Binding{{Role: role, K: k, Rows: rows}}
		if h.ProjBias {
			ws = append(ws, Binding{Role: bias, K: rows})
		}
		return g.row(name, rows, Op{Kind: OpMatVec, In: []ValueID{x}, Weight: ws})
	}
	if h.Hidden == 0 {
		g.Out = []ValueID{mv("proj", x, w, h.ProjDim, jlm.RoleVProj, jlm.RoleVProjBias)}
		return g
	}
	y := mv("proj", x, w, h.Hidden, jlm.RoleVProj, jlm.RoleVProjBias)
	y = g.row("proj_act", h.Hidden, Op{Kind: OpAct, In: []ValueID{y}, Attr: Attrs{Act: h.Act}})
	g.Out = []ValueID{mv("proj2", y, h.Hidden, h.ProjDim, jlm.RoleVProj2, jlm.RoleVProj2Bias)}
	return g
}
