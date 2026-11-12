package convert

import "github.com/jitllm/jitllm/format/jlm"

// A video's statement (jlm.Video) is expressions of its sizes and pieces of
// its prompt. These build them from what a processor does; each family's
// processor is stated with them in its tower's stater, and the engine only
// evaluates what is stated.

// sizes builds an expression of a video's sizes term by term.
type sizes struct{ jlm.ExprBuilder }

func (e *sizes) op(i int) jlm.Arg               { return jlm.Operand(uint32(i)) }
func (e *sizes) add(a, b jlm.Arg) jlm.Arg       { return e.Fn(jlm.FnAdd, a, b) }
func (e *sizes) sub(a, b jlm.Arg) jlm.Arg       { return e.Fn(jlm.FnSub, a, b) }
func (e *sizes) mul(a, b jlm.Arg) jlm.Arg       { return e.Fn(jlm.FnMul, a, b) }
func (e *sizes) div(a, b jlm.Arg) jlm.Arg       { return e.Fn(jlm.FnDiv, a, b) }
func (e *sizes) min(a, b jlm.Arg) jlm.Arg       { return e.Fn(jlm.FnMin, a, b) }
func (e *sizes) max(a, b jlm.Arg) jlm.Arg       { return e.Fn(jlm.FnMax, a, b) }
func (e *sizes) less(a, b jlm.Arg) jlm.Arg      { return e.Fn(jlm.FnLess, a, b) }
func (e *sizes) where(c, a, b jlm.Arg) jlm.Arg  { return e.Fn(jlm.FnWhere, c, a, b) }
func (e *sizes) fn(f jlm.Fn, a jlm.Arg) jlm.Arg { return e.Fn(f, a) }
func (e *sizes) expr() jlm.Expr                 { return jlm.Expr{Terms: e.T} }
func opExpr(i int) jlm.Expr                     { var e sizes; e.add(e.op(i), num(0)); return e.expr() }
func constant(v float32) jlm.Expr               { var e sizes; e.add(num(v), num(0)); return e.expr() }
func idsPiece(ids []int32) jlm.VideoPiece       { return jlm.VideoPiece{Kind: jlm.PieceIDs, IDs: ids} }
func textPiece(s string) jlm.VideoPiece         { return jlm.VideoPiece{Kind: jlm.PieceText, Text: s} }
func spanPiece() jlm.VideoPiece                 { return jlm.VideoPiece{Kind: jlm.PieceSpan} }
func valuePiece(e jlm.Expr, f jlm.ValueFormat) jlm.VideoPiece {
	return jlm.VideoPiece{Kind: jlm.PieceValue, Value: e, Format: f}
}

// everyFrame samples every frame of a video: its count N, frame i at i.
func everyFrame() (count, index jlm.Expr) { return opExpr(jlm.VideoN), opExpr(jlm.VideoAt) }

// pictureSpan is a group's rows between the picture's markers, as the tower
// span states them.
func pictureSpan(t *jlm.TowerSpan) []jlm.VideoPiece {
	var ps []jlm.VideoPiece
	if len(t.Before) > 0 {
		ps = append(ps, idsPiece(t.Before))
	}
	ps = append(ps, spanPiece())
	if len(t.After) > 0 {
		ps = append(ps, idsPiece(t.After))
	}
	return ps
}
