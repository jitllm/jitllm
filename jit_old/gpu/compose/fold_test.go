package compose

import "testing"

// TestFoldsAreChecked: a fold with no step, a loop that does not exist, and
// an iterator read outside its loop's body are refused, and a nested fold
// (a count over a loop whose key folds another) forms a kernel.
func TestFoldsAreChecked(t *testing.T) {
	g := &Row{Width: 8, Rows: 1}
	c := g.Loop(8)
	g.Fold(c, g.Const(0))
	if err := g.Check(); err == nil {
		t.Error("a fold with no step passed Check")
	}

	g = &Row{Width: 8, Rows: 1}
	g.Add(Node{Kind: NIter, At: 3})
	if err := g.Check(); err == nil {
		t.Error("the iterator of a loop that does not exist passed Check")
	}

	g = &Row{Width: 8, Rows: 1}
	out, l := g.Buf(BufOut), g.Buf(BufRow)
	c = g.Loop(8)
	f := g.Iter(c)
	s := g.Fold(c, g.Const(0))
	g.SetFold(s, g.Bin(OpAdd, s, g.LoadAt(l, f)))
	g.Store(out, g.LoadAt(l, f)) // the iterator, read outside its body
	if _, err := g.Emit(Schedule{Group: 32}); err == nil {
		t.Error("an iterator read outside its loop's body emitted")
	}

	g = &Row{Width: 8, Rows: 1}
	out, l = g.Buf(BufOutU32), g.Buf(BufRow)
	e := g.Column()
	key := func(i int) int { return g.FoldMax(4, func(j int) int { return g.LoadAt(l, g.UBin(OpAdd, i, j)) }) }
	g.StoreAt(out, e, g.Rank(4, e, key(e), key))
	if _, err := g.Emit(Schedule{Group: 32}); err != nil {
		t.Errorf("a rank over folded keys: %v", err)
	}
}
