package compose

// unrolled is the unroll pass under schedule s: each constant-trip loop
// whose folds a later node reads is unrolled, fully where its trips times
// the tile's columns stay at or below s.UnrollOver, else as a loop of
// blocks of s.Unroll trips with the remaining trips unrolled after it; each
// fold carried as s.Chains chains, trip d's step on chain d%Chains, the
// chains summed left to right where the loop's result is read. s.Unroll 0
// leaves the group as it is.
func (g *Row) unrolled(s Schedule) *Row {
	if s.Unroll <= 0 {
		return g
	}
	r := g
	for c := len(g.Loops) - 1; c >= 0; c-- {
		if r.Loops[c] > 0 {
			r = r.unrollLoop(c, s)
		}
	}
	return r
}

func (g *Row) unrollLoop(c int, s Schedule) *Row {
	r := g.clone()
	u, ub := r.UConst, r.UBin
	T, B, k := r.Loops[c], s.Unroll, max(1, s.Chains)
	vary, body := r.varies(c), r.bodyOf(c)
	iter := -1
	var folds []int
	for i, nd := range r.Nodes {
		switch {
		case nd.Kind == NIter && nd.At == c:
			iter = i
		case nd.Kind == NFold && nd.At == c:
			folds = append(folds, i)
		}
	}
	if len(folds) == 0 {
		return g
	}
	steps := map[int]int{}
	for _, f := range folds {
		steps[f] = r.Nodes[f].A
	}
	// step is fold f's step at trip index node d, the fold read as acc.
	step := func(f, acc, d int) int {
		memo := map[int]int{}
		var at func(n int) int
		at = func(n int) int {
			switch {
			case n == f:
				return acc
			case n == iter:
				return d
			case !vary[n]:
				return n
			}
			if m, ok := memo[n]; ok {
				return m
			}
			m := r.copyWith(n, at)
			memo[n] = m
			return m
		}
		return at(steps[f])
	}
	chains := func(init func(f, x int) int) map[int][]int {
		out := map[int][]int{}
		for _, f := range folds {
			for x := 0; x < k; x++ {
				out[f] = append(out[f], init(f, x))
			}
		}
		return out
	}
	start := func(f, x int) int {
		if x == 0 {
			return r.Nodes[f].B
		}
		return r.Const(0)
	}
	run := func(cur map[int][]int, d0 int, base int, n int) {
		for kk := 0; kk < n; kk++ {
			d := u(d0 + kk)
			if base >= 0 {
				d = ub(OpAdd, base, u(kk))
			}
			for _, f := range folds {
				x := (d0 + kk) % k
				cur[f][x] = step(f, cur[f][x], d)
			}
		}
	}
	nb := T / B
	blocked := s.UnrollOver > 0 && max(s.TileCols, 1)*T > s.UnrollOver && nb > 1
	var cur map[int][]int
	if blocked {
		// The loop over blocks: a chain a fold, each block's B trips on
		// them; B a multiple of the chains keeps trip d on chain d%k.
		fs := chains(func(f, x int) int {
			if x == 0 {
				return f
			}
			return r.Add(Node{Kind: NFold, At: c, B: r.Const(0)})
		})
		next := chains(func(f, x int) int { return fs[f][x] })
		run(next, 0, ub(OpMul, iter, u(B)), B)
		for _, f := range folds {
			for x := range fs[f] {
				r.Nodes[fs[f][x]].A = next[f][x]
			}
		}
		r.Loops[c] = nb
		cur = chains(func(f, x int) int { return fs[f][x] })
		run(cur, nb*B, -1, T-nb*B)
	} else {
		cur = chains(start)
		run(cur, 0, -1, T)
	}
	result := map[int]int{}
	for _, f := range folds {
		v := cur[f][0]
		for x := 1; x < k; x++ {
			v = r.Bin(OpAdd, v, cur[f][x])
		}
		result[f] = v
	}
	// What read a fold after the loop reads its result.
	for i := range r.Nodes {
		if i >= len(body) || body[i] {
			continue // the loop's own nodes, and the pass's, read the folds as meant
		}
		if nd := &r.Nodes[i]; nd.Kind == NFold && nd.At == c {
			continue
		}
		for _, f := range folds {
			r.Nodes[i].replaceOperand(f, result[f])
		}
	}
	for j, st := range r.Stores {
		for _, f := range folds {
			if st.Val == f {
				r.Stores[j].Val = result[f]
			}
		}
	}
	if !blocked {
		r.removeLoop(c, iter, folds)
	}
	return r
}

// removeLoop drops loop c, its iterator and folds no longer read, and
// renumbers the loops after it.
func (r *Row) removeLoop(c, iter int, folds []int) {
	if iter >= 0 {
		r.Nodes[iter] = Node{Kind: NUConst}
	}
	for _, f := range folds {
		r.Nodes[f] = Node{Kind: NConst}
	}
	r.Loops = append(r.Loops[:c], r.Loops[c+1:]...)
	r.Trip = append(r.Trip[:c], r.Trip[c+1:]...)
	if c < len(r.Ranges) {
		r.Ranges = append(r.Ranges[:c], r.Ranges[c+1:]...)
	}
	for i := range r.Nodes {
		if nd := &r.Nodes[i]; (nd.Kind == NIter || nd.Kind == NFold) && nd.At > c {
			nd.At--
		}
	}
}
