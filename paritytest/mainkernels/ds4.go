package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// DeepSeek V4 on the device (nn.LayerPlan.DS4; engine/model/ds4.go is the
// graph and jit/gpu/tier/ds4.go the wiring).
//
// The residual is DS4Streams streams, stream-major: stream k of row r at
// (k*rows + r)*d. Each sublayer is wrapped in a hyper-connection:
//
//	HCFlat      the row's streams side by side, which an unweighted RMSNorm
//	            and the F32 fn (RouterMatVecTok) turn into the mixes
//	HCMix       the mixes into the collapse weights, the placements and the
//	            Sinkhorn-normalised stream mixer, one thread a row
//	HCCollapse  the sublayer's input, sum_k pre_k x_k
//	HCPost      every stream's next value, post_k*y + sum_j comb[j][k] x_j
//
// The attention reads a key per window position and an entry per `rate`
// positions (engine/model/ds4.go has why): RoPETail turns the last nRot
// dimensions of each head, DS4ApeAdd adds the compressor's position bias,
// DS4Pool folds a closing window of the paged rows into an entry, which
// DS4EntWrite caches in the entries' own pool; DS4Gather copies a row's keys
// (window, then entries) into scratch pages that the paged attention reads
// through DS4Desc's descriptors. DS4EntDesc is the indexer's descriptor over
// the entries, and DS4RouteBias the router's per-row selection bias.

// DS4Streams is the hyper-connection count HCMix is unrolled for.
const DS4Streams = 4

// ds4Mix is one row's mixer width: pre and post, DS4Streams each, then the
// DS4Streams x DS4Streams mixer.
const ds4Mix = (2 + DS4Streams) * DS4Streams

func flatID(b *ir.Builder, n int) ir.Value {
	return b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
}

func negInfF32(b *ir.Builder) ir.Value {
	return b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
}

// HCFlat writes each row's streams side by side: out[r][k*d+i] = x[(k*rows +
// r)*d + i]. Params pX, pOut; rows*DS4Streams*d threads.
func HCFlat(d, rows int) (*ir.Kernel, error) {
	if d < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: HCFlat: d=%d rows=%d", d, rows)
	}
	s := DS4Streams
	b := ir.New("hcflat", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*s*d)
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(d)))
	k := b.Rem(ir.U32, b.Div(ir.U32, flat, b.Const(ir.U32, int64(d))), b.Const(ir.U32, int64(s)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(s*d)))
	src := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, k, b.Const(ir.U32, int64(rows))), r),
		b.Const(ir.U32, int64(d))), i)
	b.Store(pOut, flat, b.Load(ir.F32, pX, src, 0), 0)
	return b.Done(), nil
}

// HCMix is the hyper-connection mixer, one thread a row (jit/cpu/hc_const.go
// is the contract the host kernels share):
//
//	pre[i]  = sigma(m[i]*s[i] + b[i]) + eps
//	post[i] = 2*sigma(m[4+i]*s[4+i] + b[4+i])
//	comb    = softmax_rows(m[8:]*s[8:] + b[8:]) + eps, then divided by its
//	          column sums + eps, then iters-1 times by its row sums + eps and
//	          its column sums + eps
//
// iters 0 stops after the softmax. Params pMix [rows][24], pScale [24],
// pBase [24], pOut [rows][24]; rows threads in groups of 128, the width the
// tier launches it at: Vulkan runs the workgroup the kernel declares, not the
// launch's, so a 64-wide one computed only half of each group's rows there.
func HCMix(rows, iters int, eps float32) (*ir.Kernel, error) {
	if rows < 1 || iters < 0 {
		return nil, fmt.Errorf("kernels: HCMix: rows=%d iters=%d", rows, iters)
	}
	h := DS4Streams
	b := ir.New("hcmix", [3]int{128, 1, 1})
	pMix := b.Param("pMix", ir.F32)
	pScale := b.Param("pScale", ir.F32)
	pBase := b.Param("pBase", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	r := flatID(b, rows)
	base := b.Mul(ir.U32, r, b.Const(ir.U32, ds4Mix))
	one, two, ep := b.ConstF32(1), b.ConstF32(2), b.ConstF32(eps)
	af := func(i int) ir.Value {
		return b.Fma(b.Load(ir.F32, pMix, base, int64(i)), b.Load(ir.F32, pScale, b.Const(ir.U32, 0), int64(i)),
			b.Load(ir.F32, pBase, b.Const(ir.U32, 0), int64(i)))
	}
	sig := func(x ir.Value) ir.Value {
		return b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), x))))
	}
	for i := 0; i < h; i++ {
		b.Store(pOut, base, b.Add(ir.F32, sig(af(i)), ep), int64(i))
		b.Store(pOut, base, b.Mul(ir.F32, two, sig(af(h+i))), int64(h+i))
	}
	cm := make([]ir.Value, h*h)
	for i := 0; i < h; i++ {
		row := make([]ir.Value, h)
		for j := 0; j < h; j++ {
			row[j] = af(2*h + i*h + j)
		}
		mx := row[0]
		for j := 1; j < h; j++ {
			mx = b.Max(ir.F32, mx, row[j])
		}
		for j := 0; j < h; j++ {
			row[j] = b.Exp(b.Sub(ir.F32, row[j], mx))
		}
		sum := row[0]
		for j := 1; j < h; j++ {
			sum = b.Add(ir.F32, sum, row[j])
		}
		for j := 0; j < h; j++ {
			cm[i*h+j] = b.Add(ir.F32, b.Div(ir.F32, row[j], sum), ep)
		}
	}
	// norm divides each row (or column) by its sum + eps.
	norm := func(v []ir.Value, cols bool) []ir.Value {
		out := make([]ir.Value, h*h)
		for a := 0; a < h; a++ {
			at := func(c int) int {
				if cols {
					return c*h + a
				}
				return a*h + c
			}
			sum := v[at(0)]
			for c := 1; c < h; c++ {
				sum = b.Add(ir.F32, sum, v[at(c)])
			}
			sum = b.Add(ir.F32, sum, ep)
			for c := 0; c < h; c++ {
				out[at(c)] = b.Div(ir.F32, v[at(c)], sum)
			}
		}
		return out
	}
	if iters > 0 {
		cm = norm(cm, true)
	}
	if iters > 1 {
		// The rounds in a loop of iters-1 with the 16 entries carried: every
		// seed is emitted before the loop (ir.Validate).
		phis := make([]ir.Value, h*h)
		b.Loop(int64(iters - 1))
		for i := range phis {
			phis[i] = b.Phi(ir.F32, cm[i])
		}
		nx := norm(norm(phis, false), true)
		for i := range phis {
			b.SetPhi(phis[i], nx[i])
		}
		b.EndLoop()
		cm = phis
	}
	for i := 0; i < h*h; i++ {
		b.Store(pOut, base, cm[i], int64(2*h+i))
	}
	return b.Done(), nil
}

// HCCollapse is each row's sublayer input, sum_k pre_k x_k, pre the mixer's
// first four outputs. Params pX (the streams), pHC [rows][24], pOut
// [rows][d]; rows*d threads.
func HCCollapse(d, rows int) (*ir.Kernel, error) {
	if d < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: HCCollapse: d=%d rows=%d", d, rows)
	}
	b := ir.New("hccollapse", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pHC := b.Param("pHC", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*d)
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(d)))
	hb := b.Mul(ir.U32, r, b.Const(ir.U32, ds4Mix))
	acc := b.ConstF32(0)
	for k := 0; k < DS4Streams; k++ {
		x := b.Load(ir.F32, pX, b.Add(ir.U32, flat, b.Const(ir.U32, int64(k*rows*d))), 0)
		acc = b.Fma(b.Load(ir.F32, pHC, hb, int64(k)), x, acc)
	}
	b.Store(pOut, flat, acc, 0)
	return b.Done(), nil
}

// HCPost is every stream's next value from the sublayer output y and the
// streams before it: x'_k = post_k*y + sum_j comb[j][k] x_j (rowsFault reads
// comb[k][j], a gate's violation). Params pX, pY [rows][d], pHC, pOut (the
// streams); DS4Streams*rows*d threads.
func HCPost(d, rows int, rowsFault bool) (*ir.Kernel, error) {
	if d < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: HCPost: d=%d rows=%d", d, rows)
	}
	h := DS4Streams
	b := ir.New("hcpost", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pY := b.Param("pY", ir.F32)
	pHC := b.Param("pHC", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, h*rows*d)
	rd := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(rows*d)))
	k := b.Div(ir.U32, flat, b.Const(ir.U32, int64(rows*d)))
	r := b.Div(ir.U32, rd, b.Const(ir.U32, int64(d)))
	hb := b.Mul(ir.U32, r, b.Const(ir.U32, ds4Mix))
	acc := b.Mul(ir.F32, b.Load(ir.F32, pHC, b.Add(ir.U32, hb, b.Add(ir.U32, k, b.Const(ir.U32, int64(h)))), 0),
		b.Load(ir.F32, pY, rd, 0))
	for j := 0; j < h; j++ {
		// comb[j][k] is at 2h + j*h + k; the violation reads comb[k][j].
		ci := b.Add(ir.U32, k, b.Const(ir.U32, int64(2*h+j*h)))
		if rowsFault {
			ci = b.Add(ir.U32, b.Mul(ir.U32, k, b.Const(ir.U32, int64(h))), b.Const(ir.U32, int64(2*h+j)))
		}
		x := b.Load(ir.F32, pX, rd, int64(j*rows*d))
		acc = b.Fma(b.Load(ir.F32, pHC, b.Add(ir.U32, hb, ci), 0), x, acc)
	}
	b.Store(pOut, flat, acc, 0)
	return b.Done(), nil
}

// RoPETail turns the last nRot dimensions of each of heads heads hd wide in
// adjacent pairs, at row r's table pCS[r*nRot:] (cos, sin interleaved), and
// copies the rest; inverse turns them back (sin negated). Params pX, pCS,
// pOut, each row heads*hd wide; rows*heads*hd threads.
func RoPETail(heads, hd, nRot, rows int, inverse bool) (*ir.Kernel, error) {
	if heads < 1 || rows < 1 || nRot < 2 || nRot%2 != 0 || nRot > hd {
		return nil, fmt.Errorf("kernels: RoPETail: heads=%d hd=%d nRot=%d rows=%d", heads, hd, nRot, rows)
	}
	b := ir.New("ropetail", [3]int{128, 1, 1})
	pX := b.Param("pX", ir.F32)
	pCS := b.Param("pCS", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*heads*hd)
	e := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(hd)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(heads*hd)))
	nope := hd - nRot
	one := b.Const(ir.U32, 1)
	// t is the dimension's place in the tail, 0 in the nope half; a nope
	// dimension's "pair" is itself and the next, which nope <= hd-2 keeps in
	// its head, and its rotation is selected away.
	isNope := b.Lt(ir.U32, e, b.Const(ir.U32, int64(nope)))
	t := b.Select(ir.U32, isNope, b.Const(ir.U32, 0), b.Sub(ir.U32, e, b.Const(ir.U32, int64(nope))))
	odd := b.And(ir.U32, t, one)
	pair := b.Sub(ir.U32, flat, odd)
	ci := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(nRot))), b.Sub(ir.U32, t, odd))
	c := b.Load(ir.F32, pCS, ci, 0)
	s := b.Load(ir.F32, pCS, ci, 1)
	if inverse {
		s = b.Sub(ir.F32, b.ConstF32(0), s)
	}
	x0 := b.Load(ir.F32, pX, pair, 0)
	x1 := b.Load(ir.F32, pX, pair, 1)
	even := b.Sub(ir.F32, b.Mul(ir.F32, x0, c), b.Mul(ir.F32, x1, s))
	oddV := b.Add(ir.F32, b.Mul(ir.F32, x0, s), b.Mul(ir.F32, x1, c))
	rot := b.Select(ir.F32, b.Lt(ir.U32, b.Const(ir.U32, 0), odd), oddV, even)
	b.Store(pOut, flat, b.Select(ir.F32, isNope, b.Load(ir.F32, pX, flat, 0), rot), 0)
	return b.Done(), nil
}

// ds4Pos is row r's position: its descriptor's write position.
func ds4Pos(b *ir.Builder, pRow, r ir.Value) ir.Value { return pagedDesc(b, pRow, r, PRowWrite) }

// DS4ApeAdd adds the compressor's position bias: out[r][i] = g[r][i] +
// ape[(t_r % rate)*n + i], t_r row r's position. Params pG, pAPE, pRow, pOut;
// rows*n threads.
func DS4ApeAdd(n, rate, rows int) (*ir.Kernel, error) {
	if n < 1 || rate < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: DS4ApeAdd: n=%d rate=%d rows=%d", n, rate, rows)
	}
	b := ir.New("ds4apeadd", [3]int{128, 1, 1})
	pG := b.Param("pG", ir.F32)
	pAPE := b.Param("pAPE", ir.F32)
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*n)
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(n)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(n)))
	ph := b.Rem(ir.U32, ds4Pos(b, pRow, r), b.Const(ir.U32, int64(rate)))
	ai := b.Add(ir.U32, b.Mul(ir.U32, ph, b.Const(ir.U32, int64(n))), i)
	b.Store(pOut, flat, b.Add(ir.F32, b.Load(ir.F32, pG, flat, 0), b.Load(ir.F32, pAPE, ai, 0)), 0)
	return b.Done(), nil
}

// ds4Closing returns, for row r at position t, whether t closes an entry
// ((t+1) % rate == 0) and the entry it closes, (t+1)/rate - 1 (0 when none).
func ds4Closing(b *ir.Builder, t ir.Value, rate int) (closing, w ir.Value) {
	one := b.Const(ir.U32, 1)
	t1 := b.Add(ir.U32, t, one)
	rr := b.Const(ir.U32, int64(rate))
	closing = b.Lt(ir.U32, b.Rem(ir.U32, t1, rr), one)
	c := b.Div(ir.U32, t1, rr)
	w = b.Sub(ir.U32, b.Max(ir.U32, c, one), one)
	return closing, w
}

// poolRow is position t's row's first element in a paged row-major pool.
func poolRow(b *ir.Builder, pTab, tabOff, t ir.Value, page, rowW int) ir.Value {
	pid := pageOf(b, pTab, tabOff, t, page)
	return b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pid, b.Const(ir.U32, int64(page))), pageRem(b, t, page)),
		b.Const(ir.U32, int64(rowW)))
}

// DS4Pool folds each row's closing window into an entry, before its norm:
// per channel, the softmax over the window's slots of the gate weighting the
// kv,
//
//	out[r][i] = sum_s e[s] kv[s] / sum_s e[s],  e[s] = exp(gate[s] - max gate)
//
// read from the paged rows (rowW wide, kv at column kvo, gate at gto, wd
// channels). The window of entry w is positions w*rate..w*rate+rate-1; with
// overlap (CSA) a row's kv and gate are two heads, and the entry pools the
// window before's first head (w > 0, unless noOverlap) then its own second.
// A row that closes nothing pools positions clamped to its own; DS4EntWrite
// sends that to the dummy row. Params pPool, pTab, pRow, pOut [rows][wd];
// rows*wd threads.
func DS4Pool(rows, page, rowW, kvo, gto, wd, rate int, overlap, noOverlap bool) (*ir.Kernel, error) {
	if err := checkPage("DS4Pool", page); err != nil {
		return nil, err
	}
	half := 0
	if overlap {
		half = wd
	}
	if rows < 1 || wd < 1 || rate < 1 || kvo+half+wd > rowW || gto+half+wd > rowW {
		return nil, fmt.Errorf("kernels: DS4Pool: rows=%d row=%d kv=%d gate=%d wd=%d rate=%d", rows, rowW, kvo,
			gto, wd, rate)
	}
	b := ir.New("ds4pool", [3]int{128, 1, 1})
	pPool := b.Param("pPool", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*wd)
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(wd)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(wd)))
	t := ds4Pos(b, pRow, r)
	tab := pagedDesc(b, pRow, r, PRowTab)
	_, w := ds4Closing(b, t, rate)
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	rr := b.Const(ir.U32, int64(rate))
	p0 := b.Mul(ir.U32, w, rr)
	// The slots: n of them, the first rate the window before's when the
	// entry overlaps. A slot is position p and column offset co, and a slot
	// with no window before (w == 0) is masked to -inf.
	n := rate
	prev := overlap && !noOverlap
	if prev {
		n = 2 * rate
	}
	slot := func(s ir.Value) (pos, co ir.Value, live ir.Value) {
		co = b.Const(ir.U32, int64(half))
		pos = b.Add(ir.U32, p0, s)
		liveU := one
		if prev {
			// Slot s < rate is the window before's position p0-rate+s and its
			// first head; slot s >= rate this window's p0+s-rate and its
			// second. Both are p0+s-rate, clamped at 0.
			isPrev := b.Lt(ir.U32, s, rr)
			pos = b.Sub(ir.U32, b.Max(ir.U32, b.Add(ir.U32, p0, s), rr), rr)
			co = b.Select(ir.U32, isPrev, zero, co)
			liveU = b.Select(ir.U32, isPrev, b.Select(ir.U32, b.Lt(ir.U32, zero, w), one, zero), one)
		}
		pos = b.Min(ir.U32, pos, t)
		return pos, co, b.Lt(ir.U32, zero, liveU)
	}
	ninf := negInfF32(b)
	// Seeds hoisted: ir.Validate refuses a Phi whose init sits in its loop.
	b.Loop(int64(n))
	s := b.Phi(ir.U32, zero)
	mx := b.Phi(ir.F32, ninf)
	pos, co, live := slot(s)
	at := b.Add(ir.U32, poolRow(b, pTab, tab, pos, page, rowW), b.Add(ir.U32, co, i))
	g := b.Select(ir.F32, live, b.Load(ir.F32, pPool, at, int64(gto)), ninf)
	b.SetPhi(mx, b.Max(ir.F32, mx, g))
	b.SetPhi(s, b.Add(ir.U32, s, one))
	b.EndLoop()
	zf := b.ConstF32(0)
	b.Loop(int64(n))
	s2 := b.Phi(ir.U32, zero)
	sum := b.Phi(ir.F32, zf)
	acc := b.Phi(ir.F32, zf)
	pos2, co2, live2 := slot(s2)
	at2 := b.Add(ir.U32, poolRow(b, pTab, tab, pos2, page, rowW), b.Add(ir.U32, co2, i))
	e := b.Select(ir.F32, live2, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pPool, at2, int64(gto)), mx)), zf)
	b.SetPhi(sum, b.Add(ir.F32, sum, e))
	b.SetPhi(acc, b.Fma(e, b.Load(ir.F32, pPool, at2, int64(kvo)), acc))
	b.SetPhi(s2, b.Add(ir.U32, s2, one))
	b.EndLoop()
	b.Store(pOut, flat, b.Div(ir.F32, acc, sum), 0)
	return b.Done(), nil
}

// DS4EntWrite caches each row's n floats at column col of the entry its
// position closes, in the entries' pool (entW wide, its table at the row's
// descriptor's offset); a row that closes nothing writes the dummy page's
// first row. Params pSrc [rows][n], pDst (the entries' pool), pTab (theirs),
// pRow; rows*n threads.
func DS4EntWrite(n, rows, page, entW, col, rate int) (*ir.Kernel, error) {
	if err := checkPage("DS4EntWrite", page); err != nil {
		return nil, err
	}
	if n < 1 || rows < 1 || rate < 1 || col < 0 || col+n > entW {
		return nil, fmt.Errorf("kernels: DS4EntWrite: n=%d rows=%d entW=%d col=%d rate=%d", n, rows, entW, col, rate)
	}
	b := ir.New("ds4entwrite", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	pTab := b.Param("pTab", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	flat := flatID(b, rows*n)
	i := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(n)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(n)))
	closing, w := ds4Closing(b, ds4Pos(b, pRow, r), rate)
	zero := b.Const(ir.U32, 0)
	row := b.Select(ir.U32, closing, poolRow(b, pTab, pagedDesc(b, pRow, r, PRowTab), w, page, entW), zero)
	b.Store(pDst, b.Add(ir.U32, row, b.Add(ir.U32, i, b.Const(ir.U32, int64(col)))), b.Load(ir.F32, pSrc, flat, 0), 0)
	return b.Done(), nil
}

// ds4Visible is how many entries row r at t may read: (t+1)/rate.
func ds4Visible(b *ir.Builder, t ir.Value, rate int) ir.Value {
	return b.Div(ir.U32, b.Add(ir.U32, t, b.Const(ir.U32, 1)), b.Const(ir.U32, int64(rate)))
}

// DS4EntDesc writes each row's descriptor over its visible entries -- the
// sequence's table, keys 0..(t+1)/rate -- which the indexer's scores and
// selection read (IdxScoresPaged, IdxSelect). Params pRow, pOut;
// rows*PRowWords threads.
func DS4EntDesc(rows, rate int) (*ir.Kernel, error) {
	if rows < 1 || rate < 1 {
		return nil, fmt.Errorf("kernels: DS4EntDesc: rows=%d rate=%d", rows, rate)
	}
	b := ir.New("ds4entdesc", [3]int{128, 1, 1})
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.U32)
	flat := flatID(b, rows*PRowWords)
	wd := b.Rem(ir.U32, flat, b.Const(ir.U32, PRowWords))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, PRowWords))
	zero := b.Const(ir.U32, 0)
	tab := pagedDesc(b, pRow, r, PRowTab)
	nv := ds4Visible(b, ds4Pos(b, pRow, r), rate)
	v := b.Select(ir.U32, b.Lt(ir.U32, wd, b.Const(ir.U32, PRowTab+1)), tab,
		b.Select(ir.U32, b.Lt(ir.U32, wd, b.Const(ir.U32, PRowEnd)), zero,
			b.Select(ir.U32, b.Lt(ir.U32, wd, b.Const(ir.U32, PRowEnd+1)), nv, zero)))
	b.Store(pOut, flat, v, 0)
	return b.Done(), nil
}

// DS4Keys is how a row's keys are chosen: its window of W positions, then
// its entries -- none (a sliding block, or the WindowOnly fault), every
// visible one (HCA), or the indexer's top k of them (CSA).
type DS4Keys struct {
	Window, Rate, TopK int
	All, None          bool
}

func (k DS4Keys) counts(b *ir.Builder, t ir.Value) (nw, ne ir.Value) {
	one := b.Const(ir.U32, 1)
	t1 := b.Add(ir.U32, t, one)
	nw = b.Min(ir.U32, t1, b.Const(ir.U32, int64(k.Window)))
	switch {
	case k.None || k.Rate == 0:
		ne = b.Const(ir.U32, 0)
	case k.All:
		ne = ds4Visible(b, t, k.Rate)
	default:
		ne = b.Min(ir.U32, ds4Visible(b, t, k.Rate), b.Const(ir.U32, int64(k.TopK)))
	}
	return nw, ne
}

// DS4GatherPages is the gathered pages a row takes for kmax keys.
func DS4GatherPages(kmax, page int) int { return (kmax + page - 1) / page }

// DS4Gather copies each row's keys into its pages of a scratch pool hd wide:
// key j < nw is window position t+1-nw+j's first hd floats from the rows'
// pool (rowT wide), key nw+e an entry's from the entries' pool (rowE wide)
// -- entry e itself, or with a top-k the e-th of the selection pSel [rows]
// [TopK+1]. Key j of row r lands at page r*DS4GatherPages + j/P. A slot past
// the row's count copies something in bounds that no descriptor reads.
// Params pTok, pTabT, pEnt, pTabE, pRow, pSel, pDst; rows*kmax*hd threads.
func DS4Gather(rows, kmax, hd, page, rowT, rowE int, k DS4Keys) (*ir.Kernel, error) {
	if err := checkPage("DS4Gather", page); err != nil {
		return nil, err
	}
	if rows < 1 || kmax < 1 || hd < 1 || hd > rowT || (rowE > 0 && hd > rowE) || k.Window < 1 {
		return nil, fmt.Errorf("kernels: DS4Gather: rows=%d kmax=%d hd=%d rowT=%d rowE=%d %+v", rows, kmax,
			hd, rowT, rowE, k)
	}
	ppr := DS4GatherPages(kmax, page)
	b := ir.New("ds4gather", [3]int{128, 1, 1})
	pTok := b.Param("pTok", ir.F32)
	pTabT := b.Param("pTabT", ir.U32)
	pEnt := b.Param("pEnt", ir.F32)
	pTabE := b.Param("pTabE", ir.U32)
	pRow := b.Param("pRow", ir.U32)
	pSel := b.Param("pSel", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	flat := flatID(b, rows*kmax*hd)
	e := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(hd)))
	j := b.Rem(ir.U32, b.Div(ir.U32, flat, b.Const(ir.U32, int64(hd))), b.Const(ir.U32, int64(kmax)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(kmax*hd)))
	t := ds4Pos(b, pRow, r)
	tab := pagedDesc(b, pRow, r, PRowTab)
	nw, ne := k.counts(b, t)
	one := b.Const(ir.U32, 1)
	// The window's key: position t+1-nw+j, clamped to t.
	tp := b.Min(ir.U32, b.Add(ir.U32, b.Sub(ir.U32, b.Add(ir.U32, t, one), nw), j), t)
	kv := b.Load(ir.F32, pTok, b.Add(ir.U32, poolRow(b, pTabT, tab, tp, page, rowT), e), 0)
	val := kv
	if !(k.None || k.Rate == 0) {
		// The entry's: the (j-nw)-th, clamped into the kept ones.
		ei := b.Sub(ir.U32, b.Max(ir.U32, j, nw), nw)
		ei = b.Min(ir.U32, ei, b.Sub(ir.U32, b.Max(ir.U32, ne, one), one))
		if !k.All {
			ei = b.Load(ir.U32, pSel, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(k.TopK+1))), ei), 0)
			ei = b.Min(ir.U32, ei, b.Sub(ir.U32, b.Max(ir.U32, ds4Visible(b, t, k.Rate), one), one))
		}
		ev := b.Load(ir.F32, pEnt, b.Add(ir.U32, poolRow(b, pTabE, tab, ei, page, rowE), e), 0)
		val = b.Select(ir.F32, b.Lt(ir.U32, j, nw), kv, ev)
	}
	dst := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(ppr*page))), j),
		b.Const(ir.U32, int64(hd))), e)
	b.Store(pDst, dst, val, 0)
	return b.Done(), nil
}

// DS4Desc writes the gathered pages' descriptors: row r's table at
// r*DS4GatherPages, its keys 0..nw+ne, written nowhere. Params pRow, pOut;
// rows*PRowWords threads.
func DS4Desc(rows, kmax, page int, k DS4Keys) (*ir.Kernel, error) {
	if rows < 1 || kmax < 1 || k.Window < 1 {
		return nil, fmt.Errorf("kernels: DS4Desc: rows=%d kmax=%d %+v", rows, kmax, k)
	}
	b := ir.New("ds4desc", [3]int{128, 1, 1})
	pRow := b.Param("pRow", ir.U32)
	pOut := b.Param("pOut", ir.U32)
	flat := flatID(b, rows*PRowWords)
	wd := b.Rem(ir.U32, flat, b.Const(ir.U32, PRowWords))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, PRowWords))
	zero := b.Const(ir.U32, 0)
	nw, ne := k.counts(b, ds4Pos(b, pRow, r))
	n := b.Min(ir.U32, b.Add(ir.U32, nw, ne), b.Const(ir.U32, int64(kmax)))
	tab := b.Mul(ir.U32, r, b.Const(ir.U32, int64(DS4GatherPages(kmax, page))))
	v := b.Select(ir.U32, b.Lt(ir.U32, wd, b.Const(ir.U32, PRowTab+1)), tab,
		b.Select(ir.U32, b.Lt(ir.U32, wd, b.Const(ir.U32, PRowEnd)), zero,
			b.Select(ir.U32, b.Lt(ir.U32, wd, b.Const(ir.U32, PRowEnd+1)), n, zero)))
	b.Store(pOut, flat, v, 0)
	return b.Done(), nil
}

// DS4RouteBias writes each row's selection bias for the router
// (MoERoute.RowBias): with a table, zero at the k experts of the row's token
// (pTab [vocab][k], ids as floats) and -inf elsewhere; without, the block's
// shared bias pB copied to every row. Params pTab or pB, then pIDs (with a
// table), pOut [rows][nExpert]; rows*nExpert threads.
func DS4RouteBias(nExpert, k, vocab, rows int, table bool) (*ir.Kernel, error) {
	if nExpert < 1 || rows < 1 || (table && (k < 1 || vocab < 1)) {
		return nil, fmt.Errorf("kernels: DS4RouteBias: nExpert=%d k=%d vocab=%d rows=%d", nExpert, k, vocab, rows)
	}
	b := ir.New("ds4routebias", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	var pIDs ir.Value
	if table {
		pIDs = b.Param("pIDs", ir.U32)
	}
	pOut := b.Param("pOut", ir.F32)
	flat := flatID(b, rows*nExpert)
	e := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(nExpert)))
	if !table {
		b.Store(pOut, flat, b.Load(ir.F32, pSrc, e, 0), 0)
		return b.Done(), nil
	}
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(nExpert)))
	id := b.Min(ir.U32, b.Load(ir.U32, pIDs, r, 0), b.Const(ir.U32, int64(vocab-1)))
	base := b.Mul(ir.U32, id, b.Const(ir.U32, int64(k)))
	ef := b.CvtF32(e)
	zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
	b.Loop(int64(k))
	i := b.Phi(ir.U32, zero)
	hit := b.Phi(ir.U32, zero)
	v := b.Load(ir.F32, pSrc, b.Add(ir.U32, base, i), 0)
	ne := b.Add(ir.U32, b.Select(ir.U32, b.Lt(ir.F32, v, ef), one, zero), b.Select(ir.U32, b.Lt(ir.F32, ef, v), one, zero))
	b.SetPhi(hit, b.Add(ir.U32, hit, b.Sub(ir.U32, one, ne)))
	b.SetPhi(i, b.Add(ir.U32, i, one))
	b.EndLoop()
	b.Store(pOut, flat, b.Select(ir.F32, b.Lt(ir.U32, zero, hit), b.ConstF32(0), negInfF32(b)), 0)
	return b.Done(), nil
}
