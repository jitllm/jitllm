package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// The recurrent half of a hybrid block, as device kernels.
//
// ir.Validate refuses a param that is both loaded and stored (surplus threads
// would apply an update twice); that constrains one dispatch, not recurrence.
// These kernels are out of place: the state comes in through one param and
// leaves through another, and the caller swaps them. The convolution needs no
// loop at all: over a chunk it is a sliding window over a fixed sequence, so
// only the delta rule's state genuinely recurs.

// A device keeps every session's recurrent states in one pool per block, a
// SLOT being one sequence's state: taps-1 planes of the convolution's window,
// or the delta rule's VHeads*VDim*KDim. Every kernel that touches a state
// reads where from pN, the recurrent descriptor:
//
//	pN[0]        the real row count (Conv1dShift and the scans' chunk length)
//	pN[1+2j]     sequence j's IN slot: its state is read there
//	pN[2+2j]     sequence j's OUT slot: its next state is written there
//
// The pool is two buffers and a step reads one and writes the other, so the
// two slots are the same in the doubled form; the shared form reads its one
// state buffer at the in slot and writes a compact next-state scratch at the
// out slot (CopySlots moves it home). The convolution's window is always
// doubled and uses the in slot on both sides. A chunk is one sequence (j =
// 0); a step of several sequences reads the run descriptor below, whose
// first words are these.

// RecDescWords is the descriptor's length for seqs sequences, in u32 words.
func RecDescWords(seqs int) int { return 1 + 2*seqs }

// slotIn and slotOut read sequence j's slots from the descriptor.
func slotIn(b *ir.Builder, pN, j ir.Value) ir.Value {
	return b.Load(ir.U32, pN, b.Add(ir.U32, j, j), 1)
}

func slotOut(b *ir.Builder, pN, j ir.Value) ir.Value {
	return b.Load(ir.U32, pN, b.Add(ir.U32, j, j), 2)
}

// Conv1dRows is the causal depthwise convolution over a chunk of rows.
//
// Z is the concatenation [S_in (taps-1 planes) ; X (m planes)] along t, and
//
//	out[t][c] = sum_u W[u][c] * Z[t+u][c]
//
// The host keeps the state plane-major (st[t*chans+ch]) and transposes the
// weights at load; this keeps both, so migrating the state is a memcpy.
//
// A padded chunk submits surplus zero rows. Attention masks them, but a zero
// column shifted into a recurrent state is one the model never produced, so
// the real row count m is read from a buffer (by Conv1dShift) rather than
// baked.
func Conv1dRows(taps, chans, rows int) (*ir.Kernel, error) {
	if taps < 2 || chans <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: Conv1dRows taps=%d chans=%d rows=%d", taps, chans, rows)
	}
	b := ir.New("conv1drows", [3]int{128, 1, 1})
	pS := b.Param("pS", ir.F32) // taps-1 planes of state, plane-major
	pX := b.Param("pX", ir.F32) // rows planes of input
	pW := b.Param("pW", ir.F32) // taps planes of weight
	pOut := b.Param("pOut", ir.F32)
	pN := b.Param("pN", ir.U32) // the recurrent descriptor

	n := int64(rows * chans)
	idx := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, n-1))
	cw := b.Const(ir.U32, int64(chans))
	t := b.Div(ir.U32, idx, cw)
	c := b.Sub(ir.U32, idx, b.Mul(ir.U32, t, cw))
	pre := b.Const(ir.U32, int64(taps-1))
	base := b.Mul(ir.U32, slotIn(b, pN, b.Const(ir.U32, 0)), b.Const(ir.U32, int64((taps-1)*chans)))

	acc := b.Const(ir.F32, 0)
	for u := 0; u < taps; u++ {
		// z = t + u, into S_in while z < taps-1 and into X after it.
		z := b.Add(ir.U32, t, b.Const(ir.U32, int64(u)))
		inS := b.Lt(ir.U32, z, pre)
		sIdx := b.Add(ir.U32, b.Mul(ir.U32, z, cw), c)
		xIdx := b.Add(ir.U32, b.Mul(ir.U32, b.Sub(ir.U32, z, pre), cw), c)
		// Both loads are clamped and both run: the branch is a Select, and the
		// discarded arm is still a read that must be in range.
		sv := b.Load(ir.F32, pS, b.Add(ir.U32, base,
			b.Min(ir.U32, sIdx, b.Const(ir.U32, int64((taps-1)*chans-1)))), 0)
		xv := b.Load(ir.F32, pX, b.Min(ir.U32, xIdx, b.Const(ir.U32, int64(rows*chans-1))), 0)
		acc = b.Fma(b.Load(ir.F32, pW, b.Add(ir.U32, b.Mul(ir.U32, b.Const(ir.U32, int64(u)), cw), c), 0),
			b.Select(ir.F32, inS, sv, xv), acc)
	}
	// Every row is stored, including a padded one: guarding the store would
	// load and store one param, which ir.Validate refuses. A padded row's
	// output is never copied out; the real row count only decides the state,
	// which is Conv1dShift's job.
	b.Store(pOut, idx, acc, 0)
	return b.Done()
}

// Conv1dShift produces the next state: the last taps-1 real columns of
// [S_in ; X], where the window has moved to after m rows. It is a separate
// kernel so it writes a different buffer and stays out of place.
func Conv1dShift(taps, chans, rows int) (*ir.Kernel, error) {
	if taps < 2 || chans <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: Conv1dShift taps=%d chans=%d rows=%d", taps, chans, rows)
	}
	b := ir.New("conv1dshift", [3]int{128, 1, 1})
	pS := b.Param("pS", ir.F32)
	pX := b.Param("pX", ir.F32)
	pSOut := b.Param("pSOut", ir.F32)
	pN := b.Param("pN", ir.U32)

	pre := int64(taps - 1)
	n := pre * int64(chans)
	idx := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, n-1))
	cw := b.Const(ir.U32, int64(chans))
	t := b.Div(ir.U32, idx, cw)
	c := b.Sub(ir.U32, idx, b.Mul(ir.U32, t, cw))

	// The new state's column t is Z[m + t], with Z = [S_in ; X] and m real rows.
	// The window is read and written at the sequence's in slot.
	m := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	base := b.Mul(ir.U32, slotIn(b, pN, b.Const(ir.U32, 0)), b.Const(ir.U32, n))
	z := b.Add(ir.U32, m, t)
	preV := b.Const(ir.U32, pre)
	inS := b.Lt(ir.U32, z, preV)
	sIdx := b.Add(ir.U32, b.Mul(ir.U32, z, cw), c)
	xIdx := b.Add(ir.U32, b.Mul(ir.U32, b.Sub(ir.U32, z, preV), cw), c)
	sv := b.Load(ir.F32, pS, b.Add(ir.U32, base, b.Min(ir.U32, sIdx, b.Const(ir.U32, n-1))), 0)
	xv := b.Load(ir.F32, pX, b.Min(ir.U32, xIdx, b.Const(ir.U32, int64(rows*chans)-1)), 0)
	b.Store(pSOut, b.Add(ir.U32, base, idx), b.Select(ir.F32, inS, sv, xv), 0)
	return b.Done()
}

// A step whose rows are several sequences -- a batched decode, a step across
// sessions, and a prompt chunk riding beside them -- groups its rows into
// RUNS, a run being one sequence's rows at consecutive positions: a decoding
// sequence is a run of one row, a prompt chunk a run of several. A run chains
// through one state: its first row reads the state at the run's in slot, each
// later row the state the row before it produced, and the run's last row's
// state goes to its out slot. The run forms (DeltaScan.Runs, Conv1dRowsRuns,
// Conv1dShiftRuns), compiled for N rows, read the descriptor past the slots:
//
//	pN[1+2j], pN[2+2j]   run j's in and out slot, j < N
//	pN[1+2N+2j]          run j's first entry in the order list
//	pN[2+2N+2j]          run j's row count, at least one
//	pN[1+4N+i]           the order list: the step's rows run by run, each
//	                     run's rows in position order, so a step's rows may
//	                     stand in any order
//	pN[1+5N+r]           row r's entry in the order list
//	pN[1+6N+r]           row r's run
//
// A step of S < N runs repeats run S-1 in entries S..N-1: the threads those
// entries feed recompute run S-1 and store exactly what its own threads store,
// which is the clamp to the last work item every kernel here relies on.

// RunDescWords is the run descriptor's length for a step compiled for rows
// rows, in u32 words.
func RunDescWords(rows int) int { return 1 + 7*rows }

// runOf reads row r's run and its entry in the order list.
func runOf(b *ir.Builder, pN, r ir.Value, rows int) (run, at ir.Value) {
	return b.Load(ir.U32, pN, r, int64(1+6*rows)), b.Load(ir.U32, pN, r, int64(1+5*rows))
}

// runSpan reads run j's first entry in the order list and its row count.
func runSpan(b *ir.Builder, pN, j ir.Value, rows int) (start, count ir.Value) {
	jj := b.Add(ir.U32, j, j)
	return b.Load(ir.U32, pN, jj, int64(1+2*rows)), b.Load(ir.U32, pN, jj, int64(2+2*rows))
}

// orderRow reads the row at entry i of the order list.
func orderRow(b *ir.Builder, pN, i ir.Value, rows int) ir.Value {
	return b.Load(ir.U32, pN, i, int64(1+4*rows))
}

// Conv1dRowsRuns is Conv1dRows over a step's runs: row r's window is its run's
// state -- taps-1 planes at the run's in slot, in pS -- followed by the run's
// rows up to and including r, so the k-th row of a run reads the state's
// planes k.. and the k rows before it. A run of one row is a batched decode's
// sequence; a longer one is a prompt chunk, convolved as Conv1dRows convolves
// it alone, product for product. Launch rows*chans threads.
func Conv1dRowsRuns(taps, chans, rows int) (*ir.Kernel, error) {
	if taps < 2 || chans <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: Conv1dRowsRuns taps=%d chans=%d rows=%d", taps, chans, rows)
	}
	b := ir.New("conv1drowsruns", [3]int{128, 1, 1})
	pS := b.Param("pS", ir.F32) // the pool's windows, taps-1 planes a slot
	pX := b.Param("pX", ir.F32) // rows input rows
	pW := b.Param("pW", ir.F32) // taps planes of weight
	pOut := b.Param("pOut", ir.F32)
	pN := b.Param("pN", ir.U32) // the run descriptor
	n := int64(rows * chans)
	idx := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, n-1))
	cw := b.Const(ir.U32, int64(chans))
	t := b.Div(ir.U32, idx, cw)
	c := b.Sub(ir.U32, idx, b.Mul(ir.U32, t, cw))
	run, at := runOf(b, pN, t, rows)
	start, _ := runSpan(b, pN, run, rows)
	k := b.Sub(ir.U32, at, start) // the row's place in its run
	pre := b.Const(ir.U32, int64(taps-1))
	sBase := b.Add(ir.U32, b.Mul(ir.U32, slotIn(b, pN, run), b.Const(ir.U32, int64((taps-1)*chans))), c)
	acc := b.Const(ir.F32, 0)
	for u := 0; u < taps; u++ {
		// z = k + u into [S_in ; the run's rows]: the state while z < taps-1,
		// else the run's row z-(taps-1), which is entry at+u-(taps-1) of the
		// order list. Both loads run (the branch is a Select), so each index
		// is clamped into range; the arm not taken is a read, not a value.
		uu := b.Const(ir.U32, int64(u))
		z := b.Add(ir.U32, k, uu)
		inS := b.Lt(ir.U32, z, pre)
		sv := b.Load(ir.F32, pS, b.Add(ir.U32, sBase,
			b.Mul(ir.U32, b.Min(ir.U32, z, b.Const(ir.U32, int64(taps-2))), cw)), 0)
		ent := b.Select(ir.U32, inS, at, b.Sub(ir.U32, b.Add(ir.U32, at, uu), pre))
		xr := orderRow(b, pN, ent, rows)
		xv := b.Load(ir.F32, pX, b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, xr, cw), c), b.Const(ir.U32, n-1)), 0)
		acc = b.Fma(b.Load(ir.F32, pW, b.Add(ir.U32, b.Const(ir.U32, int64(u*chans)), c), 0),
			b.Select(ir.F32, inS, sv, xv), acc)
	}
	b.Store(pOut, idx, acc, 0)
	return b.Done()
}

// Conv1dShiftRuns is Conv1dShift for Conv1dRowsRuns' runs: each run's next
// window is the last taps-1 columns of [its state ; its rows], read and
// written at the run's in slot. Launch rows*(taps-1)*chans threads.
func Conv1dShiftRuns(taps, chans, rows int) (*ir.Kernel, error) {
	if taps < 2 || chans <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: Conv1dShiftRuns taps=%d chans=%d rows=%d", taps, chans, rows)
	}
	b := ir.New("conv1dshiftruns", [3]int{128, 1, 1})
	pS := b.Param("pS", ir.F32)
	pX := b.Param("pX", ir.F32)
	pSOut := b.Param("pSOut", ir.F32)
	pN := b.Param("pN", ir.U32) // the run descriptor
	pre := int64(taps - 1)
	per := pre * int64(chans)
	n := int64(rows) * per
	idx := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, n-1))
	cw := b.Const(ir.U32, int64(chans))
	j := b.Div(ir.U32, idx, b.Const(ir.U32, per))
	w := b.Rem(ir.U32, idx, b.Const(ir.U32, per))
	u := b.Div(ir.U32, w, cw)
	c := b.Sub(ir.U32, w, b.Mul(ir.U32, u, cw))
	start, count := runSpan(b, pN, j, rows)
	base := b.Mul(ir.U32, slotIn(b, pN, j), b.Const(ir.U32, per))
	// Plane u of the next window is column count+u of [S_in ; the run's rows].
	preV := b.Const(ir.U32, pre)
	z := b.Add(ir.U32, count, u)
	inS := b.Lt(ir.U32, z, preV)
	sv := b.Load(ir.F32, pS, b.Add(ir.U32, base,
		b.Add(ir.U32, b.Mul(ir.U32, b.Min(ir.U32, z, b.Const(ir.U32, pre-1)), cw), c)), 0)
	ent := b.Select(ir.U32, inS, start, b.Sub(ir.U32, b.Add(ir.U32, start, z), preV))
	xr := orderRow(b, pN, ent, rows)
	xv := b.Load(ir.F32, pX, b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, xr, cw), c),
		b.Const(ir.U32, int64(rows*chans)-1)), 0)
	b.Store(pSOut, b.Add(ir.U32, base, w), b.Select(ir.F32, inS, sv, xv), 0)
	return b.Done()
}

// CopySlots copies rows slots of w floats from pSrc to pDst through a
// recurrent descriptor: sequence j's slot in pDst is its IN slot, pMap[1+2j],
// and it is filled from its OUT slot of pSrc, pMap[2+2j]. That is the shared
// form's way home -- the step wrote a compact next state at the out slot, and
// the pool reads it at the in slot -- inside the step's own submission, where
// a device copy (backend.Device.Copy) cannot go. Launch rows*w threads; the
// grid is clamped.
func CopySlots(w, rows int) (*ir.Kernel, error) {
	if w <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: CopySlots w=%d rows=%d", w, rows)
	}
	b := ir.New("copyslots", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	pMap := b.Param("pMap", ir.U32)
	idx := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*w-1)))
	ww := b.Const(ir.U32, int64(w))
	j := b.Div(ir.U32, idx, ww)
	i := b.Sub(ir.U32, idx, b.Mul(ir.U32, j, ww))
	src := b.Add(ir.U32, b.Mul(ir.U32, slotOut(b, pMap, j), ww), i)
	dst := b.Add(ir.U32, b.Mul(ir.U32, slotIn(b, pMap, j), ww), i)
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, src, 0), 0)
	return b.Done()
}

// GatedDeltaStep is one token of the gated delta rule, out of place.
//
// One work item per state row, because the rows are independent: for value
// head vh the state is [vDim][kDim] and the update of row j touches only row j
// and read-only vectors. It is one token, so there is no loop and no alias
// (carrying a row across tokens would re-read what this item wrote). Decode is
// one token, so this is the whole of decode.
//
// The arithmetic order is the host's (mul, dot, sub, mul, fma, dot), which
// keeps engine/model/delta.go's Forward and Prefill in agreement.
func GatedDeltaStep(vHeads, vDim, kDim, rep int) (*ir.Kernel, error) {
	return gatedDeltaStep(vHeads, vDim, kDim, rep, false, 0, 0)
}

// GatedDeltaStepTiled is GatedDeltaStep with value head vh reading key head
// vh % kHeads instead of vh / rep: the order llama.cpp's qwen35 converter
// writes a Qwen3.5 file's value heads in (jlm.FlagDeltaKeyTiled, and
// engine/model/delta.go's keyHead on the host).
func GatedDeltaStepTiled(vHeads, vDim, kDim, kHeads int) (*ir.Kernel, error) {
	if kHeads <= 0 || vHeads%kHeads != 0 {
		return nil, fmt.Errorf("kernels: GatedDeltaStepTiled vHeads=%d kHeads=%d", vHeads, kHeads)
	}
	return gatedDeltaStep(vHeads, vDim, kDim, vHeads/kHeads, false, kHeads, 0)
}

// GatedDeltaStepTiledStride is GatedDeltaStepTiled over a pool whose slot is
// stride floats, the state its first VHeads*VDim*KDim (Qwen3.8-Flash-Next,
// whose every linear block's slot carries its n-gram tail after the state;
// nn.RecurrentPlan.TailLen).
func GatedDeltaStepTiledStride(vHeads, vDim, kDim, kHeads, stride int) (*ir.Kernel, error) {
	if kHeads <= 0 || vHeads%kHeads != 0 || stride < vHeads*vDim*kDim {
		return nil, fmt.Errorf("kernels: GatedDeltaStepTiledStride vHeads=%d kHeads=%d stride=%d",
			vHeads, kHeads, stride)
	}
	return gatedDeltaStep(vHeads, vDim, kDim, vHeads/kHeads, false, kHeads, stride)
}

// GatedDeltaStepChan is GatedDeltaStep with a per-channel decay: pDecay is
// [vHeads][kDim] and the rate varies along the row. It is Kimi-Linear's delta
// rule (KDA), where GatedDeltaStep is qwen3next's.
//
// This tree holds the state as [vDim][kDim] (the reference holds
// [kDim][vDim]), so the decay varies along i, the index both loops walk, and
// is indexed by the value head, not the key head. See oracle.GatedDeltaChan
// and jit/cpu/delta.go.
func GatedDeltaStepChan(vHeads, vDim, kDim, rep int) (*ir.Kernel, error) {
	return gatedDeltaStep(vHeads, vDim, kDim, rep, true, 0, 0)
}

// gatedDeltaStep builds the step; tiledK > 0 selects the tiled key pairing over
// that many key heads (GatedDeltaStepTiled).
//
// stride > 0 is the pool's slot width where it is wider than the state.
func gatedDeltaStep(vHeads, vDim, kDim, rep int, perChan bool, tiledK, stride int) (*ir.Kernel, error) {
	if vHeads <= 0 || vDim <= 0 || kDim <= 0 || rep <= 0 {
		return nil, fmt.Errorf("kernels: GatedDeltaStep vHeads=%d vDim=%d kDim=%d rep=%d",
			vHeads, vDim, kDim, rep)
	}
	name := "gateddeltastep"
	if perChan {
		name = "gateddeltastepchan"
	}
	if tiledK > 0 {
		name = "gateddeltasteptiled"
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pS := b.Param("pS", ir.F32)         // [vHeads][vDim][kDim] state in
	pK := b.Param("pK", ir.F32)         // [kHeads][kDim]
	pQ := b.Param("pQ", ir.F32)         // [kHeads][kDim]
	pV := b.Param("pV", ir.F32)         // [vHeads][vDim]
	pDecay := b.Param("pDecay", ir.F32) // [vHeads], or [vHeads][kDim] per-channel
	pBeta := b.Param("pBeta", ir.F32)   // [vHeads]
	pOut := b.Param("pOut", ir.F32)     // [vHeads][vDim]
	pSOut := b.Param("pSOut", ir.F32)   // [vHeads][vDim][kDim] state out
	pN := b.Param("pN", ir.U32)         // the recurrent descriptor

	rows := int64(vHeads * vDim)
	row := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, rows-1))
	vdW := b.Const(ir.U32, int64(vDim))
	// row is (vh*vDim + j); v and out are indexed by the flat row.
	vh := b.Div(ir.U32, row, vdW)
	// The key head is vh/rep; pairing a value head with the wrong key head
	// still runs, with every number finite.
	var kh ir.Value
	if tiledK > 0 {
		// Only the tiled variant takes this branch, so the grouped kernel's
		// lowering golden does not move.
		kh = b.Rem(ir.U32, vh, b.Const(ir.U32, int64(tiledK)))
	} else {
		kh = b.Div(ir.U32, vh, b.Const(ir.U32, int64(rep)))
	}
	kBase := b.Mul(ir.U32, kh, b.Const(ir.U32, int64(kDim)))
	sBase := b.Mul(ir.U32, row, b.Const(ir.U32, int64(kDim)))
	// The state is read at the sequence's in slot and written at its out slot.
	zero := b.Const(ir.U32, 0)
	slotW := rows * int64(kDim)
	if stride > 0 {
		slotW = int64(stride)
	}
	sLen := b.Const(ir.U32, slotW)
	inBase := b.Add(ir.U32, sBase, b.Mul(ir.U32, slotIn(b, pN, zero), sLen))
	outBase := b.Add(ir.U32, sBase, b.Mul(ir.U32, slotOut(b, pN, zero), sLen))

	// dBase is built only in the per-channel form, so the per-head kernel's
	// lowering golden does not move.
	var dBase, decay ir.Value
	if perChan {
		dBase = b.Mul(ir.U32, vh, b.Const(ir.U32, int64(kDim)))
	} else {
		decay = b.Load(ir.F32, pDecay, vh, 0)
	}
	decayAt := func(ci ir.Value) ir.Value {
		if perChan {
			return b.Load(ir.F32, pDecay, b.Add(ir.U32, dBase, ci), 0)
		}
		return decay
	}
	beta := b.Load(ir.F32, pBeta, vh, 0)

	// sk = dot(row*decay, k)
	sk := b.Const(ir.F32, 0)
	for i := 0; i < kDim; i++ {
		ci := b.Const(ir.U32, int64(i))
		sv := b.Mul(ir.F32, b.Load(ir.F32, pS, b.Add(ir.U32, inBase, ci), 0), decayAt(ci))
		sk = b.Fma(sv, b.Load(ir.F32, pK, b.Add(ir.U32, kBase, ci), 0), sk)
	}
	d := b.Mul(ir.F32, b.Sub(ir.F32, b.Load(ir.F32, pV, row, 0), sk), beta)

	// row = row*decay + d*k, stored out of place, and o = dot(row, q).
	o := b.Const(ir.F32, 0)
	for i := 0; i < kDim; i++ {
		ci := b.Const(ir.U32, int64(i))
		kv := b.Load(ir.F32, pK, b.Add(ir.U32, kBase, ci), 0)
		nv := b.Fma(d, kv, b.Mul(ir.F32, b.Load(ir.F32, pS, b.Add(ir.U32, inBase, ci), 0), decayAt(ci)))
		b.Store(pSOut, b.Add(ir.U32, outBase, ci), nv, 0)
		o = b.Fma(nv, b.Load(ir.F32, pQ, b.Add(ir.U32, kBase, ci), 0), o)
	}
	b.Store(pOut, row, o, 0)
	return b.Done()
}

// DeltaGateRows computes the two per-head gates a linear block needs:
//
//	decay[h] = exp(A[h] * softplus(alpha[h] + dtBias[h]))
//	beta[h]  = sigma(bRaw[h])
//
// softplus is computed without a logarithm (the IR has none), with the host's
// identity from jit/cpu/gate.go:
//
//	softplus(z) = max(z,0) + log(1 + exp(-|z|))
//
// The exp argument is never positive, so no overflow branch is needed and the
// log's argument is in (1,2], where a five-term series suffices:
//
//	log(u) = 2*atanh((u-1)/(u+1)),  atanh(x) = x + x^3/3 + x^5/5 + x^7/7 + x^9/9
//
// with x = e/(2+e) and e = exp(-|z|), so x is in (0, 1/3]. The coefficients
// are written out to match the CPU's.
//
// Over rows consecutive tokens pAlpha, pBRaw,
// pDecay and pBeta are [rows][heads], and pDt and pA -- the block's own
// constants -- are read at the head, i % heads.
//
// At rows == 1 it emits DeltaGate's instruction stream exactly: the modulo
// is only built for rows > 1.
func DeltaGateRows(heads, rows int) (*ir.Kernel, error) {
	if heads <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: DeltaGate heads=%d rows=%d", heads, rows)
	}
	b := ir.New("deltagate", [3]int{128, 1, 1})
	pAlpha := b.Param("pAlpha", ir.F32)
	pDt := b.Param("pDt", ir.F32)
	pA := b.Param("pA", ir.F32)
	pBRaw := b.Param("pBRaw", ir.F32)
	pDecay := b.Param("pDecay", ir.F32)
	pBeta := b.Param("pBeta", ir.F32)

	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*heads-1)))
	h := i
	if rows > 1 {
		h = b.Rem(ir.U32, i, b.Const(ir.U32, int64(heads)))
	}
	one := b.ConstF32(1)
	two := b.ConstF32(2)
	zero := b.ConstF32(0)

	z := b.Add(ir.F32, b.Load(ir.F32, pAlpha, i, 0), b.Load(ir.F32, pDt, h, 0))
	// |z| without an Abs op: max(z, -z). Neg is 0 - z.
	nz := b.Sub(ir.F32, zero, z)
	absz := b.Max(ir.F32, z, nz)
	e := b.Exp(b.Sub(ir.F32, zero, absz)) // exp(-|z|), in (0,1]
	x := b.Div(ir.F32, e, b.Add(ir.F32, two, e))
	x2 := b.Mul(ir.F32, x, x)
	// Horner on x^2, highest term first, matching deltaGateConsts' order.
	ser := b.ConstF32(1.0 / 9)
	ser = b.Fma(ser, x2, b.ConstF32(1.0/7))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/5))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/3))
	ser = b.Fma(ser, x2, one)
	log1pe := b.Mul(ir.F32, two, b.Mul(ir.F32, x, ser))
	sp := b.Add(ir.F32, b.Max(ir.F32, z, zero), log1pe)
	b.Store(pDecay, i, b.Exp(b.Mul(ir.F32, b.Load(ir.F32, pA, h, 0), sp)), 0)

	bl := b.Load(ir.F32, pBRaw, i, 0)
	b.Store(pBeta, i, b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, zero, bl)))), 0)
	return b.Done()
}

// SplitDeltaGatesRows deinterleaves the beta/alpha projection into one value per
// value head of each.
//
// `ssm_ba` reshapes to {2*rep, n_head_k}: beta first and alpha second within
// each key head's group of 2*rep, so value head vh = kh*rep + r takes
//
//	bRaw[vh]  = BA[kh*2*rep + r]
//	alpha[vh] = BA[kh*2*rep + rep + r]
//
// Reading it as two halves also runs and pairs every value head with the wrong
// decay (see engine/model/delta.go and SplitHeadGate for the same trap).
//
// Out of place by construction (RULE 13): every output element is a pure
// function of one read-only input element.
//
// Over rows consecutive tokens pBA is [rows][kHeads][2*rep] and both outputs
// [rows][vHeads]. Each token's group is exactly 2*vHeads wide, so the flat
// index splits as kh' = i/rep across tokens at once; the row count only widens
// the clamp.
func SplitDeltaGatesRows(kHeads, rep, rows int) (*ir.Kernel, error) {
	if kHeads <= 0 || rep <= 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: SplitDeltaGates kHeads=%d rep=%d rows=%d", kHeads, rep, rows)
	}
	vHeads := int64(kHeads * rep)
	b := ir.New("splitdeltagates", [3]int{128, 1, 1})
	pBA := b.Param("pBA", ir.F32)       // [rows][kHeads][2*rep]
	pBRaw := b.Param("pBRaw", ir.F32)   // [rows][vHeads]
	pAlpha := b.Param("pAlpha", ir.F32) // [rows][vHeads]

	vh := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows)*vHeads-1))
	repW := b.Const(ir.U32, int64(rep))
	kh := b.Div(ir.U32, vh, repW)
	r := b.Sub(ir.U32, vh, b.Mul(ir.U32, kh, repW))
	base := b.Add(ir.U32, b.Mul(ir.U32, kh, b.Const(ir.U32, int64(2*rep))), r)
	b.Store(pBRaw, vh, b.Load(ir.F32, pBA, base, 0), 0)
	b.Store(pAlpha, vh, b.Load(ir.F32, pBA, base, int64(rep)), 0)
	return b.Done()
}

// SliceRows copies one contiguous field out of a row-major buffer.
//
// A launch passes whole buffers, not views, so GatedDeltaStep's pQ, pK and pV
// must each be a buffer indexed from zero; the split happens after the
// convolution, which runs over all channels plane-major. The grid is over the
// output: gridding over the input would need a guarded store, the in-place
// shape ir.Validate refuses.
func SliceRows(n, stride, off, rows int) (*ir.Kernel, error) {
	if n <= 0 || stride < n+off || off < 0 || rows <= 0 {
		return nil, fmt.Errorf("kernels: SliceRows n=%d stride=%d off=%d rows=%d",
			n, stride, off, rows)
	}
	b := ir.New("slicerows", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	total := int64(n) * int64(rows)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, total-1))
	nw := b.Const(ir.U32, int64(n))
	r := b.Div(ir.U32, i, nw)
	j := b.Sub(ir.U32, i, b.Mul(ir.U32, r, nw))
	src := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(stride))), j)
	b.Store(pOut, i, b.Load(ir.F32, pSrc, src, int64(off)), 0)
	return b.Done()
}

// DeltaScan is the geometry of GatedDeltaScan: the gated delta rule over a
// chunk of consecutive tokens of one sequence.
type DeltaScan struct {
	VHeads, VDim, KDim, Rep int
	// TiledK > 0 pairs value head vh with key head vh % TiledK
	// (jlm.FlagDeltaKeyTiled) instead of vh / Rep.
	TiledK int
	// PerChan is KDA's per-channel decay: pDecay is [rows][VHeads][KDim]
	// instead of [rows][VHeads].
	PerChan bool
	// Rows is the widest chunk the buffers hold. The real count is read from
	// pN, exactly as Conv1dShift reads it.
	Rows int
	// Lanes is the subgroup the device guarantees: ir.SubgroupLanes, or 1 on a
	// device that guarantees none, where a state row is one thread.
	Lanes int
	// Runs makes the Rows rows a step's RUNS (the run descriptor, above)
	// instead of consecutive tokens of one sequence: run j's state is read at
	// its in slot of pS, stepped through the run's rows in position order and
	// written at its out slot of pSOut, so a batched decode's sequences step
	// once each and a prompt chunk beside them steps once a row. pN's row count
	// is not read.
	Runs bool
	// Stride is the pool's slot width where it is wider than the state
	// (GatedDeltaStepTiledStride); zero is VHeads*VDim*KDim.
	Stride int
}

// ScanLanes is how many lanes cooperate on one state row: the widest power of
// two that divides KDim and fits the subgroup.
func (d DeltaScan) ScanLanes() int {
	l := 1
	for l*2 <= d.Lanes && d.KDim%(l*2) == 0 {
		l *= 2
	}
	return l
}

// Threads is the launch's thread count: every state row times its lanes, and
// for every run when the rows are Runs.
func (d DeltaScan) Threads() int {
	n := d.VHeads * d.VDim * d.ScanLanes()
	if d.Runs {
		n *= d.Rows
	}
	return n
}

// GatedDeltaScan runs the gated delta rule over up to Rows consecutive tokens
// in one launch, each state row held in registers for the whole chunk.
//
// It is llama.cpp's sequential scan (ggml-cuda/gated_delta_net.cu),
// transcribed into this IR: a lane group owns one row of one value head's
// [vDim][kDim] state, each lane holds kDim/L of it in registers, and the
// chunk's tokens stream past. The two dots per token are butterflies over the
// lanes; global traffic per token is k, q, one v, one beta and the decay.
//
// A lane owns elements lane, lane+L, lane+2L ..., not a contiguous run, so
// every load of k, q, the state and the per-channel decay is coalesced.
//
// Out of place for GatedDeltaStep's reason: the state comes in through pS and
// leaves through pSOut. Every lane of a group stores the same output (the
// butterfly leaves the sum in all of them).
//
// m is read from pN, not Rows, for Conv1dRows' reason: a zero token run
// through a recurrence decays the whole state.
//
// The per-token arithmetic is the host's; only the order of each dot's
// additions differs.
func GatedDeltaScan(d DeltaScan) (*ir.Kernel, error) { return gatedDeltaScan(d, nil) }

// DeltaFuse is what GatedDeltaFused folds into the scan: everything between the
// convolution and the rule, which the unfused block issues as eleven launches.
type DeltaFuse struct {
	// Chans is the convolution output's row width, q | k | v.
	Chans int
	// Eps is the q/k L2 norm's epsilon as HeadNormRows bakes it -- already
	// divided by KDim, since the norm is an RMSNorm over a unit weight.
	Eps float32
	// Scale is 1/sqrt(KDim): k carries it once and q twice.
	Scale float32
	// BARep is the value heads per group of the beta/alpha projection's
	// interleave (VHeads/KHeads, SplitDeltaGates' rep). qwen3next's gates only.
	BARep int
	// SSD is Mamba-2's selective update in place of the delta rule: the
	// convolution's bias is added before the SiLU, nothing is normed or
	// scaled, dt = softplus(dt_raw + dt_bias) scales the value written along
	// the key, the key dot and the subtraction are gone, and D*x is added to
	// the output. See GatedDeltaFused for its params.
	SSD bool
	// Mamba1 is Mamba-1's selective scan: every value head is one channel of
	// one row (VDim 1) and its decay a vector over the state, exp(A*dt) with
	// A one row of KDim per channel; B and C are one key head shared by all
	// (Rep is VHeads) and arrive as buffers of their own, x already through
	// its bias and SiLU -- Mamba-1's x_proj reads the convolved x, so those
	// cannot be formed here. dt = softplus(dt_raw + dt_bias) as for SSD.
	Mamba1 bool
	// Bound is Kimi-K3's gate_lower_bound on KDA's per-channel decay
	// (PerChan): exp(Bound*sigma(-A*(alpha+dt))) in place of
	// exp(A*softplus(alpha+dt)). Zero is the softplus decay.
	Bound float32
}

// GatedDeltaFused is GatedDeltaScan with the rest of the rule's inputs formed
// in the same launch, from the convolution's raw output and the gate
// projections: the SiLU, the q|k|v slices, the two L2 norms and three scales,
// and the decay and beta gates, which unfused are about eleven launches
// between the convolution and the scan.
//
// Every value is formed by the operations the unfused kernels use, in their
// order (act's SiLU, HeadNorm's upward butterfly, DeltaGate's series
// softplus), so at KDim a multiple of 32 the rule sees the same floats. The
// norms are recomputed by every state row that reads them: two extra
// butterflies a token instead of a launch and a buffer.
//
// Params (qwen3next, qwen35): pS, pConv, pBA, pDt, pA, pOut, pSOut, pN.
// Params (KDA, PerChan): pS, pConv, pAlpha, pBRaw, pDt, pA, pOut, pSOut, pN.
// Params (Mamba-1, Mamba1): pS, pX, pB, pC, pDtRaw, pDt, pA, pD, pOut, pSOut,
// pN, where pX is [rows][VHeads], pB and pC [rows][KDim], pDtRaw [rows][VHeads]
// and pA [VHeads][KDim]. Per channel c:
//
//	S[c][s] = S[c][s]*exp(A[c][s]*dt[c]) + (x[c]*dt[c])*B[s]
//	o[c]    = dot(S[c], C) + D[c]*x[c]
//
// Params (Mamba-2, SSD): pS, pConv, pDtRaw, pDt, pA, pCB, pD, pOut, pSOut, pN,
// where pConv holds q | k | v = C | B | x before the bias and SiLU, pDtRaw is
// [rows][VHeads], pCB is the convolution's bias over Chans and pD the skip,
// one per value head. Per row j of a head, with x the value:
//
//	S[j] = S[j]*exp(A*dt) + (x[j]*dt)*B
//	o[j] = dot(S[j], C) + D*x[j]
//
// pConv is [rows][Chans] before the SiLU; pBA is [rows][kHeads][2*BARep];
// pAlpha is [rows][VHeads][KDim] and pBRaw [rows][VHeads]; pDt and pA are the
// block's constants, per value head or (KDA) per channel.
func GatedDeltaFused(d DeltaScan, f DeltaFuse) (*ir.Kernel, error) {
	if f.Mamba1 {
		if f.SSD || d.PerChan || d.TiledK > 0 || d.VDim != 1 || d.Rep != d.VHeads {
			return nil, fmt.Errorf("kernels: GatedDeltaFused: Mamba-1's scan is a channel a head "+
				"over one shared key head: %+v", d)
		}
		return gatedDeltaScan(d, &f)
	}
	if f.SSD && (d.PerChan || d.TiledK > 0) {
		return nil, fmt.Errorf("kernels: GatedDeltaFused: Mamba-2's update has one decay a head and grouped keys: %+v", d)
	}
	if f.Chans <= 0 || (!d.PerChan && !f.SSD && f.BARep <= 0) {
		return nil, fmt.Errorf("kernels: GatedDeltaFused %+v %+v", d, f)
	}
	return gatedDeltaScan(d, &f)
}

func gatedDeltaScan(d DeltaScan, f *DeltaFuse) (*ir.Kernel, error) {
	if d.VHeads <= 0 || d.VDim <= 0 || d.KDim <= 0 || d.Rep <= 0 || d.Rows <= 0 ||
		(d.Lanes != 1 && d.Lanes != ir.SubgroupLanes) {
		return nil, fmt.Errorf("kernels: GatedDeltaScan %+v", d)
	}
	if d.TiledK > 0 && d.VHeads%d.TiledK != 0 {
		return nil, fmt.Errorf("kernels: GatedDeltaScan vHeads=%d tiledK=%d", d.VHeads, d.TiledK)
	}
	L := d.ScanLanes()
	nr := d.KDim / L
	name := "gateddeltascan"
	if f != nil {
		name = "gateddeltafused"
		if f.SSD {
			name = "gatedssd"
		}
		if f.Mamba1 {
			name = "mamba1scan"
		}
	}
	if d.PerChan {
		name += "chan"
	}
	if d.TiledK > 0 {
		name += "tiled"
	}
	if d.Runs {
		name += "runs"
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pS := b.Param("pS", ir.F32) // [vHeads][vDim][kDim] state in
	var pK, pQ, pV, pDecay, pBeta ir.Value
	var pConv, pBA, pAlpha, pBRaw, pDt, pA, pDtRaw, pCB, pD ir.Value
	var pX, pB, pC ir.Value
	if f != nil && f.Mamba1 {
		pX = b.Param("pX", ir.F32)         // [rows][vHeads], through its bias and SiLU
		pB = b.Param("pB", ir.F32)         // [rows][kDim]
		pC = b.Param("pC", ir.F32)         // [rows][kDim]
		pDtRaw = b.Param("pDtRaw", ir.F32) // [rows][vHeads]
		pDt = b.Param("pDt", ir.F32)       // [vHeads], dt's bias
		pA = b.Param("pA", ir.F32)         // [vHeads][kDim]
		pD = b.Param("pD", ir.F32)         // [vHeads]
	} else if f == nil {
		pK = b.Param("pK", ir.F32)         // [rows][kHeads][kDim]
		pQ = b.Param("pQ", ir.F32)         // [rows][kHeads][kDim]
		pV = b.Param("pV", ir.F32)         // [rows][vHeads][vDim]
		pDecay = b.Param("pDecay", ir.F32) // [rows][vHeads], or [rows][vHeads][kDim]
		pBeta = b.Param("pBeta", ir.F32)   // [rows][vHeads]
	} else {
		pConv = b.Param("pConv", ir.F32) // [rows][chans], before the SiLU
		if f.SSD {
			pDtRaw = b.Param("pDtRaw", ir.F32) // [rows][vHeads]
		} else if d.PerChan {
			pAlpha = b.Param("pAlpha", ir.F32) // [rows][vHeads][kDim]
			pBRaw = b.Param("pBRaw", ir.F32)   // [rows][vHeads]
		} else {
			pBA = b.Param("pBA", ir.F32) // [rows][kHeads][2*rep]
		}
		pDt = b.Param("pDt", ir.F32)
		pA = b.Param("pA", ir.F32)
		if f.SSD {
			pCB = b.Param("pCB", ir.F32) // [chans]
			pD = b.Param("pD", ir.F32)   // [vHeads]
		}
	}
	pOut := b.Param("pOut", ir.F32)   // [rows][vHeads][vDim]
	pSOut := b.Param("pSOut", ir.F32) // [vHeads][vDim][kDim] state out
	pN := b.Param("pN", ir.U32)       // the real row count, or the run descriptor

	kHeads := d.VHeads / d.Rep
	if d.TiledK > 0 {
		kHeads = d.TiledK
	}
	qkDim := int64(kHeads * d.KDim)
	qkW := qkDim
	vW := int64(d.VHeads * d.VDim)
	decW := int64(d.VHeads)
	if d.PerChan {
		decW = int64(d.VHeads * d.KDim)
	}

	nRows := int64(d.VHeads * d.VDim)
	gid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	// seq is the run a Runs launch's thread steps; each has a state of
	// nRows*KDim floats at its slots.
	seq := b.Const(ir.U32, 0)
	if d.Runs {
		per := nRows * int64(L)
		seq = b.Min(ir.U32, b.Div(ir.U32, gid, b.Const(ir.U32, per)), b.Const(ir.U32, int64(d.Rows-1)))
		gid = b.Rem(ir.U32, gid, b.Const(ir.U32, per))
	}
	var lane, row ir.Value
	if L == 1 {
		lane = b.Const(ir.U32, 0)
		row = b.Min(ir.U32, gid, b.Const(ir.U32, nRows-1))
	} else {
		lane = b.And(ir.U32, gid, b.Const(ir.U32, int64(L-1)))
		row = b.Min(ir.U32, b.Div(ir.U32, gid, b.Const(ir.U32, int64(L))),
			b.Const(ir.U32, nRows-1))
	}
	vh := b.Div(ir.U32, row, b.Const(ir.U32, int64(d.VDim)))
	var kh ir.Value
	if d.TiledK > 0 {
		kh = b.Rem(ir.U32, vh, b.Const(ir.U32, int64(d.TiledK)))
	} else {
		kh = b.Div(ir.U32, vh, b.Const(ir.U32, int64(d.Rep)))
	}
	// Every per-lane index is a base plus r*L, and r*L is the load's immediate.
	// The state is read at the sequence's in slot and written at its out slot.
	sBase := b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(d.KDim))), lane)
	slotW := nRows * int64(d.KDim)
	if d.Stride > 0 {
		slotW = int64(d.Stride)
	}
	sLen := b.Const(ir.U32, slotW)
	inBase := b.Add(ir.U32, sBase, b.Mul(ir.U32, slotIn(b, pN, seq), sLen))
	outBase := b.Add(ir.U32, sBase, b.Mul(ir.U32, slotOut(b, pN, seq), sLen))
	kBase0 := b.Add(ir.U32, b.Mul(ir.U32, kh, b.Const(ir.U32, int64(d.KDim))), lane)
	var dBase0 ir.Value
	if d.PerChan {
		dBase0 = b.Add(ir.U32, b.Mul(ir.U32, vh, b.Const(ir.U32, int64(d.KDim))), lane)
	}
	// The fused form's per-row constants, read once: the gates' dt and A (per
	// head, or per channel on KDA), and where this row's value head sits in
	// the beta/alpha interleave.
	var dtH, aH, baOff, dH, cbV ir.Value
	var dtC, aC, cbQ, cbK []ir.Value
	if f != nil && f.Mamba1 {
		// Mamba-1's per-channel constants: dt's bias, D, and the channel's row
		// of A, at the lanes this thread owns.
		dtH, dH = b.Load(ir.F32, pDt, vh, 0), b.Load(ir.F32, pD, vh, 0)
		ab := b.Add(ir.U32, b.Mul(ir.U32, vh, b.Const(ir.U32, int64(d.KDim))), lane)
		aC = make([]ir.Value, nr)
		for r := 0; r < nr; r++ {
			aC[r] = b.Load(ir.F32, pA, ab, int64(r*L))
		}
	} else if f != nil && f.SSD {
		// Mamba-2's per-row constants: the head's dt bias, A and D, and the
		// convolution bias of every channel the row reads.
		dtH, aH, dH = b.Load(ir.F32, pDt, vh, 0), b.Load(ir.F32, pA, vh, 0), b.Load(ir.F32, pD, vh, 0)
		cbQ, cbK = make([]ir.Value, nr), make([]ir.Value, nr)
		for r := 0; r < nr; r++ {
			cbQ[r] = b.Load(ir.F32, pCB, kBase0, int64(r*L))
			cbK[r] = b.Load(ir.F32, pCB, kBase0, qkDim+int64(r*L))
		}
		cbV = b.Load(ir.F32, pCB, row, 2*qkDim)
	} else if f != nil {
		if d.PerChan {
			dtC, aC = make([]ir.Value, nr), make([]ir.Value, nr)
			for r := 0; r < nr; r++ {
				dtC[r] = b.Load(ir.F32, pDt, dBase0, int64(r*L))
				aC[r] = b.Load(ir.F32, pA, dBase0, int64(r*L))
			}
		} else {
			dtH, aH = b.Load(ir.F32, pDt, vh, 0), b.Load(ir.F32, pA, vh, 0)
			// SplitDeltaGates' deinterleave: value head vh = g*rep + r reads
			// beta at g*2*rep + r and alpha rep further on.
			rep := b.Const(ir.U32, int64(f.BARep))
			grp := b.Div(ir.U32, vh, rep)
			baOff = b.Add(ir.U32, b.Mul(ir.U32, grp, b.Const(ir.U32, int64(2*f.BARep))),
				b.Sub(ir.U32, vh, b.Mul(ir.U32, grp, rep)))
		}
	}

	init := make([]ir.Value, nr)
	for r := 0; r < nr; r++ {
		init[r] = b.Load(ir.F32, pS, inBase, int64(r*L))
	}
	zero := b.Const(ir.U32, 0)
	// A chunk's loop is its rows 0..m-1. A run's is its entries in the order
	// list, each naming the row that holds that position.
	m, t0 := b.Load(ir.U32, pN, zero, 0), zero
	if d.Runs {
		t0, m = runSpan(b, pN, seq, d.Rows)
	}
	b.LoopN(m)
	ti := b.Phi(ir.U32, t0)
	t := ti
	if d.Runs {
		t = orderRow(b, pN, ti, d.Rows)
	}
	s := make([]ir.Value, nr)
	for r := range s {
		s[r] = b.Phi(ir.F32, init[r])
	}
	// down is the scan's own reduction; up is HeadNorm's order (masks 1, 2,
	// 4, ...), which the fused norms keep so their sums are HeadNorm's.
	down := func(x ir.Value) ir.Value {
		for mask := L / 2; mask >= 1; mask /= 2 {
			x = b.Add(ir.F32, x, b.ShuffleXor(ir.F32, x, int64(mask)))
		}
		return x
	}
	up := func(x ir.Value) ir.Value {
		for mask := 1; mask < L; mask <<= 1 {
			x = b.Add(ir.F32, x, b.ShuffleXor(ir.F32, x, int64(mask)))
		}
		return x
	}
	th := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(d.VHeads))), vh)
	kv := make([]ir.Value, nr)
	qv := make([]ir.Value, nr)
	g := make([]ir.Value, nr)
	var beta, vIn, ssdDt ir.Value
	vIdx := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, vW)), row)
	if f != nil && f.Mamba1 {
		// B and C at the shared key head, x at the channel, dt through its
		// bias and softplus, and the decay a vector over the state.
		bc := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(d.KDim))), lane)
		for r := 0; r < nr; r++ {
			kv[r] = b.Load(ir.F32, pB, bc, int64(r*L))
			qv[r] = b.Load(ir.F32, pC, bc, int64(r*L))
		}
		vIn = b.Load(ir.F32, pX, vIdx, 0)
		ssdDt = softplusOf(b, b.Load(ir.F32, pDtRaw, th, 0), dtH)
		for r := range g {
			g[r] = b.Exp(b.Mul(ir.F32, aC[r], ssdDt))
		}
	} else if f != nil && f.SSD {
		// C and B at the key head's channels, x at the row's, each through
		// its bias and the SiLU; no norm and no scale.
		cb := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(f.Chans))), kBase0)
		for r := 0; r < nr; r++ {
			qv[r] = act(b, b.Add(ir.F32, b.Load(ir.F32, pConv, cb, int64(r*L)), cbQ[r]), ActSiLU)
			kv[r] = act(b, b.Add(ir.F32, b.Load(ir.F32, pConv, cb, qkDim+int64(r*L)), cbK[r]), ActSiLU)
		}
		cv := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(f.Chans))), row)
		vIn = act(b, b.Add(ir.F32, b.Load(ir.F32, pConv, cv, 2*qkDim), cbV), ActSiLU)
		var dec ir.Value
		dec, ssdDt = ssdGateOf(b, b.Load(ir.F32, pDtRaw, th, 0), dtH, aH)
		for r := range g {
			g[r] = dec
		}
	} else if f == nil {
		kb := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, qkW)), kBase0)
		for r := 0; r < nr; r++ {
			kv[r] = b.Load(ir.F32, pK, kb, int64(r*L))
			qv[r] = b.Load(ir.F32, pQ, kb, int64(r*L))
		}
		beta = b.Load(ir.F32, pBeta, th, 0)
		if d.PerChan {
			db := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, decW)), dBase0)
			for r := 0; r < nr; r++ {
				g[r] = b.Load(ir.F32, pDecay, db, int64(r*L))
			}
		} else {
			dec := b.Load(ir.F32, pDecay, th, 0)
			for r := range g {
				g[r] = dec
			}
		}
		vIn = b.Load(ir.F32, pV, vIdx, 0)
	} else {
		// q at kh*kDim, k one qkDim further, v after both, all of the
		// convolution's row t, through the SiLU the Act launch applied.
		cb := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(f.Chans))), kBase0)
		for r := 0; r < nr; r++ {
			qv[r] = act(b, b.Load(ir.F32, pConv, cb, int64(r*L)), ActSiLU)
			kv[r] = act(b, b.Load(ir.F32, pConv, cb, qkDim+int64(r*L)), ActSiLU)
		}
		cv := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(f.Chans))), row)
		vIn = act(b, b.Load(ir.F32, pConv, cv, 2*qkDim), ActSiLU)
		// HeadNorm over a unit weight, then Scale -- once for k, twice for q.
		norm := func(x []ir.Value) ir.Value {
			acc := b.ConstF32(0)
			for r := range x {
				acc = b.Fma(x[r], x[r], acc)
			}
			mean := b.Mul(ir.F32, up(acc), b.ConstF32(float32(1)/float32(d.KDim)))
			return b.Div(ir.F32, b.ConstF32(1), b.Sqrt(b.Add(ir.F32, mean, b.ConstF32(f.Eps))))
		}
		iq, ik := norm(qv), norm(kv)
		sc := b.ConstF32(f.Scale)
		for r := 0; r < nr; r++ {
			kv[r] = b.Mul(ir.F32, b.Mul(ir.F32, kv[r], ik), sc)
			qv[r] = b.Mul(ir.F32, b.Mul(ir.F32, b.Mul(ir.F32, qv[r], iq), sc), sc)
		}
		if d.PerChan {
			ab := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, decW)), dBase0)
			for r := 0; r < nr; r++ {
				if f.Bound != 0 {
					g[r] = boundDecay(b, b.Load(ir.F32, pAlpha, ab, int64(r*L)), dtC[r], aC[r], f.Bound)
				} else {
					g[r] = deltaDecay(b, b.Load(ir.F32, pAlpha, ab, int64(r*L)), dtC[r], aC[r])
				}
			}
			beta = sigmoidOf(b, b.Load(ir.F32, pBRaw, th, 0))
		} else {
			bo := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, int64(2*d.VHeads))), baOff)
			dec := deltaDecay(b, b.Load(ir.F32, pBA, bo, int64(f.BARep)), dtH, aH)
			for r := range g {
				g[r] = dec
			}
			beta = sigmoidOf(b, b.Load(ir.F32, pBA, bo, 0))
		}
	}
	// sk = dot(row*decay, k), which Mamba-2's update has no use for: there
	// the value written along the key is x*dt. The delta rule's emission
	// order is kept as it was, so its lowering goldens do not move.
	var dec []ir.Value
	var dd ir.Value
	if f != nil && (f.SSD || f.Mamba1) {
		dec = make([]ir.Value, nr)
		for r := 0; r < nr; r++ {
			dec[r] = b.Mul(ir.F32, s[r], g[r])
		}
		dd = b.Mul(ir.F32, vIn, ssdDt)
	} else {
		sk := b.Const(ir.F32, 0)
		dec = make([]ir.Value, nr)
		for r := 0; r < nr; r++ {
			dec[r] = b.Mul(ir.F32, s[r], g[r])
			sk = b.Fma(dec[r], kv[r], sk)
		}
		sk = down(sk)
		dd = b.Mul(ir.F32, b.Sub(ir.F32, vIn, sk), beta)
	}
	// row = row*decay + d*k, and o = dot(row, q).
	o := b.Const(ir.F32, 0)
	next := make([]ir.Value, nr)
	for r := 0; r < nr; r++ {
		next[r] = b.Fma(dd, kv[r], dec[r])
		o = b.Fma(next[r], qv[r], o)
	}
	o = down(o)
	if f != nil && (f.SSD || f.Mamba1) {
		o = b.Fma(dH, vIn, o)
	}
	b.Store(pOut, vIdx, o, 0)
	b.SetPhi(ti, b.Add(ir.U32, ti, b.Const(ir.U32, 1)))
	for r := range s {
		b.SetPhi(s[r], next[r])
	}
	b.EndLoop()
	for r := 0; r < nr; r++ {
		b.Store(pSOut, outBase, s[r], int64(r*L))
	}
	return b.Done()
}

// deltaDecay is DeltaGate's decay, exp(A * softplus(alpha + dt)), transcribed
// operation for operation (see there for the log-free softplus) so a fused
// kernel forms the same float the gate kernel does.
func deltaDecay(b *ir.Builder, alpha, dt, a ir.Value) ir.Value {
	one := b.ConstF32(1)
	two := b.ConstF32(2)
	zero := b.ConstF32(0)
	z := b.Add(ir.F32, alpha, dt)
	nz := b.Sub(ir.F32, zero, z)
	absz := b.Max(ir.F32, z, nz)
	e := b.Exp(b.Sub(ir.F32, zero, absz))
	x := b.Div(ir.F32, e, b.Add(ir.F32, two, e))
	x2 := b.Mul(ir.F32, x, x)
	ser := b.ConstF32(1.0 / 9)
	ser = b.Fma(ser, x2, b.ConstF32(1.0/7))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/5))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/3))
	ser = b.Fma(ser, x2, one)
	log1pe := b.Mul(ir.F32, two, b.Mul(ir.F32, x, ser))
	sp := b.Add(ir.F32, b.Max(ir.F32, z, zero), log1pe)
	return b.Exp(b.Mul(ir.F32, a, sp))
}

// boundDecay is Kimi-K3's KDA decay under gate_lower_bound,
// exp(lb / (1 + exp(A*(alpha + dt)))) -- fla's lb*sigmoid(exp(A_log)*(g +
// dt_bias)) with A = -exp(A_log) -- the host kernel's operations in its
// order (cpu.EmitDeltaDecayBound).
func boundDecay(b *ir.Builder, alpha, dt, a ir.Value, lb float32) ir.Value {
	e := b.Exp(b.Mul(ir.F32, b.Add(ir.F32, alpha, dt), a))
	return b.Exp(b.Div(ir.F32, b.ConstF32(lb), b.Add(ir.F32, e, b.ConstF32(1))))
}

// DeltaDecayBoundRows is boundDecay over n channels of rows rows: pDecay[i]
// from pAlpha[i] and the block's per-channel pDt and pA, read at i % n.
// Params pAlpha, pDt, pA, pDecay. The unfused KDA block's decay under the
// bound; beta stays DeltaGateRows'.
func DeltaDecayBoundRows(n, rows int, lb float32) (*ir.Kernel, error) {
	if n <= 0 || rows <= 0 || !(lb < 0) {
		return nil, fmt.Errorf("kernels: DeltaDecayBoundRows n=%d rows=%d bound=%v", n, rows, lb)
	}
	b := ir.New("deltadecaybound", [3]int{128, 1, 1})
	pAlpha := b.Param("pAlpha", ir.F32)
	pDt := b.Param("pDt", ir.F32)
	pA := b.Param("pA", ir.F32)
	pDecay := b.Param("pDecay", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*n-1)))
	c := i
	if rows > 1 {
		c = b.Rem(ir.U32, i, b.Const(ir.U32, int64(n)))
	}
	b.Store(pDecay, i, boundDecay(b, b.Load(ir.F32, pAlpha, i, 0), b.Load(ir.F32, pDt, c, 0),
		b.Load(ir.F32, pA, c, 0), lb), 0)
	return b.Done()
}

// ssdGateOf is Mamba-2's dt = softplus(raw + bias) and decay = exp(A*dt),
// the softplus DeltaGate's (deltaDecay's) operation for operation, so the
// fused kernel forms the floats jit/cpu's SSDGate does.
func ssdGateOf(b *ir.Builder, raw, bias, a ir.Value) (decay, dt ir.Value) {
	dt = softplusOf(b, raw, bias)
	return b.Exp(b.Mul(ir.F32, a, dt)), dt
}

// softplusOf is DeltaGate's log-free softplus of raw + bias, operation for
// operation and in ssdGateOf's emission order.
func softplusOf(b *ir.Builder, raw, bias ir.Value) ir.Value {
	one := b.ConstF32(1)
	two := b.ConstF32(2)
	zero := b.ConstF32(0)
	z := b.Add(ir.F32, raw, bias)
	nz := b.Sub(ir.F32, zero, z)
	absz := b.Max(ir.F32, z, nz)
	e := b.Exp(b.Sub(ir.F32, zero, absz))
	x := b.Div(ir.F32, e, b.Add(ir.F32, two, e))
	x2 := b.Mul(ir.F32, x, x)
	ser := b.ConstF32(1.0 / 9)
	ser = b.Fma(ser, x2, b.ConstF32(1.0/7))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/5))
	ser = b.Fma(ser, x2, b.ConstF32(1.0/3))
	ser = b.Fma(ser, x2, one)
	log1pe := b.Mul(ir.F32, two, b.Mul(ir.F32, x, ser))
	return b.Add(ir.F32, b.Max(ir.F32, z, zero), log1pe)
}

// sigmoidOf is DeltaGate's beta, 1/(1+exp(0-x)), transcribed.
func sigmoidOf(b *ir.Builder, x ir.Value) ir.Value {
	one := b.ConstF32(1)
	return b.Div(ir.F32, one, b.Add(ir.F32, one, b.Exp(b.Sub(ir.F32, b.ConstF32(0), x))))
}
