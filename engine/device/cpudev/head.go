package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// HeadParts is the head's and the samplers' forms on the CPU device: the
// head's (Head), the one also writing the norm's rows (Hidden) and the
// latter at a chunk's rows (Rows), the label pick's, the greedy sampler's,
// a top-k draw's walk, a draw's row and cut with no top-k and its top-k
// variant, their distributions, the output norm alone, the log-softmax's
// and a constrained draw's mask.
type HeadParts struct {
	Head, Hidden, Rows, Pick          Part
	Sample, Walk, Row, Cut            Part
	WalkProb, CutProb, CutK, CutKProb Part
	OutNorm, Logprobs, Mask           Part
}

// Head is the head and the samplers on the CPU device, composed once at the
// instance's placement, with the buffers their calls bind. A call is
// counted served or declined by name (nn.PartHead, nn.PartSample).
type Head struct {
	p     HeadParts
	norm  bool
	w     Weights
	dev   *Device
	steps *nn.Steps
	d, nv int
	// onPad is the output norm's rows where a call is shorter than its unit,
	// in then out.
	onPad []float32
	// lpRow and lpIDs are what the log-softmax writes.
	lpRow []float32
	lpIDs []uint32
	// probVals and probIDs are the walk's distribution.
	probVals []float32
	probIDs  []uint32
	token    [1]uint32
	// noWin is a window that holds no token, for a draw with no penalty;
	// pen is the penalty a draw reads.
	noWin []float32
	pen   [1]float32
	// walkArgs is the walk's words: k, 1/T, top-p, min-p and u; rowArgs
	// the row's (1/T, u), cutArgs the cut's (1/T, top-p, min-p, u) and
	// cutKArgs its top-k variant's (then k).
	walkArgs [5]float32
	rowArgs  [2]float32
	cutArgs  [4]float32
	cutKArgs [5]float32
	// ids is the row's ids, 0 to the vocabulary's last, which the draws with
	// no top-k read.
	ids []uint32
	// hid is the rows' norm rows where the caller keeps none; xpad, lpad and
	// hpad a call's last rows where the rows' form's units reach past them.
	hid, xpad, lpad, hpad []float32
	pickIDs               []uint32
	pickVals              []float32
	call                  [4]uint32
	zero                  []uint32
	recRows               [3]uint32
	bind                  Call
	// Wrong is the row a violated head reads in place of the caller's (a
	// gate's violation), nil for none.
	Wrong []float32
}

// NewHead is the head part over rows of width d and a vocabulary of nv on
// dev, counted in steps: its forms p, norm where the model has an output
// norm, their weights bound by w; topLog is the most alternatives the
// log-softmax selects.
func NewHead(dev *Device, steps *nn.Steps, p HeadParts, norm bool, w Weights, d, nv, topLog int) *Head {
	h := &Head{p: p, norm: norm, w: w, dev: dev, steps: steps, d: d, nv: nv}
	if f := h.p.OutNorm.F; f != nil {
		h.onPad = make([]float32, 2*max(f.Unit, 1)*d)
	}
	if f := h.p.Rows.F; f != nil {
		h.hid = make([]float32, f.Rows*d)
		if u := f.Unit; u > 1 {
			h.xpad, h.lpad, h.hpad = make([]float32, u*d), make([]float32, u*nv), make([]float32, u*d)
		}
	}
	if h.p.Pick.F != nil {
		h.pickIDs, h.pickVals = make([]uint32, nn.PickMax), make([]float32, nn.PickMax)
	}
	q := &h.p
	if q.Walk.F != nil || q.Row.F != nil || q.Cut.F != nil || q.WalkProb.F != nil || q.CutProb.F != nil || q.CutK.F != nil || q.CutKProb.F != nil {
		h.noWin = make([]float32, nv)
	}
	if q.WalkProb.F != nil {
		h.probVals, h.probIDs = make([]float32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK)
	}
	if q.Logprobs.F != nil {
		h.lpRow, h.lpIDs = make([]float32, nv), make([]uint32, min(topLog, nv))
	}
	if q.Row.F != nil || q.Cut.F != nil || q.CutProb.F != nil || q.CutK.F != nil || q.CutKProb.F != nil {
		h.ids = make([]uint32, nv)
		for i := range h.ids {
			h.ids[i] = uint32(i)
		}
	}
	h.zero = make([]uint32, 4)
	return h
}

// form is part p's form and its weights, or why it does not serve.
func (h *Head) form(p *Part, weighted bool) (*Form, []unsafe.Pointer, nn.Decline) {
	f, why := p.Form()
	var weights []unsafe.Pointer
	if why == nn.DeclineServes && weighted {
		weights, why = h.w.Bind(f)
	}
	return f, weights, why
}

// run calls form f, its weights at weights, over n rows, its buffers io.
func (h *Head) run(f *Form, weights []unsafe.Pointer, n int, io ...unsafe.Pointer) {
	c := &h.bind
	*c = Call{Own: h.dev.Own(f), N: n, Weights: weights}
	copy(c.IO[:], io)
	h.call[0] = uint32(n)
	h.recRows = [3]uint32{1, 0, 0}
	c.At[SrcCall], c.At[SrcZero] = unsafe.Pointer(&h.call[0]), unsafe.Pointer(&h.zero[0])
	c.At[SrcRecRows] = unsafe.Pointer(&h.recRows[0])
	for i := range f.Codes {
		h.dev.Call(c, &f.Codes[i])
	}
}

// in is the row a head's call reads in place of x.
func (h *Head) in(x []float32) unsafe.Pointer {
	if h.Wrong != nil {
		return unsafe.Pointer(&h.Wrong[0])
	}
	return unsafe.Pointer(&x[0])
}

// Head writes the norm of the row x, its projection and what follows it into
// logits, and the norm's rows into hidden where it is given, or says why
// not.
func (h *Head) Head(x, logits, hidden []float32) nn.Decline {
	p := &h.p.Head
	if hidden != nil {
		p = &h.p.Hidden
	}
	f, weights, why := h.form(p, true)
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartHead, why)
		return why
	}
	var hd unsafe.Pointer
	if hidden != nil {
		hd = unsafe.Pointer(&hidden[0])
	}
	h.run(f, weights, 1, h.in(x), unsafe.Pointer(&logits[0]), hd)
	h.steps.Serve(nn.PartHead)
	return nn.DeclineServes
}

// HeadRows is the head over the first n rows of x, d apart, into the rows of
// logits, and the norm's rows into hidden where it is given (n rows of it),
// a call of at most the rows' form's rows, or says why not. A call's rows
// past a whole number of the form's units run from copies, so its units
// write no row past the caller's.
func (h *Head) HeadRows(x, logits, hidden []float32, n int) nn.Decline {
	f, weights, why := h.form(&h.p.Rows, true)
	if why != nn.DeclineServes {
		h.steps.Declined[nn.PartHead][why] += n
		return why
	}
	d, nv := h.d, h.nv
	for r0 := 0; r0 < n; {
		k := min(f.Rows, n-r0)
		u := max(f.Unit, 1)
		if k >= u {
			k -= k % u
			hd := unsafe.Pointer(&h.hid[0])
			if hidden != nil {
				hd = unsafe.Pointer(&hidden[r0*d])
			}
			h.rowsCall(f, weights, unsafe.Pointer(&x[r0*d]), unsafe.Pointer(&logits[r0*nv]), hd, k)
		} else {
			copy(h.xpad, x[r0*d:(r0+k)*d])
			h.rowsCall(f, weights, unsafe.Pointer(&h.xpad[0]), unsafe.Pointer(&h.lpad[0]), unsafe.Pointer(&h.hpad[0]), k)
			copy(logits[r0*nv:(r0+k)*nv], h.lpad)
			if hidden != nil {
				copy(hidden[r0*d:(r0+k)*d], h.hpad)
			}
		}
		r0 += k
	}
	h.steps.Served[nn.PartHead] += n
	return nn.DeclineServes
}

// rowsCall runs the rows' form f over k rows of x into logits and the norm's
// rows hidden.
func (h *Head) rowsCall(f *Form, weights []unsafe.Pointer, x, logits, hidden unsafe.Pointer, k int) {
	if h.Wrong != nil {
		x = unsafe.Pointer(&h.hid[0])
	}
	h.run(f, weights, k, x, logits, hidden)
}

// OutNorm writes the output norm of n rows of x into dst -- a model with no
// output norm copies them -- or says why not.
func (h *Head) OutNorm(dst, x []float32, n int) nn.Decline {
	d := h.d
	if !h.norm {
		copy(dst[:n*d], x[:n*d])
		return nn.DeclineServes
	}
	f, weights, why := h.form(&h.p.OutNorm, true)
	if why != nn.DeclineServes {
		h.steps.Declined[nn.PartHead][why] += n
		return why
	}
	for r0 := 0; r0 < n; {
		k := min(f.Rows, n-r0)
		u := max(f.Unit, 1)
		in, out := unsafe.Pointer(&x[r0*d]), unsafe.Pointer(&dst[r0*d])
		padded := k < u
		if padded {
			copy(h.onPad[:k*d], x[r0*d:(r0+k)*d])
			in, out = unsafe.Pointer(&h.onPad[0]), unsafe.Pointer(&h.onPad[u*d])
		} else {
			k -= k % u
		}
		h.run(f, weights, k, in, out)
		if padded {
			copy(dst[r0*d:(r0+k)*d], h.onPad[u*d:(u+k)*d])
		}
		r0 += k
	}
	return nn.DeclineServes
}

// Pick is the label pick: the head projecting the ids alone from the row x
// into vals, the logits left as they were, or why not.
func (h *Head) Pick(x []float32, ids []int32, vals []float32) nn.Decline {
	f, weights, why := h.form(&h.p.Pick, true)
	if why == nn.DeclineServes && (len(ids) > len(h.pickIDs) || len(vals) < len(ids)) {
		why = nn.DeclineCaller
	}
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartHead, why)
		return why
	}
	clear(h.pickIDs)
	for i, id := range ids {
		h.pickIDs[i] = uint32(id)
	}
	h.run(f, weights, 1, h.in(x), unsafe.Pointer(&h.pickIDs[0]), unsafe.Pointer(&h.pickVals[0]))
	copy(vals, h.pickVals[:len(ids)])
	h.steps.Serve(nn.PartHead)
	return nn.DeclineServes
}

// Greedy is the token of the largest of logits, the lower id of two equal,
// or false where the sampler declines.
func (h *Head) Greedy(logits []float32) (int32, bool) {
	f, _, why := h.form(&h.p.Sample, false)
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartSample, why)
		return 0, false
	}
	h.run(f, nil, 1, unsafe.Pointer(&logits[0]), unsafe.Pointer(&h.token[0]))
	h.steps.Serve(nn.PartSample)
	return int32(h.token[0]), true
}

// window writes a's penalty where the draws read it and is the window row
// they read: a's, or a row holding no token where the penalty is off.
func (h *Head) window(a *nn.DrawArgs) unsafe.Pointer {
	h.pen[0] = a.Penalty
	if a.Penalty == 1 || len(a.Window) == 0 {
		return unsafe.Pointer(&h.noWin[0])
	}
	return unsafe.Pointer(&a.Window[0])
}

// Draw is a's sampled token over logits at the uniform u: a top-k draw's
// walk, or with no top-k the row's or, where top-p or min-p is set, the
// cut's, or with a top-k past the walk's bound the cut's top-k variant.
// False where none serves it.
func (h *Head) Draw(logits []float32, a nn.DrawArgs, u float32) (int32, bool) {
	n := len(logits)
	if a.TopK > 0 && a.TopK < n && a.TopK <= nn.SampleMaxK {
		return h.walk(logits, &a, u)
	}
	p, words := &h.p.Row, unsafe.Pointer(&h.rowArgs[0])
	h.rowArgs = [2]float32{a.InvTemp, u}
	switch {
	case a.TopK > 0 && a.TopK < n:
		p, words = &h.p.CutK, unsafe.Pointer(&h.cutKArgs[0])
		h.cutKArgs = [5]float32{a.InvTemp, a.TopP, a.MinP, u, float32(a.TopK)}
	case a.MinP > 0 || a.TopP < 1:
		p, words = &h.p.Cut, unsafe.Pointer(&h.cutArgs[0])
		h.cutArgs = [4]float32{a.InvTemp, a.TopP, a.MinP, u}
	}
	f, _, why := h.form(p, false)
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartSample, why)
		return 0, false
	}
	win := h.window(&a)
	h.run(f, nil, 1, unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0]), unsafe.Pointer(&h.ids[0]), words,
		unsafe.Pointer(&h.token[0]))
	h.steps.Serve(nn.PartSample)
	return int32(h.token[0]), true
}

// walk is a's top-k draw over logits at the uniform u, or false where it
// declines: a k past the bound the walk was formed for.
func (h *Head) walk(logits []float32, a *nn.DrawArgs, u float32) (int32, bool) {
	f, _, why := h.form(&h.p.Walk, false)
	if why == nn.DeclineServes && a.TopK > nn.SampleMaxK {
		why = nn.DeclineCaller
	}
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartSample, why)
		return 0, false
	}
	win := h.window(a)
	h.walkArgs = [5]float32{float32(a.TopK), a.InvTemp, a.TopP, a.MinP, u}
	h.run(f, nil, 1, unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0]), unsafe.Pointer(&h.walkArgs[0]),
		unsafe.Pointer(&h.token[0]))
	h.steps.Serve(nn.PartSample)
	return int32(h.token[0]), true
}

// Dist is the distribution a draws from over logits: a top-k draw's
// candidates and their share of the survivors (ids and vals, dense false),
// else every token's probability written into pall (dense true). False where
// the forms decline.
func (h *Head) Dist(logits []float32, a nn.DrawArgs, pall []float32) (dense bool, ids []uint32, vals []float32, ok bool) {
	n := len(logits)
	if n != h.nv {
		return false, nil, nil, false
	}
	topK := a.TopK > 0 && a.TopK < n
	bounded := topK && a.TopK <= nn.SampleMaxK
	p := &h.p.CutProb
	switch {
	case bounded:
		p = &h.p.WalkProb
	case topK:
		p = &h.p.CutKProb
	}
	f, _, why := h.form(p, false)
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartSample, why)
		return false, nil, nil, false
	}
	win := h.window(&a)
	if bounded {
		h.walkArgs = [5]float32{float32(a.TopK), a.InvTemp, a.TopP, a.MinP, 0}
		h.run(f, nil, 1, unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0]), unsafe.Pointer(&h.walkArgs[0]),
			unsafe.Pointer(&h.probVals[0]), unsafe.Pointer(&h.probIDs[0]))
	} else {
		words := unsafe.Pointer(&h.cutArgs[0])
		h.cutArgs = [4]float32{a.InvTemp, a.TopP, a.MinP, 0}
		if topK {
			words = unsafe.Pointer(&h.cutKArgs[0])
			h.cutKArgs = [5]float32{a.InvTemp, a.TopP, a.MinP, 0, float32(a.TopK)}
		}
		h.run(f, nil, 1, unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0]), unsafe.Pointer(&h.ids[0]), words,
			unsafe.Pointer(&pall[0]))
	}
	h.steps.Serve(nn.PartSample)
	if bounded {
		return false, h.probIDs, h.probVals, true
	}
	return true, nil, nil, true
}

// Mask writes logits plus mask into dst, or false where the form declines.
func (h *Head) Mask(dst, logits, mask []float32) bool {
	f, _, why := h.form(&h.p.Mask, false)
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartSample, why)
		return false
	}
	h.run(f, nil, 1, unsafe.Pointer(&logits[0]), unsafe.Pointer(&mask[0]), unsafe.Pointer(&dst[0]))
	return true
}

// Logprobs is the log-softmax of logits and its most likely tokens, the
// larger logit first and the lower id on a tie, or false where the form
// declines; both are the head's until its next call.
func (h *Head) Logprobs(logits []float32) (row []float32, ids []uint32, ok bool) {
	f, _, why := h.form(&h.p.Logprobs, false)
	if why != nn.DeclineServes {
		h.steps.Decline(nn.PartSample, why)
		return nil, nil, false
	}
	h.run(f, nil, 1, unsafe.Pointer(&logits[0]), unsafe.Pointer(&h.lpRow[0]), unsafe.Pointer(&h.lpIDs[0]))
	h.steps.Serve(nn.PartSample)
	return h.lpRow, h.lpIDs, true
}
