package model

import (
	"fmt"
	"slices"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// The mixture-of-experts FFN.
//
// The graph is transcribed from llama.cpp's running graph (`llama-eval-callback`
// on a qwen3moe), not from its source:
//
//	ffn_moe_logits    MUL_MAT(ffn_gate_inp.weight, ffn_norm)
//	ffn_moe_probs     SOFT_MAX(logits)          <- over ALL experts, BEFORE top-k
//	ffn_moe_argsort   ARGSORT(probs, DESC)
//	ffn_moe_topk      VIEW(argsort)[:n_used]
//	ffn_moe_weights   GET_ROWS(probs, topk)
//	..._sum / _clamped/ _norm   SUM_ROWS -> CLAMP -> DIV   <- renormalisation IS on
//	ffn_moe_gate/up   MUL_MAT_ID(ffn_{gate,up}_exps, ffn_norm)
//	ffn_moe_swiglu    SWIGLU(gate, up)
//	ffn_moe_down      MUL_MAT_ID(ffn_down_exps, swiglu)
//	ffn_moe_weighted  MUL(down, weights_norm)   then summed and added to ffn_inp
//
// Two lines are easy to get wrong and both fail silently: the softmax is over
// all experts and comes before the top-k, and qwen3moe renormalises even though
// its GGUF carries no expert_weights_norm key.
//
// llama.cpp's clamp is not reproduced; the divisor guard is DeepSeek's
// `sum + 1e-20` (see AGENTS.md's open questions).

// moe writes one mixture-of-experts FFN of the normed vector h into out.
//
// h is the vector every selected expert reads and rin the router's: the same
// vector for every mixture but Gemma 4's (Config.DenseMoE), whose router reads
// its own norm of the residual. The caller has already called NewInput for
// rin.
//
// out is added to, not written: the caller passes the residual and each
// expert's contribution goes straight in, as moeExpertsRows does, so the two produce
// the same float sum.
//
// A post-FFN norm cannot be expressed this way; forward.go refuses a mixture
// that has one.
func (s *State) moe(li int, l *layer, rin, h, out []float32) error {
	if err := s.moeRoute(li, l, rin, h); err != nil {
		return err
	}
	if err := s.moeExperts(li, l, h, out); err != nil {
		return err
	}
	// Gemma 4's dense MLP is not a shared expert's addition (see denseMoE),
	// and a latent mixture's shared expert reads the full-width input and adds
	// after the routed sum leaves the latent (k3MoE).
	if s.c.DenseMoE || s.c.ExpertLatent != 0 {
		return nil
	}
	return s.sharedExpert(l, h, out)
}

// moeRoute is the mixture's route op for one row: the router's logits over
// rin (a generated matvec), their bias, and the selection (s.route, one
// generated kernel). h is the row the experts read after it.
func (s *State) moeRoute(li int, l *layer, rin, h []float32) error {
	c := s.c
	s.traceLayer = li
	t0 := s.tick()
	if err := s.mv(s.moeProbs, l.router, rin); err != nil {
		return err
	}
	if &rin[0] != &h[0] {
		// The experts read another vector than the router did.
		s.jit.NewInput()
	}
	// gpt-oss biases the router's logits before the softmax; nil elsewhere.
	s.routerBias(l)
	// The router's logits, before the gate: what a selection's margin is read
	// from. A nil check when no tracer is installed.
	s.m.trace(li, "moe_logits", s.moeProbs[:c.NExpert])
	// s.mv already charged this to opMatVec; move it to its own bucket.
	if s.prof {
		d := time.Now().UnixNano() - t0
		opNanos[opRouter].Add(d)
		opNanos[opMatVec].Add(-d) // s.mv already charged it; do not count it twice
	}
	// The whole router decision is one generated kernel: the selection in
	// descending-probability order (lowest id wins a tie), the same ids
	// ascending, both orders' renormalised weights and the divisor. There is
	// no fallback.
	return s.routeExperts(l)
}

// moeExperts is the mixture's gather, expert matvecs, act_mul and combine
// for one row under the selection moeRoute left in s.route: the selected
// experts paged in, then one grouped region (moeFFN) adding each expert's
// contribution into out in ascending expert id.
func (s *State) moeExperts(li int, l *layer, h, out []float32) error {
	c := s.c
	sel := s.route.Sel
	// The selected experts are read here, the first moment anyone knows which.
	// pageIn deliberately skipped the banks, since a token touches only a few.
	var err error
	if sh := s.shOverlap; sh != nil {
		// The shared expert reads none of the routed pages, so it runs while
		// they are read; its contribution is added where it always was, by
		// the caller, so the sum is the same float sum (placement.md 16c-2).
		s.shOverlap = nil
		if s.expReq == nil {
			// One reader for the State's life, not a goroutine a layer: a go
			// statement with arguments allocates (TestDecodeDoesNotAllocate).
			s.expReq = make(chan expRead)
			go s.expertReader(s.expReq)
		}
		s.expWG.Add(1)
		s.expReq <- expRead{li: li, sel: sel}
		w, serr := s.sharedExpertOut(sh, s.shOverlapH)
		s.expWG.Wait()
		err = s.expErr
		if err == nil {
			err = serr
		}
		s.shW, s.shReady = w, serr == nil
		s.ShOverlaps++
	} else {
		err = s.m.ensureExperts(li, sel, &s.expHold)
	}
	defer s.expHold.release()
	if err != nil {
		return err
	}
	s.res.touch(li, c.NExpert, sel)
	// Hotness is counted here and not in moeExpertsRows: endToken() advances once
	// per Forward, so counting a prefill chunk's rows would push Uses, a
	// per-token probability tier.Place ranks by, above 1.
	if !s.rowsOfChunk {
		s.hot.add(li, sel)
	}
	// The selection is not visible in any output, so it is traced; a nil check
	// when no tracer is installed.
	if s.m.tracer != nil {
		v := make([]float32, len(sel))
		for i, e := range sel {
			v[i] = float32(e)
		}
		s.m.trace(li, "moe_topk", v)
	}

	// Renormalisation is the architecture's choice (llama and qwen3moe do,
	// olmoe does not); without it the kernel divides by one.
	//
	// Contributions are summed in ascending expert id, not selection order,
	// because moeExpertsRows visits the bank expert-major. A different order makes
	// Forward and Prefill/ForwardBatch different float sums, which in a mixture
	// can flip a near-tied top-k in the next layer. Selection order is still
	// what the trace, the counters and the divisor see.
	ord, ow := s.route.Ord, s.route.OWt
	// The k experts go out in one region per stage; the loop below is the
	// fallback. See moeFFN.
	done, err := s.moeFFN(l, ord, h, out, ow)
	if err != nil || done {
		return err
	}
	return s.moeLoop(l, ord, ow, h, out)
}

// denseMoERows is denseMoE over the n rows of a batched chunk -- prefill's
// successive positions or a batch's sequences -- one row at a time, so every
// entry point computes Gemma 4's block as decode does, bit for bit.
//
// It runs row by row, where moeExpertsRows visits a chunk's experts once each;
// batching the experts here is the lever if a Gemma 4 prompt's mixture shows in
// a profile.
func (s *State) denseMoERows(li int, l *layer, h, x []float32, n int) error {
	d := s.c.NEmbd
	s.rowsOfChunk = true
	defer func() { s.rowsOfChunk = false }()
	for i := 0; i < n; i++ {
		s.jit.NewInput()
		if err := s.denseMoE(li, l, h[i*d:(i+1)*d], x[i*d:(i+1)*d]); err != nil {
			return err
		}
	}
	return nil
}

// denseMoE is Gemma 4's mixture block (Config.DenseMoE) for one row, after the
// caller has normed x by ffn_norm into h and called NewInput:
//
//	d = post_norm_1(mlp(h))
//	e = post_norm_2(moe(router: rmsnorm(x) * router_norm, experts: pre_norm_2(x)))
//	x += post_norm(d + e)
//
// The dense MLP is the shared expert's tensors, ungated. h is consumed.
func (s *State) denseMoE(li int, l *layer, h, x []float32) error {
	c := s.c
	if err := s.mv(s.shGate, l.shGate, h); err != nil {
		return err
	}
	if err := s.mv(s.shUp, l.shUp, h); err != nil {
		return err
	}
	s.actmulAll(s.shGate, s.shUp, c.Act)
	s.jit.NewInput()
	if err := s.mv(s.shOut, l.shDown, s.shGate); err != nil {
		return err
	}
	s.rmsnorm(s.shOut, s.shOut, l.postFFNNorm1, c.RMSEps)
	rin, sum := s.dmoe[0], s.dmoe[1]
	s.rmsnorm(rin, x, l.routerNorm, c.RMSEps)
	s.rmsnorm(h, x, l.ffnNorm2, c.RMSEps)
	s.jit.NewInput()
	clear(sum)
	if err := s.moe(li, l, rin, h, sum); err != nil {
		return err
	}
	s.rmsnorm(sum, sum, l.postFFNNorm2, c.RMSEps)
	s.addInto(s.shOut, sum)
	s.rmsnorm(s.shOut, s.shOut, l.postFFNNorm, c.RMSEps)
	s.addInto(x, s.shOut)
	s.jit.NewInput()
	return nil
}

// moeFFN runs the k selected experts' feed-forward with one pool region per
// stage instead of one per expert, and reports whether it served the call.
//
// It is the same arithmetic as the per-expert loop (same bytes, same order;
// only which worker takes which rows moves), so Forward, Prefill and
// ForwardBatch stay bit-identical on a mixture. The shared activation is
// quantized once for all k experts.
//
// It declines whenever the batch is not uniform: gate and up must share a
// type, a shape and the fused kernel's row-group alignment. See AGENTS.md
// ("batched CPU MoE dispatch") for the measurements.
func (s *State) moeFFN(l *layer, ord []int32, h, out []float32, dw []float32) (bool, error) {
	c := s.c
	m := len(ord)
	if m < 2 || s.moePad == 0 || len(s.moeGate) < m*s.moePad ||
		cap(s.moeW) < 2*m || cap(s.moeO) < 2*m {
		return false, nil
	}
	g0 := &l.experts[ord[0]].gate
	if l.ungatedExp {
		g0 = &l.experts[ord[0]].up
	}
	if g0.packed == nil || g0.rows > s.moePad || g0.rows != c.NFFNExp {
		return false, nil
	}
	typ, rows, k := g0.typ, g0.rows, g0.k
	// The capacities above make these appends allocation-free.
	ws, os := s.moeW[:0], s.moeO[:0]
	for i, e := range ord {
		x := &l.experts[e]
		lo := i * s.moePad
		if l.ungatedExp {
			// Up alone, into the gate's slot: the activation runs there in
			// place and down reads it, as for a gated expert.
			t := &x.up
			if t.packed == nil || t.typ != typ || t.rows != rows || t.k != k {
				return false, nil
			}
			ws = append(ws, t.packed)
			os = append(os, s.moeGate[lo:lo+rows])
			continue
		}
		for _, t := range [...]*tensor{&x.gate, &x.up} {
			if t.packed == nil || t.typ != typ || t.rows != rows || t.k != k {
				return false, nil
			}
			ws = append(ws, t.packed)
		}
		os = append(os, s.moeGate[lo:lo+rows], s.moeUp[lo:lo+rows])
	}
	if !s.jit.MatVecPackedMulti(os, typ, ws, h, rows, k) {
		return false, nil
	}
	s.moeBatched.Add(1)
	// Llama 4 weights the expert's INPUT; see weightIn. dw becomes the ones the
	// down projections are then summed at.
	if c.ExpertWeightIn {
		for i := range ord {
			lo := i * s.moePad
			s.weightIn(dw[i], s.moeGate[lo:lo+rows], s.moeUp[lo:lo+rows])
		}
		dw = s.moeOnes[:m]
	}
	// gpt-oss biases every expert's gate and up; nil for every other mixture.
	// Only the expert's own rows, so the slack past them stays zero.
	for i, e := range ord {
		lo := i * s.moePad
		s.addBias(s.moeGate[lo:lo+rows], expBias(l.expGateB, int(e), rows))
		s.addBias(s.moeUp[lo:lo+rows], expBias(l.expUpB, int(e), rows))
	}
	s.traceExperts(ord, "moe_gate", s.moeGate, s.moePad, rows)
	s.traceExperts(ord, "moe_up", s.moeUp, s.moePad, rows)
	// One SwiGLU over every expert. The slack between rows and moePad is
	// zero and the activation maps zero to zero, so it stays zero.
	n := m * s.moePad
	// A gated bank's SwiGLU goes into the down projections' quantize region
	// (MatVecPackedGatherAct): one pool region fewer a call, the same kernel
	// and bits. Not while tracing, which reads the activation between.
	fuseAct := !l.ungatedExp && s.m.tracer == nil
	switch {
	case l.ungatedExp:
		s.actAll(s.moeGate[:n], c.Act)
	case !fuseAct:
		s.actmulAll(s.moeGate[:n], s.moeUp[:n], c.Act)
	}
	s.traceExperts(ord, "moe_act", s.moeGate, s.moePad, rows)
	// dw is the routed weights in the ascending-expert-id order the
	// contributions are summed in -- the router kernel's second weight output,
	// not a re-derivation of it from moeProbs and a divisor.
	// The down projection batches under a different contract: each expert
	// reads its own SwiGLU output, so MatVecPackedGather quantizes k
	// activations in one region and runs k matvecs in one more.
	d0 := &l.experts[ord[0]].down
	ew := c.ExpWidth()
	ok := d0.packed != nil && d0.rows == ew && d0.k == rows &&
		len(s.moeDown) >= m*ew
	ps, os := s.moeW[:0], s.moeO[:0]
	if ok {
		for _, e := range ord {
			t := &l.experts[e].down
			if t.packed == nil || t.typ != d0.typ || t.rows != d0.rows || t.k != d0.k {
				ok = false
				break
			}
			ps = append(ps, t.packed)
		}
	}
	if ok {
		for j := 0; j < m; j++ {
			os = append(os, s.moeDown[j*ew:(j+1)*ew])
		}
		if fuseAct {
			ok = s.jit.MatVecPackedGatherAct(os, d0.typ, ps, s.moeGate, s.moeUp, c.Act, s.moePad, ew, rows)
		} else {
			ok = s.jit.MatVecPackedGather(os, d0.typ, ps, s.moeGate, s.moePad, ew, rows)
		}
	}
	if !ok && fuseAct {
		// Declined before it touched an activation: the SwiGLU on its own.
		s.actmulAll(s.moeGate[:n], s.moeUp[:n], c.Act)
	}
	if !ok {
		// The per-expert down, for a non-uniform or unpacked bank;
		// bit-identical to the batch by construction.
		s.moeDownLooped.Add(1)
		for i, e := range ord {
			lo := i * s.moePad
			s.jit.NewInput()
			if err := s.mv(s.moeDown[:ew], l.experts[e].down, s.moeGate[lo:lo+rows]); err != nil {
				return true, err
			}
			s.addBias(s.moeDown[:ew], expBias(l.expDownB, int(e), ew))
			s.axpy(out, s.moeDown[:ew], dw[i])
		}
		return true, nil
	}
	s.moeDownBatched.Add(1)
	// The down bias joins each expert's projection BEFORE its weight, as the
	// looped arm and moeExpertsRows add it, so the three stay the same float sum.
	if l.expDownB != nil {
		for j, e := range ord {
			s.addBias(s.moeDown[j*ew:(j+1)*ew], expBias(l.expDownB, int(e), ew))
		}
	}
	s.traceExperts(ord, "moe_down", s.moeDown, ew, ew)
	s.moeReduce(out, dw, ew)
	return true, nil
}

// weightIn applies an expert's routed weight w to its INPUT when the
// architecture says so (Llama 4), and returns the weight its output is then
// summed at: w itself for every other mixture, and one here.
//
// It scales the gate and up projections rather than h, which is the same
// function since both are linear, and keeps h's single shared quantization.
// The bias is added after, where W(w*h)+b puts it. It is not w*silu(g)*u:
// silu(w*g)*(w*u) is a different curve.
func (s *State) weightIn(w float32, gate, up []float32) float32 {
	if !s.c.ExpertWeightIn {
		return w
	}
	s.scale(gate, w)
	s.scale(up, w)
	return 1
}

// moeReduce adds every expert's down projection into the residual, weighted, in
// one pool region, rather than k serial nn.Axpy32JIT calls on the caller while
// the workers spin.
//
// Each worker owns a disjoint row range and walks the experts ascending inside
// it, so every element sees exactly the per-expert loop's sequence of adds
// (see moe()).
func (s *State) moeReduce(out, w []float32, n int) {
	lanes := nn.ElemLanes
	v := n / lanes
	if v == 0 {
		for j := range w {
			s.axpy(out[:n], s.moeDown[j*n:(j+1)*n], w[j])
		}
		return
	}
	j := &s.rg.elem
	j.op, j.a, j.b, j.n, j.v = elemReduce, out, w, n, v
	s.elemRun()
}

// sharedExpert adds the always-on feed-forward, scaled by its own sigmoid gate.
//
// The gate is a vector producing one logit whose sigmoid scales the whole
// contribution; nothing is selected or renormalised. It is added after the
// routed sum, so the routed renormalisation does not rescale it.
func (s *State) sharedExpert(l *layer, h, out []float32) error {
	// Presence is the weights, not the gate: DeepSeek has a shared expert and
	// no gate (weight 1). Test rows, not data: a paged matrix's data is nil
	// while its page is out, but rows is set when the tensor is bound.
	if l.shGate.rows == 0 && l.shUp.rows == 0 {
		return nil
	}
	w, err := s.sharedExpertOut(l, h)
	if err != nil {
		return err
	}
	s.axpy(out, s.shOut, w)
	s.jit.NewInput()
	return nil
}

// expRead is one layer's routed read handed to the State's reader.
type expRead struct {
	li  int
	sel []int32
}

// expertReader runs ensureExperts for each request until the channel closes
// (State.Close), its error left in s.expErr: the routed read moe runs
// behind the shared expert.
func (s *State) expertReader(req chan expRead) {
	for r := range req {
		s.expErr = s.m.ensureExperts(r.li, r.sel, &s.expHold)
		s.expWG.Done()
	}
}

// sharedExpertOut computes the shared expert's output into s.shOut and
// returns the weight it is added at, adding nothing: sharedExpert adds it,
// and moe's overlap leaves the add to its caller.
func (s *State) sharedExpertOut(l *layer, h []float32) (float32, error) {
	if l.shGate.rows == 0 && l.shUp.rows == 0 {
		return 0, nil
	}
	// The gate logit is a 1 x NEmbd F32 matvec written straight into s.sg[0].
	// MatVecHost, not MatVec: this only runs for a block the host is running,
	// so offering one row to the device would be a pointless round trip.
	if l.shRouter != nil {
		if !s.jit.MatVecHost(s.sg[:1], quant.F32, f32Bytes(l.shRouter), h, 1, len(l.shRouter)) {
			return 0, fmt.Errorf("model: no kernel for the shared expert's gate (F32, 1 x %d)",
				len(l.shRouter))
		}
	}
	if l.shGate.rows == 0 {
		// Ungated (Nemotron 3): down(act(up(h))).
		if err := s.mv(s.shGate, l.shUp, h); err != nil {
			return 0, err
		}
		s.actAll(s.shGate, s.c.Act)
	} else {
		if err := s.mv(s.shGate, l.shGate, h); err != nil {
			return 0, err
		}
		if err := s.mv(s.shUp, l.shUp, h); err != nil {
			return 0, err
		}
		s.actmulAll(s.shGate, s.shUp, s.c.Act)
	}
	s.jit.NewInput()
	if err := s.mv(s.shOut, l.shDown, s.shGate); err != nil {
		return 0, err
	}
	// The gate is one logit, and the generated sigmoid takes it as a
	// one-element vector: sg[0] = sigma(logit) * sg[1], with sg[1] = 1.
	//
	// An ungated shared expert contributes at weight 1 and skips the sigmoid;
	// a zero logit would give 0.5.
	w := float32(1)
	if l.shRouter != nil {
		s.sg[1] = 1
		nn.SigmoidMul32JIT(s.sg[:1], s.sg[1:])
		w = s.sg[0]
	}
	// Granite's residual scale, which the mixture cannot take at an add of
	// its own: it adds straight into the residual. See addInto.
	w *= float32(s.c.ResidualScale)
	s.jit.NewInput()
	return w, nil
}

// moeBatchFrom is moeBatch with the router reading rin (n rows of n_embd)
// and the experts h (n rows of the expert width), writing resid at the
// expert width: a latent mixture's routed half (k3MoEBatch), which adds its
// shared expert itself.
func (s *State) moeBatchFrom(li int, l *layer, rin, h, resid []float32, n int) error {
	if err := s.moeRouteRows(li, l, rin, n); err != nil {
		return err
	}
	if err := s.moeExpertsRows(li, l, h, resid, n); err != nil {
		return err
	}
	if s.c.ExpertLatent != 0 {
		return nil
	}
	return s.sharedExpertRows(l, h, resid, n)
}

// moeRouteRows is the route op over n rows of rin: every row's selection
// and weights, in s.bsel and s.bw, before any expert runs.
func (s *State) moeRouteRows(li int, l *layer, rin []float32, n int) error {
	c := s.c
	k := c.NExpertUsed

	// Routing every row in one batched pass over the router was measured and
	// is not faster: a router is ~1 MiB per layer and stays in cache across
	// rows, and at a 128-row chunk the batch was 0.90x.
	//
	// The selection has to be known for all rows before the experts can be
	// visited in order.
	for i := 0; i < n; i++ {
		s.jit.NewInput()
		if err := s.mv(s.moeProbs, l.router, rin[i*c.NEmbd:(i+1)*c.NEmbd]); err != nil {
			return err
		}
		s.routerBias(l)
		if err := s.routeExperts(l); err != nil {
			return err
		}
		sel := s.route.Sel
		if s.m.tracer != nil {
			v := make([]float32, len(sel))
			for j, e := range sel {
				v[j] = float32(e)
			}
			s.m.trace(li, "moe_topk", v)
		}
		s.res.touch(li, c.NExpert, sel)
		// Selection-order weights: the storage order here is only an index,
		// since the bank is then visited expert-major, matching decode's
		// ascending sum.
		for j, e := range sel {
			s.bsel[i*k+j] = e
			s.bw[i*k+j] = s.route.Wt[j]
		}
	}
	return nil
}

// moeExpertsRows is the gather, expert matvecs, act_mul and combine over n
// rows under moeRouteRows' selections, expert-major: each selected expert
// paged in once and run over every row that chose it, into resid.
func (s *State) moeExpertsRows(li int, l *layer, h, resid []float32, n int) error {
	c := s.c
	k := c.NExpertUsed
	s.growMoEBatch(n)
	for e := 0; e < c.NExpert; e++ {
		// One expert held at a time: a chunk routes to most of a layer's
		// experts, and pinning them all would exceed a budget sized for one
		// token's selection.
		used := false
		for i := 0; i < n*k && !used; i++ {
			used = s.bsel[i] == int32(e)
		}
		if !used {
			continue
		}
		one := [1]int32{int32(e)}
		err := s.m.ensureExperts(li, one[:], &s.expHold)
		if err == nil {
			err = s.moeBatchExpert(l, e, h, resid, n)
		}
		s.expHold.release()
		if err != nil {
			return err
		}
	}
	return nil
}

// sharedExpertRows is the shared expert over n rows, one row at a time, so
// prefill and decode compute the same function.
func (s *State) sharedExpertRows(l *layer, h, resid []float32, n int) error {
	c := s.c
	for i := 0; i < n; i++ {
		row := h[i*c.NEmbd : (i+1)*c.NEmbd]
		out := resid[i*c.NEmbd : (i+1)*c.NEmbd]
		s.jit.NewInput()
		if err := s.sharedExpert(l, row, out); err != nil {
			return err
		}
	}
	return nil
}

// f32Bytes is a float32 slice addressed as the bytes a kernel reads. The
// container stores an F32 router verbatim, so this is the same memory and no
// copy: nn.JIT.MatVec takes a weight as bytes because every other format
// arrives that way.
func f32Bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// routerBias adds the router's bias to its logits in moeProbs.
func (s *State) routerBias(l *layer) {
	if l.routerB != nil {
		s.axpy(s.moeProbs[:len(l.routerB)], l.routerB, 1)
	}
}

// traceExperts hands the tracer each selected expert's row of buf, laid out
// at stride pad, as "name/e". A nil check when no tracer is installed.
func (s *State) traceExperts(ord []int32, name string, buf []float32, pad, n int) {
	if s.m.tracer == nil {
		return
	}
	for i, e := range ord {
		s.m.trace(s.traceLayer, name+"/"+itoa(int(e)), buf[i*pad:i*pad+n])
	}
}

// expBias is expert e's slice of a bank bias that holds n values per expert,
// or nil when the model has none.
func expBias(b []float32, e, n int) []float32 {
	if b == nil {
		return nil
	}
	return b[e*n : (e+1)*n]
}

// MoEDispatch reports how many mixture FFNs ran batched (one region per stage)
// and how many ran as the per-expert loop. A model with no mixture reports
// (0, 0); a nonzero second number means the fallback ran.
func (s *State) MoEDispatch() (batched, looped int64) {
	return s.moeBatched.Load(), s.moeLooped.Load()
}

// MoEDownDispatch reports the same for the DOWN projection, which batches under
// its own contract (one activation per expert) and declines independently.
//
// A separate counter, because the loop computes the same answer and only a
// counter can show which path ran.
func (s *State) MoEDownDispatch() (batched, looped int64) {
	return s.moeDownBatched.Load(), s.moeDownLooped.Load()
}

// moeRouteGate describes this model's router to the generated kernel.
//
// Its shape is compiled into the kernel key (nn.MoEGate.baked()); the bias
// values and the scale's magnitude are read at run time, so one kernel serves
// every block.
//
// The two biases are different tensors: l.routerB (gpt-oss) is added to the
// router's logits by routerBias; l.expProbsB (DeepSeek) is added to the gated
// scores for selection only. See jlm.RoleExpProbsB.
func (s *State) moeRouteGate(l *layer) nn.MoEGate {
	c := s.c
	bias := l.expProbsB
	// DeepSeek V4's hash blocks select the token's own experts: a mask as the
	// selection bias (ds4HashBias), the weights still the gate's.
	if l.ds4 != nil && l.ds4.hash != nil && c.ds4Fault != ds4FaultScoreRoute {
		bias = s.ds.hashBias
	}
	return nn.MoEGate{
		Sigmoid:      c.ExpertSigmoid,
		SqrtSoftplus: c.ExpertSqrtSoftplus,
		Bias:         bias,
		NGroup:       c.NExpertGroup,
		NGroupUsed:   c.NExpertGroupUsed,
		Norm:         !c.NoExpertNorm,
		Scale:        float32(c.routedScale()),
		ExpScale:     l.expScale,
		SparseMixer:  c.SparseMixer,
	}
}

// routeExperts gates the router's logits and selects the top k, in one
// generated kernel. It takes raw logits: MoERouteJIT applies the gating
// itself, so softmaxing first would compute softmax(softmax(x)).
func (s *State) routeExperts(l *layer) error {
	c := s.c
	if !nn.MoERouteJIT(s.moeProbs, s.moeRouteGate(l), s.route) {
		return fmt.Errorf("model: no kernel for the mixture router's top-k "+
			"(%d experts, %d used, %d group(s))", c.NExpert, c.NExpertUsed, c.NExpertGroup)
	}
	return nil
}

// expSelBias is a representative selection bias, or nil when this model has
// none: it answers the shape question NewMoERouteFor asks, not a value one.
// A model must not mix biased and unbiased blocks (loadConfig refuses it),
// since the kernel bakes bias presence into its key.
func (m *Model) expSelBias() []float32 {
	for li := range m.layers {
		if m.Cfg.MoEAt(li) && m.layers[li].expProbsB != nil {
			return m.layers[li].expProbsB
		}
	}
	return nil
}

// moeBatchExpert runs expert e of a prefill chunk's layer over every row that
// selected it, adding each result into that row's residual. The caller holds
// the expert's page.
func (s *State) moeBatchExpert(l *layer, e int, h, resid []float32, n int) error {
	c := s.c
	k := c.NExpertUsed
	x := &l.experts[e]
	F, D := c.NFFNExp, c.ExpWidth()
	// The rows that chose this expert go through it as one batch, as ggml's
	// mul_mat_id does, so its matrices are read once per token block rather
	// than once per row. It is bit-identical to the per-row loop
	// (nn.TestMatMulPackedMatchesTheRowLoopExactly), and the residual adds
	// keep their order: expert-major, rows ascending.
	m := 0
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			if s.bsel[i*k+j] == int32(e) {
				s.bmRow[m] = int32(i)
				s.bmW[m] = s.bw[i*k+j]
				m++
				break
			}
		}
	}
	if m == 0 {
		return nil
	}
	X, G, U, Y := s.bmX[:m*D], s.bmG[:m*F], s.bmU[:m*F], s.bmY[:m*D]
	for r := 0; r < m; r++ {
		i := int(s.bmRow[r])
		copy(X[r*D:(r+1)*D], h[i*D:(i+1)*D])
	}
	s.jit.NewInput()
	if l.ungatedExp {
		// Up alone into G, the activation in place: what down reads.
		if err := s.mm(G, x.up, X, m); err != nil {
			return err
		}
		s.actAll(G, c.Act)
	} else {
		if err := s.mm(G, x.gate, X, m); err != nil {
			return err
		}
		if err := s.mm(U, x.up, X, m); err != nil {
			return err
		}
		for r := 0; r < m; r++ {
			g, u := G[r*F:(r+1)*F], U[r*F:(r+1)*F]
			s.bmW[r] = s.weightIn(s.bmW[r], g, u)
			s.addBias(g, expBias(l.expGateB, e, F))
			s.addBias(u, expBias(l.expUpB, e, F))
		}
		s.actmulAll(G, U, c.Act)
	}
	s.jit.NewInput()
	if err := s.mm(Y, x.down, G, m); err != nil {
		return err
	}
	for r := 0; r < m; r++ {
		i := int(s.bmRow[r])
		y := Y[r*D : (r+1)*D]
		s.addBias(y, expBias(l.expDownB, e, D))
		// Straight into the residual, in ascending expert id, the order
		// moe() sums in.
		s.axpy(resid[i*D:(i+1)*D], y, s.bmW[r])
	}
	return nil
}

// growMoEBatch sizes the per-expert gather moeBatchExpert runs a chunk of n
// rows through. Grows, never shrinks.
func (s *State) growMoEBatch(n int) {
	c := s.c
	if len(s.bmRow) >= n {
		return
	}
	// A previous State's set where it is big enough; see hostBatch.
	h := s.spareHost
	s.spareHost.bmRow = nil
	if cap(h.bmRow) >= n {
		s.bmRow = h.bmRow[:n]
	} else {
		s.bmRow = make([]int32, n)
	}
	s.bmW = scratch(h.bmW, n, 0)
	s.bmX = scratch(h.bmX, n*c.NEmbd, 0)
	s.bmY = scratch(h.bmY, n*c.NEmbd, 0)
	s.bmG = scratch(h.bmG, n*c.NFFNExp, ffnPad(n*c.NFFNExp))
	s.bmU = scratch(h.bmU, n*c.NFFNExp, ffnPad(n*c.NFFNExp))
}

// hostExpertsFor is block li's nn.LayerWeights.HostExperts: the routed
// experts on the host for a device running the rest of the block. nil for a
// block with no mixture, and for a mixture whose experts read anything but
// the one vector they are given (a dense MLP beside them, per-expert scales,
// ungated experts): those stay whole on one tier.
func (s *State) hostExpertsFor(li int) func(sel []uint32, w, in, out []float32) error {
	c := s.c
	if !c.MoEAt(li) || c.DenseMoE || s.m.layers[li].ungatedExp {
		return nil
	}
	return func(sel []uint32, w, in, out []float32) error { return s.hostExperts(li, sel, w, in, out) }
}

// hostExpertsRowsFor is block li's nn.LayerWeights.HostExpertsRows, nil where
// hostExpertsFor is.
func (s *State) hostExpertsRowsFor(li int) func(sel []uint32, w, in, out []float32, rows, k int) error {
	if s.hostExpertsFor(li) == nil {
		return nil
	}
	return func(sel []uint32, w, in, out []float32, rows, k int) error {
		return s.hostExpertsRows(li, sel, w, in, out, rows, k)
	}
}

// hostRowsByRow is the most rows a chunk's host experts run a row at a time
// (hostExperts per row) rather than expert-major: a short prompt's rows share
// few experts, so the expert-major pass reads about the same bytes in a
// region or more per expert.
const hostRowsByRow = 8

// hostExpertsRows is moeExpertsRows' expert-major pass with the device's
// selections and weights: each expert the chunk chose is read once and run
// over its rows as one batch (moeBatchExpert), summed into out in ascending
// expert id, rows ascending -- the order the host's own batched mixture sums.
func (s *State) hostExpertsRows(li int, sel []uint32, w, in, out []float32, rows, k int) error {
	c := s.c
	l := &s.m.layers[li]
	if k != c.NExpertUsed {
		return fmt.Errorf("model: block %d: %d routed experts a row, the model routes %d", li, k, c.NExpertUsed)
	}
	if rows <= hostRowsByRow {
		// A few rows read nearly every expert they choose once each anyway,
		// and the decode path runs a row's selection as a handful of grouped
		// regions where the expert-major pass runs regions per expert.
		D := c.ExpWidth()
		for i := 0; i < rows; i++ {
			if err := s.hostExperts(li, sel[i*(k+1):i*(k+1)+k], w[i*k:(i+1)*k],
				in[i*D:(i+1)*D], out[i*D:(i+1)*D]); err != nil {
				return err
			}
		}
		return nil
	}
	s.hostOnly = true
	defer func() { s.hostOnly = false }()
	if len(s.bsel) < rows*k {
		s.bsel = make([]int32, rows*k)
		s.bw = make([]float32, rows*k)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < k; j++ {
			s.bsel[i*k+j] = int32(sel[i*(k+1)+j])
			s.bw[i*k+j] = w[i*k+j]
		}
	}
	s.growMoEBatch(rows)
	clear(out)
	for e := 0; e < c.NExpert; e++ {
		used := false
		for i := 0; i < rows*k && !used; i++ {
			used = s.bsel[i] == int32(e)
		}
		if !used {
			continue
		}
		one := [1]int32{int32(e)}
		err := s.m.pageInSelected(li, []uint32{uint32(e)})
		if err == nil {
			err = s.m.ensureExperts(li, one[:], &s.expHold)
		}
		if err == nil {
			err = s.moeBatchExpert(l, e, in, out, rows)
		}
		s.expHold.release()
		if err != nil {
			return err
		}
	}
	return nil
}

// hostExperts is hostExpertsFor's body: the selection put in ascending id
// with its weights, as moe sums them (the batched path visits the bank in
// that order), the pages read, and moeFFN over in into out. It is moe() from
// its selection on, with the device's selection and weights in place of the
// host router's.
func (s *State) hostExperts(li int, sel []uint32, w, in, out []float32) error {
	l := &s.m.layers[li]
	if l.ungatedExp {
		return fmt.Errorf("model: block %d: ungated experts do not run split across tiers", li)
	}
	s.hostCalls++
	s.m.hostExperts.Store(true)
	k := len(sel)
	s.hyOrd = slices.Grow(s.hyOrd[:0], k)[:k]
	s.hyW = slices.Grow(s.hyW[:0], k)[:k]
	for i, e := range sel {
		// Insertion by id: k is a handful.
		j := i
		for j > 0 && s.hyOrd[j-1] > int32(e) {
			s.hyOrd[j], s.hyW[j] = s.hyOrd[j-1], s.hyW[j-1]
			j--
		}
		s.hyOrd[j], s.hyW[j] = int32(e), w[i]
	}
	s.hostOnly = true
	defer func() { s.hostOnly = false }()
	// A bank not split into expert pages lives in the block's own page, which
	// nothing else on the host has reason to keep in: read the selected
	// sheets' ranges of it (a no-op for a bank in expert pages).
	if err := s.m.pageInSelected(li, sel); err != nil {
		return err
	}
	if err := s.m.ensureExperts(li, s.hyOrd, &s.expHold); err != nil {
		s.expHold.release()
		return err
	}
	defer s.expHold.release()
	clear(out)
	s.jit.NewInput()
	done, err := s.moeFFN(l, s.hyOrd, in, out, s.hyW)
	if err != nil {
		return err
	}
	if !done {
		if err := s.moeLoop(l, s.hyOrd, s.hyW, in, out); err != nil {
			return err
		}
	}
	s.jit.NewInput()
	return nil
}

// moeLoop is moeFFN's fallback, one generated matvec per expert and
// projection, summing each expert's weighted output into out: the path a
// mixture with no fused expert kernel takes (an F32 bank).
func (s *State) moeLoop(l *layer, ord []int32, ow []float32, h, out []float32) error {
	c := s.c
	// The fallback loops the down projection too, so it is counted here;
	// moeFFN returned before reaching its own counter.
	s.moeLooped.Add(1)
	s.moeDownLooped.Add(1)
	ff := c.NFFNExp
	for i, e := range ord {
		w := ow[i]
		x := &l.experts[e]
		if l.ungatedExp {
			// No gate: the activation alone, in the gate's buffer, which
			// down reads.
			if err := s.mv(s.moeGate[:ff], x.up, h); err != nil {
				return err
			}
			s.actAll(s.moeGate[:ff], c.Act)
		} else {
			// gate and up read h, which is still the cached quantization.
			if err := s.mv(s.moeGate[:ff], x.gate, h); err != nil {
				return err
			}
			if err := s.mv(s.moeUp[:ff], x.up, h); err != nil {
				return err
			}
			w = s.weightIn(w, s.moeGate[:ff], s.moeUp[:ff])
			s.addBias(s.moeGate[:ff], expBias(l.expGateB, int(e), ff))
			s.addBias(s.moeUp[:ff], expBias(l.expUpB, int(e), ff))
			s.actmulAll(s.moeGate[:ff], s.moeUp[:ff], c.Act)
		}
		s.jit.NewInput()
		ew := c.ExpWidth()
		if err := s.mv(s.moeDown[:ew], x.down, s.moeGate[:ff]); err != nil {
			return err
		}
		s.addBias(s.moeDown[:ew], expBias(l.expDownB, int(e), ew))
		s.axpy(out, s.moeDown[:ew], w)
		s.jit.NewInput()
	}
	return nil
}
