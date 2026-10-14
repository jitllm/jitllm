package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
)

// ForwardBatch runs one token for each of the session's nseq sequences and
// returns their logits, nseq rows of NVocab.
//
// Decode is memory-bound: one pass over the weights serves one token. Running
// N sequences in lockstep spends that pass on N tokens, which is Prefill's
// trade and the same batched kernels (every matvec a matmul over the batch).
// Attention is the one op that differs: Prefill's rows are one sequence at
// successive positions, while a batch's rows are different sequences, each
// attending into its own cache at its own position.
func (s *State) ForwardBatch(tokens []int32) ([]float32, error) {
	if !s.batched {
		return nil, errBatch{}
	}
	n := s.nseq
	if len(tokens) != n {
		return nil, errBatchWidth{len(tokens), n}
	}
	seq, pos := make([]int, n), make([]int, n)
	for i := range seq {
		seq[i], pos[i] = i, s.bpos[i]
	}
	return s.forwardRows(tokens, seq, pos, n)
}

// forwardRows is one batch step over rows that need not be one per sequence:
// row r is tokens[r] for the sequence in slot seq[r] at position pos[r]. A
// sequence may take several rows, which is a prompt chunk riding in a decode
// step (Scheduler): its positions in the step must be exactly its next ones,
// bpos[seq].. upward, each once, in any row order. Every row writes its K and
// V before any row attends, on the host and on the device, so a chunk's row
// sees the rows of its own sequence at lower positions in the same step, and
// its own position bounds what it reads.
//
// Only the first nlogit rows' logits are wanted, and only they are projected
// and returned, nlogit rows of NVocab: a prompt chunk needs the logits of its
// last row at most, and a 128256-wide head over every row of a chunk would
// cost a chunk's worth of projections and, on a device, their readback.
//
// Each sequence's position moves on by the rows it took.
func (s *State) forwardRows(tokens []int32, seq, pos []int, nlogit int) ([]float32, error) {
	defer s.m.enterPager()()
	c := s.c
	if !s.batched {
		return nil, errBatch{}
	}
	n := len(tokens)
	if len(seq) != n || len(pos) != n || n == 0 || nlogit < 0 || nlogit > n {
		return nil, fmt.Errorf("model: a batch step of %d tokens, %d slots, %d positions and %d logit rows",
			n, len(seq), len(pos), nlogit)
	}
	if err := s.retireErr; err != nil {
		s.retireErr = nil
		return nil, err
	}
	took, err := s.rowsTake(seq, pos)
	if err != nil {
		return nil, err
	}
	for _, id := range tokens {
		if int(id) < 0 || int(id) >= c.NVocab {
			return nil, errToken{id, c.NVocab}
		}
	}
	// A device that cannot hold the batch's history hands blocks to the host
	// (or a later card) before the step, as Forward does; every row's history
	// travels with them.
	s.relocateRows(took)
	// A model wholly on one device runs the ragged step there. Otherwise the
	// device's blocks go as rows with their own slots (nn.RowsDevice): the
	// ordinary per-sequence call would write every row's history into one
	// sequence's cache.
	if _, ok := s.onRowsDevice(); ok && s.devCount() > 0 {
		return s.forwardRowsDevice(tokens, seq, pos, took, nlogit)
	}
	// A split placement runs its host blocks here and hands each run of device
	// blocks to the device as rows (rowsRun); the projection rides the last run
	// when the device holding the last block holds it.
	_, rowsDev := s.ld.(nn.RowsDevice)
	if s.devCount() > 0 && !rowsDev {
		return nil, errBatchDevice{}
	}
	lg, err := s.rowsHost(tokens, seq, pos, nlogit, s.nseq)
	if err != nil {
		return nil, err
	}
	s.advanceSeqs(took)
	// Seal and trim against the slowest row, not s.pos: a page holds every
	// row's slots, so a page the furthest row has left is one a lagging row may
	// still be writing. (Positions advance in advanceSeqs, not via advance().)
	if low := s.lowPos(); low > 0 {
		s.kv.seal(low)
		s.kv.trim(low)
	}
	// A page the store could not return fails the request; see kvFault.
	if err := s.kvCheck(); err != nil {
		return nil, err
	}
	return lg, nil
}

// rowOwner is the State whose history row i of a ragged step reads and
// writes, and the slot there: the batch's own slot seq[i], or in a step across
// sessions (stepHost) the row's own session, whose one sequence is slot 0.
func (s *State) rowOwner(i int, seq []int) (*State, int) {
	if len(s.rowOwn) > 0 {
		return s.rowOwn[i], 0
	}
	return s, seq[i]
}

// rowsHost runs a ragged step's blocks and head and returns the wanted rows'
// logits; the caller moves the positions on. seqs is how many sequences the
// rows hold: a step of one row per sequence fans attention out over (row,
// head) units.
func (s *State) rowsHost(tokens []int32, seq, pos []int, nlogit, seqs int) ([]float32, error) {
	m, c := s.m, s.c
	n := len(tokens)
	// The ragged head applies the final softcap itself, as decode's head does.
	// AltUp's head reads the streams' mean, which the host takes (altup.go).
	headOnDev := s.devCount() > 0 && s.headWithLastBlock() && !c.streamHead()
	s.growBatch(n)
	if len(s.bsel) < n*c.NExpertUsed && c.MoE() {
		s.bsel = make([]int32, n*c.NExpertUsed)
		s.bw = make([]float32, n*c.NExpertUsed)
	}
	// Sized apart: a device batch step allocates blogits and never the score
	// rows, so a batch whose blocks came home finds the one without the other.
	if len(s.blogits) < nlogit*c.NVocab {
		s.blogits = make([]float32, nlogit*c.NVocab)
	}
	// Attention fans out over (row, head) units, each with its own scores row.
	// A step of one row per sequence is that; one carrying a prompt chunk goes
	// a row per unit, heads in turn, as Prefill does -- a scores row per (row,
	// head) would be rows x heads x context, a quarter of a GiB for 512 rows of
	// 32 heads at a 4096 context.
	unitHeads := 1
	if n > seqs {
		unitHeads = c.NHead
	}
	units := n * c.NHead / unitHeads
	// A row reaches keys below its own position plus one (rowsStride).
	reach := 0
	for i := 0; i < n; i++ {
		reach = max(reach, pos[i]+1)
	}
	astride := s.rowsStride(reach)
	// Across sessions a row's reach is its own session's, which may run
	// past the leading State's context, where rowsStride stops.
	if len(s.rowOwn) > 0 && astride < attStride(reach+s.m.sinkSlot()) {
		r := scoreReachMin
		for r < reach {
			r *= 2
		}
		astride = attStride(r + s.m.sinkSlot())
	}
	if len(s.bathf) < units*astride {
		s.bathf = make([]float32, units*astride)
	}

	// c.AttnScale rather than 1/sqrt(hd): they differ under YaRN (DeepSeek).
	// The widths are per layer (HeadDimAt).
	scale := c.AttnScale
	nh := c.NHead
	nrot := c.RopeW()
	// One rotary table per row, rows being at different positions: the prompt
	// chunk's tables, since a step may carry more rows than there are slots.
	cs, csSWA := s.bcs[:n*nrot], s.bswaTable(n, c.NRotSWA)

	t0 := s.tick()
	for i := 0; i < n; i++ {
		row := s.bx[i*c.NEmbd : (i+1)*c.NEmbd]
		if err := m.embedRow(row, int(tokens[i])); err != nil {
			return nil, err
		}
		if c.EmbdScale != 1 {
			s.scale(row, float32(c.EmbdScale))
		}
		// Each row at its own position (starcoder's learned table).
		if err := m.addPos(row, s.prow, pos[i]); err != nil {
			return nil, err
		}
		if c.PLEDim != 0 {
			w := c.pleWidth()
			s.growBPLE(n)
			if err := s.pleInputs(s.bple[i*w:(i+1)*w], tokens[i], row); err != nil {
				return nil, err
			}
		}
	}

	if c.AltUp != 0 {
		if err := s.altExpand(s.bx[:n*c.ResidW()], n); err != nil {
			return nil, err
		}
	}
	if c.DSV4() {
		s.ds4Expand(s.bx[:n*c.ResidW()], n)
	}
	if c.ResAttn() {
		s.k3Expand(s.bx[:n*c.ResidW()], n)
	}
	if c.Q4E() {
		s.growQ4E(n)
		s.q4eExpand(s.bx[:n*c.ResidW()], n)
	}
	s.tock(opResid, t0)

	// At each row's rotary position, which an image earlier in its sequence
	// has moved off its cache position (mrope.go).
	for i := 0; i < n; i++ {
		o, slot := s.rowOwner(i, seq)
		s.ropeRow(i, o.ropeOf(slot, pos[i]))
	}

	var devSlot []int
	if s.devCount() > 0 {
		devSlot = s.flatSlots(seq, pos)
	}
	// A recurrent block steps a sequence's summary one row at a time, so its
	// rows go in position order: a chunk's rows may stand in any order in the
	// step (its logit row leads), and its summary must see them as written.
	var causal []int
	if s.recurrent() {
		causal = slices.Grow(s.rowCausal[:0], n)[:n]
		s.rowCausal = causal
		for i := range causal {
			causal[i] = i
		}
		slices.SortStableFunc(causal, func(a, b int) int { return pos[a] - pos[b] })
	}
	for li := s.lo; li < s.hi; li++ {
		if s.devAt(li) {
			hi := li + 1
			for hi < s.hi && s.devAt(hi) {
				hi++
			}
			withHead := headOnDev && hi == s.hi
			s.tokenIDs(tokens)
			// Qwen3.8-Flash-Next's n-gram rows, when the run holds its n-gram
			// block, each sequence's in position order.
			if c.NGramBlock != 0 {
				s.q4.setRows(s, seq, tokens)
				if err := s.ngramToDevice(li, hi, n, causal, s.q4.rowsOwner, s.q4.callTok); err != nil {
					return nil, err
				}
			}
			if err := s.rowsRun(li, hi, pos, devSlot, s.bx[:n*c.ResidW()], cs, csSWA,
				withHead, s.blogits[:nlogit*c.NVocab], nlogit); err != nil {
				return nil, err
			}
			li = hi - 1
			continue
		}
		// Fault the block in. Skipping this is invisible while the model fits,
		// and under paging reads another block's bytes from a reused frame
		// (see TestForwardBatchFaultsItsBlocksIn).
		if err := m.pageIn(li); err != nil {
			return nil, err
		}
		l := &m.layers[li]
		// DeepSeek V4's block is its own (ds4.go): each row its sequence's
		// slot, position and token.
		if c.DSV4() {
			s.growDS4(n)
			for i := 0; i < n; i++ {
				s.ds.rslot[i], s.ds.rpos[i], s.ds.rid[i] = seq[i], pos[i], tokens[i]
			}
			if err := s.ds4Block(li, l, s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
			continue
		}
		if c.AltUp != 0 {
			s.growAltUpRows(n)
			if err := s.altPredict(l, s.bx[:n*c.ResidW()], s.baltPred, n); err != nil {
				return nil, err
			}
		}

		t0 = s.tick()
		// Kimi-K3's attention reads its mix over the bank (k3.go).
		attnIn := s.bx
		if c.Q4E() {
			// The n-gram embedding in each sequence's position order, then
			// the attention's hyper-connection mix (q4e.go).
			x := s.bx[:n*c.ResidW()]
			if c.NGramAt(li) {
				if err := s.ngramApply(li, l, x, n, causal, func(i int) (*State, int) { return s.rowOwner(i, seq) },
					func(i int) int32 { return tokens[i] }); err != nil {
					return nil, err
				}
			}
			if err := s.hcMix(x, n, &l.q4e.attn); err != nil {
				return nil, err
			}
			attnIn = s.q4.mix
		}
		if c.ResAttn() {
			var err error
			if attnIn, err = s.k3AttnIn(li, l, s.bx[:n*c.ResidW()], n); err != nil {
				return nil, err
			}
		}
		s.batchNormB(s.bh, attnIn, l.attnNorm, l.attnNormB, c.NEmbd, n)
		if c.AltUp != 0 {
			for r := 0; r < n; r++ {
				if err := s.laurel(l, s.bh[r*c.NEmbd:(r+1)*c.NEmbd], s.baltLaur[r*c.NEmbd:(r+1)*c.NEmbd]); err != nil {
					return nil, err
				}
			}
		}
		// A parallel block's FFN input; see forward.go.
		if c.Parallel {
			if l.ffnNorm != nil {
				s.batchNormB(s.bhf, s.bx, l.ffnNorm, l.ffnNormB, c.NEmbd, n)
			} else {
				copy(s.bhf[:n*c.NEmbd], s.bh[:n*c.NEmbd])
			}
		}
		s.tock(opRMSNorm, t0)
		// Each row passes its sequence's slot as the recurrent-state slot;
		// sharing one state would give every row slot 0's history.
		// A block that attends as well (Falcon-H1) leaves its mixer's rows
		// in bhf and runs attention from the same bh.
		kind := c.LayerKind(li)
		if kind.Recurrent() {
			t0 = s.tick()
			for _, r := range causal {
				h := s.bh[r*c.NEmbd : (r+1)*c.NEmbd]
				out := h
				if kind.Attends() {
					out = s.bhf[r*c.NEmbd : (r+1)*c.NEmbd]
				}
				o, slot := s.rowOwner(r, seq)
				o.jit.NewInput()
				if err := o.linearAttn(li, l, slot, h, out); err != nil {
					return nil, err
				}
			}
			s.tock(opAttn, t0)
		}
		if kind.Attends() {
			if c.MLA() {
				// MLA: one call replaces q, k, v and attnPrep, shared with
				// prefill. Here each row names its slot; in prefill all rows
				// share one.
				if err := s.mlaProjectRows(li, l, n, cs, csSWA,
					func(i int) (int, int) { return seq[i], pos[i] }); err != nil {
					return nil, err
				}
			} else {
				kvDim, qDim := c.KVDimAt(li), c.QDimAt(li)
				if _, err := s.projectQ(l, s.bq, s.ogate, s.bh, n, true); err != nil {
					return nil, err
				}
				// A KV-sharing block: q alone, then its source's history.
				if c.KVShared(li) {
					for i := 0; i < n; i++ {
						o, slot := s.rowOwner(i, seq)
						o.attnPrep(l, s.bq[i*qDim:(i+1)*qDim], nil, nil,
							s.ropeTable(li, cs, csSWA, i), li, slot, pos[i])
					}
					goto attend
				}
				if err := s.mm(s.bk, l.wk, s.bh, n); err != nil {
					return nil, err
				}
				s.addBiasRows(s.bk, l.bk, n)
				if l.vFromK {
					// v is k's projection (Gemma 4's global layers).
					copy(s.bv[:n*kvDim], s.bk[:n*kvDim])
				} else if err := s.mm(s.bv, l.wv, s.bh, n); err != nil {
					return nil, err
				}
				s.addBiasRows(s.bv, l.bv, n)
				// MiniMax Sparse Attention's indexer key and query (msa.go).
				msa := c.MSAAt(li)
				if msa {
					if err := s.msaProjectRows(l, n, cs, csSWA, li); err != nil {
						return nil, err
					}
				}
				qsa := c.QSAAt(li)
				if qsa {
					if err := s.qsaProjectRows(l, n, cs, csSWA, li); err != nil {
						return nil, err
					}
				}

				// No outer timer: attnPrep charges its own RoPE.
				for i := 0; i < n; i++ {
					k, v := s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim]
					if msa {
						k, v = s.msaRowOf(li, i, k, v)
					}
					if qsa {
						k, v = s.qsaRowOf(li, i, k, v)
					}
					// Each row has its own position, table slice and cache slot.
					// Every row's K and V land here, before any row attends.
					o, slot := s.rowOwner(i, seq)
					o.attnPrep(l, s.bq[i*qDim:(i+1)*qDim], k, v,
						s.ropeTable(li, cs, csSWA, i), li, slot, pos[i])
				}
			}
		attend:

			// Across sequences and heads: at n=2 a split over sequences alone would
			// leave four of six cores idle, and attention is the op that takes over
			// as the context grows. A step with a chunk has rows enough.
			t0 = s.tick()
			bq, bxb, bqf, bxbf := s.bq, s.bxb, s.bqf, s.bxbf
			bathf := s.bathf
			hd, qDim, gqa := c.HeadDimAt(li), c.QDimAt(li), c.GQAAt(li)
			// MLA reads a wider query than it writes, and every head reads the
			// same cached row: qw is the whole row, ow its latent prefix, and
			// gqaN = NHead makes hh/gqaN 0 for every head.
			qw, ow, mla := hd, hd, c.MLA()
			qsrc, odst, gqaN := bqf, bxbf, gqa
			if mla {
				qw, ow = c.KVLoraRank+c.NRot, c.KVLoraRank
				qsrc, odst, gqaN = s.bmlaAbs, s.bmlaAcc, nh
			}
			// Residency before the fan-out; see kvEnsureWindow. Rows share the
			// layer's pages, so the widest row decides.
			maxNp := 0
			for i := 0; i < n; i++ {
				maxNp = max(maxNp, pos[i]+1)
			}
			kvli := c.KVSource(li)
			if len(s.rowOwn) == 0 {
				s.kvEnsureWindow(kvli, 0, maxNp)
			} else {
				for i, o := range s.rowOwn {
					if s.rowWin[i] {
						o.kvEnsureWindow(kvli, 0, pos[i]+1)
					}
				}
			}
			masks, mstride := s.idxMasks, 0
			if c.Indexer() {
				s.idxRows(kvli, n, func(i int) int { return seq[i] }, func(i int) int { return pos[i] + 1 }, masks)
			}
			if c.MSAAt(li) {
				// The masks are laid out at the State's own stride (msa.go).
				masks, mstride = s.msaMasks, s.attStride
				s.msaRows(kvli, n, func(i int) int { return seq[i] }, func(i int) int { return pos[i] + 1 }, masks)
			}
			if c.QSAAt(li) {
				masks, mstride = s.qsa().masks, 0
				s.qsaRows(kvli, l, n, func(i int) (*State, int) { return s.rowOwner(i, seq) },
					func(i int) int { return pos[i] + 1 }, masks)
			}
			// The fan-out's arguments are State fields and its function a
			// method value made once: a closure over these locals escapes,
			// an allocation per block per step.
			s.ra = rowsAttn{s: s, l: l, li: li, kvli: kvli, unitHeads: unitHeads, nh: nh,
				seq: seq, pos: pos, qsrc: qsrc, odst: odst, bq: bq, bxb: bxb, bathf: bathf,
				astride: astride, qw: qw, ow: ow, hd: hd, qDim: qDim, gqaN: gqaN, mla: mla,
				scale: scale, masks: masks, mstride: mstride}
			if s.raRun == nil {
				s.raRun = s.ra.run
			}
			s.jit.Parallel(units, 1, s.raRun)
			s.ra = rowsAttn{}
			s.tock(opAttn, t0)
			s.jit.NewInput()

			if mla {
				// W_v un-absorbs out of the latent and wo follows, both inside
				// mlaOutRows, with Kimi-K3's output gate between them.
				if err := s.mlaOutRows(l, n); err != nil {
					return nil, err
				}
			} else {
				asrc := s.bxb
				if c.AttnOutGate {
					s.applyOutGate(s.ogate, s.bxb, n)
					s.jit.NewInput()
					asrc = s.ogate
				}
				if err := s.mm(s.bh, l.wo, asrc[:n*c.QDimAt(li)], n); err != nil {
					return nil, err
				}
			}
			s.addBiasRows(s.bh, l.bo, n)
			if kind.Recurrent() {
				s.addInto(s.bh[:n*c.NEmbd], s.bhf[:n*c.NEmbd])
			}
		}
		// gemma2/gemma3 post-norms; every entry point must apply them.
		if l.postAttnNorm != nil {
			s.batchNorm(s.bh, s.bh, l.postAttnNorm, c.NEmbd, n)
		}
		t0 = s.tick()
		if c.Q4E() {
			s.hcPlace(s.bx[:n*c.ResidW()], n, s.bh)
		} else if c.ResAttn() {
			s.k3Resid(li, s.bx, s.bh, n)
		} else {
			s.addInto(s.bx[:n*c.NEmbd], s.bh[:n*c.NEmbd])
		}
		if c.AltUp != 0 {
			s.laurelJoin(s.bx[:n*c.NEmbd], s.baltLaur[:n*c.NEmbd])
		}
		s.tock(opResid, t0)
		// A block with no FFN ends at the first residual; see forward.go.
		if l.noFFN {
			continue
		}

		t0 = s.tick()
		ffnIn := s.bh
		if c.Parallel {
			ffnIn = s.bhf
			s.jit.NewInput()
		} else if c.ResAttn() {
			in, err := s.k3FFNIn(li, l, s.bx[:n*c.ResidW()], n)
			if err != nil {
				return nil, err
			}
			s.batchNormB(s.bh, in, l.ffnNorm, l.ffnNormB, c.NEmbd, n)
		} else if c.Q4E() {
			if err := s.hcMix(s.bx[:n*c.ResidW()], n, &l.q4e.ffn); err != nil {
				return nil, err
			}
			copy(s.bh[:n*c.NEmbd], s.q4.mix[:n*c.NEmbd])
		} else {
			s.batchNormB(s.bh, s.bx, l.ffnNorm, l.ffnNormB, c.NEmbd, n)
		}
		s.tock(opRMSNorm, t0)
		// Rows are different sequences and pick different experts, so the
		// mixture runs per row.
		if l.router.data != nil && c.DenseMoE {
			if err := s.denseMoERows(li, l, ffnIn, s.bx, n); err != nil {
				return nil, err
			}
		} else if l.router.data != nil && c.ExpertLatent != 0 {
			if err := s.k3MoEBatch(li, l, ffnIn, s.bx, n); err != nil {
				return nil, err
			}
		} else if l.router.data != nil && c.Q4E() {
			out := s.q4.out[:n*c.NEmbd]
			clear(out)
			if err := s.moeBatch(li, l, ffnIn, out, n); err != nil {
				return nil, err
			}
			s.hcPlace(s.bx[:n*c.ResidW()], n, out)
		} else if l.router.data != nil {
			if err := s.moeBatch(li, l, ffnIn, s.bx, n); err != nil {
				return nil, err
			}
		} else {
			if l.gate.e == nil {
				// The ungated FFN (C6); see forward.go.
				if err := s.mm(s.bup, l.up, ffnIn, n); err != nil {
					return nil, err
				}
				s.addBiasRows(s.bup, l.upB, n)
				t0 = s.tick()
				s.ungatedAct(l, s.bup[:n*c.NFFN])
				s.jit.NewInput()
				s.tock(opAct, t0)
				if err := s.mm(s.bh, l.down, s.bup, n); err != nil {
					return nil, err
				}
				s.addBiasRows(s.bh, l.downB, n)
			} else {
				if err := s.mm(s.bgate, l.gate, ffnIn, n); err != nil {
					return nil, err
				}
				if err := s.mm(s.bup, l.up, ffnIn, n); err != nil {
					return nil, err
				}
				t0 = s.tick()
				gate, up, act := s.bgate[:n*c.NFFNAt(li)], s.bup[:n*c.NFFNAt(li)], c.Act
				s.gaussTopK(li, gate, c.NFFNAt(li))
				s.actmulAll(gate, up, act)
				s.jit.NewInput()
				s.tock(opAct, t0)
				if err := s.mm(s.bh, l.down, s.bgate, n); err != nil {
					return nil, err
				}
			}
			if l.postFFNNorm != nil {
				s.batchNorm(s.bh, s.bh, l.postFFNNorm, c.NEmbd, n)
			}
			t0 = s.tick()
			s.addInto(s.bx[:n*c.NEmbd], s.bh[:n*c.NEmbd])
			s.tock(opResid, t0)
		}
		if c.AltUp != 0 {
			if err := s.altCorrect(li, l, s.bx[:n*c.ResidW()], s.baltPred, s.bple, n); err != nil {
				return nil, err
			}
		} else if err := s.pleRows(li, l, s.bx, n); err != nil {
			return nil, err
		}
		s.layerOutScale(l, s.bx[:n*c.NEmbd])
	}
	if c.AltUp != 0 && nlogit > 0 {
		if err := s.altCollapse(s.bx[:n*c.ResidW()], n); err != nil {
			return nil, err
		}
	}
	if c.DSV4() && nlogit > 0 {
		if err := s.ds4Collapse(s.bx[:n*c.ResidW()], n); err != nil {
			return nil, err
		}
	}
	if c.ResAttn() && nlogit > 0 {
		if err := s.k3Collapse(s.bx[:n*c.ResidW()], n); err != nil {
			return nil, err
		}
	}
	if c.Q4E() && nlogit > 0 {
		if err := s.q4eCollapse(s.bx[:n*c.ResidW()], n); err != nil {
			return nil, err
		}
	}

	// Unlike Prefill, every decoding row's logits are wanted: each one is a
	// live sequence that has to be sampled. So lm_head runs for those rows,
	// which lead -- the widest matmul of the token, and the shape it likes.
	if !headOnDev && nlogit > 0 {
		t0 = s.tick()
		s.batchNormB(s.bh, s.bx, s.outNorm, m.outNormB, c.NEmbd, nlogit)
		s.tock(opRMSNorm, t0)
		if err := s.mm(s.blogits, *s.outW, s.bh, nlogit); err != nil {
			return nil, err
		}
	}
	// The wanted rows' logits, which is one contiguous run of nlogit*NVocab:
	// capped already where the device ran the head.
	if headOnDev {
		s.finishDevLogits(s.blogits[:nlogit*c.NVocab])
	} else {
		s.finishLogits(s.blogits[:nlogit*c.NVocab])
	}
	return s.blogits[:nlogit*c.NVocab], nil
}

// BatchLogits is row i of the last ForwardBatch.
func (s *State) BatchLogits(i int) []float32 {
	nv := s.c.NVocab
	return s.blogits[i*nv : (i+1)*nv]
}

type errBatch struct{}

func (errBatch) Error() string {
	return "model: this session came from NewBatch; use ForwardBatch"
}

type errBatchWidth struct{ got, want int }

func (e errBatchWidth) Error() string {
	return "model: ForwardBatch got " + itoa(e.got) + " tokens for " + itoa(e.want) + " sequences"
}

type errSlot struct{ got, nseq int }

func (e errSlot) Error() string {
	return fmt.Sprintf("model: sequence slot %d out of range for a batch of %d", e.got, e.nseq)
}

type errBatchDevice struct{}

func (errBatchDevice) Error() string {
	return "model: a batch runs on a device only with every block and the head on one device " +
		"that runs ragged rows; this placement splits the model, and a split device holds one sequence's KV cache"
}

// rowsAttn is a ragged step's attention fan-out over (row, head) units, one
// block's: what rowsHost hands the pool.
type rowsAttn struct {
	s                               *State
	l                               *layer
	li, kvli, unitHeads, nh         int
	seq, pos                        []int
	qsrc, odst, bq, bxb, bathf      []float32
	astride, qw, ow, hd, qDim, gqaN int
	mla                             bool
	scale                           float64
	masks                           [][]float32
	mstride                         int
}

func (j *rowsAttn) run(lo, hi int) {
	s, c, l, li, kvli, unitHeads, nh := j.s, j.s.c, j.l, j.li, j.kvli, j.unitHeads, j.nh
	seq, pos, qsrc, odst, bq, bxb, bathf := j.seq, j.pos, j.qsrc, j.odst, j.bq, j.bxb, j.bathf
	astride, qw, ow, hd, qDim, gqaN, mla := j.astride, j.qw, j.ow, j.hd, j.qDim, j.gqaN, j.mla
	scale, masks, mstride := j.scale, j.masks, j.mstride
	for idx := lo * unitHeads; idx < hi*unitHeads; idx++ {
		i, hh := idx/nh, idx%nh
		// The row's own length, not the batch's: past pos[i] the
		// slot holds stale or unwritten keys -- or, for a prompt
		// chunk, the keys of its own later rows -- and this bound is
		// what keeps them unreachable (which is why Retire copies
		// nothing). The sliding window applies here as in decode.
		w0, np := c.AttnWindow(li, pos[i])
		qh := qsrc[i*nh*qw+hh*qw : i*nh*qw+(hh+1)*qw]
		if !mla {
			copy(qh, bq[i*qDim+hh*hd:])
		}
		kvh := hh / gqaN
		// The unit's scores row: a unit's heads run in turn on it.
		u := idx / unitHeads
		af := bathf[u*astride : u*astride+np]
		o, slot := s.rowOwner(i, seq)
		fast := o.attnAt(li)
		o.kvScores(fast, kvli, slot, kvh, qh, af, w0, np)
		s.scale(af, float32(scale*c.AttnTemp(li, pos[i])))
		softcap(af, c.AttnSoftcap)
		if masks != nil && masks[i] != nil {
			s.idxApply(af, masks[i][kvh*mstride:], w0)
		}
		s.softmaxSink(af, l.sinks, hh)
		out := odst[i*nh*ow+hh*ow : i*nh*ow+(hh+1)*ow]
		o.kvAcc(fast, kvli, slot, kvh, out, af, w0, np)
		// MLA's result stays in bmlaAcc, where the un-absorb
		// reads it; bxb is the wrong width for it.
		if !mla {
			for j, v := range out {
				bxb[i*qDim+hh*hd+j] = v
			}
		}
	}
}
