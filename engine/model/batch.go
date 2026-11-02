package model

import (
	"slices"

	"github.com/jitllm/jitllm/jit"
)

// rowOwner is the session whose history row i of a ragged step reads and
// writes, and the slot there: the batch's own slot seq[i], or in a step across
// sessions (stepHost) the row's own session, whose one sequence is slot 0.
func (s *Instance) rowOwner(i int, seq []int) (*Session, int) {
	if len(s.rowOwn) > 0 {
		return s.rowOwn[i], 0
	}
	return s.Session, seq[i]
}

// rowsHost runs a ragged step's blocks and head and returns the wanted rows'
// logits; the caller moves the positions on. seqs is how many sequences the
// rows hold: a step of one row per sequence fans attention out over (row,
// head) units.
func (s *Instance) rowsHost(tokens []int32, seq, pos []int, nlogit, seqs int) ([]float32, error) {
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
	// past the leading instance's context, where rowsStride stops.
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

	nrot := c.RopeW()
	// One rotary table per row, rows being at different positions: the prompt
	// chunk's tables, since a step may carry more rows than there are slots.
	cs, csSWA := s.bcs[:n*nrot], s.bswaTable(n, c.NRotSWA)

	t0 := s.tick()
	// The rows' input rows, scale, positions and per-layer inputs through the
	// composed graph (cpuembed.go), or the old engine path where it declines.
	var bple []float32
	if c.PLEDim != 0 {
		s.growBPLE(n)
		bple = s.bple[:n*c.pleWidth()]
	}
	composed := s.cpuEmbed(s.bx[:n*c.NEmbd], bple, tokens[:n], pos[:n]) == jit.DeclineServes
	for i := 0; i < n && !composed; i++ {
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

	if c.PredStreams != 0 {
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

	// The step's rotary tables are the CPU's to make where a block runs
	// there (cpuRowTables); a device makes its own.
	s.cpu.tabsFresh = false

	var devSlot []int
	if s.devCount() > 0 && len(s.rowOwn) == 0 {
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
			run := s.rowsRun
			if len(s.rowOwn) > 0 {
				run = s.rowsSessions
			}
			if err := run(li, hi, pos, devSlot, s.bx[:n*c.ResidW()], cs, csSWA,
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
		var why jit.Decline
		var err error
		if c.NHashLayers > 0 && li == s.lo {
			s.cpuTokens(n, func(i int) int32 { return tokens[i] })
		}
		switch {
		case c.LayerKind(li).Recurrent():
			if c.NGramAt(li) {
				s.q4.setRows(s, seq, tokens)
				if err := s.ngramForCPU(li, n, causal, s.q4.rowsOwner, s.q4.callTok); err != nil {
					return nil, err
				}
			}
			why, err = s.cpuRecRows(li, l, s.bx[:n*c.ResidW()], cs, seq, pos, causal, -1, 0)
		case len(s.rowOwn) > 0:
			why, err = s.cpuSessions(li, l, s.bx[:n*c.ResidW()], cs, csSWA, seq, pos, nlogit)
		default:
			why, err = s.cpuRows(li, l, s.bx[:n*c.ResidW()], cs, csSWA, seq, pos)
		}
		if err != nil {
			return nil, err
		}
		if why == jit.DeclineServes {
			s.cpu.served++
			continue
		}
		return nil, s.cpuRefused(li, why)
	}
	if c.PredStreams != 0 && nlogit > 0 {
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
	// The CPU's composed head runs the wanted rows and applies every stage
	// of the head's graph.
	if !headOnDev && nlogit > 0 {
		if why := s.cpuHeadRows(s.bx, s.blogits, nil, nlogit); why != jit.DeclineServes {
			return nil, cpuHeadRefused(why)
		}
	}
	// The wanted rows' logits, which is one contiguous run of nlogit*NVocab:
	// capped already where the device ran the head.
	if headOnDev {
		s.finishDevLogits(s.blogits[:nlogit*c.NVocab])
	}
	return s.blogits[:nlogit*c.NVocab], nil
}

type errBatchDevice struct{}

func (errBatchDevice) Error() string {
	return "model: a batch runs on a device only with every block and the head on one device " +
		"that runs ragged rows; this placement splits the model, and a split device holds one sequence's KV cache"
}
