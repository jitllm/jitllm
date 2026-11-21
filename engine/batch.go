package engine

import (
	"github.com/jitllm/jitllm/nn"
)

// rowOwner is the session whose history row i of a ragged step reads and
// writes, and the slot there: the batch's own slot seq[i], or in a step across
// sessions (stepRouted) the row's own session, whose one sequence is slot 0.
func (s *Instance) rowOwner(i int, seq []int) (*Session, int) {
	if len(s.rowOwn) > 0 {
		return s.rowOwn[i], 0
	}
	return s.Session, seq[i]
}

// rowsRouted runs a ragged step's blocks and head and returns the wanted rows'
// logits; the caller moves the positions on. seqs is how many sequences the
// rows hold: a step of one row per sequence fans attention out over (row,
// head) units.
func (s *Instance) rowsRouted(tokens []int32, seq, pos []int, nlogit, seqs int) ([]float32, error) {
	c := s.c
	n := len(tokens)
	// The ragged head applies the final softcap itself, as decode's head does.
	headOnDev := s.devCount() > 0 && s.headWithLastBlock() && !s.wantsResidual() && !s.cutAt(s.hi-1)
	s.growRows(n)
	if len(s.blogits) < nlogit*c.NVocab {
		s.blogits = make([]float32, nlogit*c.NVocab)
	}

	// One rotary table per row, rows being at different positions: the prompt
	// chunk's tables, since a step may carry more rows than there are slots.
	cs := s.batchTables(n)

	t0 := s.tick()
	// The rows' input rows, scale, positions and per-layer inputs through the
	// composed graph (cpuembed.go).
	bple := s.inputSide(n)
	if why := s.embedRows(s.bx[:n*c.ResidW()], bple, tokens[:n], pos[:n], s.plane(n)); why != nn.DeclineServes {
		return nil, s.embedDeclined(why)
	}
	s.tock(opResid, t0)

	// The step's rotary tables are the CPU's to make where a block runs
	// there (rowTables); a device makes its own.
	s.rowsFresh = false

	var slot []int
	if len(s.rowOwn) == 0 {
		slot = s.flatSlots(seq, pos)
	}
	// A run of blocks on one device is one call, whichever device: the
	// rows' call, or across sessions every row through its own session's
	// view of the device.
	for li := s.lo; li < s.hi; li++ {
		k := s.routeOf(li)
		r := &s.routes[k]
		hi := li + 1
		for hi < s.hi && s.routeOf(hi) == k {
			hi++
		}
		hi = s.runEnd(li, hi)
		if err := s.openRows(r, seq, pos, tokens); err != nil {
			return nil, err
		}
		if err := s.crossRows(k, li, hi, pos, slot); err != nil {
			return nil, err
		}
		withHead := headOnDev && hi == s.hi
		s.cutFor(r, hi-1, n)
		var err error
		if len(s.rowOwn) > 0 {
			err = s.sessionsOn(k, li, hi, pos, s.bx[:n*c.ResidW()], cs, withHead, s.blogits[:nlogit*c.NVocab], nlogit)
		} else {
			err = s.rowsOn(r, li, hi, pos, slot, s.bx[:n*c.ResidW()], cs, withHead, s.blogits[:nlogit*c.NVocab], nlogit)
		}
		// A block cut at its routed experts: their half on the CPU device, over
		// every row the card's half left in the cut.
		if err == nil && s.cutAt(hi-1) {
			cr := &s.routes[routeLanes]
			if err = s.openRows(cr, seq, pos, tokens); err == nil {
				s.cutFor(cr, hi-1, n)
				if len(s.rowOwn) > 0 {
					err = s.sessionsOn(routeLanes, hi-1, hi, pos, s.bx[:n*c.ResidW()], cs, false, nil, 0)
				} else {
					err = s.rowsOn(cr, hi-1, hi, pos, slot, s.bx[:n*c.ResidW()], cs, false, nil, 0)
				}
			}
		}
		if err != nil {
			return nil, err
		}
		li = hi - 1
	}
	// Unlike Prefill, every decoding row's logits are wanted: each one is a
	// live sequence that has to be sampled. So lm_head runs for those rows,
	// which lead -- the widest matmul of the token, and the shape it likes.
	// The CPU's composed head runs the wanted rows and applies every stage
	// of the head's graph.
	if !headOnDev && nlogit > 0 {
		if why := s.headRows(s.bx, s.blogits, nil, nlogit, s.plane(n)); why != nn.DeclineServes {
			return nil, cpuHeadRefused(why)
		}
	}
	// The wanted rows' logits, which is one contiguous run of nlogit*NVocab.
	return s.blogits[:nlogit*c.NVocab], nil
}

type errBatchDevice struct{}

func (errBatchDevice) Error() string {
	return "model: a batch runs on a device only with every block and the head on one device " +
		"that runs ragged rows; this placement splits the model, and a split device holds one sequence's KV cache"
}
