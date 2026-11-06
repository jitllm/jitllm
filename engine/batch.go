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
	// AltUp's head reads the streams' mean, which the host takes (altup.go).
	headOnDev := s.devCount() > 0 && s.headWithLastBlock() && !c.StreamHead()
	s.growRows(n)
	if len(s.blogits) < nlogit*c.NVocab {
		s.blogits = make([]float32, nlogit*c.NVocab)
	}

	nrot := c.RopeW()
	// One rotary table per row, rows being at different positions: the prompt
	// chunk's tables, since a step may carry more rows than there are slots.
	cs, csSWA := s.bcs[:n*nrot], s.bswaTable(n, c.LocalNRot())

	t0 := s.tick()
	// The rows' input rows, scale, positions and per-layer inputs through the
	// composed graph (cpuembed.go).
	var bple []float32
	if c.PLEDim != 0 {
		s.growBPLE(n)
		bple = s.bple[:n*c.PleWidth()]
	}
	s.tokenSide(n)
	if why := s.embedRows(s.bx[:n*c.NEmbd], bple, tokens[:n], pos[:n]); why != nn.DeclineServes {
		return nil, s.embedDeclined(why)
	}

	if c.StreamHead() {
		if err := s.expandStreams(s.bx[:n*c.ResidW()], n, 0); err != nil {
			return nil, err
		}
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
		if err := s.openRows(r, seq, pos, tokens); err != nil {
			return nil, err
		}
		withHead := headOnDev && hi == s.hi
		var err error
		if len(s.rowOwn) > 0 {
			err = s.sessionsOn(k, li, hi, pos, s.bx[:n*c.ResidW()], cs, csSWA, withHead, s.blogits[:nlogit*c.NVocab], nlogit)
		} else {
			err = s.rowsOn(r, li, hi, pos, slot, s.bx[:n*c.ResidW()], cs, csSWA, withHead, s.blogits[:nlogit*c.NVocab], nlogit)
		}
		if err != nil {
			return nil, err
		}
		li = hi - 1
	}
	if c.StreamHead() && nlogit > 0 {
		if err := s.collapseStreams(s.bx[:n*c.ResidW()], n); err != nil {
			return nil, err
		}
	}

	// Unlike Prefill, every decoding row's logits are wanted: each one is a
	// live sequence that has to be sampled. So lm_head runs for those rows,
	// which lead -- the widest matmul of the token, and the shape it likes.
	// The CPU's composed head runs the wanted rows and applies every stage
	// of the head's graph.
	if !headOnDev && nlogit > 0 {
		if why := s.headRows(s.bx, s.blogits, nil, nlogit); why != nn.DeclineServes {
			return nil, cpuHeadRefused(why)
		}
	}
	// The wanted rows' logits, which is one contiguous run of nlogit*NVocab:
	// capped already where the device ran the head.
	if headOnDev {
		if err := s.devLogits(); err != nil {
			return nil, err
		}
	}
	return s.blogits[:nlogit*c.NVocab], nil
}

type errBatchDevice struct{}

func (errBatchDevice) Error() string {
	return "model: a batch runs on a device only with every block and the head on one device " +
		"that runs ragged rows; this placement splits the model, and a split device holds one sequence's KV cache"
}
