package model

import "fmt"

// Deepstack (Qwen3-VL): a picture's rows carry, beside the rows that enter
// the text model, one more row per tower tap, and tap k's row adds into the
// picture's row of the residual after text block k -- transformers'
// _deepstack_process, llama.cpp's deepstack_out. It is an addend riding the
// rows, not a second input path: the rows are prefilled as any embedding span,
// and after each of the first blocks the runner adds a span's tap into the
// rows of the chunk that hold it. A device runs those blocks one submission
// each so the addend lands between them; the add itself is the generated
// axpy, on the host, over the residual every tier already hands back.

// deepRun is one span's deepstack rows within a prefill: the prefill's rows
// [lo, hi) hold its picture rows first.. onward.
type deepRun struct {
	lo, hi int
	// rows is the span's Deep, [taps][n][NEmbd]; n is its picture's rows.
	rows     []float32
	n, first int
	taps     int
}

// splitDeep cuts a deepstack tower's output -- n merged rows of width wide,
// then each tap's n rows -- into the merged rows and the taps'. rows with no
// taps behind them come back whole with nil.
func splitDeep(rows []float32, n, wide int) (embd, deep []float32) {
	if n <= 0 || len(rows) <= n*wide {
		return rows, nil
	}
	return rows[:n*wide], rows[n*wide:]
}

// deepRuns records the deepstack rows of spans, whose prefill starts at the
// spans' row from (a restore covered the rows before it). It returns the
// reset.
func (s *Instance) deepRuns(spans []Span, from int) (func(), error) {
	s.deep = s.deep[:0]
	at := 0
	for i, sp := range spans {
		n := sp.n(s.c.NEmbd)
		if sp.Deep != nil && at+n > from {
			w := n * s.c.NEmbd
			if sp.Embd == nil || w == 0 || len(sp.Deep)%w != 0 {
				return nil, fmt.Errorf("model: span %d's deepstack rows are %d floats, not taps of %d rows of %d",
					i, len(sp.Deep), n, s.c.NEmbd)
			}
			taps := len(sp.Deep) / w
			if taps > s.hi-s.lo {
				return nil, fmt.Errorf("model: span %d carries %d deepstack taps and the model has %d blocks",
					i, taps, s.hi-s.lo)
			}
			lo := max(at, from)
			s.deep = append(s.deep, deepRun{lo: lo - from, hi: at + n - from, rows: sp.Deep,
				n: n, first: lo - at, taps: taps})
		}
		at += n
	}
	return func() { s.deep = s.deep[:0] }, nil
}

// deepIn reports whether the prefill rows [base, base+n) hold a deepstack row
// still to be added after block li.
func (s *Instance) deepIn(li, base, n int) bool {
	for _, r := range s.deep {
		if li-s.lo < r.taps && r.lo < base+n && r.hi > base {
			return true
		}
	}
	return false
}

// addDeep adds, after block li has run over the prefill rows [base, base+n)
// in s.bx, each deepstack run's tap li into the rows it holds there.
func (s *Instance) addDeep(li, base, n int) {
	k, E := li-s.lo, s.c.NEmbd
	for _, r := range s.deep {
		lo, hi := max(r.lo, base), min(r.hi, base+n)
		if k >= r.taps || lo >= hi {
			continue
		}
		if cap(s.deepIdx) < hi-lo {
			s.deepIdx = make([]int32, max(hi-lo, s.maxSeq))
		}
		idx := s.deepIdx[:hi-lo]
		for i := range idx {
			idx[i] = int32(r.first + lo - r.lo + i)
		}
		tap := r.rows[k*r.n*E : (k+1)*r.n*E]
		s.axpyRows(s.bx[(lo-base)*E:(hi-base)*E], tap, idx, E, hi-lo, s.rowChunk(hi-lo, E), 1)
	}
}
