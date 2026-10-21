//go:build amd64 || arm64

package nn

import (
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// AheadSet is the weights a fused run reads next: the matrices of the op after
// the current one (a block's o projection while its q/k/v and attention run,
// its gate and up while o runs, ...). The caller owns it and fills it between
// phases; SetAhead publishes it.
type AheadSet struct {
	P [MaxGroup]*Packed
}

// aheadPiece is how many lines a lookahead hints before it asks whether the
// phase is over: 4 KiB, about a microsecond of one core's DRAM rate.
const aheadPiece = 64

// lookahead is the JIT's sched.Lookahead: a worker that finds the current
// phase drained hints the next op's weights into the outer caches until the
// phase is over, so the bandwidth the phase's tail leaves idle reads what the
// next phase will. It writes nothing a kernel reads, and a phase computes the
// same bits with or without it (model.TestFusedBlocksMatchPerOp runs with it).
//
// The weights are split into pieces dealt round robin across the phase's
// workers, so the start of the next matrix is covered first whoever is idle;
// each worker resumes where it stopped at the next phase that leaves it idle.
type lookahead struct {
	set   atomic.Pointer[AheadSet]
	gen   atomic.Uint64 // bumped by SetAhead: a worker restarts its cursor
	lines atomic.Int64  // lines hinted, for the gate that it ran
	pos   []aheadPos    // per worker
}

// aheadPos is one worker's cursor, padded to a line of its own.
type aheadPos struct {
	gen   uint64
	piece int
	_     [48]byte
}

// SetAhead names the weights the fused run reads next; nil hints nothing. A
// call outside a fused run, or with the lookahead off, only records it.
func (f *JIT) SetAhead(s *AheadSet) {
	if f == nil {
		return
	}
	f.look.set.Store(s)
	f.look.gen.Add(1)
}

// SetLookahead turns the fused runs' lookahead on or off (on by default);
// takes effect at the next Fuse.
func (f *JIT) SetLookahead(on bool) {
	if f != nil {
		f.lookOff = !on
	}
}

// AheadLines is how many 64-byte lines the lookahead has hinted.
func (f *JIT) AheadLines() int64 {
	if f == nil {
		return 0
	}
	return f.look.lines.Load()
}

// Ahead implements sched.Lookahead.
func (l *lookahead) Ahead(p *sched.Pool, worker, workers int, tag uint64) {
	s := l.set.Load()
	if s == nil || worker >= len(l.pos) || workers < 1 {
		return
	}
	gen := l.gen.Load()
	me := &l.pos[worker]
	if me.gen != gen {
		me.gen, me.piece = gen, worker
	}
	e := elemFor()
	var args cpu.Args
	for {
		// Piece me.piece of the planes laid end to end.
		at := me.piece * aheadPiece * cpu.PrefetchLine
		var b []byte
		for i := 0; i < MaxGroup && b == nil; i++ {
			pk := s.P[i]
			if pk == nil {
				continue
			}
			for _, plane := range [3][]byte{pk.QS, pk.D, pk.SC} {
				if at < len(plane) {
					b = plane[at:]
					break
				}
				at -= len(plane)
			}
		}
		if b == nil {
			return // every piece is this worker's no more
		}
		n := min(aheadPiece, (len(b)+cpu.PrefetchLine-1)/cpu.PrefetchLine)
		args.Out = (*float32)(unsafe.Pointer(&b[0]))
		args.K = int64(n)
		e.pflines.Call(&args)
		l.lines.Add(int64(n))
		me.piece += workers
		if p.Moved(tag) {
			return
		}
	}
}
