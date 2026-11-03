package metal

import "time"

// Opts are the choices a context is opened with. Zero is the shipping default.
// They are per context, so two tiers in one process open theirs differently.
type Opts struct {
	// FastMath compiles libraries with Metal's fast math instead of the IEEE
	// rounding the other backends lower to; see strictMath.
	FastMath bool
	// Spin is how long a Wait polls a command buffer's status before it blocks
	// in waitUntilCompleted. Zero takes the default, 20ms; negative blocks at
	// once, which is waitUntilCompleted's own behaviour.
	Spin time.Duration
	// Concurrent opens every compute encoder concurrent, with a buffer
	// barrier only where a dispatch reads or writes what an earlier one since
	// the last barrier wrote, or writes what it read (Batch.order). The
	// default is serial: on the M4, Ministral-3-3B Q4_K_M, concurrent read
	// 0.9924 of serial in decode (316 barriers among 370 dispatches a token:
	// a block is a chain) and 1.0053 in a 415-token prefill.
	Concurrent bool
	// Unretained selects -commandBufferWithUnretainedReferences for every
	// batch; see Ctx.Unretained for what it trades. Default off.
	Unretained bool
	// Residency attaches an MTLResidencySet to the context's queues (see
	// Ctx.Residency); the open fails on a host or queue that cannot carry one,
	// rather than silently measuring without it.
	Residency bool
	// Stats is what the context counts into; nil gives it its own.
	Stats *Stats
}

func (o Opts) stats() *Stats {
	if o.Stats != nil {
		return o.Stats
	}
	return &Stats{}
}

func (o Opts) spin() time.Duration {
	switch {
	case o.Spin < 0:
		return 0
	case o.Spin == 0:
		return 20 * time.Millisecond
	}
	return o.Spin
}

// CopyOp is one copy of a Ctx.CopyList: N bytes of Src at SrcOff into Dst at
// DstOff.
type CopyOp struct {
	Dst       *Buf
	DstOff    int
	Src       *Buf
	SrcOff, N int
}
