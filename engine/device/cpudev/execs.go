package cpudev

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// Execs is the CPU device as an instance opens it (nn.ExecutionDevice), its
// executions composed with the CPU's own options: float forms and the
// serving split. One per option set, so two instances on one CPU with the
// same options hold the same device.
type Execs struct {
	c            *CPU
	float, split bool
}

// Execs is c with float forms and the serving split as asked.
func (c *CPU) Execs(float, split bool) *Execs {
	c.mu.Lock()
	defer c.mu.Unlock()
	k := [2]bool{float, split}
	if c.execs == nil {
		c.execs = map[[2]bool]*Execs{}
	}
	if x := c.execs[k]; x != nil {
		return x
	}
	x := &Execs{c: c, float: float, split: split}
	c.execs[k] = x
	return x
}

// CPU is the device x composes for.
func (x *Execs) CPU() *CPU { return x.c }

// HoldModel and ReleaseModel are the CPU's (nn.ModelOwner).
func (x *Execs) HoldModel(model uint64)    { x.c.HoldModel(model) }
func (x *Execs) ReleaseModel(model uint64) { x.c.ReleaseModel(model) }

// Attached is the CPU's (nn.ModelScratch).
func (x *Execs) Attached(model uint64, mk func() any) any { return x.c.Attached(model, mk) }

// NewQuota and Frames are the CPU's (nn.FrameKeeper).
func (x *Execs) NewQuota(model, bytes uint64) nn.Quota { return x.c.NewQuota(model, bytes) }
func (x *Execs) Frames(model uint64, mk func(...jlm.FramesOption) (*jlm.Frames, error)) (*jlm.Frames, error) {
	return x.c.Frames(model, mk)
}

// Close closes the CPU.
func (x *Execs) Close() { x.c.Close() }

// AttachExecution attaches an instance's execution (nn.ExecutionAttacher).
func (x *Execs) AttachExecution(st nn.ExecutionSetup) nn.BlockPlacer {
	q, _ := st.Quota.(*Quota)
	return x.c.Attach(Setup{Model: st.Model, Frames: st.Frames, Float: x.float, Split: x.split, NoFuse: st.NoFuse,
		Chunk: st.Chunk, Tries: st.Tries, Runs: st.Runs, Quota: q, Width: st.Width, Seqs: st.Seqs,
		Streams: st.Streams, ConvLen: st.ConvLen, Trace: st.Trace})
}
