package probe

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// hostLanes is a vector's floats.
func (e emitter) hostLanes() int {
	if e.sse {
		return 4
	}
	return 8
}

// formLanes is the words one vector instruction the estimator counts in a
// form covers: a vector's.
func (e emitter) formLanes() int { return e.hostLanes() }

// hostCheck refuses an x86 host below the SSE tier.
func hostCheck(h *cpu.Target) error {
	if t := h.Tier(); t != cpu.TierAVX2 && t != cpu.TierSSE {
		return fmt.Errorf("probe: the host's measuring kernels need SSE4.1 and SSSE3 at least; this host is %v", t)
	}
	return nil
}

// buf is a kernel's buffer, declared SSE-tier on a host that runs that
// tier, so Map lets it run there.
func (e emitter) buf() *x86.Buf {
	var a x86.Buf
	if e.sse {
		a.DeclareISA(x86.ISATierSSE)
	}
	return &a
}

// zero clears vector register r.
func (e emitter) zero(a *x86.Buf, r x86.Reg) {
	if e.sse {
		a.XORPS(r, r, r)
		return
	}
	a.VPXOR(r, r, r)
}

// done ends a kernel that used vector registers.
func (e emitter) done(a *x86.Buf) {
	if !e.sse {
		a.VZEROUPPER()
	}
	a.RET()
}

// lanesKernel runs [0] trips of eight swaps of a vector's pairs of lanes,
// each on a register of its own, or each of the one before's result where
// chain is set; in SSE's encoding on a host at that tier.
func (e emitter) lanesKernel(chain bool) ([]byte, error) {
	a := e.buf()
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		e.zero(a, x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for j := range 8 {
		r := x86.Y0
		if !chain {
			r += x86.Reg(j)
		}
		if e.sse {
			a.SHUFPS(r, r, r, 0xB1)
		} else {
			a.VPERMILPS(r, r, 0xB1)
		}
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	e.done(a)
	return a.Bytes()
}
