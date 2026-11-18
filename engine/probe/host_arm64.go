package probe

import (
	"github.com/jitllm/jitllm/composer/a64"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// hostLanes is a NEON vector's floats.
func (e emitter) hostLanes() int { return 4 }

// formLanes is the words one vector instruction the estimator counts in a
// form covers: the composer's arm64 vector is two NEON registers, eight
// lanes, counted as one.
func (e emitter) formLanes() int { return 8 }

// hostCheck accepts every arm64 host: the measuring kernels are base NEON.
func hostCheck(*cpu.Target) error { return nil }

// lanesKernel runs [0] trips of eight pairwise adds of a vector's lanes,
// each on a register of its own, or each of the one before's result where
// chain is set.
func (e emitter) lanesKernel(chain bool) ([]byte, error) {
	var a a64.A64
	return e.loop(&a, func() {
		for r := range 8 {
			v := a64.V0
			if !chain {
				v += a64.VReg(r)
			}
			a.FADDP4s(v, v, v)
		}
	})
}

// loop is body run [0] times, the vector and general registers it uses
// cleared first.
func (e emitter) loop(a *a64.A64, body func()) ([]byte, error) {
	a.LDRx(a64.X2, a64.X0, 0)
	for r := range 16 {
		a.MOVIzero(a64.V0 + a64.VReg(r))
	}
	for r := a64.X3; r <= a64.X7; r++ {
		a.MOVimm(r, 0)
	}
	top := a.Label()
	a.Bind(top)
	body()
	a.SUBimm(a64.X2, a64.X2, 1)
	a.CBNZ(a64.X2, top)
	a.RET()
	return a.Bytes()
}
