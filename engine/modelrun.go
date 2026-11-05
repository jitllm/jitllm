package engine

import (
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// qkNorms is the pair block li offers a device: its weights when QKNormAt says
// the block norms q and k, and none when it does not, so the device's launch
// follows the same derivation as the host's.
func (s *Instance) qkNorms(li int) (q, k []float32) {
	if !s.c.QKNormAt(li) {
		return nil, nil
	}
	l := &s.m.Layers[li]
	return l.QNorm, l.KNorm
}

// cpuISA is the extensions the model's CPU device forms for (its Host's).
func (s *Instance) cpuISA() cpu.ISA { return s.eng.cpu.Target.ISA() }

// Trace installs a callback invoked with every intermediate vector, for diffing
// against an independent oracle layer by layer. nil disables it.
func (s *Instance) Trace(fn func(layer int, name string, v []float32)) { s.opt.tracer = fn }

func (s *Instance) trace(l int, name string, v []float32) {
	if s.opt.tracer != nil {
		s.opt.tracer(l, name, v)
	}
}

// poolOptions is what an instance's CPU device's workers are made with: the
// caller's, and the serving split unless the model turned it off.
func (s *Instance) poolOptions() []cpu.PoolOption {
	return append(s.eng.cpu.PoolOptions(), cpu.WithServingSplit(!s.opt.noServingSplit))
}
