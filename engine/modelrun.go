package engine

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

// Trace installs a callback invoked with every intermediate vector, for diffing
// against an independent oracle layer by layer. nil disables it.
func (s *Instance) Trace(fn func(layer int, name string, v []float32)) { s.opt.tracer = fn }

func (s *Instance) trace(l int, name string, v []float32) {
	if s.opt.tracer != nil {
		s.opt.tracer(l, name, v)
	}
}
