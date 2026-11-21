package engine

// Trace installs a callback invoked with every intermediate vector, for diffing
// against an independent oracle layer by layer. nil disables it.
func (s *Instance) Trace(fn func(layer int, name string, v []float32)) { s.opt.tracer = fn }

func (s *Instance) trace(l int, name string, v []float32) {
	if s.opt.tracer != nil {
		s.opt.tracer(l, name, v)
	}
}
