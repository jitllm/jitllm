package engine

// convertKV narrows the floats f of head rows hd wide into the history's
// bytes h, or widens h into f, on the device the instance places the
// conversions on (nn.KVConvDevice); false where they do not serve.
func (s *Instance) convertKV(narrow bool, hd int, f []float32, h []byte) bool {
	return s.kvConv != nil && s.kvConv.Convert(narrow, hd, f, h)
}
