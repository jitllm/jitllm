package layout

// byteOrder says a sub-block's m8n8k4 steps take consecutive elements
// (voltaSteps): an 8-bit format's, and a two-byte float's.
func (qi qinfo) byteOrder() bool { return qi.bits == 8 || qi.float == 2 }
