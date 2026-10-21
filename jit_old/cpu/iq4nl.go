//go:build amd64

package cpu

// EmitRowIQ4NL is the AVX2 tier's IQ4_NL row decode (EmitRowIQ4NLAVX2).
func EmitRowIQ4NL() []byte { return EmitRowIQ4NLAVX2() }
