//go:build amd64

package cpu

// EmitNGramHash is the AVX2 tier's n-gram hash (emitNGramHashX86).
func EmitNGramHash(n, per int, reset bool) ([]byte, error) {
	return emitNGramHashX86(n, per, reset, false)
}
