//go:build amd64 && jitllmtest

package cpu

// The SSE twin of the pre-VNNI violation hook (prevnni_hook_amd64.go), armed
// by the SAME switch -- SetNaiveSignedQ8ForTest -- so a gate that arms it sees
// both tiers' Q8_0 kernels go wrong together, and a release build carries
// neither sequence (sse_packed_release.go).

func (d sseDot) naiveActive() bool { return naiveSignedQ8 }

// naiveSigned emits the centred-and-saturating sequence a mechanical port of
// the VNNI kernel produces -- XOR 0x80 into the payload, then PMADDUBSW -- and
// reports whether it did. 2*255*127 = 64,770 saturates int16, so the dot
// products come back slightly wrong and nothing crashes. The 0x80 bytes are
// built in t (PCMPEQB, PABSB, PSLLW 7: 0x0101 << 7 = 0x8080), because no
// register holds a sign flip on this path.
func (d sseDot) naiveSigned(a *Buf, acc, pay, act, t Reg) bool {
	if !naiveSignedQ8 {
		return false
	}
	a.PCMPEQB(t, t, t)
	a.PABSB(t, t)
	a.PSLLW(t, t, 7)
	a.PXOR(pay, pay, t)
	a.PMADDUBSW(pay, pay, act)
	a.PMADDWD(pay, pay, d.ones)
	a.PADDD(acc, acc, pay)
	return true
}
