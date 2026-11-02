//go:build amd64

package cpu

// runnable is Baseline's question one level finer: not "can this host run any
// code we emit" but "can it run this code". It sits at the choke point (Map)
// rather than at the emitters, because a VNNI question asked at nn's entry
// points let VPDPBUSD reach an AVX2 host with no VNNI through paths that
// skipped them.
//
// It scans bytes because Map takes bytes. A constant pool could in principle
// spell the pattern; the cost is one extra decline on a host already declining
// that family, while missing a real site is a SIGILL, so the scan errs eager.
//
// VPDPBUSD is VEX.128/256.66.0F38.W0 50 /r. The two-byte VEX form (0xC5) cannot
// encode the 0F38 map at all, so only the three-byte form can carry it:
//
//	C4  RXB.mmmmm  W.vvvv.L.pp  50
//	        mmmmm=00010 (0F38)      pp=01 (66), W=0
func runnable(code []byte) error {
	if CPU().AVXVNNI {
		return nil
	}
	for i := 0; i+3 < len(code); i++ {
		if code[i] != 0xC4 || code[i+3] != 0x50 {
			continue
		}
		if code[i+1]&0x1F != 0x02 { // mmmmm != 0F38
			continue
		}
		if code[i+2]&0x03 != 0x01 || code[i+2]&0x80 != 0 { // pp != 66, or W != 0
			continue
		}
		return ErrNoVNNI
	}
	return nil
}
