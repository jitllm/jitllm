package cpu

import "testing"

// TestAttnF16KernelSize records what a binary16 KV cache costs in
// instructions. On amd64 VCVTPH2PS widens straight from memory, so the f16
// kernel is about the same size; on arm64 LDRd plus FCVTL is two instructions
// where LDRq was one, so the kernel is ~25% bigger and f16 does not pay (see
// docs/engineering-history/cpu-kernels.md). A log, not a bar.
func TestAttnF16KernelSize(t *testing.T) {
	for _, hd := range []int{64, 128, 256} {
		s32, err := EmitAttnScores(hd, hd*4, KVF32)
		if err != nil {
			t.Fatal(err)
		}
		s16, _ := EmitAttnScores(hd, hd*4, KVF16)
		a32, _ := EmitAttnAcc(hd, hd*4, KVF32)
		a16, _ := EmitAttnAcc(hd, hd*4, KVF16)
		t.Logf("%s hd=%3d  scores %d -> %d (%+.0f%%)  acc %d -> %d (%+.0f%%)",
			NativeArch(), hd, len(s32), len(s16),
			100*float64(len(s16)-len(s32))/float64(len(s32)),
			len(a32), len(a16), 100*float64(len(a16)-len(a32))/float64(len(a32)))
	}
}
