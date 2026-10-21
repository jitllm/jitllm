package nn

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestRowIQ4NLMatchesTheDecoder holds the generated decode to quant's IQ4_NL
// decoder, bit for bit (quant.TestIQ4NLMatchesGGUFPy holds that one to
// gguf-py), on the host tier and, on an AVX2 host, the SSE tier too: random
// blocks with planted finite scales at row widths of one block through
// Qwen3.8-Flash-Next's five and more, with a guard after the output; and it
// fails with one table entry wrong.
func TestRowIQ4NLMatchesTheDecoder(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	halves := []uint16{0x3C00, 0x3800, 0x0001, 0x03FF, 0x0400, 0x1000, 0x7BFF, 0xBC00, 0x8001, 0x0000}
	scratch := make([]byte, 32)
	for _, tier := range sseAndHost() {
		code, err := cpu.EmittersFor(tier).RowIQ4NL()
		if err != nil {
			t.Fatal(err)
		}
		c, err := cpu.Map(code)
		if err != nil {
			t.Fatal(err)
		}
		run := func(dst []float32, raw []byte, konst []byte) {
			c.Call(&cpu.Args{Out: &dst[0], W: &raw[0], K: int64(len(raw) / 18), Scr: &konst[0], Scratch: &scratch[0]})
		}
		for nb := 1; nb <= 9; nb++ {
			raw := make([]byte, 18*nb)
			for i := range raw {
				raw[i] = byte(rng.Uint32())
			}
			for b := 0; b < nb; b++ {
				h := halves[(b+nb)%len(halves)]
				raw[18*b], raw[18*b+1] = byte(h), byte(h>>8)
			}
			want := make([]float32, 32*nb)
			if err := quant.Dequant32(quant.IQ4_NL, raw, want); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, 32*nb+1)
			got[32*nb] = 12345
			run(got, raw, cpu.IQ4NLConsts(quant.IQ4NLValues))
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%v, %d blocks: element %d is %g, the decoder %g", tier, nb, i, got[i], want[i])
				}
			}
			if got[32*nb] != 12345 {
				t.Fatalf("%v: the decode wrote past %d blocks", tier, nb)
			}
			bad := quant.IQ4NLValues
			bad[raw[2]&0xF]++
			run(got, raw, cpu.IQ4NLConsts(bad))
			same := true
			for i := range want {
				same = same && got[i] == want[i]
			}
			if same {
				t.Fatalf("%v: a wrong table entry decodes the same", tier)
			}
		}
		c.Close()
	}
}

// sseAndHost is the host's tier and, where it is above SSE on amd64, SSE too.
func sseAndHost() []cpu.Tier {
	ts := []cpu.Tier{cpu.HostTier()}
	if runtime.GOARCH == "amd64" && cpu.HostTier() != cpu.TierSSE {
		ts = append(ts, cpu.TierSSE)
	}
	return ts
}
