package backend_test

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// The gate for a device unpack is byte equality against the host packer
// (mainkernels.PackWeightsInto), whose bytes every device matvec reads. An NMSE
// bound would admit a permuted layout, where every magnitude is right and every
// position is wrong.
//
// The shapes are real: the mapping is a function of nrows, baked into the
// kernel as a stride, and a stride error is invisible at any other row count.

// unpackCase is one tensor shape to unpack, with its source bytes.
type unpackCase struct {
	name    string
	rows, k int
	src     []byte
}

// q4kShapes is every distinct 2-D Q4_K (rows, k) that
// Llama-3.3-70B-Instruct-Q4_K_M and tinyllama-1.1b-q3_K_M hold, read out of
// those files once and recorded here.
//
// The shapes are baked and the bytes are generated: byte equality is exact for
// any input, so only the row counts need to be real, and no GGUF is needed.
// 128256x8192 is Llama-3.3-70B's lm_head.
var q4kShapes = []struct {
	rows, k int
	from    string
}{
	{256, 2048, "tinyllama attn_k/attn_v"},
	{2048, 2048, "tinyllama attn_q/attn_output"},
	{1024, 8192, "Llama-3.3-70B attn_k/attn_v"},
	{2048, 5632, "tinyllama ffn_down"},
	{8192, 8192, "Llama-3.3-70B attn_q/attn_output"},
	{28672, 8192, "Llama-3.3-70B ffn_gate/ffn_up"},
	{8192, 28672, "Llama-3.3-70B ffn_down"},
	{128256, 8192, "Llama-3.3-70B lm_head"},
}

// q4kSource builds rows*k elements of valid Q4_K, deterministically.
//
// The f16 scales are constructed rather than random, so none is NaN or Inf (a
// shared non-finite word would agree on both sides). The sub-block bytes and
// the payload are arbitrary, since only a permutation reads them.
func q4kSource(rows, k int) []byte {
	const blockE, blockB = 256, 144
	nblk := rows * k / blockE
	src := make([]byte, nblk*blockB)
	x := uint32(0x9E3779B9)
	next := func() uint32 {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		return x
	}
	for b := 0; b < nblk; b++ {
		o := b * blockB
		// d and dmin: f16 with a small exponent and a varying mantissa, so
		// every one is finite and no two consecutive blocks share a scale.
		d := uint16(0x2C00 | (next() & 0x03FF))
		dmin := uint16(0x2000 | (next() & 0x03FF))
		src[o], src[o+1] = byte(d), byte(d>>8)
		src[o+2], src[o+3] = byte(dmin), byte(dmin>>8)
		for i := 4; i < blockB; i += 4 {
			v := next()
			for j := 0; j < 4 && i+j < blockB; j++ {
				src[o+i+j] = byte(v >> (8 * j))
			}
		}
	}
	return src
}

// q4kCases is one case per shape above, plus the two ragged row counts.
func q4kCases(t *testing.T) []unpackCase {
	t.Helper()
	var out []unpackCase
	for _, sh := range q4kShapes {
		out = append(out, unpackCase{
			name: fmt.Sprintf("%dx%d", sh.rows, sh.k),
			rows: sh.rows, k: sh.k, src: q4kSource(sh.rows, sh.k),
		})
	}
	sort.Slice(out, func(i, j int) bool {
		return int64(out[i].rows)*int64(out[i].k) < int64(out[j].rows)*int64(out[j].k)
	})

	// Every real shape's thread count divides by 128, so the surplus-thread
	// clamp (RULE 13) never runs. Two ragged prefixes exercise it: 1000 rows
	// leaves 64 surplus threads in the last group, 1 row leaves 120. The assert
	// below checks the configuration was actually selected. The base is the
	// smallest tensor that still holds 1000 rows.
	base := out[len(out)-1]
	for _, c := range out {
		if c.rows >= 1000 && int64(c.rows)*int64(c.k) < int64(base.rows)*int64(base.k) {
			base = c
		}
	}
	if base.rows < 1000 {
		t.Fatalf("no shape holds 1000 rows; the ragged cases would not build")
	}
	for _, rows := range []int{1000, 1} {
		c := unpackCase{
			name: fmt.Sprintf("ragged/%dx%d", rows, base.k),
			rows: rows, k: base.k, src: base.src[:rows*base.k/256*144],
		}
		if got := mainkernels.UnpackThreads(mainkernels.Q4_K, c.rows, c.k) % 128; got == 0 {
			t.Fatalf("%s was meant to be ragged and its thread count is a multiple of 128; "+
				"the surplus-thread clamp would go unexercised under a green line", c.name)
		}
		out = append(out, c)
	}
	return out
}

// TestUnpackMatchesHostPacker is the gate.
func TestUnpackMatchesHostPacker(t *testing.T) {
	gpuLock(t)
	// Every device, not backend.Open's discrete-first pick: a second driver's
	// access-chain lowering is exactly what an unpack can disagree about.
	devs := importDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	cases := q4kCases(t)
	if len(cases) < 5 {
		t.Fatalf("%d shapes collected; the two models hold eight between them, so this "+
			"would be a gate that passes having checked almost nothing", len(cases))
	}
	for _, d := range devs {
		d := d
		t.Run(devTag(d), func(t *testing.T) {
			defer d.Close()
			t.Logf("device: %s", d.Name())
			for _, c := range cases {
				c := c
				t.Run(c.name, func(t *testing.T) {
					qs, dw, sc, err := mainkernels.PackWeights(mainkernels.Q4_K, c.src, c.rows, c.k)
					if err != nil {
						t.Fatalf("host pack: %v", err)
					}
					kk, err := mainkernels.Unpack(mainkernels.Q4_K, c.rows, c.k)
					if err != nil {
						t.Fatalf("Unpack: %v", err)
					}
					gq, gd, gs, over := runUnpack(t, d, kk, c, len(qs), len(dw), len(sc))
					if over != 0 {
						t.Fatalf("%s: the unpack wrote %d words past the ends of its destinations", c.name, over)
					}
					cmpWords(t, "qs", c, qs, gq)
					cmpWords(t, "d", c, dw, gd)
					cmpWords(t, "sc", c, sc, gs)
					t.Logf("%s: %d payload + %d d + %d sc words identical to PackWeightsInto",
						c.name, len(qs), len(dw), len(sc))
				})
			}
		})
	}
}

// runUnpack uploads the raw GGUF bytes, launches the kernel and reads the three
// destination arrays back, with overrun the words it wrote past their ends.
//
// The destinations are poisoned before the launch (RULE 13) with 0xA5 bytes,
// which the packer cannot produce for a whole array, so an unwritten word
// cannot match a host word that happens to be zero.
//
// Each destination carries a poisoned guard of unpackGuard(c) words past its
// end. A bent kernel's store lands there rather than past the allocation: on
// llvmpipe a buffer is host memory, and a store past it corrupted the process
// and killed the package with SIGSEGV. A write into a guard is an overrun the
// shipping kernel must never make.
func runUnpack(t *testing.T, d backend.Device, kk *ir.Kernel, c unpackCase,
	nq, nd, nsc int) (qs, dw, sc []uint32, overrun int) {
	t.Helper()
	if err := kk.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kern.Close()

	bs, err := d.Alloc(len(c.src))
	if err != nil {
		t.Skipf("DEVICE FULL: cannot allocate %d MiB for the source (%v) -- this shape "+
			"proved nothing; run this package alone", len(c.src)>>20, err)
	}
	defer bs.Free()
	guard := unpackGuard(c)
	outs := make([]backend.Buf, 3)
	for i, n := range []int{nq, nd, nsc} {
		n += guard
		b, err := d.Alloc(n * 4)
		if err != nil {
			// A card with no free VRAM (another package's tests may hold it)
			// cannot answer this gate. Only an allocation refusal skips, and
			// it names the size; every comparison below still fails loudly.
			t.Skipf("DEVICE FULL: cannot allocate %d MiB for out %d (%v) -- this shape "+
				"proved nothing; run this package alone", n*4>>20, i, err)
		}
		defer b.Free()
		poison := make([]byte, n*4)
		for j := range poison {
			poison[j] = 0xA5
		}
		if err := b.Write(poison); err != nil {
			t.Fatalf("poison out %d: %v", i, err)
		}
		outs[i] = b
	}
	if err := bs.Write(c.src); err != nil {
		t.Fatalf("upload src: %v", err)
	}
	const width = 128
	threads := mainkernels.UnpackThreads(mainkernels.Q4_K, c.rows, c.k)
	if threads != c.rows*c.k/256 {
		t.Fatalf("UnpackThreads says %d, the mapping is one thread per (row, super-block) = %d",
			threads, c.rows*c.k/256)
	}
	groups := (threads + width - 1) / width
	if err := kern.Launch(groups, width, bs, outs[0], outs[1], outs[2]); err != nil {
		t.Fatalf("launch %d x %d: %v", groups, width, err)
	}
	qs, dw, sc = make([]uint32, nq+guard), make([]uint32, nd+guard), make([]uint32, nsc+guard)
	for i, v := range [][]uint32{qs, dw, sc} {
		if err := outs[i].Read(u32bytes(v)); err != nil {
			t.Fatalf("read out %d: %v", i, err)
		}
		for _, w := range v[len(v)-guard:] {
			if w != 0xA5A5A5A5 {
				overrun++
			}
		}
	}
	return qs[:nq], dw[:nd], sc[:nsc], overrun
}

// unpackGuard is the words of poison past each destination's end: more than
// the farthest a violation bends a store (four sub-blocks of rows).
func unpackGuard(c unpackCase) int { return 4*c.rows + 64 }

// cmpWords reports the first differing word with enough structure to name the
// sub-block, because "byte 41,238,016 differs" is not a diagnosis.
func cmpWords(t *testing.T, what string, c unpackCase, want, got []uint32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: %d words on the host, %d from the device", what, len(want), len(got))
	}
	if bytes.Equal(u32bytes(want), u32bytes(got)) {
		return
	}
	bad := 0
	first := -1
	for i := range want {
		if want[i] != got[i] {
			bad++
			if first < 0 {
				first = i
			}
		}
	}
	i := first
	row, lane := i%c.rows, i/c.rows
	extra := ""
	switch what {
	case "qs":
		extra = fmt.Sprintf(" super=%d sub=%d word=%d", lane/32, (lane%32)/4, lane%4)
	case "sc":
		extra = fmt.Sprintf(" super=%d pair=%d", lane/4, lane%4)
	case "d":
		extra = fmt.Sprintf(" super=%d", lane)
	}
	t.Errorf("%s %s: %d of %d words differ; first at %d (row=%d%s): host %08x, device %08x",
		c.name, what, bad, len(want), i, row, extra, want[i], got[i])
}

// devTag names a subtest for a device, since this host has three SPIR-V devices
// and "spirv" alone would let Go number them rather than say which is which.
func devTag(d backend.Device) string {
	for _, f := range strings.Fields(d.Name()) {
		if len(f) == 0 || !isLetter(f[0]) {
			continue
		}
		n := 0
		for n < len(f) && (isLetter(f[n]) || (f[n] >= '0' && f[n] <= '9')) {
			n++
		}
		return d.API() + "-" + f[:n]
	}
	return d.API()
}

func isLetter(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }
