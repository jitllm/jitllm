package lowertest

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/paritytest/mainkernels"
	"github.com/jitllm/jitllm/jit/gpu/msl"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
	"github.com/jitllm/jitllm/jit/gpu/spirv"
)

// Which formats the device can unpack is a list in the repo. A page-in with no
// unpack kernel silently falls back to the much slower host packer, so a
// format is either implemented and lowers on every backend, or it declines
// with a reason.
var (
	unpackImplemented = []mainkernels.Quant{mainkernels.Q4_K}
	unpackDeclined    = []mainkernels.Quant{
		mainkernels.Q4_0, mainkernels.Q5_0, mainkernels.Q5_1, mainkernels.Q8_0, mainkernels.Q3_K, mainkernels.Q5_K,
		mainkernels.Q6_K, mainkernels.TQ2, mainkernels.TQ2F,
		// MXFP4's 17-byte block straddles words.
		mainkernels.MXFP4,
		// A float format's pack is a copy of its rows.
		mainkernels.Float32, mainkernels.Float16, mainkernels.BFloat16,
	}
)

// TestUnpackCoversExactlyWhatItClaims is the sweep.
func TestUnpackCoversExactlyWhatItClaims(t *testing.T) {
	seen := map[mainkernels.Quant]int{}
	for _, q := range unpackImplemented {
		seen[q]++
	}
	for _, q := range unpackDeclined {
		seen[q]++
	}
	// The enum's length is pinned, so a new format needs a decision here.
	n := 0
	for {
		if !quantExists(mainkernels.Quant(n)) {
			break
		}
		n++
	}
	if n != len(seen) {
		t.Fatalf("mainkernels.Quant has %d values and this file accounts for %d; a format added "+
			"without an entry above gets the 0.90 GB/s host packer and no red line", n, len(seen))
	}
	for q, c := range seen {
		if c != 1 {
			t.Errorf("%s appears %d times across the implemented and declined lists", q, c)
		}
	}

	for _, q := range unpackImplemented {
		if !mainkernels.UnpackSupported(q) {
			t.Errorf("%s is in the implemented list and UnpackSupported says no", q)
			continue
		}
		k, err := mainkernels.Unpack(q, 256, 2048)
		if err != nil {
			t.Errorf("%s: %v", q, err)
			continue
		}
		if err := k.Validate(); err != nil {
			t.Errorf("%s: validate: %v", q, err)
		}
		// Validate (just run) proves pSrc is only loaded and the three
		// destinations only stored.
		if _, err := ptx.Lower(k, "sm_86"); err != nil {
			t.Errorf("%s: ptx: %v", q, err)
		}
		if _, err := spirv.Emit(k); err != nil {
			t.Errorf("%s: spirv: %v", q, err)
		}
		if _, err := msl.Emit(k); err != nil {
			t.Errorf("%s: msl: %v", q, err)
		}
		t.Logf("%s: unpack lowers on ptx, spirv and msl", q)
	}

	for _, q := range unpackDeclined {
		if mainkernels.UnpackSupported(q) {
			t.Errorf("%s is in the declined list and UnpackSupported says yes", q)
		}
		k, err := mainkernels.Unpack(q, 256, 2048)
		if err == nil {
			t.Errorf("%s built an unpack kernel and is listed as declined; move it", q)
			_ = k
			continue
		}
		// The decline must say why, so a real constraint (a block length not
		// a multiple of 4, whose fields straddle u32 boundaries) can be told
		// from work not done yet.
		if !strings.Contains(err.Error(), q.String()) {
			t.Errorf("%s: the refusal does not name the format: %v", q, err)
		}
		if !strings.Contains(err.Error(), "multiple of 4") &&
			!strings.Contains(err.Error(), "not written yet") &&
			!strings.Contains(err.Error(), "nothing to unpack") {
			t.Errorf("%s: the refusal gives no reason: %v", q, err)
		}
		t.Logf("%s declines: %v", q, err)
	}
}

// quantExists reports whether n is a value of mainkernels.Quant, by asking the
// String method, which indexes a fixed array and panics past the end.
func quantExists(q mainkernels.Quant) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	_ = q.String()
	return true
}
