//go:build amd64

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestTileTokensPinNarrowsAndNeverDisables is the gate on WithTileTokens.
//
// It asserts the width, not a counter or a rate: a pin that never reached the
// emitter and a pin above the register file that emits nothing both look like
// "no difference".
func TestTileTokensPinNarrowsAndNeverDisables(t *testing.T) {
	const k, nrows = 2048, 2048
	for _, gt := range []quant.Type{quant.Q4_K, quant.Q8_0, quant.Q6_K} {
		maxTok := cpu.MaxTiledTokensNative(gt)
		if maxTok < 2 {
			continue
		}
		t.Run(gt.String(), func(t *testing.T) {
			def := widthWithPin(t, gt, k, nrows, 0)
			if def < 2 {
				t.Fatalf("%s has no default tile at k=%d nrows=%d, so this gate "+
					"would be comparing nothing", gt, k, nrows)
			}
			// Narrowing must be observed, not assumed.
			if got := widthWithPin(t, gt, k, nrows, 2); got != 2 {
				t.Errorf("%s pinned to 2 ran at %d -- the pin did not reach the emitter", gt, got)
			}
			// A pin wider than the register file must not disable the tile.
			// tiledFor's narrowing loop is what holds this; the violation is
			// the pin becoming the loop's floor, as want does.
			if got := widthWithPin(t, gt, k, nrows, 32); got != def {
				t.Errorf("%s pinned to 32 ran at %d, want the unpinned %d -- an "+
					"out-of-range pin disabled the tile instead of clamping", gt, got, def)
			}
		})
	}
}

func widthWithPin(t *testing.T, gt quant.Type, k, nrows, pin int) int {
	t.Helper()
	opts := []Option{WithTune(TuneOff), WithQuietTuner(true)}
	if pin > 0 {
		opts = append(opts, WithTileTokens(pin))
	}
	f := NewJIT(k, nrows, []quant.Type{gt}, opts...)
	if f == nil {
		t.Skip("no JIT on this build")
	}
	defer f.Close()
	return f.TiledWidth(gt, k, nrows)
}
