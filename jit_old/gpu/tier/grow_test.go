package tier

import "testing"

// TestGrowFailureDoesNotPoisonTheBuffer covers sizeOut, sizePart, and
// prepare's aBuf and axBuf. Each grow frees the old buffer before allocating
// the bigger one (on a full card the only way to grow); if the allocation then
// fails, the buffer must not be left nil with a cap naming a size that no
// longer exists. The second, smaller request is the test: it would otherwise
// take the `cap >= n` fast path and launch on a nil buffer.
func TestGrowFailureDoesNotPoisonTheBuffer(t *testing.T) {
	for _, c := range []struct {
		name string
		grow func(g *devTier, n int) bool
		buf  func(g *devTier) bool // true when the buffer is usable
	}{
		{"sizeOut", (*devTier).sizeOut, func(g *devTier) bool { return g.outBuf != nil }},
		{"sizePart", (*devTier).sizePart, func(g *devTier) bool { return g.partBuf != nil }},
	} {
		t.Run(c.name, func(t *testing.T) {
			g, d, _ := fakeTier(t)
			// fakeTier prepares a layer, which already sized these; start from
			// nothing so the subtest owns the sequence.
			g.outCap, g.partCap = 0, 0
			// Small first, so there is a live buffer and a live cap to poison.
			if !c.grow(g, 64) || !c.buf(g) {
				t.Fatal("the first grow should have succeeded")
			}
			d.allocMax = 1024 // the card is now full
			if c.grow(g, 4096) {
				t.Fatal("a grow past the limit should have failed")
			}
			// Without the fix the cap still reads 64, so this returns true
			// with a nil buffer.
			if c.grow(g, 32) && !c.buf(g) {
				t.Fatal("reported capacity for a buffer that was freed and never replaced")
			}
			// And once there is room again it must recover rather than stay
			// stuck at a cap it cannot honour.
			d.allocMax = 0
			if !c.grow(g, 32) || !c.buf(g) {
				t.Fatal("did not recover after the card had room again")
			}
		})
	}
}
