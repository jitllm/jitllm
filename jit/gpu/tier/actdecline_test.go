package tier

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestDeclineReasonReachesLastErr: prepLayer must record its decline reason, or
// a block sent to the host looks exactly like one the tier never saw.
func TestDeclineReasonReachesLastErr(t *testing.T) {
	g := newDevice(&devShared{})
	p := visionPlan()
	p.Act = nn.ActSwiGLUOAI // a gated kind in an ungated block: still declined
	// The recover is part of the assertion: with no device behind this
	// devTier, prepLayer can only return by declining before any allocation,
	// where declineReason is placed on purpose. A removed decline walks into
	// the scratch builder and panics.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("prepLayer went past the decline and panicked (%v): a declined "+
				"block reached allocation instead of being refused before it", r)
		}
	}()
	if g.prepLayer(0, 0, p, &nn.LayerWeights{}, false) {
		t.Fatal("prepLayer accepted a vision block with a gated activation")
	}
	if g.LastErr == "" || !strings.Contains(g.LastErr, "swiglu-oai") {
		t.Fatalf("LastErr is %q; a decline nobody can read is an absence", g.LastErr)
	}
	t.Logf("LastErr: %s", g.LastErr)
}
