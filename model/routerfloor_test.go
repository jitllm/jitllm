package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// TestRouterFloorMatchesTheOracle holds the composed router's renormalising
// floor to the oracle's: an oracle that guards its divide differently from the
// form it gates is not an oracle.
func TestRouterFloorMatchesTheOracle(t *testing.T) {
	if routerSumFloor != oracle.MoESumFloorForTest {
		t.Fatalf("the router floors its sum at %g, the oracle at %g", routerSumFloor, oracle.MoESumFloorForTest)
	}
}
