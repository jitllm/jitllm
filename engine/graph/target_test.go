package graph

import (
	"strings"
	"testing"
)

// The target guards: gate2in is offered on a PTX decode, refused at a chunk
// row class (it removes nothing there) and on a backend with no contract,
// each with a reason; the verdict key carries every guard dimension.
func TestGate2InOfferIsGuardedByTarget(t *testing.T) {
	f := stated(false)
	g := f.Config.Build()
	r := Apply(g, Gate2In)[0].Region
	dec := Target{API: "ptx", ISA: "sm_70", Subgroup: 32, SharedMem: 48 << 10, Rows: RowsDecode}
	if err := Gate2In.Offer(g, r, dec); err != nil {
		t.Fatal(err)
	}
	chunk := dec
	chunk.Rows = RowsChunk
	if err := Gate2In.Offer(g, r, chunk); err == nil || !strings.Contains(err.Error(), "row class") {
		t.Fatalf("a chunk: %v", err)
	}
	spirv := dec
	spirv.API = "spirv"
	if err := Gate2In.Offer(g, r, spirv); err == nil || !strings.Contains(err.Error(), "no numerical contract") {
		t.Fatalf("spirv: %v", err)
	}
	streams := dec
	streams.Layout.Streams = 4
	mma := dec
	mma.MMA = "m16n8k16"
	for _, o := range []Target{chunk, spirv, streams, mma} {
		if o.Key() == dec.Key() {
			t.Fatalf("%+v shares the decode key %s", o, dec.Key())
		}
	}
}
