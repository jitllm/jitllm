package graph

import (
	"strings"
	"testing"
)

// The MLP partition is found on the dense FFN by its edges and refused by
// its static bound at every row class: on the dense facts (a 2048 row, an
// 8192 intermediate) the partials are 8*2048*32 bytes a row against the
// 8*8192 the intermediate cost. A residual narrower than one slice passes
// the bound and is then refused for want of a contract, not accepted. The
// gated delta sublayer, whose act_mul reads a head norm, is no match.
func TestMLPPartitionIsRefusedByItsBound(t *testing.T) {
	f := stated(false)
	f.Config.NEmbd, f.Config.NFFN, f.Config.NHead, f.Config.NKVHead, f.Config.HeadDim = 2048, 8192, 32, 8, 64
	g := f.Config.Build()
	os := Match(g, MLPPartition)
	if len(os) != 1 {
		t.Fatalf("%d matches", len(os))
	}
	b := MLPPartition.Bounds(g, os[0].Region)
	ff, h := int64(f.Config.NFFN), int64(f.Config.NEmbd)
	if want := 8*ff - 8*h*((ff+PartitionSlice-1)/PartitionSlice); b.TrafficBytes != want || b.Launches != 0 || want >= 0 {
		t.Fatalf("bounds %+v, want %d bytes (negative)", b, want)
	}
	for _, rows := range []RowClass{RowsDecode, RowsFew, RowsChunk} {
		err := os[0].Offer(Target{API: "ptx", Rows: rows})
		if err == nil || !strings.Contains(err.Error(), "static bound") {
			t.Fatalf("rows %d: %v", rows, err)
		}
	}
	narrow := stated(false) // a row of 64 under one slice of 96
	ng := narrow.Config.Build()
	no := Match(ng, MLPPartition)
	if len(no) != 1 {
		t.Fatalf("narrow: %d matches", len(no))
	}
	if err := no[0].Offer(Target{API: "ptx", Rows: RowsChunk}); err == nil || !strings.Contains(err.Error(), "contract") {
		t.Fatalf("narrow: %v", err)
	}
	df := deltaFacts(ActSiLU, 0)
	if os := Match(df.Delta.Build(), MLPPartition); len(os) != 0 {
		t.Fatalf("delta sublayer: %d matches", len(os))
	}
}
