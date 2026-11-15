package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestBlockGraphsRoundTrip holds the blocks' graphs through the config's
// encoding, with and without a joint head before them, and refuses a
// statement that leaves a block out, names a graph it does not hold or
// holds one no block reads.
func TestBlockGraphsRoundTrip(t *testing.T) {
	g := BlockGraphs{Graphs: []string{"first", "second"}, Of: []uint16{0, 1, 1, 0}}
	c := &Config{NLayer: 3, NMTP: 1, NEmbd: 8, NHead: 1, NKVHead: 1, HeadDim: 8, NRot: 8, NFFN: 16,
		NVocab: 32, NCtx: 64, Blocks: g}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Blocks, g) || got.JointHead != nil {
		t.Fatalf("came back as %+v with joint head %+v, want %+v and none", got.Blocks, got.JointHead, g)
	}
	if got.Blocks.Graph(2) != "second" || got.Blocks.Graph(4) != "" {
		t.Fatalf("block 2's graph %q, block 4's %q", got.Blocks.Graph(2), got.Blocks.Graph(4))
	}
	j := &JointHead{TypeRows: [3]uint8{1, 2, 0}, SortKeys: true, Confidence: 1}
	c.JointHead = j
	if got, err = decodeConfig(encodeConfig(c)); err != nil || !reflect.DeepEqual(got.JointHead, j) || !reflect.DeepEqual(got.Blocks, g) {
		t.Fatalf("with a joint head: %+v, %+v (%v)", got.JointHead, got.Blocks, err)
	}
	c.Blocks = BlockGraphs{}
	if got, err = decodeConfig(encodeConfig(c)); err != nil || len(got.Blocks.Of) != 0 || !reflect.DeepEqual(got.JointHead, j) {
		t.Fatalf("with no graphs: %+v, %+v (%v)", got.Blocks, got.JointHead, err)
	}
	for _, b := range []struct {
		name, want string
		g          BlockGraphs
	}{
		{"a block left out", "for 3 blocks of 4", BlockGraphs{Graphs: []string{"a"}, Of: []uint16{0, 0, 0}}},
		{"a graph it does not hold", "is 2 of 2", BlockGraphs{Graphs: []string{"a", "b"}, Of: []uint16{0, 1, 2, 0}}},
		{"a graph no block reads", "no block's", BlockGraphs{Graphs: []string{"a", "b"}, Of: []uint16{0, 0, 0, 0}}},
		{"an empty graph", "0 bytes", BlockGraphs{Graphs: []string{""}, Of: []uint16{0, 0, 0, 0}}},
	} {
		c.Blocks = b.g
		if _, err := decodeConfig(encodeConfig(c)); err == nil || !strings.Contains(err.Error(), b.want) {
			t.Errorf("%s: %v, want an error naming %q", b.name, err, b.want)
		}
	}
}

// TestBlockGraphsStateTheHeadBlocks: a marker head's blocks follow the
// trunk's in the statement; one written before they were stated leaves them
// out, and any other count is refused.
func TestBlockGraphsStateTheHeadBlocks(t *testing.T) {
	c := &Config{NLayer: 3, NEmbd: 8, NHead: 1, NKVHead: 1, HeadDim: 8, NRot: 8, NFFN: 16, NVocab: 32, NCtx: 64,
		Flags2: Flag2HeadBlocks, DecisionBlocks: 2}
	if c.HeadBlocks() != 2 {
		t.Fatalf("%d head blocks, want 2", c.HeadBlocks())
	}
	for _, b := range []struct {
		n  int
		ok bool
	}{{5, true}, {3, true}, {4, false}, {6, false}} {
		c.Blocks = BlockGraphs{Graphs: []string{"a"}, Of: make([]uint16, b.n)}
		got, err := decodeConfig(encodeConfig(c))
		switch {
		case b.ok && err != nil:
			t.Errorf("%d blocks' graphs: %v", b.n, err)
		case b.ok && len(got.Blocks.Of) != b.n:
			t.Errorf("%d blocks' graphs came back as %d", b.n, len(got.Blocks.Of))
		case !b.ok && (err == nil || !strings.Contains(err.Error(), "blocks of 5")):
			t.Errorf("%d blocks' graphs: %v, want a refusal", b.n, err)
		}
	}
	c.Flags2 = 0
	if c.HeadBlocks() != 0 {
		t.Fatalf("%d head blocks without Flag2HeadBlocks", c.HeadBlocks())
	}
}
