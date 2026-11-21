package sched

import (
	"sync/atomic"
	"testing"
)

// fakeNUMA gives a pool n nodes without a multi-node host: every CPU is dealt
// to node cpu%n, or all to node `all` when it is >= 0. That is enough to gate
// the claiming arithmetic -- the part that must cover every index exactly once
// whatever the placement says.
func fakeNUMA(p *Pool, n, all int) {
	p.nodeIDs = make([]int, n)
	for i := range p.nodeIDs {
		p.nodeIDs[i] = i
	}
	p.cpuNode = make([]int, 4096)
	for c := range p.cpuNode {
		p.cpuNode[c] = c % n
		if all >= 0 {
			p.cpuNode[c] = all
		}
	}
	p.nnext = make([]atomic.Int64, n)
}

// TestDoNodesCoversEveryIndexOnce: a node-affine claim that skips an index
// leaves rows of a matvec unwritten, and one that claims an index twice adds a
// row's product into it twice -- both are finite, plausible and wrong. Ragged
// totals, a first node other than 0, and chunks smaller than the block.
func TestDoNodesCoversEveryIndexOnce(t *testing.T) {
	p := New([]int{0, 1, 2, 3, 4, 5})
	defer p.Close()
	for _, n := range []int{2, 3} {
		fakeNUMA(p, n, -1)
		for _, total := range []int{64, 65, 128, 1000, 896, 7} {
			// {10, 64}, {3, 4} and {100, 64} do not divide the block, so chunks
			// cross from one of a node's blocks into its next.
			for _, c := range []struct{ chunk, block int }{{64, 64}, {16, 64}, {1, 4}, {8, 64}, {10, 64}, {3, 4}, {100, 64}} {
				for first := 0; first < n; first++ {
					hits := make([]atomic.Int32, total)
					p.DoNodes("", total, c.chunk, c.block, first, func(_, lo, hi int) {
						for i := lo; i < hi; i++ {
							hits[i].Add(1)
						}
					})
					for i := range hits {
						if h := hits[i].Load(); h != 1 {
							t.Fatalf("nodes %d total %d chunk %d block %d first %d: index %d claimed %d times",
								n, total, c.chunk, c.block, first, i, h)
						}
					}
				}
			}
		}
	}
}

// TestDoNodesPrefersTheLocalNode: with one participant per node and no
// stealing pressure, a node's own blocks go to its own worker. Checked with
// the caller alone (participant 0, node 0): it must drain node 0's blocks
// before any other node's, which is the whole point of the claim order.
func TestDoNodesPrefersTheLocalNode(t *testing.T) {
	p := New([]int{0})
	defer p.Close()
	fakeNUMA(p, 2, 0) // the caller is on node 0 wherever it runs
	var order []int
	// total > chunk so the region is dispatched rather than run inline.
	p.DoNodes("", 256, 64, 64, 1, func(_, lo, _ int) { order = append(order, lo) })
	// first=1: block 0 is node 1's, block 1 node 0's, so node 0 owns 64 and 192.
	want := []int{64, 192, 0, 128}
	for i := range want {
		if i >= len(order) || order[i] != want[i] {
			t.Fatalf("claim order %v, want %v -- the caller's node first", order, want)
		}
	}
}
