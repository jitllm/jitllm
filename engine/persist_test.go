package engine

// carried says whether the positions in over (ascending, each past a gate's
// bound) are damage that persists, out of n compared in order: more than a
// quarter of them, three in a row, or the last one, which has nothing after it
// to show whether it carried.
//
// A defect in a stored row or a kernel reaches every position after the one it
// hits, since every later position reads that row. A model can also amplify a
// rounding at a few positions and pass it no further: a router that takes one
// decision the other way, or a neuron that moves 25% for a 4% input change.
// A windowed or chunked layer lets a corrupted row fall out of reach, so the
// damage can stop carrying after a few positions, which is why three in a row
// is enough.
func carried(over []int, n int) bool {
	c := len(over) > n/4 || len(over) > 0 && over[len(over)-1] == n-1
	for i := 0; i+2 < len(over); i++ {
		c = c || over[i+2] == over[i]+2
	}
	return c
}
