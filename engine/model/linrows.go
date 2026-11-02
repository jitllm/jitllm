package model

// A prompt chunk through a recurrent block.
//
// Only the recurrence is sequential: each row's state depends on the row
// before it, and so does the short convolution's window. The projections into
// the block and out of it read nothing but the normed residual and the gated
// output, so over a chunk they are GEMMs: the weights cross the memory bus
// once per chunk instead of once per row. linearPre runs the input
// projections over the chunk, the row loop runs the convolution, the gates,
// the recurrence and the gated norm per row reading row r of each
// (lproj, lrow), and linearPost runs the output projection over the gated
// rows the loop left (lout).
//
// Mamba-1's dt, B and C projections read the convolved x, which is row
// sequential, so they stay matvecs inside the loop.
//
// WithLinearRowByRow keeps every projection a matvec per row, decode's form;
// it is the arm the batched one is held to (TestLinearRowsMatchRowByRow).

// linRows is one chunk's projections for the block being run. on is set only
// between linearPre and linearPost.
type linRows struct {
	// ref is WithLinearRowByRow, copied when the instance is made; a test sets
	// it on one of two instances over one model.
	ref bool
	// chunks counts the block-chunks run batched, so a gate can assert the
	// arm it compared was the one selected.
	chunks int64
}
