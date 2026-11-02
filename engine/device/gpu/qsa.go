package gpu

// QSA on the device: Qwen3.8-Flash-Next's block selection
// (nn.LayerPlan.QSA; engine/model/qsa.go is the graph).
//
// The cache row carries the indexer's raw key as one more kv head, as
// MiniMax-M3's does (msa.go), but unrotated: the row is copied whole and the
// rotary then turns k's heads alone. Each call:
//
//	the key: idx_k on the block's normed input, assembled after k's normed
//	  heads into one row
//	the query: idx_q on the same input, each head's RMSNorm, the rotary
//	the scores (kernels.QSAScores): each complete block's keys pooled,
//	  normed, rotated at the block's first position and scored against the
//	  query's heads; the tail block forced in
//	the selection, the compaction, the gather and the descriptors (MSA's,
//	  under QSA's forced tail)
//
// and the attention then runs over the gathered pages. Below the budget every
// block is kept, which is the reference's dense attention.
//
// The block-start rotary table is one table for every row: a block starts at
// a multiple of the ratio, which is its rotary position on a sequence with
// no picture.
