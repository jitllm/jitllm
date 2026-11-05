package engine

// A prompt's last chunk is often a few rows: 514 tokens at a device chunk of
// 512 are a chunk of 512 and one of 2. The device runs a chunk at its grain
// (32 or 64 rows), so those 2 rows ran every batched kernel at that width --
// a staged GEMM dequantizing all its weights for 2 live rows, the attention
// over the padding. On a V100 with Llama-3.2-1B Q4_K_M the 2-row chunk ran
// 6.7 ms of kernels where a decode token runs about 1.6, and a 514-token
// prompt took 30% longer than a 512-token one. llama.cpp runs a batch that
// small as a matvec.
//
// A ragged step of the chunk's rows (nn.RowsDevice, one sequence, the slot its
// position) is that matvec: the tier's decode kernel carrying every live row
// in one pass over the weights (groupMV, up to four rows), each row attending
// to its own history. The same 2 rows ran 2.2 ms of kernels there. The head is
// the prompt's tail call, as it is for any chunk that did not fold it.
//
// Not for a hybrid (a recurrent block's state is stepped through a session's
// seat, which the single-sequence call does not name), a speculation step
// (its rows' logits are read back), a bidirectional run or a split
// placement: those keep the batched chunk, as does a device that refuses the
// step. WithTailRows(false) is the control arm.

// tailRowsMax is the widest chunk taken as a ragged step: the tier's one-pass
// matvec carries up to four rows (groupTokMax); past it the step's matvecs pad
// to the grain as the chunk's do.
const tailRowsMax = 4
