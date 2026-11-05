package model

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The n-gram embedding: Qwen3.8-Flash-Next's PLE (transformers'
// Qwen4ExpTextPLELayer, llama.cpp's build_ple), on one linear block, before
// its attention mix.
//
// Each position's token and its NGram-1 predecessors are hashed per head
// (Config.NGramMul, NGramOff, NGramVoc; a predecessor at or before an
// end-of-sequence token, or before the sequence, reads as that token) into a
// row of a shared table, read a row at a time from the container
// (jlm.RowTable): it is hundreds of millions of rows, of which a position
// reads (NGram-1)*NGramHeadsPer, each head's from its own range. The graph
// states all of it (frontngram.go); the CPU device reads the rows a call's
// hash names into the table's view between its gathers' halves (cpuForm).
// The rows, concatenated, project to a key
// per stream and one value; each stream's normed key and normed residual give
// a gate, sigmoid of the signed square root of their scaled dot product; the
// gated value is added to the stream, and so is a depthwise convolution over
// it (normed), dilated by NGram, through silu.
//
// The predecessors and the convolution's history are the block's recurrent
// state, after the delta rule's (Config.ngramStateLen), so they page,
// relocate and are prefix-cached with it.

// ngramTable is the container's n-gram table.
type ngramTable struct {
	E        *jlm.Entry
	typ      quant.Type
	RowBytes int
}
