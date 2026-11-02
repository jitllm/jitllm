package model

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The gated delta net: the other half of a hybrid's block loop.
//
// The graph is llama.cpp's running graph (layer 0 of Qwen3-Next-80B-A3B under
// `llama-eval-callback`), and six of its steps are ones a from-paper reading
// gets wrong:
//
//	(1) SiLU on the convolution's output, before it is split
//	(2) q and k L2-normalised: no weight, divided by the vector's own length
//	(3) 16 key heads and 32 value heads, interleaved: value head h reads h/2
//	(4) the decay softplus(a + dt_bias) * ssm_a, ssm_a already -exp(A_log)
//	(5) beta a plain sigmoid, no bias
//	(6) the output gate SiLU(z), where the full-attention layers use sigmoid
//
// The state is not a KV cache: it is the same size at every position, so
// residency, paging and placement cannot treat it as KV.
//
// The node-by-node transcript: docs/engineering-history/model-correctness.md,
// "engine/model/delta.go".

// kdaL2Eps is the epsilon Kimi Delta Attention's q/k L2 normalisation adds to
// the sum of squares. It is the reference helper's own default, not the
// model's rms_norm_eps (1e-5 on the checkpoints shipped so far).
const kdaL2Eps = 1e-6

// deltaGeom is the recurrent block's derived geometry. Every field is a
// function of Config.SSM; they are computed once per call rather than stored so
// that a Config stays a description of the file.
type deltaGeom struct {
	conv    int // convolution width, in positions
	chans   int // convolved channels: q + k + v
	kHeads  int // key/query heads
	vHeads  int // value heads, and the number of decay rates
	kDim    int // key/query head width, which is also the state's key extent
	vDim    int // value head width, which is also the state's value extent
	rep     int // value heads per key head
	qkBytes int // q and k each occupy kHeads*kDim of the convolved channels
	// chanDecay is Kimi Delta Attention rather than qwen3next's gated delta
	// rule: the decay is a vector per value head instead of one rate, and both
	// gates come off a rank-kDim bottleneck (f_a_proj and g_a_proj project to
	// linear_head_dim, which is kDim).
	chanDecay bool
	rank      int
	// tiled pairs value head vh with key head vh % kHeads instead of vh / rep:
	// the order llama.cpp's qwen35 converter writes the value heads in. See
	// keyHead.
	tiled bool
	// ssd is Mamba-2's selective update in place of the delta rule (see
	// ssdAttn): the same state, k is B, q is C and v is x, and the convolved
	// channels arrive in that order because the converter put them there.
	ssd bool
	// scale is 1/sqrt(kDim), which the L2 norm's rescale and the delta rule's
	// query scale both want.
	scale float32
	// mamba1 is Mamba-1's selective scan (mamba1Attn): every channel a value
	// head of one row (vHeads is Inner, vDim one), B and C one key head
	// shared by all, and only x convolved -- B and C come off it afterwards,
	// so chans is Inner. rank is its dt bottleneck.
	mamba1 bool
	// tail is the recurrent state's tail per sequence (Config.recTailLen).
	tail int
}

func (c *Config) delta() deltaGeom {
	g := deltaGeom{
		conv:   int(c.SSM.ConvKernel),
		kHeads: int(c.SSM.Groups),
		vHeads: int(c.SSM.NHeadV),
		kDim:   int(c.SSM.StateSize),
	}
	g.qkBytes = g.kHeads * g.kDim
	g.chans = int(c.SSM.Inner) + 2*g.qkBytes
	if g.vHeads > 0 {
		g.vDim = int(c.SSM.Inner) / g.vHeads
	}
	if g.kHeads > 0 {
		g.rep = g.vHeads / g.kHeads
	}
	if g.kDim > 0 {
		g.scale = 1 / float32(math.Sqrt(float64(g.kDim)))
	}
	g.chanDecay, g.rank = c.ChanDecay(), g.kDim
	g.tiled = c.DeltaKeyTiled
	g.ssd = c.SSD()
	g.tail = c.recTailLen()
	if c.Mamba1() {
		g.mamba1, g.chans, g.qkBytes, g.rank = true, int(c.SSM.Inner), 0, c.dtRank
	}
	return g
}

// mixer is a recurrent block's mixer by its structure, as the front end
// states it.
type mixer uint8

const (
	mixNone      mixer = iota
	mixDelta           // the gated delta rule (jlm.LayerLinearAttn)
	mixSelective       // the additive update, one decay a head (jlm.LayerSSD, LayerSSDAttn)
	mixScan            // the additive update, one decay a state element of each channel (jlm.LayerMamba1)
	mixShortConv       // the gated short convolution (jlm.LayerShortConv)
)

// mixerAt is block li's recurrent mixer, mixNone for a block with none.
func (c *Config) mixerAt(li int) mixer {
	switch c.LayerKind(li) {
	case jlm.LayerLinearAttn:
		return mixDelta
	case jlm.LayerSSD, jlm.LayerSSDAttn:
		return mixSelective
	case jlm.LayerMamba1:
		return mixScan
	case jlm.LayerShortConv:
		return mixShortConv
	}
	return mixNone
}

// keyHead is the key/query head value head vh reads.
//
// Two pairings ship, and both are fluent when wrong. qwen3next's GGUF keeps
// HuggingFace's grouped order (value heads kh*rep .. kh*rep+rep-1 share key
// head kh); llama.cpp's qwen35 converter reorders to tiled order (value head
// vh shares key head vh % kHeads). They differ only when rep > 1.
func (g deltaGeom) keyHead(vh int) int {
	if g.tiled {
		return vh % g.kHeads
	}
	return vh / g.rep
}

// convStateLen and deltaStateLen are one sequence's recurrent state for one
// linear layer. Both are constant in the context length.
func (g deltaGeom) convStateLen() int  { return (g.conv - 1) * g.chans }
func (g deltaGeom) deltaStateLen() int { return g.vHeads * g.vDim * g.kDim }

// stateStride is one sequence's whole recurrent state row: the delta rule's
// and the tail after it (Config.recTailLen).
func (g deltaGeom) stateStride() int { return g.deltaStateLen() + g.tail }

// allocRecurrent sizes the recurrent state and the scratch the delta rule
// needs. Both are nil for every architecture that is not a hybrid.
//
// It is per sequence, like the KV cache: a batched session's rows share the
// weights and nothing else.
func (s *Instance) allocRecurrent(nseq int) {
	c := s.c
	if !c.Hybrid() || !s.linearIn() {
		return
	}
	g := c.delta()
	s.dg = g
	s.lin.ref = s.m.opt.linRowByRow
	s.rconv = make([][]float32, c.NLayer)
	s.rstate = make([][]float32, c.NLayer)
	for i := 0; i < c.NLayer; i++ {
		if !c.LayerKind(i).Recurrent() {
			continue
		}
		s.rconv[i] = make([]float32, nseq*g.convStateLen())
		s.rstate[i] = make([]float32, nseq*g.stateStride())
	}
	s.dMixed = make([]float32, g.chans)
	if g.mamba1 {
		s.dLowRank = make([]float32, g.rank)
		s.dSB, s.dSC = make([]float32, g.kDim), make([]float32, g.kDim)
		s.dZeroA = make([]float32, g.vHeads)
	}
	if c.ShortConv() {
		// B and C, the short convolution's two gates.
		s.dZ = make([]float32, 2*g.chans)
	} else {
		s.dZ = make([]float32, g.vHeads*g.vDim)
	}
	// KDA's decay is per channel, so its gate buffers are vHeads*kDim wide;
	// qwen3next's are vHeads.
	nGate := g.vHeads
	if g.chanDecay {
		nGate = g.vHeads * g.kDim
	}
	s.dBA = make([]float32, 2*g.vHeads)
	s.dDecay = make([]float32, nGate)
	s.dBeta = make([]float32, g.vHeads)
	s.dAlpha = make([]float32, nGate)
	s.dBRaw = make([]float32, g.vHeads)
	if g.chanDecay {
		s.dLowRank = make([]float32, g.rank)
		// The gate kernel writes both outputs, and KDA wants them at two
		// different lengths: the decay over vHeads*kDim and beta over vHeads.
		// dGateWaste receives whichever half the call in question is not for.
		s.dGateWaste = make([]float32, nGate)
	}
	s.dOut = make([]float32, g.vHeads*g.vDim)
	s.dOnes = make([]float32, g.kDim)
	for i := range s.dOnes {
		s.dOnes[i] = 1
	}
}

// linearIn reports whether any block this instance runs is a linear one. A draft
// Instance over a prediction block runs attention alone, so it keeps no summary
// and recurrent() is false for it.
func (s *Instance) linearIn() bool {
	for li := s.lo; li < s.hi; li++ {
		if s.c.LayerKind(li).Recurrent() {
			return true
		}
	}
	return false
}

// ResetRecurrent clears one row's recurrent state, which is what retiring a
// sequence means for a linear layer: there is no per-position cache to
// invalidate, only a running summary to forget.
func (s *Instance) ResetRecurrent(row int) {
	for li := range s.rconv {
		if s.rconv[li] == nil {
			continue
		}
		n := s.dg.convStateLen()
		clear(s.rconv[li][row*n : (row+1)*n])
		n = s.dg.stateStride()
		clear(s.rstate[li][row*n : (row+1)*n])
	}
}

// recTail is sequence slot's tail of layer li's recurrent state (recTailLen),
// after its delta-rule state; empty on a model with none.
func (s *Instance) recTail(li, slot int) []float32 { return s.recTailOf(s.Session, li, slot) }

// recTailOf is recTail of session o's state.
func (s *Instance) recTailOf(o *Session, li, slot int) []float32 {
	if s.dg.tail == 0 || o.rstate == nil || o.rstate[li] == nil {
		return nil
	}
	n := s.dg.stateStride()
	return o.rstate[li][slot*n+s.dg.deltaStateLen() : (slot+1)*n]
}

// recTailLen is what each linear block's recurrent state carries per
// sequence after the delta rule's: room for Qwen3.8-Flash-Next's n-gram
// predecessors and convolution history (ngram.go), which its n-gram block
// fills and every other leaves unread, nothing on any other model. Every
// linear block has the same row, [delta rule | tail], so a device keeps one
// slot width for all of them (nn.RecurrentPlan.TailLen).
func (c *Config) recTailLen() int {
	if c.NGramBlock != 0 {
		return c.ngramStateLen()
	}
	return 0
}
