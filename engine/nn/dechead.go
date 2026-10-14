package nn

// ClefHead is Clef's joint schema head as a device runs it
// (engine/model/decideclef.go has the graph; jit/gpu/tier/dechead.go runs it).
// Every matrix is f32, row-major, one row per output; the model dequantizes
// them once when the head is first offered, and the device keeps them for the
// life of the value (keyed by its pointer).
type ClefHead struct {
	D, W, Heads, FFN, Routing   int
	Eps, RMSEps                 float32
	OutNorm                     []float32 // the backbone's output RMSNorm, D
	HidW, HidB                  []float32
	PMem, PQ, POQ, PG, POC, POL []float32 // W x D each
	SumW, SumB                  []float32
	FieldW, FieldB              []float32
	OptW, OptB                  []float32
	Scales                      [3]float32 // prior, joint, gate: the values used
	Types                       []float32  // 3 x W
	Sc, ScB                     []float32  // W x 4W, W
	ScOut                       []float32  // 1 x W
	ScOutB                      []float32  // 1
	Blocks                      []ClefHeadBlock
}

// ClefHeadBlock is one head block: a routing block's KVW/KVB set and no
// self-attention, a joint block's the other way round.
type ClefHeadBlock struct {
	SelfW, SelfB, SQ, SK, SV, SO, SBQ, SBK, SBV, SBO []float32
	CrossW, CrossB, KVW, KVB                         []float32
	CQ, CK, CV, CO, CBQ, CBK, CBV, CBO               []float32
	FFNW, FFNB, Up, BUp, Down, BDown                 []float32
}

// ClefHeadInput is one request's part: the backbone's final residual rows
// (before its output norm), N of them; the pooling rows (NQ questions then
// one row picking the last position, then NO options, each N wide, a span's
// rows weighted 1/len); each option's question; each question's type row;
// each option's lexical mean (the output embedding's rows for its tokens,
// D wide); and the per-question mask over the options (0 for its own, -Inf
// elsewhere, NQ x NO).
type ClefHeadInput struct {
	N, NQ, NO int
	Resid     []float32
	Pool      []float32
	Owner     []uint32
	QType     []uint32
	Lex       []float32
	Mask      []float32
}

// ClefHeadDevice is a device that runs Clef's head over the rows it is
// handed and writes one logit per option into out, in the input's order. It
// returns false having written nothing it should be read for, and the
// caller runs the head on the host.
type ClefHeadDevice interface {
	ClefHead(h *ClefHead, in *ClefHeadInput, out []float32) bool
}
