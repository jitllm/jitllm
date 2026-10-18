package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The vision tower: an encoder carried in the model's own container.
//
// A tower is a transformer over patches, so it is the prefill path (batched
// matmuls, the pool) with four substitutions: LayerNorm instead of RMSNorm, an
// ungated GELU MLP instead of SwiGLU, bidirectional attention, and no KV cache.
//
// The projector list is a list (idefics3, mlp, qwen2vl_merger,
// qwen2.5vl_merger, gemma3, internvl); an unknown one is refused, not
// defaulted.

// TowerConfig is the clip.* header. The names are ggml's, not invented here.
type TowerConfig struct {
	NLayer  int
	NEmbd   int // clip.vision.embedding_length
	NFFN    int // clip.vision.feed_forward_length
	NHead   int
	HeadDim int
	ImageSz int // clip.vision.image_size
	PatchSz int // clip.vision.patch_size
	// ProjDim is what the LAST projector matrix emits and must equal the text
	// model's NEmbd. It is NOT clip.vision.projection_dim, which llava-phi-3
	// sets to 768 while its mm.2 emits 3072 -- see convert.visionOf.
	ProjDim int
	Scale   int // clip.vision.projector.scale_factor, the idefics3 pixel shuffle
	Eps     float64
	// Act is the MLP's non-linearity: CLIP's tower is quick-GELU, SmolVLM's is
	// GELU-tanh.
	Act nn.ActKind
	// CLS says a class embedding is prepended to the patch sequence, so the
	// tower runs Seq() = Patches()+1 rows while the projector still emits
	// Tokens(): the class row is dropped rather than projected.
	CLS bool
	// Rope says position is 2-D rotary over the patch grid rather than a
	// learned table (Qwen2-VL): half a head's rotary pairs carry the patch row
	// and half its column. It comes from the container's flag, not from posW
	// being nil, so a truncated file is not mistaken for a rotary tower.
	Rope  bool
	PreLN bool // a LayerNorm over the whole sequence before block 0
	// PosBuckets, when set, says the position table is PosBuckets x
	// PosBuckets rows indexed by bucketing each patch's fractional (row,
	// column), so a grid of any shape reads it (MiniCPM-V's SigLIP; see
	// bucketPositions). Zero is a table with one row per patch of the square
	// ImageSz grid.
	PosBuckets int
	// Queries is how many tokens the resampler emits per image or slice,
	// whatever its grid; zero for every other projector.
	Queries   int
	Mean      [3]float64
	Std       [3]float64
	Projector string
	// Kind is Projector as the container's code, which is what the graph
	// switches on.
	Kind jlm.Projector
	// MinTiles and MaxTiles bound a tiling preprocessor (InternVL); zero for
	// a tower that takes one square.
	MinTiles, MaxTiles int
	// fault is a violation a gate runs to show it discriminates; zero in
	// every real use.
	fault towerFault
	// RMS says the tower's norms -- each block's two and the merger's -- are
	// RMSNorms with no bias (Qwen2.5-VL, Pixtral), not LayerNorms.
	RMS bool
	// Gated says the tower's MLP is gated (SwiGLU): Qwen2.5-VL's and
	// Pixtral's, every other tower's is up, activation, down.
	Gated bool
	// CLSLast says the class token is the LAST row, with the position
	// table's last row (Llama 4), where CLIP's is the first.
	CLSLast bool
	// WinPattern and WinSize are window attention (jlm.Vision.WinPattern):
	// every block but each WinPattern'th attends only inside its window of
	// WinSize pixels. Zero is every block over every patch.
	WinPattern, WinSize int
	// PosAxis is the rows of each of Gemma 4's two position tables (one per
	// axis); zero for every other tower.
	PosAxis int
	// MinPixels and MaxPixels bound a dynamic-resolution tower's input: its
	// patch grid follows the picture's aspect, resized to whole merge units
	// inside the bounds (smart_resize, qwenvit.go). Zero for a fixed-size tower.
	MinPixels, MaxPixels int
	// PosSide is a learned position table's side when the table is resampled
	// to each picture's grid (Qwen3-VL, qwen3vit.go); zero otherwise.
	PosSide int
	// PosBicubic says the table is resampled bicubic (GLM-4.xV, grid_sample
	// with align_corners False), not bilinear with aligned corners.
	PosBicubic bool
	// TokenLimit is Kimi-VL's processor's in_token_limit: a picture of more
	// whole patches is scaled down to it (kimivit.go).
	TokenLimit int
	// TileSize and Tolerance are LFM2-VL's tiling (lfm2vision.go): a picture
	// past Tolerance times MaxPixels is cut into TileSize squares. Zero for
	// every other tower.
	TileSize  int
	Tolerance float64
	// Bicubic says the pictures are resampled bicubic (FlagVisionBicubic).
	Bicubic bool
	// Deep is the segment-local blocks whose output a deepstack merger reads,
	// ascending (Qwen3-VL); tap k's rows add into the text model after its
	// block k.
	Deep []int
}

// towerFault names one wrong step a vision gate must catch.
type towerFault uint8

const (
	towerFaultNone towerFault = iota
	// towerFaultPoolCorner takes each pooled group's first patch instead of
	// its average: gemma3 with the pool skipped.
	towerFaultPoolCorner
	// towerFaultShuffleSwap groups dx outer and dy inner: the transposed
	// shuffle.
	towerFaultShuffleSwap
	// towerFaultTilesReversed hands InternVL's squares over in reverse, the
	// thumbnail first.
	towerFaultTilesReversed
	// faultNoBuckets reads the position table at the patch's own index, as a
	// tower with one row per patch of a square grid would.
	faultNoBuckets
	// faultNoKeyPos gives the resampler's keys no position.
	faultNoKeyPos
	// faultQueryRaw skips the queries' LayerNorm.
	faultQueryRaw
	// faultSliceOrder encodes a picture's slices in reverse.
	faultSliceOrder
	// faultNoNorm feeds the tower 8-bit samples scaled to [0, 1], with no mean
	// or std.
	faultNoNorm
	// faultDeepPreNorm skips the deepstack mergers' LayerNorm, the norm that
	// sets them apart from the main merger (after the shuffle, not before).
	faultDeepPreNorm
	// faultNoPosLerp adds the position table's top-left rows unresampled, as a
	// tower reading the table at the patch's own index would.
	faultNoPosLerp
	// faultNoEmbdNorm skips GLM-4.xV's RMSNorm over the patch embedding.
	faultNoEmbdNorm
	// faultBilinearTable resamples a bicubic table bilinearly.
	faultBilinearTable
	// faultKimiAxes turns MoonViT's even pairs by the row and odd by the
	// column.
	faultKimiAxes
	// faultKimiNoPreNorm skips Kimi-VL's projector LayerNorm over each patch.
	faultKimiNoPreNorm
	// faultHyNoNewline puts each merged row's last unit where HunyuanVL's
	// newline row goes.
	faultHyNoNewline
	// The remaining faults are gates' violations of one family's step, each
	// false in every real run: what the name says is what the tower does wrong.
	faultKeyAliases        // imageKey leaves the configuration and the grid out of the name
	faultVideoFirstFrame   // each video pair's second frame is its first
	faultQwenNoUnwindow    // qwen's merged rows stay in window order
	faultGemma4NoClamp     // gemma4 skips every clamp
	faultGemma4NoPool      // gemma4 takes each group's first patch
	faultGemma4RowFirst    // gemma4's row turns the first half of the head
	faultLFM2NoProjNorm    // LFM2-VL skips the projector's LayerNorm
	faultLFM2TanhGELU      // LFM2-VL's projector GELU in llama.cpp's tanh form
	faultLFM2NoThumb       // LFM2-VL leaves a tiled picture's thumbnail out
	faultLFM2Bilinear      // LFM2-VL resamples bilinear whatever the file says
	faultLlama4FromZero    // llama4's positions start at zero
	faultLlama4RowFirst    // llama4's row turns the first half
	faultPhi4Bicubic       // phi4 resamples with Pillow's BICUBIC
	faultPhi4NoResize      // phi4 reads the square table's first rows
	faultPhi4PosTransposed // phi4 resizes the table with the grid transposed
	faultPixtralFreq       // pixtral's rotary frequencies wrong
	faultPixtralMerge      // pixtral's merge wrong
	faultPixtralNoBreaks   // pixtral leaves the break rows out
	// MobileNet-V5's (mobilenetrun.go).
	mnFaultSymmetric  // pads every window k/2 on both sides, PyTorch's static padding, in place of TF SAME's smaller half first
	mnFaultNoResidual // drops every block's residual add
	mnFaultUpsampleT  // reads the coarser stage's row at the transposed position when the fusion adapter upsamples it
	mnFaultPoolFirst  // takes each pooling window's first row for its mean
)

// tfault is the fault of the tower this State runs or encodes for: the
// tower's configuration is the instance a gate sets it on.
func (s *State) tfault() towerFault {
	if s.vis != nil && s.vis.t != nil {
		return s.vis.t.Cfg.fault
	}
	if s.m != nil && s.m.tower != nil {
		return s.m.tower.Cfg.fault
	}
	return towerFaultNone
}

// Patches is how many patch embeddings one image produces before the
// projector -- the most a dynamic-resolution tower's grid may hold, which is
// what its scratch and its device blocks are sized for.
func (c TowerConfig) Patches() int {
	if c.MaxPixels > 0 {
		return c.MaxPixels / (c.PatchSz * c.PatchSz)
	}
	return (c.ImageSz / c.PatchSz) * (c.ImageSz / c.PatchSz)
}

// Seq is how many rows the block loop runs over: the patches plus the class
// token when the tower has one.
func (c TowerConfig) Seq() int {
	if c.CLS {
		return c.Patches() + 1
	}
	return c.Patches()
}

// Grid is the shape of the embeddings the projector emits for one image, in
// the order it emits them: a single frame of the merged patch grid. It is
// where those rows turn on an M-RoPE text model (ImageGrid, mrope.go).
func (c TowerConfig) Grid() ImageGrid {
	side := c.ImageSz / c.PatchSz / max(c.Scale, 1)
	return c.gridOf(side, side)
}

// Tokens is how many embeddings reach the language model. The idefics3 projector
// groups Scale x Scale neighbouring patches into one, so it divides by Scale^2;
// llava's MLP projector has Scale 1 and projects each patch on its own; the
// resampler emits Queries whatever the grid.
func (c TowerConfig) Tokens() int {
	if c.Queries > 0 {
		return c.Queries
	}
	return c.Patches() / (c.Scale * c.Scale)
}

// MaxTokens is the most embeddings one encode emits: Pixtral's add a break
// row after every merged grid row but the last.
func (c TowerConfig) MaxTokens() int {
	if c.Kind == jlm.ProjPixtral {
		return c.Tokens() + c.maxSide()/max(c.Scale, 1) - 1
	}
	if c.Kind == jlm.ProjHunyuanVL {
		return c.hyMaxTokens()
	}
	return c.Tokens()
}

// MaxSeq is the most rows one encode runs. A tower that buckets its table
// (PosBuckets) takes the slices of a llava-uhd plan, which aim at ImageSz^2
// pixels and round each side to a whole patch: the rounding can grow a side
// of 1.5 patches to 2, so a third over the square grid plus a row and a column of rounding bounds
// every plan, and Layout refuses a grid past it. Every other tower reads its
// one square grid.
func (c TowerConfig) MaxSeq() int {
	if c.PosBuckets > 0 {
		side := c.ImageSz / c.PatchSz
		return c.Patches()*4/3 + 2*side
	}
	return c.Seq()
}

// projK is the projector's first matrix's k: a shuffle concatenates Scale^2
// patches into one row, gemma3's pool averages them into NEmbd.
func (c TowerConfig) projK() int {
	if c.Kind == jlm.ProjGemma3 {
		return c.NEmbd
	}
	return c.NEmbd * c.Scale * c.Scale
}

// projEps is the projector norm's epsilon. InternVL's is torch's LayerNorm
// default, 1e-5, not the tower's 1e-6: the reference constructs it without
// one, and llama.cpp hardcodes the same; LFM2-VL's nn.LayerNorm is the same
// default. gemma3's soft_emb_norm takes the tower's.
func (c TowerConfig) projEps() float64 {
	if c.Kind == jlm.ProjInternVL || c.Kind == jlm.ProjLFM2 {
		return 1e-5
	}
	return c.Eps
}

// vmats is a vision block's matrices: the attention four, the MLP's up and
// down, and the gate's when the MLP is gated.
func vmats(l *layer) []*tensor {
	ws := []*tensor{&l.wq, &l.wk, &l.wv, &l.wo, &l.up, &l.down}
	if l.gate.e != nil {
		ws = append(ws, &l.gate)
	}
	return ws
}

// Tower is the vision segment's entry and head: what turns a picture into the
// rows its blocks run over, and their last residual into the rows the text
// segment reads. The blocks themselves are the model's, m.layers[base,
// base+NLayer): ordinary layers in the one block space, paged by the one pager,
// offered through the one placement, and run by the one runner (hostRows) on a
// State whose configuration is the segment's (cfg).
type Tower struct {
	// container is the jlm the tower was read from.
	container *jlm.File
	// jit is the model's WithJITOptions set, so a State the tower makes on its
	// own (no text State to share a JIT with) is configured as the text
	// model's is.
	jit []nn.Option
	// opt points at the model's options. A pointer and not a copy, because
	// some knobs are set after Open and the tower must see them.
	opt *modelOpts
	// model is the model the tower belongs to.
	model *Model
	Cfg   TowerConfig
	// base is the first vision block in the model's block space: the
	// container's NBlocks, past every text and prediction block.
	base int
	// cfg is the segment's block geometry, the Config a vision State runs
	// with (TowerConfig.segConfig).
	cfg *Config

	// patchW is the conv-as-matmul, 2-D by the time it reaches a container:
	// [k, NEmbd] with k >= PatchSz*PatchSz*channels, zero-padded up to a whole
	// quantization block by convert.reshapePatchEmbd.
	patchW tensor
	// patchW1 is a conv3d patch embedding's second temporal plane alone
	// (convert.foldPatchPlanes), which a video's second frame reads; patchW
	// is the planes' sum. Absent on every other tower, and on a container
	// converted before the plane was kept.
	patchW1 tensor
	patchB  []float32
	clsW    []float32 // the class embedding, one NEmbd row, or nil
	posW    []float32 // learned position embeddings, Seq x NEmbd
	preLnW  []float32 // the pre-loop LayerNorm, or nil
	preLnB  []float32
	postLnW []float32
	postLnB []float32
	// projW is the FIRST projector matrix and the only one idefics3 has.
	// projW2 is llava's second, with a GELU-tanh between them.
	projW  tensor
	projB  []float32
	projW2 tensor
	projB2 []float32
	// projW3 is a third projector matrix: Llama 4's projector after its
	// adapter's two.
	projW3 tensor
	// g4 is Gemma 4's head and block constants, nil for every other tower
	// (gemma4vision.go).
	g4 *gemma4Head
	// mn is Gemma 3n's MobileNet-V5 stem, fusion adapter and embedder, nil
	// for every other tower (mobilenet.go).
	mn *mnTower
	// projNorm is the norm a projector applies before its first matrix:
	// gemma3's RMSNorm (no bias) and InternVL's LayerNorm (with projNormB).
	projNorm, projNormB []float32
	// rs is MiniCPM-V's resampler, in place of projW, or nil.
	rs *resampler
	// mergeW is Mistral 3's patch merger, NEmbd*Scale^2 -> NEmbd, between the
	// shuffle and projW; imgBreak is the [IMG_BREAK] row the head writes after
	// each merged grid row but the last (Pixtral). See pixtral.go.
	mergeW   tensor
	imgBreak []float32
	// deep is Qwen3-VL's deepstack mergers, one per TowerConfig.Deep entry.
	deep []deepstack
	// embdNorm is GLM-4.xV's RMSNorm after the patch embedding, nil for
	// every other tower; glm its merging convolution and projector MLP.
	embdNorm []float32
	glm      *glmProj
	// hy is HunyuanVL's merger (hunyuanvit.go), nil for every other tower.
	hy *hyProj
	// pending is the vision blocks buildTower read, until Open moves them into
	// the model's layers at base.
	pending []layer
}

// weights is every matrix of the entry and the head, in no particular order.
// It exists so binding the container and auditing the layout walk the same
// list rather than two copies of it. The blocks' are the model's layers.
func (t *Tower) weights() []*tensor {
	ws := []*tensor{&t.patchW}
	if t.patchW1.e != nil {
		ws = append(ws, &t.patchW1)
	}
	if t.mergeW.e != nil {
		ws = append(ws, &t.mergeW)
	}
	if t.projW.e != nil {
		ws = append(ws, &t.projW)
	}
	if t.projW2.e != nil {
		ws = append(ws, &t.projW2)
	}
	if t.projW3.e != nil {
		ws = append(ws, &t.projW3)
	}
	if t.rs != nil {
		ws = append(ws, t.rs.weights()...)
	}
	for i := range t.deep {
		ws = append(ws, &t.deep[i].fc1, &t.deep[i].fc2)
	}
	if g := t.glm; g != nil {
		ws = append(ws, &g.merge, &g.gate, &g.up, &g.down)
	}
	if t.hy != nil {
		ws = append(ws, &t.hy.merge)
	}
	if t.mn != nil {
		ws = append(ws, &t.mn.fuseExp, &t.mn.fuseProj)
	}
	return ws
}

// layers is the vision blocks, in the model's block space.
func (t *Tower) layers() []layer { return t.model.layers[t.base : t.base+t.Cfg.NLayer] }

// segConfig is the vision segment's block geometry as the runner reads it: a
// ViT block is the text kit's classic block (C6) -- LayerNorm with biases,
// biased q/k/v/o, an ungated MLP with biases -- or, for Qwen2.5-VL, an RMSNorm
// block with a gated MLP, and its attention is bidirectional over the image's
// rows (vision.go's key runs) with no rotary, or the 2-D one per patch.
func (c TowerConfig) segConfig() *Config {
	sc := &Config{
		Arch:          "vision:" + c.Projector,
		NonCausal:     true,
		NLayer:        c.NLayer,
		NEmbd:         c.NEmbd,
		NHead:         c.NHead,
		NKVHead:       c.NHead,
		HeadDim:       c.HeadDim,
		NFFN:          c.NFFN,
		NCtx:          c.MaxSeq(),
		RMSEps:        c.Eps,
		Act:           c.Act,
		LayerNorm:     !c.RMS,
		NoPosEnc:      !c.Rope,
		RopeNeox:      c.ropeNeox(),
		AttnScale:     1 / math.Sqrt(float64(c.HeadDim)),
		ResidualScale: 1,
		LogitScale:    1,
		HeadDimV:      c.HeadDim,
	}
	if c.Rope {
		sc.NRot = c.HeadDim
	}
	if c.Kind == jlm.ProjGemma4V {
		// A per-head q and k norm, a weightless v norm, and scores at scale
		// one: the norms carry it.
		sc.QKNorm, sc.VNorm, sc.AttnScale = true, true, 1
	}
	return sc
}

func buildTower(c0 *jlm.File) (*Tower, error) {
	t := &Tower{}
	c := &t.Cfg

	// The tower's description is the container's vision section, filled by
	// convert.visionOf, which also refuses an unimplemented projector.
	v := c0.Vision()
	if v == nil {
		return nil, fmt.Errorf("model: OpenTower: this container is not a vision tower")
	}
	switch v.Projector {
	case jlm.ProjIdefics3, jlm.ProjMLP, jlm.ProjQwen2VL, jlm.ProjQwen25VL, jlm.ProjGemma3, jlm.ProjInternVL,
		jlm.ProjResampler, jlm.ProjJanus, jlm.ProjPixtral, jlm.ProjLlama4, jlm.ProjGemma4V, jlm.ProjPhi4,
		jlm.ProjQwen3VL, jlm.ProjGLM4V, jlm.ProjGemma3nV, jlm.ProjKimiVL, jlm.ProjHunyuanVL, jlm.ProjLFM2:
	default:
		return nil, fmt.Errorf("model: OpenTower: projector %v unsupported "+
			"(only idefics3, mlp, qwen2vl_merger, qwen2.5vl_merger, qwen3vl_merger, glm4v, kimivl, hunyuanvl, gemma3, "+
			"internvl, resampler, janus_pro, pixtral, llama4, gemma4v, phi4, gemma3nv, lfm2)", v.Projector)
	}
	c.NLayer, c.NEmbd, c.NFFN = int(v.NLayer), int(v.NEmbd), int(v.NFFN)
	c.NHead, c.ImageSz, c.PatchSz = int(v.NHead), int(v.ImageSize), int(v.PatchSize)
	c.ProjDim, c.Scale, c.Eps = int(v.ProjDim), int(v.Scale), float64(v.Eps)
	c.Projector, c.Kind = v.Projector.String(), v.Projector
	c.MinTiles, c.MaxTiles = int(v.MinTiles), int(v.MaxTiles)
	// The activation is a kind the container states. Both flags set cannot be
	// honoured and is refused rather than resolved by precedence.
	switch {
	case v.Flags.Has(jlm.FlagGELU) && v.Flags.Has(jlm.FlagQuickGELU):
		return nil, fmt.Errorf("model: OpenTower: the container asks for GELU and quick-GELU at once")
	case v.Flags.Has(jlm.FlagGELU):
		c.Act = nn.ActGELU
	case v.Flags.Has(jlm.FlagQuickGELU):
		c.Act = nn.ActQuickGELU
	default:
		c.Act = nn.ActSiLU
	}
	c.CLS = c0.Has(jlm.RoleVClassEmbd, jlm.DenseBlock, -1)
	c.CLSLast = v.Projector == jlm.ProjLlama4
	c.Rope = v.Flags.Has(jlm.FlagVisionRope)
	c.RMS = v.Projector == jlm.ProjQwen25VL || v.Projector == jlm.ProjPixtral || v.Projector == jlm.ProjGemma4V ||
		v.Projector == jlm.ProjGLM4V
	c.WinPattern, c.WinSize = int(v.WinPattern), int(v.WinSize)
	if c.WinPattern > 0 && c.WinSize <= 0 {
		return nil, fmt.Errorf("model: OpenTower: window attention every %d blocks with no window size", c.WinPattern)
	}
	switch v.Projector {
	case jlm.ProjQwen2VL, jlm.ProjQwen25VL, jlm.ProjQwen3VL, jlm.ProjGLM4V:
		dynamicPixels(c)
	case jlm.ProjPixtral:
		pixtralPixels(c)
	case jlm.ProjGemma4V:
		gemma4Pixels(c)
	case jlm.ProjPhi4:
		if err := phi4Pixels(c, int(v.MinPixels), int(v.MaxPixels)); err != nil {
			return nil, err
		}
	case jlm.ProjHunyuanVL:
		// Read at its own size between the two pixel counts the file states,
		// to whole merge units (smart_resize, ResizeTo).
		c.MinPixels, c.MaxPixels = int(v.MinPixels), int(v.MaxPixels)
	case jlm.ProjLFM2:
		if err := lfm2Pixels(c, v); err != nil {
			return nil, err
		}
	}
	if v.Projector == jlm.ProjGLM4V {
		c.MinPixels = min(glmMinPixels, c.MaxPixels)
	}
	if v.Projector == jlm.ProjKimiVL {
		// No floor and no resize short of the limit: the tower holds what any
		// picture under it pads to.
		c.TokenLimit = kimiTokenLimit
		c.MaxPixels = kimiCapacity(c.TokenLimit) * c.PatchSz * c.PatchSz
	}
	if v.Projector == jlm.ProjQwen3VL {
		// Its processor's floor, under the ceiling: a floor above the ceiling
		// would grow every picture past what the tower holds.
		c.MinPixels = min(qwen3MinPixels, c.MaxPixels)
	}
	c.PreLN = c0.Has(jlm.RoleVPreNorm, jlm.DenseBlock, -1)
	for i := 0; i < 3; i++ {
		c.Mean[i], c.Std[i] = float64(v.Mean[i]), float64(v.Std[i])
	}
	if c.Scale <= 0 {
		c.Scale = 1
	}
	if c.NLayer == 0 || c.NEmbd == 0 || c.NHead == 0 {
		return nil, fmt.Errorf("model: OpenTower: the container's vision section is empty")
	}
	c.HeadDim = c.NEmbd / c.NHead
	if c.NHead*c.HeadDim != c.NEmbd {
		return nil, fmt.Errorf("model: OpenTower: n_embd %d is not divisible by %d heads", c.NEmbd, c.NHead)
	}
	if c.PatchSz == 0 || c.ImageSz%c.PatchSz != 0 {
		return nil, fmt.Errorf("model: OpenTower: image %d is not a multiple of patch %d", c.ImageSz, c.PatchSz)
	}
	if c.Patches()%(c.Scale*c.Scale) != 0 {
		return nil, fmt.Errorf("model: OpenTower: %d patches do not group into %dx%d", c.Patches(), c.Scale, c.Scale)
	}

	getAt := func(role jlm.Role, block, index int32) (tensor, error) {
		e, ok := c0.Find(role, block, index)
		if !ok {
			return tensor{}, fmt.Errorf("model: OpenTower: missing %v (block %d)", role, block)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok || !quant.Dequantable(typ) {
			return tensor{}, fmt.Errorf("model: OpenTower: %v is %v, no dequantizer", role, e.Type)
		}
		rows := 1
		for d := 1; d < int(e.NDim); d++ {
			rows *= int(e.Dims[d])
		}
		qs, _, _ := c0.Span(e)
		tn := tensor{e: e, typ: typ, data: qs, rows: rows, k: int(e.Dims[0])}
		if jlm.Packed(e.Type) {
			tn.packed = &nn.Packed{}
		}
		return tn, nil
	}
	get := func(role jlm.Role, block int32) (tensor, error) { return getAt(role, block, -1) }
	vecAt := func(role jlm.Role, block, index int32) ([]float32, error) {
		e, ok := c0.Find(role, block, index)
		if !ok {
			return nil, fmt.Errorf("model: OpenTower: missing %v (block %d)", role, block)
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok {
			return nil, fmt.Errorf("model: OpenTower: %v is %v, no dequantizer", role, e.Type)
		}
		n := 1
		for d := 0; d < int(e.NDim); d++ {
			n *= int(e.Dims[d])
		}
		qs, _, _ := c0.Span(e)
		out := make([]float32, n)
		if err := quant.Dequant32(typ, qs, out); err != nil {
			return nil, err
		}
		return out, nil
	}
	vec := func(role jlm.Role, block int32) ([]float32, error) { return vecAt(role, block, -1) }
	optVec := func(role jlm.Role, block int32) ([]float32, error) {
		if !c0.Has(role, block, -1) {
			return nil, nil
		}
		return vec(role, block)
	}

	var err error
	// Gemma 3n's tower is convolutional: its stem, blocks and head are its
	// own (mobilenet.go).
	if v.Projector == jlm.ProjGemma3nV {
		if err := t.loadGemma3nV(c0, get, vec, optVec); err != nil {
			return nil, err
		}
		return t, t.packedWeights()
	}
	if t.patchW, err = get(jlm.RoleVPatchEmbd, jlm.DenseBlock); err != nil {
		return nil, err
	}
	// The patch embedding is 2-D in a container: a conv whose stride equals its
	// kernel is a matmul, so convert.reshapePatchEmbd flattens it and pads k to
	// a whole quantization block. k is therefore >= PatchSz*PatchSz*channels;
	// the extra columns are zero and Encode leaves them zero.
	if t.patchW.e.NDim != 2 || t.patchW.rows != c.NEmbd {
		return nil, fmt.Errorf("model: OpenTower: patch_embd is %v (%d rows), want a 2-D [k %d]",
			t.patchW.e.Dims[:t.patchW.e.NDim], t.patchW.rows, c.NEmbd)
	}
	if want := c.PatchSz * c.PatchSz * 3; t.patchW.k < want {
		return nil, fmt.Errorf("model: OpenTower: patch_embd k is %d, want at least %d (%dx%dx3)",
			t.patchW.k, want, c.PatchSz, c.PatchSz)
	}
	if t.patchB, err = optVec(jlm.RoleVPatchBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if c0.Has(jlm.RoleVPatchEmbd, jlm.DenseBlock, 1) {
		if t.patchW1, err = getAt(jlm.RoleVPatchEmbd, jlm.DenseBlock, 1); err != nil {
			return nil, err
		}
		if t.patchW1.rows != t.patchW.rows || t.patchW1.k != t.patchW.k {
			return nil, fmt.Errorf("model: OpenTower: the second temporal plane is %dx%d, the first %dx%d",
				t.patchW1.rows, t.patchW1.k, t.patchW.rows, t.patchW.k)
		}
	}
	if c.CLS {
		if t.clsW, err = vec(jlm.RoleVClassEmbd, jlm.DenseBlock); err != nil {
			return nil, err
		}
		if len(t.clsW) != c.NEmbd {
			return nil, fmt.Errorf("model: OpenTower: the class embedding is %d floats, want %d",
				len(t.clsW), c.NEmbd)
		}
	}
	// resampled is a table read at any grid by its family's loader
	// (loadQwen3VL, loadGLM4V, loadKimiVL), whose rows are not the patches.
	resampled := v.Projector == jlm.ProjQwen3VL || v.Projector == jlm.ProjGLM4V || v.Projector == jlm.ProjKimiVL ||
		v.Projector == jlm.ProjHunyuanVL
	if !c.Rope || v.Projector == jlm.ProjLlama4 || v.Projector == jlm.ProjGemma4V {
		// Llama 4 adds its table AND turns q and k: both are the reference's.
		if t.posW, err = vec(jlm.RoleVPosEmbd, jlm.DenseBlock); err != nil {
			return nil, err
		}
	} else if resampled {
		// Qwen3-VL, GLM-4.xV and Kimi-VL have both: the rotary, and a table
		// resampled to the grid.
		if t.posW, err = vec(jlm.RoleVPosEmbd, jlm.DenseBlock); err != nil {
			return nil, err
		}
	} else if c0.Has(jlm.RoleVPosEmbd, jlm.DenseBlock, -1) {
		// Refused rather than ignored: the two halves of the container
		// disagree about what position means.
		return nil, fmt.Errorf("model: OpenTower: the container asks for rotary " +
			"positions AND carries a position table")
	}
	// The resampler's tower reads its table by bucket, so the table is a
	// square of buckets and not the patch grid: MiniCPM-V's is 70x70 (a 980
	// image of 14-pixel patches) under a 448 slice.
	if v.Projector == jlm.ProjResampler && !c.Rope {
		rows := len(t.posW) / c.NEmbd
		b := int(math.Sqrt(float64(rows)))
		if b*b != rows || rows*c.NEmbd != len(t.posW) {
			return nil, fmt.Errorf("model: OpenTower: the position table is %d floats, "+
				"not a square of %d-wide rows", len(t.posW), c.NEmbd)
		}
		c.PosBuckets = b
	}
	// Seq(), not Patches(): a CLIP tower's table has a row for the class
	// token.
	// Gemma 4's is two tables, one per axis, a patch adding the column's row
	// of the first and the row's of the second.
	if v.Projector == jlm.ProjGemma4V {
		if c.PosAxis = len(t.posW) / (2 * c.NEmbd); c.PosAxis == 0 || 2*c.PosAxis*c.NEmbd != len(t.posW) {
			return nil, fmt.Errorf("model: OpenTower: gemma4v's position tables are %d floats, not two tables "+
				"of %d-wide rows", len(t.posW), c.NEmbd)
		}
	} else if side := c.ImageSz / c.PatchSz; v.Projector == jlm.ProjPhi4 || v.Projector == jlm.ProjLFM2 {
		// Phi-4's and LFM2-VL's are a square table resized to each grid
		// (phi4Table).
		if len(t.posW) != side*side*c.NEmbd {
			return nil, fmt.Errorf("model: OpenTower: %s's position table is %d floats, want %dx%d rows of %d",
				c.Projector, len(t.posW), side, side, c.NEmbd)
		}
	} else if n := c.Seq() * c.NEmbd; t.posW != nil && c.PosBuckets == 0 && !resampled && len(t.posW) != n {
		return nil, fmt.Errorf("model: OpenTower: position embeddings are %d floats, want %d (%d rows x %d)",
			len(t.posW), n, c.Seq(), c.NEmbd)
	}
	if c.PreLN {
		if t.preLnW, err = vec(jlm.RoleVPreNorm, jlm.DenseBlock); err != nil {
			return nil, err
		}
		if t.preLnB, err = optVec(jlm.RoleVPreNormBias, jlm.DenseBlock); err != nil {
			return nil, err
		}
	}

	ls := make([]layer, c.NLayer)
	for i := range ls {
		b := &ls[i]
		bi := int32(i)
		for _, w := range []struct {
			role jlm.Role
			dst  *tensor
		}{
			{jlm.RoleVAttnQ, &b.wq}, {jlm.RoleVAttnK, &b.wk},
			{jlm.RoleVAttnV, &b.wv}, {jlm.RoleVAttnOut, &b.wo},
		} {
			if *w.dst, err = get(w.role, bi); err != nil {
				return nil, err
			}
		}
		// The MLP matrices are identified by shape, not by name: SmolVLM's
		// mmproj names the expansion ffn_down and the contraction ffn_up, the
		// opposite of llama's convention. Exactly one matrix has rows == NFFN
		// and exactly one has k == NFFN, which the checks below assert.
		na, errA := get(jlm.RoleVFC1, bi)
		if errA != nil {
			return nil, errA
		}
		nb, errB := get(jlm.RoleVFC2, bi)
		if errB != nil {
			return nil, errB
		}
		aUp, aDn := jlm.RoleVFC1Bias, jlm.RoleVFC2Bias
		// A width that is not a whole number of quantization blocks reaches
		// the container padded with zero columns (convert.quantizeTower):
		// the matrices reading the residual are read at NEmbd.
		for _, w := range []*tensor{&b.wq, &b.wk, &b.wv, &b.wo, &na, &nb} {
			unpadK(w, c.NEmbd)
		}
		if na.rows == c.NFFN && nb.k == c.NFFN {
			b.up, b.down = na, nb
		} else if nb.rows == c.NFFN && na.k == c.NFFN {
			b.up, b.down, aUp, aDn = nb, na, jlm.RoleVFC2Bias, jlm.RoleVFC1Bias
		} else {
			return nil, fmt.Errorf("model: OpenTower: block %d MLP is %dx%d and %dx%d; neither expands to NFFN %d",
				i, na.rows, na.k, nb.rows, nb.k, c.NFFN)
		}
		if b.up.k != c.NEmbd || b.down.rows != c.NEmbd {
			return nil, fmt.Errorf("model: OpenTower: block %d MLP does not round-trip NEmbd %d", i, c.NEmbd)
		}
		if b.upB, err = optVec(aUp, bi); err != nil {
			return nil, err
		}
		if b.downB, err = optVec(aDn, bi); err != nil {
			return nil, err
		}
		for _, v := range []struct {
			role jlm.Role
			dst  *[]float32
		}{
			{jlm.RoleVAttnNorm, &b.attnNorm}, {jlm.RoleVAttnNormBias, &b.attnNormB},
			{jlm.RoleVFFNNorm, &b.ffnNorm}, {jlm.RoleVFFNNormBias, &b.ffnNormB},
			{jlm.RoleVAttnQBias, &b.bq}, {jlm.RoleVAttnKBias, &b.bk},
			{jlm.RoleVAttnVBias, &b.bv}, {jlm.RoleVAttnOutBias, &b.bo},
			// Gemma 4's sandwich norms, q/k norms and clipped-linear bounds;
			// absent on every other tower.
			{jlm.RoleVPostAttnNorm, &b.postAttnNorm}, {jlm.RoleVPostFFNNorm, &b.postFFNNorm},
			{jlm.RoleVAttnQNorm, &b.qNorm}, {jlm.RoleVAttnKNorm, &b.kNorm},
			{jlm.RoleVClamp, &b.clamp},
		} {
			if *v.dst, err = optVec(v.role, bi); err != nil {
				return nil, err
			}
		}
		if b.attnNorm == nil || b.ffnNorm == nil {
			return nil, fmt.Errorf("model: OpenTower: block %d has no layer norms", i)
		}
		// An RMSNorm has no shift: a bias beside one is a container that
		// disagrees with itself.
		if c.RMS && (b.attnNormB != nil || b.ffnNormB != nil) {
			return nil, fmt.Errorf("model: OpenTower: block %d's RMSNorms carry biases", i)
		}
		// A gated MLP's gate is the second expansion beside fc1.
		if c0.Has(jlm.RoleVFFNGate, bi, -1) {
			if b.gate, err = get(jlm.RoleVFFNGate, bi); err != nil {
				return nil, err
			}
			unpadK(&b.gate, c.NEmbd)
			if b.gate.rows != c.NFFN || b.gate.k != c.NEmbd {
				return nil, fmt.Errorf("model: OpenTower: block %d's MLP gate is %dx%d, want %d rows of %d",
					i, b.gate.rows, b.gate.k, c.NFFN, c.NEmbd)
			}
			if b.gateB, err = optVec(jlm.RoleVFFNGateBias, bi); err != nil {
				return nil, err
			}
		} else if c.RMS {
			return nil, fmt.Errorf("model: OpenTower: block %d of a %s tower has no MLP gate", i, c.Projector)
		}
	}
	t.pending = ls
	c.Gated = ls[0].gate.e != nil
	// An RMSNorm has no shift, before block 0 as in it.
	if c.RMS && t.preLnB != nil {
		return nil, fmt.Errorf("model: OpenTower: the %s tower's pre-norm is an RMSNorm and carries a bias", c.Projector)
	}

	if t.postLnW, err = optVec(jlm.RoleVPostNorm, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if t.postLnB, err = optVec(jlm.RoleVPostNormBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if v.Projector == jlm.ProjResampler {
		if c.CLS || c.Rope || c.Scale != 1 {
			return nil, fmt.Errorf("model: OpenTower: a resampler tower with a class token, " +
				"rotary positions or a pixel shuffle is not one this engine knows")
		}
		if t.rs, err = loadResampler(c, get, vec); err != nil {
			return nil, err
		}
		c.Queries = t.rs.queries
		return t, t.packedWeights()
	}
	if v.Projector == jlm.ProjGLM4V {
		if err := t.loadGLM4V(c0, get, vec, optVec); err != nil {
			return nil, err
		}
		return t, t.packedWeights()
	}
	if v.Projector == jlm.ProjHunyuanVL {
		if err := t.loadHunyuanVL(get, vec, optVec); err != nil {
			return nil, err
		}
		return t, t.packedWeights()
	}
	// Qwen2.5-VL's merger norm (ln_q) is what v.post_ln carries: required,
	// and an RMSNorm. Mistral 3's input norm is the same position (absent on
	// Pixtral-12B).
	if v.Projector == jlm.ProjQwen25VL && t.postLnW == nil || c.RMS && t.postLnB != nil {
		return nil, fmt.Errorf("model: OpenTower: a %s tower's merger norm is an RMSNorm "+
			"(v.post_ln, weight only)", c.Projector)
	}
	if v.Projector == jlm.ProjPixtral {
		if err := t.loadPixtral(c0, get, vec); err != nil {
			return nil, err
		}
		return t, t.packedWeights()
	}
	if v.Projector == jlm.ProjLlama4 {
		if err := t.loadLlama4(get); err != nil {
			return nil, err
		}
		return t, t.packedWeights()
	}
	if v.Projector == jlm.ProjGemma4V {
		if err := t.loadGemma4V(c0, get, vec); err != nil {
			return nil, err
		}
		return t, t.packedWeights()
	}
	if t.projW, err = get(jlm.RoleVProj, jlm.DenseBlock); err != nil {
		return nil, err
	}
	if t.projB, err = optVec(jlm.RoleVProjBias, jlm.DenseBlock); err != nil {
		return nil, err
	}
	// A shuffling projector consumes Scale^2 patches at once, which is why its
	// k is NEmbd*Scale^2 rather than NEmbd; gemma3's pool averages them and
	// reads NEmbd. Checking it here turns a layout misunderstanding into a
	// load error instead of fluent wrong embeddings.
	if want := c.projK(); t.projW.k != want {
		return nil, fmt.Errorf("model: OpenTower: %s projector k is %d, want %d (n_embd %d, scale %d)",
			c.Projector, t.projW.k, want, c.NEmbd, c.Scale)
	}
	// The projector's own norm: gemma3's RMSNorm and InternVL's LayerNorm are
	// required, every other projector has none.
	if v.Projector == jlm.ProjGemma3 || v.Projector == jlm.ProjInternVL ||
		v.Projector == jlm.ProjLFM2 && c0.Has(jlm.RoleVProjNorm, jlm.DenseBlock, -1) {
		if t.projNorm, err = vec(jlm.RoleVProjNorm, jlm.DenseBlock); err != nil {
			return nil, err
		}
		if len(t.projNorm) != c.projK() {
			return nil, fmt.Errorf("model: OpenTower: the projector norm is %d floats, want %d",
				len(t.projNorm), c.projK())
		}
		if v.Projector == jlm.ProjInternVL {
			if t.projNormB, err = vec(jlm.RoleVProjNormBias, jlm.DenseBlock); err != nil {
				return nil, err
			}
		}
		// LFM2-VL's LayerNorm may come without its bias (llama.cpp reads
		// each on its own).
		if v.Projector == jlm.ProjLFM2 {
			if t.projNormB, err = optVec(jlm.RoleVProjNormBias, jlm.DenseBlock); err != nil {
				return nil, err
			}
		}
	} else if v.Projector == jlm.ProjLFM2 && c0.Has(jlm.RoleVProjNormBias, jlm.DenseBlock, -1) {
		return nil, fmt.Errorf("model: OpenTower: lfm2's projector norm has a bias and no weight")
	}
	// The MLP projector is two matrices and a GELU; only the last has to emit
	// ProjDim, since the hidden width may differ.
	if v.Projector == jlm.ProjMLP || v.Projector == jlm.ProjQwen2VL || v.Projector == jlm.ProjQwen25VL ||
		v.Projector == jlm.ProjInternVL || v.Projector == jlm.ProjJanus || v.Projector == jlm.ProjPhi4 ||
		v.Projector == jlm.ProjQwen3VL || v.Projector == jlm.ProjKimiVL || v.Projector == jlm.ProjLFM2 {
		if t.projW2, err = get(jlm.RoleVProj2, jlm.DenseBlock); err != nil {
			return nil, err
		}
		if t.projB2, err = optVec(jlm.RoleVProj2Bias, jlm.DenseBlock); err != nil {
			return nil, err
		}
		if t.projW2.k != t.projW.rows {
			return nil, fmt.Errorf("model: OpenTower: projector mm.0 emits %d and mm.2 reads %d",
				t.projW.rows, t.projW2.k)
		}
		if t.projW2.rows != c.ProjDim {
			return nil, fmt.Errorf("model: OpenTower: the projector emits %d, want %d",
				t.projW2.rows, c.ProjDim)
		}
	} else if t.projW.rows != c.ProjDim {
		return nil, fmt.Errorf("model: OpenTower: projector emits %d, want projection_dim %d", t.projW.rows, c.ProjDim)
	}
	if v.Projector == jlm.ProjKimiVL {
		if err := t.loadKimiVL(vec); err != nil {
			return nil, err
		}
	}
	if v.Projector == jlm.ProjQwen3VL {
		if t.postLnW == nil || t.postLnB == nil {
			return nil, fmt.Errorf("model: OpenTower: a qwen3vl_merger tower needs its merger's LayerNorm (v.post_ln)")
		}
		if err := t.loadQwen3VL(c0, getAt, vecAt); err != nil {
			return nil, err
		}
	}

	return t, t.packedWeights()
}

// unpadK reads a packed matrix at width want when the container padded it
// to the next whole quantization block (convert.quantizeTower): its k becomes
// want, and kp the packed width every product pads its rows to. A matrix at
// any other width is left for the shape checks to name.
func unpadK(t *tensor, want int) {
	const towerBlock = 32 // Q8_0's, the type convert.quantizeTower writes
	if t.packed == nil || want%towerBlock == 0 || t.k == want {
		return
	}
	if kp := (want + towerBlock - 1) / towerBlock * towerBlock; t.k == kp {
		t.k, t.kp = want, kp
	}
}

// packedWeights refuses a tower with an unpacked matrix: the runner drives a
// tower's packed weights only, so one is named here rather than crashing inside a
// kernel dispatch.
func (t *Tower) packedWeights() error {
	ws := t.weights()
	for i := range t.pending {
		ws = append(ws, vmats(&t.pending[i])...)
	}
	for _, w := range ws {
		// A block holds the matrices its kind has: a MobileNet block one
		// pair or the four projections.
		if w.e == nil && t.mn != nil {
			continue
		}
		if w.packed == nil {
			return fmt.Errorf("model: OpenTower: a tower matrix is %v, which is not a "+
				"packed container type; the engine reads one weight layout and the "+
				"converter is what produces it (re-run jitllm convert)", w.typ)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// The forward pass: a State over the vision segment.
//
// An image is a sequence of patch rows through the model's own runner. The
// State that encodes it is a State like any other -- the same block body
// (hostRows), the same JIT, the same KV pages, the same device placement --
// over the vision blocks and with the segment's configuration. What is the
// tower's own is the entry (patches to rows) and the head (rows to the text
// model's embeddings), and the mask: an image's rows attend to every row of
// their piece, or on a windowed block (Qwen2.5-VL) to every row of their
// window, rather than causally.

// visRun is what an encode needs on a State beyond its blocks' scratch: the
// picture's grid and window order, the entry's and the head's buffers, the
// bidirectional attention's per-head scratch, and the tracer hooks.
type visRun struct {
	t *Tower
	// gh and gw are the image's patch grid, which Preprocess sets for a
	// dynamic-resolution tower (qwenvit.go); zero is the square ImageSz grid.
	gh, gw int
	// ord, for a tower that runs its patches in window order, is the raster
	// patch row r holds, and segs the windows' row ranges as [lo, hi) pairs;
	// both nil for a raster tower with no windows (windowOrder).
	ord  []int
	segs []int
	// ropeAt is the grid ropeSC was built for; ropeSC is every row's
	// {cos, sin} row of the 2-D rotary (visRope).
	ropeAt [2]int
	ropeSC []float32
	// The attention's scratch, per head and per query tile so the head loop
	// runs on the pool: scores, one padded row each, and weighted sums.
	att, acc32 []float32
	attStride  int
	tiled      *nn.AttnTiledSet
	// kp and vp are the one K/V pair every vision block writes in turn. A
	// block's keys are read only inside that block, so each block borrows the
	// pair for its run and hands it on (lend, unlend).
	kp, vp []float32
	gather []float32
	// px2 is the encode in progress's second frame (a video's temporal pair,
	// Picture.Pixels2), nil for a still picture; gather2 and pair are its
	// gathered difference from the first frame and that difference's rows.
	px2           []float32
	gather2, pair []float32
	// posTab is Phi-4's position table resized to the grid posAt
	// (phi4Table), nil for every other tower.
	posTab []float32
	posAt  [2]int
	// posRows is the position-table row each patch row reads, and whole the
	// one key run [0, n) of a block that attends over the whole picture:
	// both kept here so an encode allocates nothing.
	posRows []int32
	whole   [2]int
	// winRuns is segs as the device takes them (nn.KeyRunDevice).
	winRuns []nn.KeyRun
	out     []float32 // [Tokens][ProjDim]
	grp     []float32 // [Tokens][NEmbd*Scale^2] the pixel-shuffled rows
	// mrg is Mistral 3's merged rows, [Tokens][NEmbd], nil for every other
	// projector.
	mrg []float32
	// hid is the MLP projector's middle, [Tokens][projW.rows], nil for
	// idefics3. It is ffnPad'd so the GELU between the matrices can write its
	// ragged tail into slack.
	hid []float32
	// rs is the resampler's working memory, nil for every other projector.
	rs *rsState
	// mn is the MobileNet-V5 tower's working memory (mobilenetrun.go), nil
	// for every other projector.
	mn *mnRun
	// borrowed says the JIT is a text State's (State.Vision), so
	// closing this State leaves it.
	borrowed bool
	// afterBlock, when set, receives the residual after each block, by its
	// index in the segment (-1 is the entry's rows): the tower's tracer, for
	// localising a device/host disagreement per block.
	afterBlock func(li int, x []float32)
	// chunkX is a row-chunked tower call's phase-1 scratch (towerLayers).
	chunkX []float32
	// atStage, when set, sees the rows at the named points llama.cpp's graph
	// dump names too: "pos_embed" (block 0's input), "post_ln", "proj_in"
	// (the projector's first matrix's input) and "out". The tracer's other
	// half, for holding the tower to llama-mtmd-debug node by node.
	atStage func(stage string, x []float32)
	// beforeProj, when set, receives the tower's output -- the rows the
	// projector reads, after the post-LayerNorm -- for the gate that holds it
	// to the reference's last_hidden_state.
	beforeProj func(x []float32)
	// fault breaks one step on purpose, for the gates that must be seen to
	// fail (RULE 10). Zero in every real run.
	fault towerFault
	// hostBlocks and devBlocks count the block runs this State has made on
	// each side, which is what a gate reads to know an encode went through
	// the runner and the placement rather than around them.
	hostBlocks, devBlocks int64
	// cacheHits counts the encodes the image cache answered (encodeNamed).
	cacheHits int64
	// prog is the encode in progress, between steps (encodeBlocks).
	prog encodeProg
	// lerpI and lerpW are a resampled position table's rows and weights for
	// the grid and tap count lerpAt, a plane per tap in raster order
	// (posLerp); axI and axW the two axes' they cross; lerpTab the rows they
	// make.
	lerpI   []int32
	lerpW   []float32
	lerpAt  [3]int
	axI     []int32
	axW     []float32
	lerpTab []float32
	// glmGate and glmUp are GLM-4.xV's projector MLP's two halves.
	glmGate, glmUp []float32
	// hyLines is HunyuanVL's merged rows with their newlines (projectHunyuan).
	hyLines []float32
}

// stage reports rows to atStage when one is set.
func (v *visRun) stage(name string, x []float32) {
	if v.atStage != nil {
		v.atStage(name, x)
	}
}

// release drops the State's hold on the transient K/V pair.
func (v *visRun) release() { v.kp, v.vp = nil, nil }

// towerQT is the query tile the bidirectional attention asks for. Eight costs
// qt+2 of AVX2's 16 vector registers in the score kernel and divides the pass
// over K by eight; the accumulate is capped at two by its own register budget
// either way.
const towerQT = 8

// newState is a State over the vision segment, for encoding pictures, on the
// JIT j: a text State's (State.Vision), so a model has one JIT and one worker
// pool however many segments a session runs. owned says the State closes j
// with itself. Its blocks are offered to a device through its own SetDevice,
// which is State.SetDevice: the one placement path.
func (t *Tower) newState(j *nn.JIT, owned bool) *State {
	m, c, tc := t.model, t.cfg, t.Cfg
	n := tc.MaxSeq()
	// relocate: a picture the device cannot hold takes tower blocks home
	// (reserveTowerRows), as a history that does not fit takes text blocks.
	s := &State{m: m, c: c, lo: t.base, hi: t.base + tc.NLayer, gpuLayers: t.base,
		maxSeq: n, reqSeq: n, nseq: 1, prof: m.opt.profile, outW: &tensor{}, relocate: true}
	s.holdHostRuns()
	v := &visRun{t: t, attStride: nn.SoftmaxPad(n), borrowed: !owned}
	s.vis = v
	s.jit = j
	// The image's rows are one page: K and V are contiguous for every query
	// tile, and the page is the pair every block borrows in turn (lend).
	s.kvl = kvLayout{maxSeq: n, nKV: c.NKVHead, headDim: c.HeadDim}
	s.kv = newKVCacheRange(c, nextCacheID(), 1, s.kvl, nil, align(n, kvKeyTile), s.lo, s.hi)
	qDim := c.NHead * c.HeadDim
	s.attnG = s.jit.AttnSetFor(c.HeadDim, c.HeadDim, s.kvl.Stride(), cpu.KVF32)
	s.attnL = s.attnG
	// The tiled score kernel, usable only by a bidirectional caller: the
	// queries arrive as a block against one K. The accumulate stays at two
	// queries because its accumulator is qt*hd floats and must fit the
	// register file.
	v.tiled = s.jit.AttnTiledFor(c.HeadDim, s.kvl.Stride(), qDim, v.attStride, towerQT, cpu.KVF32)
	// The text attention's per-row score rows are not used: an image's rows
	// attend through visAttend's tiles.
	s.attStride = attStride(1)
	v.att = make([]float32, towerQT*c.NHead*v.attStride)
	v.acc32 = make([]float32, towerQT*c.NHead*c.HeadDim)
	projK, projRows := tc.projK(), t.projW.rows
	if t.rs != nil {
		projK, projRows = c.NEmbd, c.NEmbd
	}
	v.out = make([]float32, tc.MaxTokens()*tc.RowWidth())
	v.grp = make([]float32, tc.MaxTokens()*max(projK, tc.ProjDim))
	if t.projW2.e != nil {
		h := tc.MaxTokens() * projRows
		v.hid = make([]float32, h, ffnPad(h))
	}
	if t.rs != nil {
		v.rs = newRsState(t.rs, s.jit, n)
	}
	if t.mergeW.e != nil {
		v.mrg = make([]float32, tc.Tokens()*tc.NEmbd)
	}
	if t.g4 != nil && t.g4.std != nil {
		// The standardisation's transpose (projectGemma4V).
		v.hid = make([]float32, tc.Tokens()*tc.NEmbd)
	}
	if g := t.glm; g != nil {
		v.hid = make([]float32, tc.MaxTokens()*tc.ProjDim)
		v.glmGate = make([]float32, tc.MaxTokens()*g.gate.rows)
		v.glmUp = make([]float32, tc.MaxTokens()*g.gate.rows)
	}
	if h := t.hy; h != nil {
		v.hid = make([]float32, tc.Tokens()*h.mergeW)
		v.hyLines = make([]float32, tc.MaxTokens()*h.lineWidth)
	}
	s.memBudget = sched.MemBudget()
	return s
}

// lend hands the transient K/V pair to block li for its run: the page every
// vision block writes and reads in turn, allocated once per State.
func (s *State) lend(li int) {
	v, pg := s.vis, &s.kv.layers[li]
	if pg.p == 0 {
		return
	}
	if v.kp == nil {
		v.kp, v.vp = s.m.kvPool.get(pg.pp), s.m.kvPool.get(pg.pp)
	}
	pg.extend(0)
	pg.k[0], pg.v[0] = v.kp, v.vp
}

// unlend takes the pair back from block li once its attention has run.
func (s *State) unlend(li int) {
	if pg := &s.kv.layers[li]; len(pg.k) > 0 {
		pg.k[0], pg.v[0] = nil, nil
		pg.written = 0
	}
}

// visWindowed reports whether vision block li attends inside windows: every
// block but each WinPattern'th (llama.cpp: full when (il+1) % n_wa_pattern ==
// 0), counted from the segment's first block.
func (s *State) visWindowed(li int) bool {
	p := s.vis.t.Cfg.WinPattern
	return p > 0 && (li-s.lo+1)%p != 0
}

// Encode turns a preprocessed image into the embeddings the language model
// splices in with PrefillMixed.
//
// px is the image's pixels, already resized and normalised, laid out
// [y][x][channel] -- the order Patchify produces: ImageSz x ImageSz for a
// fixed-size tower, the grid Preprocess chose for a dynamic one (Grid).
//
// A tiling tower (InternVL) takes several ImageSz squares back to back, as
// Preprocess produces them, and returns their embeddings in the same order;
// each square is its own pass through the blocks, since attention never
// crosses a tile.
func (s *State) Encode(px []float32) ([]float32, error) {
	if s.vis == nil {
		return nil, fmt.Errorf("model: Encode on a State that is not the vision segment's (State.Vision)")
	}
	defer s.m.enterPager()()
	return s.encode(px)
}

// encode is Encode for a caller that holds the pager (enterPager).
func (s *State) encode(px []float32) ([]float32, error) {
	c := s.vis.t.Cfg
	if c.Dynamic() {
		gh, gw := s.patchGrid()
		return s.encodeGrid(px, gh, gw)
	}
	one := c.ImageSz * c.ImageSz * 3
	if len(px) == one || c.MaxTiles == 0 {
		return s.encodeOne(px)
	}
	if len(px)%one != 0 || len(px)/one > c.MaxTiles+1 {
		return nil, fmt.Errorf("model: Encode: %d pixels is not 1..%d squares of %d", len(px), c.MaxTiles+1, one)
	}
	n := len(px) / one
	out := make([]float32, 0, n*c.Tokens()*c.ProjDim)
	for i := 0; i < n; i++ {
		e, err := s.encodeOne(px[i*one : (i+1)*one])
		if err != nil {
			return nil, err
		}
		out = append(out, e...)
	}
	return out, nil
}

// encodeOne is Encode for one square.
func (s *State) encodeOne(px []float32) ([]float32, error) {
	side := s.vis.t.Cfg.ImageSz / s.vis.t.Cfg.PatchSz
	return s.encodeGrid(px, side, side)
}

// EncodeGrid is Encode for a picture of gh x gw patches: px is
// (gh*PatchSz) x (gw*PatchSz) pixels, [y][x][channel]. Only a tower whose
// position table is read by bucket (TowerConfig.PosBuckets) or whose grid
// follows the picture takes a grid other than its square one.
func (s *State) EncodeGrid(px []float32, gh, gw int) ([]float32, error) {
	defer s.m.enterPager()()
	return s.encodeGrid(px, gh, gw)
}

// encodeGrid is EncodeGrid for a caller that holds the pager.
func (s *State) encodeGrid(px []float32, gh, gw int) ([]float32, error) {
	if err := s.encodeBegin(px, gh, gw); err != nil {
		return nil, err
	}
	if _, err := s.encodeBlocks(0); err != nil {
		s.vis.prog = encodeProg{}
		return nil, err
	}
	return s.encodeFinish()
}

// encodeBegin is an encode's first part: the picture's rows, out of the
// entry (the patch embedding, the class row, the positions, the pre-norm)
// into s.bx, and the encode in progress set to its first block.
func (s *State) encodeBegin(px []float32, gh, gw int) error {
	if s.vis == nil {
		return fmt.Errorf("model: EncodeGrid on a State that is not the vision segment's")
	}
	v := s.vis
	t, c := v.t, v.t.Cfg
	E := c.NEmbd
	side := c.ImageSz / c.PatchSz
	if c.PosBuckets == 0 && !c.Dynamic() && (gh != side || gw != side) {
		return fmt.Errorf("model: Encode: a %dx%d grid; this tower reads only its %dx%d one",
			gh, gw, side, side)
	}
	if gh < 1 || gw < 1 || gh*gw > c.MaxSeq() {
		return fmt.Errorf("model: Encode: a %dx%d grid; this tower runs at most %d patches",
			gh, gw, c.MaxSeq())
	}
	if c.Kind == jlm.ProjGemma3nV {
		return s.mnBegin(px)
	}
	// The grid is the state's for this encode: the rotary table, the window
	// partition and the merger read it (patchGrid).
	v.gh, v.gw = gh, gw
	np := gh * gw
	n := np
	// first is where the patch rows start in the residual: 1 when a class
	// token occupies row 0, 0 otherwise.
	//
	// Row 0 and position row 0 belong to the class token, which is HF's
	// layout (modeling_clip.py). llama.cpp's llava graph concatenates
	// [patches, class] and so misassigns position rows; jitllm follows the
	// weights' training layout, a deliberate divergence (RULE 7m).
	// cls is the class token's row: 0 in the reference's layout, np in
	// llama.cpp's (towerCLSLast).
	first, cls := 0, -1
	if c.CLS {
		first, cls = 1, 0
		if t.opt.towerCLSLast || c.CLSLast {
			first, cls = 0, np
		}
		n++
	}
	if n > c.MaxSeq() {
		return fmt.Errorf("model: Encode: a %dx%d patch grid is %d rows, and this state holds %d",
			gh, gw, n, c.MaxSeq())
	}
	ph, pw := gh*c.PatchSz, gw*c.PatchSz
	if len(px) != ph*pw*3 {
		return fmt.Errorf("model: Encode: %d pixels, want %d (%dx%dx3)", len(px), ph*pw*3, ph, pw)
	}
	s.windowOrder(gh, gw)
	s.growVision(n)

	// 1. Patch embedding. A conv2d whose stride equals its kernel is a matmul
	// over non-overlapping patches, so the gather is a permutation and the
	// arithmetic is the GEMM path -- no convolution kernel anywhere.
	// The gather row is the weight's k, not the patch's: the converter pads k
	// to a whole quantization block, and the slack must stay zero.
	pk := t.patchW.k
	// The picture's patches, not the tower's largest grid.
	if len(v.gather) < np*pk {
		v.gather = make([]float32, np*pk)
	}
	gather := v.gather[:np*pk]
	// The flattened patch is channel-planar, not pixel-interleaved: the conv
	// kernel is [kx, ky, channel, NEmbd], so one output row reads element
	// kx + ky*PatchSz + ch*PatchSz^2, channel slowest. px is [y][x][channel],
	// so a contiguous copy would be a silent permutation: each channel of a
	// patch is a strided gather, PatchSz rows of PatchSz samples three apart
	// (nn.Copy32JIT). Permuting the weight's columns at conversion instead
	// would regroup a quantized patch weight's blocks and change its values.
	P := c.PatchSz
	for r := 0; r < np; r++ {
		p := s.patchAt(r)
		py, pxi := p/gw, p%gw
		at := (py*P*pw + pxi*P) * 3
		for ch := 0; ch < 3; ch++ {
			nn.Copy32JIT(gather[r*pk+ch*P*P:r*pk+(ch+1)*P*P], P, 1, px[at+ch:], 3*pw, 3, P, P)
		}
	}
	x := s.bx[:n*E]
	s.jit.NewInput()
	if err := s.mm(x[first*E:], t.patchW, gather, np); err != nil {
		return err
	}
	// A video's temporal pair: the planes' sum read the first frame, and the
	// second plane reads the second frame's difference from it, so the row is
	// x0*W0 + x1*W1 (convert.foldPatchPlanes).
	if v.px2 != nil {
		if err := s.patchPair(x[first*E:first*E+np*E], gather, pw, np, pk); err != nil {
			return err
		}
	}
	// 1b. The class token, if there is one: one learned embedding in row 0,
	// before the position table is added, so it gets position row 0.
	if c.CLS {
		copy(x[cls*E:(cls+1)*E], t.clsW)
	}
	// The patch bias, if any, belongs to the patch rows only: it is part of the
	// convolution, which never produced the class row.
	s.visBiasRows(x[first*E:], t.patchB, np)
	// The position embedding is a different row per position, so it is its
	// own pass. A bucketed table is read at each patch's bucket; every other
	// table at the patch's own row -- but a table resampled to the grid
	// (PosSide) is posLerp's, below.
	if t.posW != nil && c.PosSide == 0 {
		// Row p reads table row p, or its bucket's: the index is setup, once
		// per grid, and the add is one region over the rows.
		if cap(v.posRows) < n {
			v.posRows = make([]int32, n)
		}
		rows := v.posRows[:n]
		table := t.posW
		if c.Kind == jlm.ProjPhi4 || c.Kind == jlm.ProjLFM2 {
			table = s.phi4Table(gh, gw)
		}
		if c.PosAxis > 0 {
			// Gemma 4: the column's row of the first table, then the row's
			// row of the second, two passes over the rows.
			for p := range rows {
				rows[p] = int32(s.patchAt(p) % gw)
			}
			s.axpyRows(x, t.posW, rows, E, n, s.rowChunk(n, E), 1)
			for p := range rows {
				rows[p] = int32(c.PosAxis + s.patchAt(p)/gw)
			}
		} else if c.PosBuckets > 0 && v.fault != faultNoBuckets {
			bucketPositions(rows, gh, gw, c.PosBuckets)
		} else {
			for p := range rows {
				rows[p] = int32(p)
			}
		}
		s.axpyRows(x, table, rows, E, n, s.rowChunk(n, E), 1)
	}
	// GLM-4.xV's RMSNorm over the patch embedding, before its table.
	if t.embdNorm != nil && c.fault != faultNoEmbdNorm {
		s.normRows(x, x, t.embdNorm, nil, E, n, false, c.Eps, s.rowChunk(n, E))
	}
	// A table resampled to the grid (Qwen3-VL, GLM-4.xV), on top of the rotary.
	if c.PosSide > 0 {
		s.posLerp(x, gh, gw, n)
	}
	// 1c. The pre-loop LayerNorm, over the whole sequence, after the position
	// embedding and before block 0, as both modeling_clip.py and llama.cpp's
	// llava graph order it.
	if t.preLnW != nil {
		s.normRows(x, x, t.preLnW, t.preLnB, E, n, !c.RMS, c.Eps, s.rowChunk(n, E))
	}
	v.stage("pos_embed", x)
	// Block -1 is the embedding: the patches with their positions, before
	// block 0 (llama.cpp's pos_embed node, the tracer's first stop).
	if v.afterBlock != nil {
		v.afterBlock(-1, x)
	}
	// The 2-D rotary rows every block turns q and k by, on either tier.
	if c.Rope {
		copy(s.bcs[:n*c.HeadDim], s.visRope(n))
	}
	v.prog = encodeProg{on: true, n: n, first: first, next: s.lo, gh: gh, gw: gw}
	return nil
}

// encodeProg is an encode in progress: the entry has run, its rows are in
// s.bx, and next is the first block still to run. Between two steps it is
// all that holds the picture -- an image's rows attend to each other, so an
// encode is split by blocks, never by rows.
type encodeProg struct {
	on                     bool
	n, first, next, gh, gw int
}

// encodeBlocks runs up to max of the blocks the encode in progress has left,
// all of them at max <= 0, and reports whether none is left. Each run of
// device blocks is one submission (attention is bidirectional, so the whole
// image is one call), the rest run on the host through the runner's block
// body, the residual crossing once per seam. Where a block runs is decided
// when it runs: a block the placement moved between two calls runs where it
// went.
func (s *State) encodeBlocks(max int) (bool, error) {
	v := s.vis
	p, c := &v.prog, v.t.Cfg
	if !p.on {
		return false, fmt.Errorf("model: no encode in progress")
	}
	x := s.bx[:p.n*c.NEmbd]
	var cs []float32
	if c.Rope {
		cs = s.bcs[:p.n*c.HeadDim]
	}
	// The device's tower blocks are sized for this picture before any of
	// them runs, or come home.
	s.reserveTowerRows(p.n)
	ran := 0
	for p.next < s.hi && (max <= 0 || ran < max) {
		li := p.next
		if c.Kind == jlm.ProjGemma3nV {
			// A MobileNet block's input and output are the encode's own
			// buffers, each its own shape (mobilenetrun.go).
			hi := li + 1
			if s.devAt(li) {
				for hi < s.hi && s.devAt(hi) && (max <= 0 || ran+hi-li < max) {
					hi++
				}
				if err := s.mnDevRun(li, hi); err != nil {
					return false, err
				}
				v.devBlocks += int64(hi - li)
			} else {
				if err := s.mnBlockRun(li); err != nil {
					return false, err
				}
				v.hostBlocks++
			}
			ran += hi - li
			p.next = hi
			if v.afterBlock != nil {
				v.afterBlock(hi-1-s.lo, v.mn.act[v.mn.cur][:v.mn.h*v.mn.w*v.mn.c])
			}
			continue
		}
		if s.devAt(li) {
			hi := li + 1
			// A deepstack tap reads the residual after its block, on the host.
			stop := s.hi
			if tap := c.tapAfter(li - s.lo); tap >= 0 {
				stop = s.lo + tap + 1
			}
			for hi < stop && s.devAt(hi) && (max <= 0 || ran+hi-li < max) {
				hi++
			}
			// The picture's windows, for its windowed blocks (key runs; a
			// tower with none hands over none).
			if c.WinPattern > 0 {
				v.winRuns = v.winRuns[:0]
				for w := 0; w+1 < len(v.segs); w += 2 {
					v.winRuns = append(v.winRuns, nn.KeyRun{Lo: v.segs[w], Hi: v.segs[w+1]})
				}
				kd, ok := s.ld.(nn.KeyRunDevice)
				if !ok || !kd.SetKeyRuns(nil, v.winRuns) {
					return false, fmt.Errorf("model: the device refused the tower's %d window(s)", len(v.winRuns))
				}
			}
			// The image's rows and no more: the scratch may hold more rows
			// than the picture (reserveTowerRows rounds them up), and a row
			// past the image is a key every row would attend to.
			if !s.towerLayers(li, hi, p.n, x, cs) {
				// A device that fails mid-image has left the residual in an
				// unknown state, so this is an error rather than a fallback.
				why := "no reason given"
				if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
					why = e.Err()
				}
				return false, fmt.Errorf("model: the device failed tower block(s) [%d,%d): %s",
					li-s.lo, hi-s.lo, why)
			}
			v.devBlocks += int64(hi - li)
			ran += hi - li
			p.next = hi
		} else {
			s.lend(li)
			err := s.hostRows(li, 0, 0, 0, p.n, p.n, false)
			s.unlend(li)
			if err != nil {
				return false, err
			}
			v.hostBlocks++
			ran++
			p.next = li + 1
		}
		if v.afterBlock != nil {
			v.afterBlock(p.next-1-s.lo, x)
		}
		for k, b := range c.Deep {
			if b == p.next-1-s.lo {
				if err := s.deepstackAt(k, x); err != nil {
					return false, err
				}
			}
		}
	}
	return p.next >= s.hi, nil
}

// reserveTowerRows sizes the device's tower blocks for a picture of n rows
// (nn.RowsReserver) before any of them runs: the device builds them for the
// picture, not the tower's largest grid. Where the budget cannot hold that,
// tower blocks go home, the highest first, until it can (relocateUntil) -- a
// picture too large for what is left of the card encodes on the host rather
// than out of memory -- and come back through reclaim when the device
// reports room.
func (s *State) reserveTowerRows(n int) {
	rr, ok := s.ld.(nn.RowsReserver)
	if !ok || s.devCount() == 0 {
		return
	}
	s.relocateUntil(0, func() bool { return rr.ReserveRows(n) })
}

// encodeFinish runs the head over the encode whose blocks have all run: the
// post-LayerNorm and the projector. The rows are this State's own buffer,
// which the next encode overwrites.
func (s *State) encodeFinish() ([]float32, error) {
	v := s.vis
	p, t, c := v.prog, v.t, v.t.Cfg
	v.prog = encodeProg{}
	if !p.on || p.next < s.hi {
		return nil, fmt.Errorf("model: the encode's blocks have not all run")
	}
	if c.Kind == jlm.ProjGemma3nV {
		return s.mnFinish()
	}
	E, n := c.NEmbd, p.n
	x := s.bx[:n*E]
	if t.postLnW != nil {
		s.visNorm(x, x, t.postLnW, t.postLnB, n)
	}
	v.stage("post_ln", x)

	// The class row is dropped here and nowhere else: it carries information
	// through the blocks but is not a patch, so it is not projected.
	patches := x[p.first*E : n*E]
	if v.beforeProj != nil {
		v.beforeProj(patches)
	}
	if t.rs != nil {
		if err := s.resample(v.out, patches, p.gh, p.gw); err != nil {
			return nil, err
		}
		v.stage("out", v.out)
		return v.out, nil
	}
	if t.glm != nil {
		return s.projectGLM(patches)
	}
	if t.hy != nil {
		return s.projectHunyuan(patches)
	}
	out, err := s.project(patches)
	if err != nil || len(c.Deep) == 0 {
		return out, err
	}
	// The merged rows, then each tap's: deepstackAt wrote them behind.
	return v.out[:len(out)*(1+len(c.Deep))], nil
}

// growVision sizes the State's block scratch for n rows of the segment. The
// text State's spare buffers are another shape, so none are taken.
func (s *State) growVision(n int) {
	c := s.c
	if len(s.bx) < n*c.NEmbd {
		s.bx = make([]float32, n*c.NEmbd)
		s.bcs = make([]float32, n*max(c.NRot, 1))
	}
	s.growBatch(n)
}

// visPrep is the prologue (runPrep's steps) over a picture's n rows of block li, on the pool: each
// row's q and k turned by its rotary row where the tower has one, and its k
// and v written to the page the block was lent (lend), which holds every row
// of the picture -- so no row grows the page or faults it, and the rows are
// independent. A tower block has no clamp, q/k norm or v norm, which is all
// runPrep would add.
func (s *State) visPrep(li, n int) {
	pg := &s.kv.layers[li]
	pg.grow(0)
	j := &s.rg.rows
	j.op, j.li = rowPrep, li
	s.rowRun(n, s.rowChunk(n, s.c.QDimAt(li)))
	pg.written = max(pg.written, n)
}

// visPrepRows is rows lo..hi-1 of visPrep.
func (s *State) visPrepRows(lo, hi int) {
	c, li := s.c, s.rg.rows.li
	hd, qDim, kvDim := c.HeadDimAt(li), c.QDimAt(li), c.KVDimAt(li)
	pg := &s.kv.layers[li]
	pl := pg.layout(s.kvlAt(li))
	rope := c.RopeAt(li)
	l := &s.m.layers[li]
	for i := lo; i < hi; i++ {
		q, k, v := s.bq[i*qDim:(i+1)*qDim], s.bk[i*kvDim:(i+1)*kvDim], s.bv[i*kvDim:(i+1)*kvDim]
		// Gemma 4's per-head q and k norms before the rotary, and its
		// weightless v norm; no other tower has any.
		if l.qNorm != nil {
			for h := 0; h < c.NHead; h++ {
				nn.RMSNorm32JIT(q[h*hd:(h+1)*hd], q[h*hd:(h+1)*hd], l.qNorm, c.RMSEps)
			}
			for h := 0; h < c.NKVHeadAt(li); h++ {
				nn.RMSNorm32JIT(k[h*hd:(h+1)*hd], k[h*hd:(h+1)*hd], l.kNorm, c.RMSEps)
			}
		}
		if c.VNorm {
			ones := s.vis.t.g4.ones[:hd]
			for h := 0; h < c.NKVHeadAt(li); h++ {
				nn.RMSNorm32JIT(v[h*hd:(h+1)*hd], v[h*hd:(h+1)*hd], ones, c.RMSEps)
			}
		}
		if rope {
			cs := s.ropeTable(li, s.bcs, s.bcsSWA, i)
			nn.RoPE32JIT(q[:c.NHead*hd], hd, cs, c.RopeNeox)
			nn.RoPE32JIT(k[:c.NKVHeadAt(li)*hd], hd, cs, c.RopeNeox)
		}
		pl.Write(pg.k[0], 0, i, k)
		pl.Write(pg.v[0], 0, i, v)
	}
}

// visAttend is a vision block's attention: every row over the keys of its
// run -- the whole image, or on a windowed block its window -- from the K/V
// page the block wrote, into s.bxb. On the pool over heads: each head writes
// a disjoint HeadDim slice and has its own scratch.
func (s *State) visAttend(li, n int) {
	v := s.vis
	v.whole = [2]int{0, n}
	j := &s.rg.rows
	j.op, j.li, j.segs, j.alpha = rowHeads, li, v.whole[:], float32(s.c.AttnScale)
	if s.visWindowed(li) {
		j.segs = v.segs
	}
	s.rowRun(s.c.NHead, 1)
}

// visAttendHead is one head's attention for the queries [w0, w1) over the
// keys [w0, w1).
func (s *State) visAttendHead(li, h, w0, w1 int, scale float32) {
	c, v := s.c, s.vis
	m := w1 - w0
	qd := c.NHead * c.HeadDim
	off := h * c.HeadDim
	pg := &s.kv.layers[li]
	sp := kvSpan{page: 0, first: w0, n: m}
	kb, vb := pg.kSpan(s.kvl, sp, 0, h), pg.vSpan(s.kvl, sp, 0, h)
	b0 := towerQT * h * v.attStride
	row := func(j int) []float32 { return v.att[b0+j*v.attStride : b0+j*v.attStride+m] }
	acc := func(j int) []float32 {
		o := (towerQT*h + j) * c.HeadDim
		return v.acc32[o : o+c.HeadDim]
	}
	att0, att1 := row(0), row(1)
	a0, a1 := acc(0), acc(1)
	store := func(i int, out []float32) {
		copy(s.bxb[i*qd+off:i*qd+off+c.HeadDim], out)
	}
	norm := func(att []float32) {
		s.scale(att, scale)
		nn.Softmax32JIT(att, m)
	}
	q := s.bq
	i := w0
	// One pass over K for towerQT queries: the one-query kernel re-reads all
	// of K per query and is memory-bound. The accumulate still goes two at a
	// time (see nn.AttnTiledShape).
	if qt := v.tiled.Qt(); qt >= 2 {
		for ; i+qt <= w1; i += qt {
			if !v.tiled.Scores(row(0), kb, q[i*qd+off:], m) {
				for j := 0; j < qt; j++ {
					s.attnG.AttnScores(row(j), kb, q[(i+j)*qd+off:], m)
				}
			}
			for j := 0; j < qt; j++ {
				norm(row(j))
			}
			for j := 0; j+1 < qt; j += 2 {
				if !s.attnG.AttnAcc2(acc(j), acc(j+1), vb, row(j), row(j+1), m) {
					s.attnG.AttnAcc(acc(j), vb, row(j), m)
					s.attnG.AttnAcc(acc(j+1), vb, row(j+1), m)
				}
			}
			for j := 0; j < qt; j++ {
				store(i+j, acc(j))
			}
		}
	}
	// The remainder two queries per pass over K, which halves the reads.
	for ; i+1 < w1; i += 2 {
		q0, q1 := q[i*qd+off:], q[(i+1)*qd+off:]
		if !s.attnG.AttnScores2(att0, att1, kb, q0, q1, m) {
			s.attnG.AttnScores(att0, kb, q0, m)
			s.attnG.AttnScores(att1, kb, q1, m)
		}
		norm(att0)
		norm(att1)
		if !s.attnG.AttnAcc2(a0, a1, vb, att0, att1, m) {
			s.attnG.AttnAcc(a0, vb, att0, m)
			s.attnG.AttnAcc(a1, vb, att1, m)
		}
		store(i, a0)
		store(i+1, a1)
	}
	for ; i < w1; i++ { // odd tail
		qi := q[i*qd+off:]
		s.attnG.AttnScores(att0, kb, qi, m)
		norm(att0)
		s.attnG.AttnAcc(a0, vb, att0, m)
		store(i, a0)
	}
}

// project runs the projector over one picture's patch rows, [Patches][NEmbd]
// with the class row already dropped. It is the end of EncodeGrid and its own
// method so a gate can hold it to the reference on inputs of its choosing.
func (s *State) project(patches []float32) ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	switch c.Kind {
	case jlm.ProjPixtral:
		return s.projectPixtral(patches)
	case jlm.ProjLlama4:
		return s.projectLlama4(patches)
	case jlm.ProjGemma4V:
		return s.projectGemma4V(patches)
	}
	side := c.ImageSz / c.PatchSz
	gh, gw := s.patchGrid()
	nt := gh * gw / (c.Scale * c.Scale)

	// The shuffle and the matrix count are independent axes: idefics3
	// shuffles and runs one matrix, llava runs two and does not shuffle,
	// qwen2vl_merger does both. Scale decides the shuffle, projW2 the matrices.
	//
	// The grouping is spatial: group (gy, gx) collects the Scale x Scale block
	// at (gy*Scale + dy, gx*Scale + dx) of the row-major patch grid. A tower in
	// window order already holds each group in Scale^2 consecutive rows, which
	// is a grouping that needs no copy at all.
	src := patches
	// Kimi-VL's LayerNorm over each patch, before the shuffle.
	if c.Kind == jlm.ProjKimiVL && c.fault != faultKimiNoPreNorm {
		s.normRows(patches, patches, t.projNorm, t.projNormB, c.NEmbd, gh*gw, true, 1e-5,
			s.rowChunk(gh*gw, c.NEmbd))
	}
	switch {
	case c.Kind == jlm.ProjGemma3:
		// gemma3: average-pool each Scale x Scale block of the patch grid into
		// one row -- the sum, then one multiply by 1/Scale^2 (exact: a power of
		// two), which is ggml's pool_2d order -- then the RMSNorm.
		w := c.NEmbd
		r := &s.rg.rows
		r.op, r.src, r.dim, r.side = rowPool, patches, w, side
		s.rowRun(c.Tokens(), s.rowChunk(c.Tokens(), w*c.Scale*c.Scale))
		src = v.grp
	case c.Scale > 1 && v.ord == nil:
		gsh, gsw, w := gh/c.Scale, gw/c.Scale, c.NEmbd*c.Scale*c.Scale
		for gy := 0; gy < gsh; gy++ {
			for gx := 0; gx < gsw; gx++ {
				dst := v.grp[(gy*gsw+gx)*w : (gy*gsw+gx+1)*w]
				for dy := 0; dy < c.Scale; dy++ {
					for dx := 0; dx < c.Scale; dx++ {
						p := (gy*c.Scale+dy)*gw + gx*c.Scale + dx
						slot := dy*c.Scale + dx
						if c.fault == towerFaultShuffleSwap {
							slot = dx*c.Scale + dy
						}
						copy(dst[slot*c.NEmbd:], patches[p*c.NEmbd:(p+1)*c.NEmbd])
					}
				}
			}
		}
		src = v.grp
		// InternVL's LayerNorm over the grouped row, before the MLP, and
		// LFM2-VL's where its file carries one.
		if c.Kind == jlm.ProjInternVL || c.Kind == jlm.ProjLFM2 && t.projNorm != nil && s.tfault() != faultLFM2NoProjNorm {
			w := c.projK()
			s.normRows(v.grp, v.grp, t.projNorm, t.projNormB, w, c.Tokens(), true, c.projEps(),
				s.rowChunk(c.Tokens(), w))
		}
	}
	v.stage("proj_in", src[:nt*c.projK()])
	s.jit.NewInput()
	if t.projW2.e == nil {
		// idefics3: one matmul over the grouped rows.
		if err := s.mm(v.out, t.projW, src, nt); err != nil {
			return nil, err
		}
		s.visBiasRows(v.out, t.projB, nt)
		v.stage("out", v.out[:nt*c.ProjDim])
		return v.out[:nt*c.ProjDim], nil
	}

	// llava's MLP projector and the Qwen mergers: linear, GELU, linear. llava
	// runs it over each patch on its own (scale_factor is absent from the file,
	// so Scale is 1 and Tokens() is Patches()); the merger runs it over the
	// grouped rows.
	//
	// The activation here is GELU-tanh, not the tower's own: llama.cpp uses
	// ggml_gelu for the projector. HF's "gelu" is the erf-exact form; the two
	// differ by <1e-3 and the divergence is deliberate. LFM2-VL's is the erf
	// form, its reference's (Lfm2VlMultiModalProjector, ACT2FN["gelu"]), where
	// llama.cpp's lfm2 graph runs the tanh form (RULE 7m).
	if err := s.mm(v.hid, t.projW, src, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(v.hid, t.projB, nt)
	act := nn.ActGELU
	if c.Kind == jlm.ProjLFM2 && s.tfault() != faultLFM2TanhGELU {
		act = nn.ActGELUErf
	}
	s.actAll(v.hid[:nt*t.projW.rows], act)
	s.jit.NewInput()
	out := v.out
	if v.ord != nil {
		out = v.grp // the window-order rows, put back below
	}
	if err := s.mm(out, t.projW2, v.hid, nt); err != nil {
		return nil, err
	}
	s.visBiasRows(out, t.projB2, nt)
	if v.ord != nil {
		s.unwindow(v.out, out, nt)
	}
	v.stage("out", v.out[:nt*c.ProjDim])
	return v.out[:nt*c.ProjDim], nil
}

// visNorm applies the tower's norm -- a LayerNorm, or an RMSNorm on a tower
// whose blocks carry them (TowerConfig.RMS) -- to each of n rows, on the pool.
func (s *State) visNorm(y, x, w, b []float32, n int) {
	c := s.vis.t.Cfg
	s.normRows(y, x, w, b, c.NEmbd, n, !c.RMS, c.Eps, s.rowChunk(n, c.NEmbd))
}

// visBiasRows adds b to each of n rows of width len(b), the rows on the pool.
func (s *State) visBiasRows(out, b []float32, n int) {
	if b == nil {
		return
	}
	s.axpyRows(out, b, nil, len(b), n, s.rowChunk(n, len(b)), 1)
}

// poolRows is gemma3's pool for the projector's groups lo..hi-1 (rowPool):
// the sum of each Scale x Scale block of patch rows, one multiply by
// 1/Scale^2 (exact: a power of two), which is ggml's pool_2d order, then the
// RMSNorm.
func (s *State) poolRows(lo, hi int) {
	j, v := &s.rg.rows, s.vis
	t, c := v.t, v.t.Cfg
	w, side := j.dim, j.side
	gside := side / c.Scale
	inv := 1 / float32(c.Scale*c.Scale)
	for g := lo; g < hi; g++ {
		gy, gx := g/gside, g%gside
		dst := v.grp[g*w : (g+1)*w]
		clear(dst)
		for dy := 0; dy < c.Scale; dy++ {
			for dx := 0; dx < c.Scale; dx++ {
				p := (gy*c.Scale+dy)*side + gx*c.Scale + dx
				if c.fault == towerFaultPoolCorner && (dy|dx) != 0 {
					continue
				}
				nn.Axpy32JIT(dst, j.src[p*w:(p+1)*w], 1)
			}
		}
		if c.fault != towerFaultPoolCorner {
			nn.Scale32JIT(dst, inv)
		}
		nn.RMSNorm32JIT(dst, dst, t.projNorm, c.projEps())
	}
}

// elemChunk gives each worker a few chunks: enough to absorb a late worker, few
// enough that the claim is not the work.
//
// Four chunks per worker is a heuristic; the constraint is that a chunk's
// execution must dominate the claim, callback and kernel entry.
func elemChunk(items, workers int) int {
	if workers < 1 {
		workers = 1
	}
	if c := items / (4 * workers); c > 1 {
		return c
	}
	return 1
}

// rowChunk is elemChunk for a loop whose unit is a ROW of `width` elements:
// 1024 rows at chunk 1 is 1024 atomic claims around one kernel call each.
func (s *State) rowChunk(rows, width int) int {
	if !s.worthPooling(rows * width) {
		return rows // one chunk: the pool runs it on the caller
	}
	return elemChunk(rows, s.jit.Workers())
}

// worthPooling is the crossover: parallelise only when the serial work exceeds
// the dispatch cost by enough to pay for it, Tserial > D/(1 - 1/S). With a
// region dispatch of ~3 us and a speedup of ~3 that is ~4.5 us of serial work,
// expressed in elements at a conservative streaming rate. It reads the model's
// WithTowerTiling.
func (s *State) worthPooling(elems int) bool {
	if !s.m.opt.elemPool {
		return false
	}
	const elemsPerUs = 2000 // conservative f32 streaming rate, one core
	const crossoverUs = 10  // ~3x the region dispatch, with margin
	return elems >= crossoverUs*elemsPerUs
}

// Regions reports how many elementwise/matmul passes this State dispatched to
// the pool and how many ran inline, which is what prices a dispatch-bound
// encode. See sched.Pool.Regions.
func (s *State) Regions() (parallel, serial int64) { return s.jit.Regions() }

// ResetRegions zeroes the counters so a measurement can exclude the warm-up
// encode, which pays for codegen and first-touch faults that no later one does.
func (s *State) ResetRegions() { s.jit.ResetRegions() }

// GPUBlocks is how many of this State's blocks a device took.
func (s *State) GPUBlocks() int { return s.devCount() }

// towerRope is the ViT's 2-D rotary as a multi-axis table (nn.Rope.Runs): a
// half-head rotary whose HeadDim/4 frequencies turn the first quarter of a
// head's pairs by the patch's grid row and then, from the first frequency
// again, the second quarter by its column, NEOX-paired (i, i+HeadDim/2). HF
// builds VisionRotaryEmbedding with dim = head_dim/2 and uses each frequency
// twice; llama.cpp's ggml_rope_multi in vision mode restarts the frequency
// per section, which is the same table.
func (c TowerConfig) towerRope() nn.Rope {
	switch c.Kind {
	case jlm.ProjPixtral:
		return c.pixtralRope()
	case jlm.ProjLlama4:
		return c.llama4Rope()
	case jlm.ProjGemma4V:
		return c.gemma4Rope()
	case jlm.ProjKimiVL:
		return c.kimiRope()
	}
	quart := c.HeadDim / 4
	return nn.Rope{NRot: c.HeadDim / 2, Base: 10000, Neox: true,
		Runs: []nn.RopeRun{{Pairs: quart, Axis: 0}, {Pairs: quart, Axis: 1}}}
}

// visRope is every row's {cos, sin} row -- HeadDim floats, the row's patch's
// (grid row, grid column) on towerRope's two axes -- built once per grid by
// the generated table kernel. The text block's rotation turns q and k by it
// (visPrep) on the host, and the device by the same rows (Layers' cs).
func (s *State) visRope(n int) []float32 {
	v := s.vis
	c := v.t.Cfg
	gh, gw := s.patchGrid()
	if v.ropeAt == [2]int{gh, gw} && len(v.ropeSC) >= n*c.HeadDim {
		return v.ropeSC[:n*c.HeadDim]
	}
	if cap(v.ropeSC) < n*c.HeadDim {
		v.ropeSC = make([]float32, n*c.HeadDim)
	}
	v.ropeSC = v.ropeSC[:n*c.HeadDim]
	r := c.towerRope()
	for i := 0; i < n; i++ {
		if c.Kind == jlm.ProjLlama4 {
			var pos [2]int
			llama4Coords(s.tfault(), i, gh*gw, gw, pos[:])
			s.jit.RopeTableAt(r, v.ropeSC[i*c.HeadDim:(i+1)*c.HeadDim], pos[:])
			continue
		}
		if c.Kind == jlm.ProjGemma4V {
			// (column, row): the column turns the first half.
			p := s.patchAt(i)
			pos := []int{p % gw, p / gw}
			if s.tfault() == faultGemma4RowFirst {
				pos[0], pos[1] = pos[1], pos[0]
			}
			s.jit.RopeTableAt(r, v.ropeSC[i*c.HeadDim:(i+1)*c.HeadDim], pos)
			continue
		}
		p := s.patchAt(i)
		s.jit.RopeTableAt(r, v.ropeSC[i*c.HeadDim:(i+1)*c.HeadDim], []int{p / gw, p % gw})
	}
	v.ropeAt = [2]int{gh, gw}
	return v.ropeSC
}

// visPlan is vision block li's plan for a device: the text kit's fields read
// off the segment's Config -- LayerNorm or RMSNorm, an ungated MLP or a gated
// one, a per-row rotary (NRot over Layers' cs) or none -- and NonCausal, so a
// call over the block is the picture's rows as one run, its windowed blocks
// inside the windows the call was handed (nn.KeyRunDevice).
func (s *State) visPlan(li int) *nn.LayerPlan {
	c, tc := s.c, s.vis.t.Cfg
	l := &s.m.layers[li]
	plan := &nn.LayerPlan{
		Model:     s.m.id,
		NonCausal: true,
		Windowed:  s.visWindowed(li),
		NEmbd:     c.NEmbd, NHead: c.NHead, NKVHead: c.NKVHead, HeadDim: c.HeadDim,
		NFFN: c.NFFN, MaxSeq: tc.MaxSeq(),
		// The segment's own activation, as the host reads it. A device that
		// cannot emit this kind declines the block; see tier.PrepLayer.
		Act: c.Act,
		// The JIT's window, not a constant: both tiers must quantize the
		// activation the same way (see nn.LayerPlan.ActWin).
		ActWin:     s.jit.ActWindow(),
		RMSEps:     c.RMSEps,
		LayerNorm:  c.LayerNorm,
		UngatedFFN: l.gate.e == nil,
		KPad:       l.wq.kp,
	}
	if s.m.kpadViolation {
		plan.KPad = 0
	}
	// A 2-D rotary tower turns the whole head, NEOX-paired, by the per-patch
	// table EncodeGrid hands Layers (visRope).
	if tc.Rope {
		plan.NRot, plan.RopeNeox = c.HeadDim, c.RopeNeox
	}
	// Gemma 4's block: sandwich norms, the per-head q/k norm, the weightless
	// v norm, scores at the segment's own scale, the clipped linears.
	plan.PostNorm = l.postAttnNorm != nil
	plan.QKNorm, plan.VNorm = l.qNorm != nil, c.VNorm
	plan.AttnScale = c.AttnScale
	plan.Clamps = l.clamp != nil
	if b := l.mn; b != nil {
		plan.Conv = b.plan(tc.Eps, plan.ActWin)
	}
	return plan
}

// visWeights is vision block li's weights for a device.
func (s *State) visWeights(li int) nn.LayerWeights {
	l := &s.m.layers[li]
	w := nn.LayerWeights{
		AttnNorm: l.attnNorm, AttnNormB: l.attnNormB,
		FFNNorm: l.ffnNorm, FFNNormB: l.ffnNormB,
		Bq: l.bq, Bk: l.bk, Bv: l.bv, Bo: l.bo,
		Wq: wt(l.wq), Wk: wt(l.wk), Wv: wt(l.wv), Wo: wt(l.wo),
		Up: wt(l.up), Down: wt(l.down), BUp: l.upB, BDown: l.downB,
	}
	if l.gate.e != nil {
		w.Gate, w.BGate = wt(l.gate), l.gateB
	}
	w.QNorm, w.KNorm = l.qNorm, l.kNorm
	w.PostAttnNorm, w.PostFFNNorm = l.postAttnNorm, l.postFFNNorm
	w.Clamp = l.clamp
	if b := l.mn; b != nil {
		w.Conv = &nn.ConvWeights{Norm1: b.norm1, Norm2: b.norm2, DwStart: b.dwStart, DwStartNorm: b.dwStartNorm,
			DwMid: b.dwMid, DwMidNorm: b.dwMidNorm, KDown: b.kDown, KDownNorm: b.kDownNorm,
			VDown: b.vDown, VDownNorm: b.vDownNorm}
	}
	return w
}

// Vision is the State this session encodes pictures in: a State over the
// model's vision segment, on this State's JIT and pool, its blocks offered to
// this State's device through the one placement path (State.SetDevice). It is
// made on first use and closed with this State.
func (s *State) Vision() (*State, error) {
	if s.vis != nil {
		return s, nil
	}
	if s.m.tower == nil {
		return nil, fmt.Errorf("model: this model has no vision tower")
	}
	if s.visState == nil {
		v := s.m.tower.newState(s.jit, false)
		if s.device != nil {
			if err := v.SetDevice(s.device); err != nil {
				v.Close()
				return nil, err
			}
		}
		s.visState = v
	}
	return s.visState, nil
}

// towerLayers runs tower blocks [li, hi) over the picture's n rows on the
// device: one Layers call, or -- where the device built the non-causal set
// row-chunked because the whole picture's did not fit (nn.RowChunker) --
// block by block, every chunk's phase 1 before any chunk's phase 2. x and cs
// are the picture's rows; cs may be nil (a tower with no rotary).
func (s *State) towerLayers(li, hi, n int, x, cs []float32) bool {
	rc, ok := s.ld.(nn.RowChunker)
	c := 0
	if ok {
		c = rc.TowerChunk()
	}
	if c <= 0 || c >= n {
		return s.ld.Layers(li, hi, 0, n, x, cs, nil, nil)
	}
	e := len(x) / n
	rw := 0
	if cs != nil {
		rw = len(cs) / n
	}
	tmp := s.towerChunkX(c * e)
	for b := li; b < hi; b++ {
		for phase := 1; phase <= 2; phase++ {
			for at := 0; at < n; at += c {
				m := min(c, n-at)
				xs := x[at*e : (at+m)*e]
				if phase == 1 {
					// Phase 1 leaves the rows as they were; the device's
					// copy goes to scratch so nothing can touch x.
					xs = append(tmp[:0], xs...)
				}
				var cr []float32
				if cs != nil {
					cr = cs[at*rw : (at+m)*rw]
				}
				if !rc.LayersPhase(b, b+1, at, m, n, phase, xs, cr, nil) {
					return false
				}
			}
		}
	}
	return true
}

// towerChunkX is phase 1's scratch for one chunk's rows, kept on the vision
// segment so a warm encode allocates nothing.
func (s *State) towerChunkX(n int) []float32 {
	if cap(s.vis.chunkX) < n {
		s.vis.chunkX = make([]float32, n)
	}
	return s.vis.chunkX[:n]
}
