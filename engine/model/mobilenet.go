package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// Gemma 3n's vision: timm's MobileNet-V5 encoder (jlm.ProjGemma3nV), a
// convolutional tower on the one runner.
//
// Rows are positions and a row is its channels, so a pointwise convolution
// is the packed matmul every tower runs, a full one is that matmul over an
// im2col of the padded input (tap-major, as the converter lays the weight
// out), timm's RmsNorm2d is the row RMSNorm, and the only arithmetic of its
// own is the depthwise convolution, one generated row kernel per shape
// (nn.DWConvRow). Every block is a block of the model -- paged, bound and
// run in order like a transformer tower's -- whose shape is its own: the
// tower's activations change size from one block to the next, so a block's
// input and output live in the encode's own buffers (mnRun) and not in the
// segment's residual.
//
// What is the family's own, read off timm's MobileNetV5Encoder, its blocks
// and transformers' Gemma3nMultimodalEmbedder:
//
//	padding       TF SAME everywhere (pad_type 'same'): a stride-2 window
//	              pads (out-1)*s + K - in, the smaller half first
//	stride        each stage's first block downsamples: an edge residual in
//	              its full convolution, an inverted residual in its
//	              depthwise middle (or start, if it has no middle)
//	attention     multi-query: every head reads one key and one value head,
//	              both a stride-2 depthwise downsample of the normed input
//	              where the block has one; scores at 1/sqrt(key dim)
//	fusion        the last two stages' outputs, the coarser one upsampled
//	              (nearest) to the finer one's size, concatenated by
//	              channel, an inverted-residual FFN with no skip, a k x k
//	              average pool to 16x16, an RMSNorm
//	embedder      the 256 rows times sqrt(NEmbd), an RMSNorm, one matrix and
//	              an unweighted RMSNorm

type mnKind uint8

const (
	mnEdge mnKind = iota + 1
	mnInverted
	mnAttention
)

// mnBlock is a MobileNet block's own description: its kind and place, its
// shapes at the tower's picture size, and the vectors it reads (norms and
// depthwise filters, copied out at load). Its matrices are the layer's, so
// the pager binds them as any block's: an edge residual's full and pointwise
// convolutions and an inverted residual's expansion and projection are up and
// down, an attention block's projections wq, wk, wv and wo.
type mnBlock struct {
	kind       mnKind
	stage, idx int
	// The block's input and output, and its widest intermediate.
	hin, win, cin    int
	hout, wout, cout int
	cexp             int
	// norm1 and norm2 are the norms after the block's two matrices: an edge
	// residual's bn1 and bn2, an inverted residual's expansion and projection
	// norms. An attention block's norm1 is its input norm.
	norm1, norm2 []float32
	// The convolutions' geometry: the full convolution's stride, and each
	// depthwise filter's side, stride and the size it writes.
	stride               int
	dwStart, dwStartNorm []float32
	kStart, sStart       int
	dwMid, dwMidNorm     []float32
	kMid, sMid           int
	hmid, wmid           int // after the depthwise start: the expansion's grid
	// Attention: heads of key and value dim kd reading one key/value head, its
	// keys and values a kvK x kvK stride-2 downsample when kDown is set.
	heads, kd        int
	kDown, kDownNorm []float32
	vDown, vDownNorm []float32
	kvK, hkv, wkv    int
	residual         bool
}

// mnTower is the tower's own pieces beside its blocks: the stem, the fusion
// adapter and the embedder's norms.
type mnTower struct {
	stemNorm []float32
	stemOut  int // the stem's channels
	hs, ws   int // the stem's output grid
	// tapA and tapB are the blocks whose outputs the fusion adapter reads (the
	// last of the last two stages), by index in the segment.
	tapA, tapB          int
	fuseExp, fuseProj   tensor
	fuseExpNorm         []float32
	fuseProjNorm, fNorm []float32
	// grid is the output side (16), pool the average pool's side.
	grid, pool int
	// ones is the unweighted norm's weight, ProjDim of them.
	ones []float32
	// dw is every depthwise row shape the tower runs, provisioned into a
	// State's JIT before its first encode.
	dw []cpu.DWShape
	// hard is the text embedding of the hard vision tokens from hardOff on
	// (jlm.RoleVMNHardEmbd), ProjDim floats a token; nil when the container
	// carries none.
	hard    []float32
	hardOff int
}

// hardEmbd is Gemma 3n's hard vision tokens' text embedding, ready for
// embedRow: divided by the embedding scale its callers apply.
type hardEmbd struct {
	rows   []float32
	off, n int
}

// hardRows is the model's hardEmbd from the tower's table.
func (mt *mnTower) hardRows(scale float64) *hardEmbd {
	if len(mt.ones) == 0 {
		return nil
	}
	w := len(mt.ones)
	rows := append([]float32(nil), mt.hard...)
	nn.Scale32JIT(rows, float32(1/scale))
	return &hardEmbd{rows: rows, off: mt.hardOff, n: len(mt.hard) / w}
}

// mnRun is an encode's working memory for the tower, sized at the first
// encode and reused: the activation's two halves, the expansion, the padded
// input and the im2col tile, the attention's buffers and the fusion taps.
type mnRun struct {
	act      [2][]float32
	cur      int
	h, w, c  int
	exp, pad []float32
	col      []float32
	q, k, v  []float32
	ao       []float32
	kin      []float32
	scores   []float32
	tapA     []float32
	cat      []float32
	pooled   []float32
	// The region a pool run executes: one depthwise output row per item, or
	// one query row of the attention.
	op       uint8
	dwOut    []float32
	dwIn     []float32
	dwFilt   []float32
	dwShape  cpu.DWShape
	dwStride int
	att      *nn.AttnSet
	attM     int
	attHeads int
	attKD    int
	chunk    int
	fn       func(lo, hi int)
	// provisioned says the JIT has every depthwise shape.
	provisioned bool
}

const (
	mnOpDW uint8 = iota + 1
	mnOpAttn
)

// mnSame is TF SAME padding along one axis: the output size, and the padding
// before the first sample (the smaller half of the total).
func mnSame(in, k, stride int) (out, before, total int) {
	out = (in + stride - 1) / stride
	total = max((out-1)*stride+k-in, 0)
	return out, total / 2, total
}

// loadGemma3nV reads the tower: its stem, every block's kind and shape at the
// tower's picture size, the fusion adapter and the embedder.
func (t *Tower) loadGemma3nV(c0 *jlm.File, get func(jlm.Role, int32) (tensor, error),
	vec, optVec func(jlm.Role, int32) ([]float32, error)) error {
	c := &t.Cfg
	var err error
	mt := &mnTower{grid: c.ImageSz / c.PatchSz}
	if t.patchW, err = get(jlm.RoleVMNStem, jlm.DenseBlock); err != nil {
		return err
	}
	if t.patchB, err = optVec(jlm.RoleVMNStemBias, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.stemNorm, err = vec(jlm.RoleVMNStemNorm, jlm.DenseBlock); err != nil {
		return err
	}
	mt.stemOut = t.patchW.rows
	if t.patchW.k < 27 || len(mt.stemNorm) != mt.stemOut || t.patchB != nil && len(t.patchB) != mt.stemOut {
		return fmt.Errorf("model: OpenTower: gemma3nv's stem is %dx%d, its norm %d", t.patchW.rows, t.patchW.k, len(mt.stemNorm))
	}
	h, _, _ := mnSame(c.ImageSz, 3, 2)
	mt.hs, mt.ws = h, h
	w, ch := h, mt.stemOut
	ls := make([]layer, c.NLayer)
	maxStage := 0
	for i := range ls {
		bi := int32(i)
		l := &ls[i]
		g, err := vec(jlm.RoleVMNGeom, bi)
		if err != nil {
			return err
		}
		if len(g) != 2 {
			return fmt.Errorf("model: OpenTower: block %d's place is %d numbers", i, len(g))
		}
		b := &mnBlock{stage: int(g[0]), idx: int(g[1]), hin: h, win: w, cin: ch}
		maxStage = max(maxStage, b.stage)
		stride := 1
		if b.idx == 0 {
			stride = 2
		}
		switch {
		case c0.Has(jlm.RoleVMNConvExp, bi, -1):
			b.kind, b.stride = mnEdge, stride
			if l.up, err = get(jlm.RoleVMNConvExp, bi); err != nil {
				return err
			}
			if l.down, err = get(jlm.RoleVMNConvPwl, bi); err != nil {
				return err
			}
			if b.norm1, err = vec(jlm.RoleVMNNorm1, bi); err != nil {
				return err
			}
			if b.norm2, err = vec(jlm.RoleVMNNorm2, bi); err != nil {
				return err
			}
			b.cexp, b.cout = l.up.rows, l.down.rows
			b.hout, _, _ = mnSame(h, 3, stride)
			b.wout, _, _ = mnSame(w, 3, stride)
			if l.up.k != pad32(9*ch) || l.down.k != b.cexp || len(b.norm1) != b.cexp || len(b.norm2) != b.cout {
				return fmt.Errorf("model: OpenTower: block %d's edge residual is %dx%d and %dx%d for %d channels",
					i, l.up.rows, l.up.k, l.down.rows, l.down.k, ch)
			}
			b.residual = stride == 1 && ch == b.cout
		case c0.Has(jlm.RoleVAttnQ, bi, -1):
			b.kind = mnAttention
			for _, m := range []struct {
				role jlm.Role
				dst  *tensor
			}{{jlm.RoleVAttnQ, &l.wq}, {jlm.RoleVAttnK, &l.wk}, {jlm.RoleVAttnV, &l.wv}, {jlm.RoleVAttnOut, &l.wo}} {
				if *m.dst, err = get(m.role, bi); err != nil {
					return err
				}
			}
			if b.norm1, err = vec(jlm.RoleVAttnNorm, bi); err != nil {
				return err
			}
			b.kd = l.wk.rows
			if b.kd == 0 || l.wq.rows%b.kd != 0 || l.wv.rows != b.kd || l.wo.k != l.wq.rows || l.wo.rows != ch ||
				l.wq.k != ch || l.wk.k != ch || l.wv.k != ch || len(b.norm1) != ch {
				return fmt.Errorf("model: OpenTower: block %d's attention is q %dx%d, k %dx%d, v %dx%d, o %dx%d over %d channels",
					i, l.wq.rows, l.wq.k, l.wk.rows, l.wk.k, l.wv.rows, l.wv.k, l.wo.rows, l.wo.k, ch)
			}
			b.heads = l.wq.rows / b.kd
			b.hkv, b.wkv = h, w
			if c0.Has(jlm.RoleVMNKDown, bi, -1) {
				if b.kDown, err = vec(jlm.RoleVMNKDown, bi); err != nil {
					return err
				}
				if b.kDownNorm, err = vec(jlm.RoleVMNKDownNorm, bi); err != nil {
					return err
				}
				if b.vDown, err = vec(jlm.RoleVMNVDown, bi); err != nil {
					return err
				}
				if b.vDownNorm, err = vec(jlm.RoleVMNVDownNorm, bi); err != nil {
					return err
				}
				if b.kvK, err = mnSide(b.kDown, ch); err != nil || len(b.vDown) != len(b.kDown) {
					return fmt.Errorf("model: OpenTower: block %d's key and value downsamples are %d and %d floats over %d channels",
						i, len(b.kDown), len(b.vDown), ch)
				}
				b.hkv, _, _ = mnSame(h, b.kvK, 2)
				b.wkv, _, _ = mnSame(w, b.kvK, 2)
			}
			b.cout, b.hout, b.wout, b.residual = ch, h, w, true
		default:
			b.kind = mnInverted
			if l.up, err = get(jlm.RoleVMNPwExp, bi); err != nil {
				return err
			}
			if l.down, err = get(jlm.RoleVMNPwProj, bi); err != nil {
				return err
			}
			if b.norm1, err = vec(jlm.RoleVMNPwExpNorm, bi); err != nil {
				return err
			}
			if b.norm2, err = vec(jlm.RoleVMNPwProjNorm, bi); err != nil {
				return err
			}
			if b.dwStart, err = optVec(jlm.RoleVMNDwStart, bi); err != nil {
				return err
			}
			if b.dwMid, err = optVec(jlm.RoleVMNDwMid, bi); err != nil {
				return err
			}
			b.cexp, b.cout = l.up.rows, l.down.rows
			if l.up.k != ch || l.down.k != b.cexp || len(b.norm1) != b.cexp || len(b.norm2) != b.cout {
				return fmt.Errorf("model: OpenTower: block %d's inverted residual is %dx%d and %dx%d over %d channels",
					i, l.up.rows, l.up.k, l.down.rows, l.down.k, ch)
			}
			// The stride goes to the depthwise middle, or to the start where
			// there is no middle (timm's UniversalInvertedResidual).
			b.sStart, b.sMid = 1, stride
			if b.dwMid == nil {
				b.sStart, b.sMid = stride, 1
			}
			b.hmid, b.wmid = h, w
			if b.dwStart != nil {
				if b.dwStartNorm, err = vec(jlm.RoleVMNDwStartNorm, bi); err != nil {
					return err
				}
				if b.kStart, err = mnSide(b.dwStart, ch); err != nil || len(b.dwStartNorm) != ch {
					return fmt.Errorf("model: OpenTower: block %d's depthwise start is %d floats over %d channels", i, len(b.dwStart), ch)
				}
				b.hmid, _, _ = mnSame(h, b.kStart, b.sStart)
				b.wmid, _, _ = mnSame(w, b.kStart, b.sStart)
			} else if b.sStart != 1 {
				return fmt.Errorf("model: OpenTower: block %d downsamples and has no depthwise filter to do it", i)
			}
			b.hout, b.wout = b.hmid, b.wmid
			if b.dwMid != nil {
				if b.dwMidNorm, err = vec(jlm.RoleVMNDwMidNorm, bi); err != nil {
					return err
				}
				if b.kMid, err = mnSide(b.dwMid, b.cexp); err != nil || len(b.dwMidNorm) != b.cexp {
					return fmt.Errorf("model: OpenTower: block %d's depthwise middle is %d floats over %d channels", i, len(b.dwMid), b.cexp)
				}
				b.hout, _, _ = mnSame(b.hmid, b.kMid, b.sMid)
				b.wout, _, _ = mnSame(b.wmid, b.kMid, b.sMid)
			}
			b.residual = b.hout == h && b.wout == w && b.cout == ch
		}
		l.mn = b
		h, w, ch = b.hout, b.wout, b.cout
		mt.dw = append(mt.dw, b.dwShapes()...)
	}
	// The fusion adapter reads the last block of the last two stages.
	mt.tapA, mt.tapB = -1, len(ls)-1
	for i := range ls {
		if ls[i].mn.stage == maxStage-1 {
			mt.tapA = i
		}
	}
	if mt.tapA < 0 {
		return fmt.Errorf("model: OpenTower: gemma3nv's tower has one stage; the fusion adapter reads two")
	}
	a, b := ls[mt.tapA].mn, ls[mt.tapB].mn
	if a.hout%b.hout != 0 || a.wout%b.wout != 0 || a.hout != a.wout || a.hout%mt.grid != 0 {
		return fmt.Errorf("model: OpenTower: gemma3nv fuses a %dx%d and a %dx%d grid into %dx%d, which is not a whole factor of each",
			a.hout, a.wout, b.hout, b.wout, mt.grid, mt.grid)
	}
	mt.pool = a.hout / mt.grid
	if mt.fuseExp, err = get(jlm.RoleVMNFusionExp, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.fuseProj, err = get(jlm.RoleVMNFusionProj, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.fuseExpNorm, err = vec(jlm.RoleVMNFusionExpNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.fuseProjNorm, err = vec(jlm.RoleVMNFusionProjNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.fNorm, err = vec(jlm.RoleVMNFusionNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.fuseExp.k != a.cout+b.cout || mt.fuseProj.k != mt.fuseExp.rows || mt.fuseProj.rows != c.NEmbd ||
		len(mt.fuseExpNorm) != mt.fuseExp.rows || len(mt.fuseProjNorm) != c.NEmbd || len(mt.fNorm) != c.NEmbd {
		return fmt.Errorf("model: OpenTower: gemma3nv's fusion adapter is %dx%d and %dx%d for %d+%d channels to %d",
			mt.fuseExp.rows, mt.fuseExp.k, mt.fuseProj.rows, mt.fuseProj.k, a.cout, b.cout, c.NEmbd)
	}
	if t.projNorm, err = vec(jlm.RoleVProjNorm, jlm.DenseBlock); err != nil {
		return err
	}
	if t.projW, err = get(jlm.RoleVProj, jlm.DenseBlock); err != nil {
		return err
	}
	if len(t.projNorm) != c.NEmbd || t.projW.k != c.NEmbd || t.projW.rows != c.ProjDim {
		return fmt.Errorf("model: OpenTower: gemma3nv's embedder is %dx%d with a %d-wide norm, want %d to %d",
			t.projW.rows, t.projW.k, len(t.projNorm), c.NEmbd, c.ProjDim)
	}
	mt.ones = make([]float32, c.ProjDim)
	for i := range mt.ones {
		mt.ones[i] = 1
	}
	if mt.hard, err = optVec(jlm.RoleVMNHardEmbd, jlm.DenseBlock); err != nil {
		return err
	}
	if mt.hard != nil {
		off, err := vec(jlm.RoleVMNHardOffset, jlm.DenseBlock)
		if err != nil {
			return err
		}
		if len(off) != 1 || len(mt.hard)%c.ProjDim != 0 {
			return fmt.Errorf("model: OpenTower: gemma3nv's hard-token table is %d floats at %d offsets, for %d-wide rows",
				len(mt.hard), len(off), c.ProjDim)
		}
		mt.hardOff = int(off[0])
	}
	t.mn = mt
	t.pending = ls
	return nil
}

// mnSide is a depthwise filter's side: K*K rows of c channels.
func mnSide(filt []float32, c int) (int, error) {
	if c == 0 || len(filt)%c != 0 {
		return 0, fmt.Errorf("a filter of %d floats over %d channels", len(filt), c)
	}
	k := int(math.Round(math.Sqrt(float64(len(filt) / c))))
	if k*k*c != len(filt) {
		return 0, fmt.Errorf("a filter of %d floats over %d channels is not square", len(filt), c)
	}
	return k, nil
}

// pad32 is n rounded up to a whole quantization block.
func pad32(n int) int { return (n + 31) / 32 * 32 }

// dwShapes is every depthwise row the block runs.
func (b *mnBlock) dwShapes() []cpu.DWShape {
	var out []cpu.DWShape
	add := func(w, k, s, c int) {
		wo, _, _ := mnSame(w, k, s)
		out = append(out, cpu.DWShape{K: k, Stride: s, WP: (wo-1)*s + k, Chans: c, WOut: wo})
	}
	switch b.kind {
	case mnInverted:
		if b.dwStart != nil {
			add(b.win, b.kStart, b.sStart, b.cin)
		}
		if b.dwMid != nil {
			add(b.wmid, b.kMid, b.sMid, b.cexp)
		}
	case mnAttention:
		if b.kDown != nil {
			add(b.win, b.kvK, 2, b.cin)
		}
	}
	return out
}

// plan is the block's geometry as a device takes it (nn.ConvPlan).
func (b *mnBlock) plan(eps float64, actWin int) *nn.ConvPlan {
	p := &nn.ConvPlan{HIn: b.hin, WIn: b.win, CIn: b.cin, HOut: b.hout, WOut: b.wout, COut: b.cout,
		CExp: b.cexp, Stride: b.stride, HMid: b.hmid, WMid: b.wmid, Heads: b.heads, KD: b.kd,
		HKV: b.hkv, WKV: b.wkv, Residual: b.residual, Eps: eps, ActWin: actWin}
	switch b.kind {
	case mnEdge:
		p.Kind = nn.ConvEdge
	case mnInverted:
		p.Kind = nn.ConvInverted
		if b.dwStart != nil {
			p.KStart, p.SStart = b.kStart, b.sStart
		}
		if b.dwMid != nil {
			p.KMid, p.SMid = b.kMid, b.sMid
		}
	case mnAttention:
		p.Kind = nn.ConvAttention
		if b.kDown != nil {
			p.KVK = b.kvK
		}
	}
	return p
}
