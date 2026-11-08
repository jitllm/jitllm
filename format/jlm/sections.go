package jlm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// The encoders and decoders for the container's own sections. One encoding
// for every section: little-endian fixed widths; a string is u32 length then
// bytes; a slice is u32 count then elements. There is no tag or type byte
// inside a record: the schema files beside this one are the description, and
// the container's version says which revision of them to read.

type wbuf struct{ b []byte }

func (w *wbuf) u8(v uint8)    { w.b = append(w.b, v) }
func (w *wbuf) u16(v uint16)  { w.b = binary.LittleEndian.AppendUint16(w.b, v) }
func (w *wbuf) u32(v uint32)  { w.b = binary.LittleEndian.AppendUint32(w.b, v) }
func (w *wbuf) i32(v int32)   { w.u32(uint32(v)) }
func (w *wbuf) u64(v uint64)  { w.b = binary.LittleEndian.AppendUint64(w.b, v) }
func (w *wbuf) f32(v float32) { w.u32(math.Float32bits(v)) }
func (w *wbuf) bool(v bool) {
	if v {
		w.u8(1)
	} else {
		w.u8(0)
	}
}
func (w *wbuf) str(s string) { w.u32(uint32(len(s))); w.b = append(w.b, s...) }

type rbuf struct {
	b   []byte
	err error
}

// want is the one bounds check. A container is untrusted input: every
// accessor goes through here, the error is sticky, and a short read yields a
// zero rather than a panic.
func (r *rbuf) want(n int) bool {
	if r.err != nil {
		return false
	}
	if len(r.b) < n {
		r.err = fmt.Errorf("jlm: section truncated: wanted %d bytes, %d left", n, len(r.b))
		return false
	}
	return true
}
func (r *rbuf) u8() uint8 {
	if !r.want(1) {
		return 0
	}
	v := r.b[0]
	r.b = r.b[1:]
	return v
}
func (r *rbuf) u16() uint16 {
	if !r.want(2) {
		return 0
	}
	v := binary.LittleEndian.Uint16(r.b)
	r.b = r.b[2:]
	return v
}
func (r *rbuf) u32() uint32 {
	if !r.want(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.b)
	r.b = r.b[4:]
	return v
}
func (r *rbuf) u64() uint64 {
	lo := uint64(r.u32())
	return lo | uint64(r.u32())<<32
}
func (r *rbuf) i32() int32   { return int32(r.u32()) }
func (r *rbuf) f32() float32 { return math.Float32frombits(r.u32()) }
func (r *rbuf) bool() bool   { return r.u8() != 0 }
func (r *rbuf) str() string {
	n := int(r.u32())
	// Size nothing from untrusted input: the length is checked against what
	// is left, before any allocation.
	if !r.want(n) {
		return ""
	}
	s := string(r.b[:n])
	r.b = r.b[n:]
	return s
}

// count reads a slice length and refuses one the remaining bytes cannot hold,
// given a minimum per element. Same rule as str, one level up.
func (r *rbuf) count(minPer int) int {
	n := int(r.u32())
	if r.err != nil {
		return 0
	}
	if n < 0 || (minPer > 0 && n > len(r.b)/minPer) {
		r.err = fmt.Errorf("jlm: section claims %d elements, %d bytes left", n, len(r.b))
		return 0
	}
	return n
}

// --- Config -------------------------------------------------------------

func encodeConfig(c *Config) []byte { return encodeConfigAt(c, Version) }

// encodeConfigAt encodes c as a version v container's record: a migration
// writes the version after the one it read (migrate.go). Write refuses a
// config that version cannot state (writeAt).
func encodeConfigAt(c *Config, v uint32) []byte {
	w := &wbuf{}
	w.u16(uint16(c.Arch))
	for _, v := range []uint32{c.NLayer, c.NEmbd, c.NHead, c.NKVHead, c.HeadDim, c.NRot,
		c.NFFN, c.NVocab, c.NCtx, c.NExpert, c.NExpertUsed, c.NFFNExp, c.SWAWindow, c.SWAPeriod} {
		w.u32(v)
	}
	for _, v := range []float32{c.RMSEps, c.RopeBase, c.AttnFactor, c.EmbdScale, c.RopeBaseSWA} {
		w.f32(v)
	}
	w.u32(uint32(c.Flags))
	w.u32(c.NFFNShExp)
	for _, v := range []uint32{c.SSM.ConvKernel, c.SSM.Groups, c.SSM.Inner, c.SSM.StateSize, c.SSM.NHeadV} {
		w.u32(v)
	}
	w.u32(uint32(len(c.LayerKinds)))
	for _, k := range c.LayerKinds {
		w.u8(uint8(k))
	}
	for _, v := range c.RopeSections {
		w.u32(v)
	}
	// Fields are appended at the end; the version bump stops an older
	// container being decoded with a newer layout.
	w.f32(c.AttnSoftcap)
	w.f32(c.FinalSoftcap)
	w.f32(c.YarnFactor)
	w.u32(c.YarnOrigCtx)
	w.f32(c.YarnBetaFast)
	w.f32(c.YarnBetaSlow)
	w.f32(c.RopeLinear)
	// v23: MLA, the leading dense run, and the DeepSeek router's three knobs.
	w.u32(c.HeadDimV)
	w.u32(c.NDenseLead)
	w.u32(c.QLoraRank)
	w.u32(c.KVLoraRank)
	w.f32(c.YarnLogMul)
	w.f32(c.ExpertScale)
	w.u32(c.NExpertGroup)
	w.u32(c.NExpertGroupUsed)
	// v28: the prediction blocks after the trunk.
	if v >= 28 {
		w.u32(c.NMTP)
	}
	// (ERNIE 4.5's mixtures write the interleave too, under ArchErnie45MoE,
	// another code an older reader refuses.)
	// Llama 4's interleave and temperature are an optional tail, costing no
	// version bump: the record is length-delimited (CfgLen), an older reader
	// stops before it, and the only writer is ArchLlama4, a code such a reader
	// refuses. Every other model writes the same bytes as before.
	//
	// The second group (the three scales) follows the first and implies it,
	// zeros and all, so a reader never guesses which group a lone tail is. Its
	// writers are ArchGranite (refused by older readers) and gemma2/gemma3 27B,
	// whose scale an older reader drops, as it did before the field existed.
	// The third, DBRX's clamp, implies both; its only writer is ArchDBRX.
	// The fourth, Gemma 4's sliding geometry and Flags2, implies the first
	// three; its only writer is ArchGemma4.
	// The fifth, DeepSeek V3.2's indexer, implies the first four; its only
	// writer is ArchDeepseek32. The sixth, Gemma 3n's AltUp and sparsity,
	// implies the first five; its only writer is ArchGemma3n. The seventh,
	// MiniMax-M3's block selection, implies the first six; its only writer is
	// ArchMiniMaxM3. The eighth, DeepSeek V4's hyper-connections, grouped
	// output and compressed blocks, implies the first seven; its only writer
	// is ArchDeepseek4. The ninth, Kimi-K3's residual attention, latent
	// mixture, situ bounds, decay bound and latent norm epsilon, implies the
	// first eight; ArchKimiK3 writes all of it, and ArchKimiLinear the latent
	// norm epsilon alone, which an older reader drops and norms at RMSEps, as
	// it did before the field existed. The tenth, a decision model's readout
	// and temperatures, implies the first nine; an older reader drops it and
	// runs the model as the backbone it is, a model that generates.
	// The twelfth, Clef's routing blocks and head epsilon, implies the first
	// eleven; an older reader refuses its decision kind by code.
	// The thirteenth, the stated activations, implies the first twelve; an
	// older reader drops it and runs the activation its flags name. The
	// fourteenth, the router's stated score, implies the first thirteen; an
	// older reader drops it and routes by the gating its flags name. The
	// fifteenth, the router's stated weight, implies the first fourteen; an
	// older reader drops it and weighs by the router its flags name. The
	// sixteenth, the residual's stated streams, implies the first fifteen;
	// an older reader drops it and refuses the streams it cannot state. The
	// seventeenth, the statement revision with the sliding layers' rotary
	// magnitude and the layers a picture's rows see each other on, implies
	// the first sixteen; an older reader drops it.
	// The eighteenth, the activation quantization, implies the first
	// seventeen; an older reader drops it.
	// The nineteenth, the streams' expansions and writes, implies the first
	// eighteen; the twentieth, the selection, the first nineteen; the
	// twenty-first, the blocks' additions before their attention, the first
	// twenty. An older reader drops each.
	// The twenty-second, the streams' posts, implies the first twenty-one;
	// the twenty-third, the blocks' own geometries (Geoms), the first
	// twenty-two; the twenty-fourth, the delta rule's stated decay and gate,
	// the first twenty-three. An older reader drops each.
	// The twenty-fifth, a label readout's statement, implies the first
	// twenty-four; an older reader drops it and opens the model as the
	// backbone it is. The twenty-sixth, a marker head's prompt, implies the
	// first twenty-five, an empty label statement standing for none.
	// The twenty-seventh, a joint head's statement, implies the first
	// twenty-six, an empty marker prompt standing for none.
	joint := c.JointHead != nil
	marker := joint || c.MarkerPrompt != nil
	labels := marker || c.DecisionLabels != nil
	delta := labels || c.DeltaDecay != nil || c.DeltaGate != nil
	geoms := delta || len(c.Geoms) != 0
	post := geoms || c.Streams.sharesWrites()
	pre := post || c.Streams.addsBefore()
	sel := pre || c.Selection != nil
	sw := sel || c.Streams.writesStreams()
	aq := sw || c.ActQuant != ActQuantNone
	rev := aq || c.Revision != 0 || c.RopeScaleSWA != 0 || len(c.BidirLayers) != 0
	streams := rev || c.Streams.N > 1
	weight := streams || len(c.RouterWeight.Terms) != 0
	score := weight || len(c.RouterScore.Terms) != 0
	acts := score || len(c.Acts) != 0 || len(c.ActLayers) != 0
	clef := acts || c.DecisionRouting != 0 || c.DecisionEps != 0
	q4e := clef || c.HCLowRank != 0 || c.QSARatio != 0 || c.NGramBlock != 0
	dec := q4e || c.Decision != DecisionNone || c.DecisionBlocks != 0 || c.DecisionHeadTokens != 0 || len(c.DecisionTemps) != 0 ||
		c.DecisionHeads != 0
	k3 := dec || c.AttnResBlock != 0 || c.ExpertLatent != 0 || c.SituBeta != 0 || c.SituLinearBeta != 0 ||
		c.KDALowerBound != 0 || c.LatentNormEps != 0
	ds4 := k3 || c.HCMult != 0 || len(c.CompKinds) != 0
	msa := ds4 || c.IdxBlock != 0 || c.IdxLocal != 0
	alt := msa || c.PredStreams != 0 || c.NSparse != 0
	idx := alt || c.IdxHeads != 0
	geom := idx || c.Flags2 != 0 || c.NKVShared != 0 || c.PLEDim != 0
	clamp := geom || c.ClampKQV != 0
	scales := clamp || c.AttnScale != 0 || c.ResidualScale != 0 || c.LogitScale != 0
	if scales || c.MoEStep != 0 || c.AttnTempScale != 0 || c.AttnTempFloor != 0 || c.AttnTempOffset != 0 {
		w.u32(c.MoEStep)
		w.f32(c.AttnTempScale)
		w.u32(c.AttnTempFloor)
		w.f32(c.AttnTempOffset)
	}
	if scales {
		w.f32(c.AttnScale)
		w.f32(c.ResidualScale)
		w.f32(c.LogitScale)
	}
	if clamp {
		w.f32(c.ClampKQV)
	}
	if geom {
		// Three slots an older writer filled with the sliding layers'
		// geometry; decodeConfigAt reads them into Geoms.
		w.u32(0)
		w.u32(0)
		w.u32(0)
		w.u32(uint32(c.Flags2))
		w.u32(c.NKVShared)
		w.u32(c.PLEDim)
	}
	if idx {
		w.u32(c.IdxHeads)
		w.u32(c.IdxHeadDim)
		w.u32(c.IdxTopK)
	}
	if alt {
		w.u32(c.PredStreams)
		w.u32(c.NSparse)
		w.f32(c.SparseStd)
	}
	if msa {
		w.u32(c.IdxBlock)
		w.u32(c.IdxLocal)
	}
	if ds4 {
		w.u32(c.HCMult)
		w.u32(c.HCIters)
		w.f32(c.HCEps)
		w.u32(c.OGroups)
		w.u32(c.OLoraRank)
		w.u32(c.CompRateCSA)
		w.u32(c.CompRateHCA)
		w.u32(c.NHashLayers)
		w.u32(uint32(len(c.CompKinds)))
		for _, k := range c.CompKinds {
			w.u8(uint8(k))
		}
	}
	if k3 {
		w.u32(c.AttnResBlock)
		w.u32(c.ExpertLatent)
		w.f32(c.SituBeta)
		w.f32(c.SituLinearBeta)
		w.f32(c.KDALowerBound)
		w.f32(c.LatentNormEps)
	}
	if dec {
		w.u8(uint8(c.Decision))
		w.u32(c.DecisionBlocks)
		w.u32(c.DecisionHeadTokens)
		w.u32(c.DecisionHeads)
		w.u32(uint32(len(c.DecisionTemps)))
		for _, d := range c.DecisionTemps {
			w.u8(uint8(d.Type))
			w.u32(d.MinOptions)
			w.u32(d.MaxOptions)
			w.f32(d.T)
		}
	}
	// The eleventh, Qwen3.8-Flash-Next's, implies the first ten; its only
	// writer is ArchQwen4Exp.
	if q4e {
		for _, v := range []uint32{c.HCLowRank, c.QSARatio, c.NGramBlock, c.NGram, c.NGramHeadsPer,
			c.NGramRowDim, c.NGramConv, c.NGramEOS, c.NGramImage} {
			w.u32(v)
		}
		for _, a := range [][]uint64{c.NGramMul, c.NGramOff, c.NGramVoc} {
			w.u32(uint32(len(a)))
			for _, v := range a {
				w.u64(v)
			}
		}
	}
	if clef {
		w.u32(c.DecisionRouting)
		w.f32(c.DecisionEps)
	}
	if acts {
		encodeExprs(w, c)
	}
	if score {
		encodeExpr(w, &c.RouterScore)
	}
	if weight {
		encodeExpr(w, &c.RouterWeight)
	}
	if streams {
		encodeStreams(w, &c.Streams)
	}
	if rev {
		w.u8(c.Revision)
		w.f32(c.RopeScaleSWA)
		w.u32(uint32(len(c.BidirLayers)))
		for _, b := range c.BidirLayers {
			w.bool(b)
		}
	}
	if aq {
		w.u8(uint8(c.ActQuant))
	}
	if sw {
		encodeStreamWrites(w, &c.Streams)
	}
	if sel {
		encodeSelection(w, c.Selection)
	}
	if pre {
		encodeStreamPre(w, &c.Streams)
	}
	if post {
		encodeStreamPosts(w, &c.Streams)
	}
	if geoms {
		w.u32(uint32(len(c.Geoms)))
		for _, g := range c.Geoms {
			w.u32(g.NKVHead)
			w.u32(g.HeadDim)
			w.u32(g.NRot)
			w.u32(g.NFFN)
		}
	}
	if delta {
		encodePrograms(w, []*Program{c.DeltaDecay, c.DeltaGate})
	}
	if labels {
		l := c.DecisionLabels
		if l == nil {
			l = &DecisionLabels{}
		}
		encodeDecisionLabels(w, l)
	}
	if marker {
		p := c.MarkerPrompt
		if p == nil {
			p = &MarkerPrompt{}
		}
		encodeMarkerPrompt(w, p)
	}
	if joint {
		encodeJointHead(w, c.JointHead)
	}
	return w.b
}

func decodeConfig(b []byte) (*Config, error) { return decodeConfigAt(b, Version) }

// decodeConfigAt decodes the config record of a version v container: a
// migration reads an older one (migrate.go).
func decodeConfigAt(b []byte, v uint32) (*Config, error) {
	r := &rbuf{b: b}
	c := &Config{Arch: Arch(r.u16())}
	for _, p := range []*uint32{&c.NLayer, &c.NEmbd, &c.NHead, &c.NKVHead, &c.HeadDim, &c.NRot,
		&c.NFFN, &c.NVocab, &c.NCtx, &c.NExpert, &c.NExpertUsed, &c.NFFNExp, &c.SWAWindow, &c.SWAPeriod} {
		*p = r.u32()
	}
	for _, p := range []*float32{&c.RMSEps, &c.RopeBase, &c.AttnFactor, &c.EmbdScale, &c.RopeBaseSWA} {
		*p = r.f32()
	}
	c.Flags = Flags(r.u32())
	c.NFFNShExp = r.u32()
	for _, p := range []*uint32{&c.SSM.ConvKernel, &c.SSM.Groups, &c.SSM.Inner, &c.SSM.StateSize, &c.SSM.NHeadV} {
		*p = r.u32()
	}
	if n := r.count(1); n > 0 {
		c.LayerKinds = make([]LayerKind, n)
		for i := range c.LayerKinds {
			c.LayerKinds[i] = LayerKind(r.u8())
		}
	}
	for i := range c.RopeSections {
		c.RopeSections[i] = r.u32()
	}
	c.AttnSoftcap = r.f32()
	c.FinalSoftcap = r.f32()
	c.YarnFactor = r.f32()
	c.YarnOrigCtx = r.u32()
	c.YarnBetaFast = r.f32()
	c.YarnBetaSlow = r.f32()
	c.RopeLinear = r.f32()
	c.HeadDimV = r.u32()
	c.NDenseLead = r.u32()
	c.QLoraRank = r.u32()
	c.KVLoraRank = r.u32()
	c.YarnLogMul = r.f32()
	c.ExpertScale = r.f32()
	c.NExpertGroup = r.u32()
	c.NExpertGroupUsed = r.u32()
	// v28 added the prediction blocks' count; an older record has none.
	if v >= 28 {
		c.NMTP = r.u32()
	}
	// The optional tail; see encodeConfig. Absent means all zero, which is what
	// every architecture but Llama 4 is.
	if r.err == nil && len(r.b) > 0 {
		c.MoEStep = r.u32()
		c.AttnTempScale = r.f32()
		c.AttnTempFloor = r.u32()
		c.AttnTempOffset = r.f32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.AttnScale = r.f32()
		c.ResidualScale = r.f32()
		c.LogitScale = r.f32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.ClampKQV = r.f32()
	}
	var swa Geom
	if r.err == nil && len(r.b) > 0 {
		swa.HeadDim = r.u32()
		swa.NKVHead = r.u32()
		swa.NRot = r.u32()
		c.Flags2 = Flags2(r.u32())
		c.NKVShared = r.u32()
		c.PLEDim = r.u32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.IdxHeads = r.u32()
		c.IdxHeadDim = r.u32()
		c.IdxTopK = r.u32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.PredStreams = r.u32()
		c.NSparse = r.u32()
		c.SparseStd = r.f32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.IdxBlock = r.u32()
		c.IdxLocal = r.u32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.HCMult = r.u32()
		c.HCIters = r.u32()
		c.HCEps = r.f32()
		c.OGroups = r.u32()
		c.OLoraRank = r.u32()
		c.CompRateCSA = r.u32()
		c.CompRateHCA = r.u32()
		c.NHashLayers = r.u32()
		if n := r.count(1); n > 0 {
			c.CompKinds = make([]CompKind, n)
			for i := range c.CompKinds {
				c.CompKinds[i] = CompKind(r.u8())
			}
		}
	}
	if r.err == nil && len(r.b) > 0 {
		c.AttnResBlock = r.u32()
		c.ExpertLatent = r.u32()
		c.SituBeta = r.f32()
		c.SituLinearBeta = r.f32()
		c.KDALowerBound = r.f32()
		c.LatentNormEps = r.f32()
	}
	if r.err == nil && len(r.b) > 0 {
		c.Decision = DecisionKind(r.u8())
		c.DecisionBlocks = r.u32()
		c.DecisionHeadTokens = r.u32()
		c.DecisionHeads = r.u32()
		if n := r.count(13); n > 0 {
			c.DecisionTemps = make([]DecisionTemp, n)
			for i := range c.DecisionTemps {
				d := &c.DecisionTemps[i]
				d.Type = QuestionType(r.u8())
				d.MinOptions = r.u32()
				d.MaxOptions = r.u32()
				d.T = r.f32()
			}
		}
	}
	if r.err == nil && len(r.b) > 0 {
		for _, p := range []*uint32{&c.HCLowRank, &c.QSARatio, &c.NGramBlock, &c.NGram, &c.NGramHeadsPer,
			&c.NGramRowDim, &c.NGramConv, &c.NGramEOS, &c.NGramImage} {
			*p = r.u32()
		}
		for _, a := range []*[]uint64{&c.NGramMul, &c.NGramOff, &c.NGramVoc} {
			if n := r.count(8); n > 0 {
				*a = make([]uint64, n)
				for i := range *a {
					(*a)[i] = r.u64()
				}
			}
		}
	}
	if r.err == nil && len(r.b) > 0 {
		c.DecisionRouting = r.u32()
		c.DecisionEps = r.f32()
	}
	if r.err == nil && len(r.b) > 0 {
		decodeExprs(r, c)
	}
	if r.err == nil && len(r.b) > 0 {
		decodeExpr(r, &c.RouterScore)
	}
	if r.err == nil && len(r.b) > 0 {
		decodeExpr(r, &c.RouterWeight)
	}
	if r.err == nil && len(r.b) > 0 {
		decodeStreams(r, &c.Streams)
	}
	if r.err == nil && len(r.b) > 0 {
		c.Revision = r.u8()
		c.RopeScaleSWA = r.f32()
		for range r.count(1) {
			c.BidirLayers = append(c.BidirLayers, r.bool())
		}
	}
	if r.err == nil && len(r.b) > 0 {
		c.ActQuant = ActQuant(r.u8())
	}
	if r.err == nil && len(r.b) > 0 {
		decodeStreamWrites(r, &c.Streams)
	}
	if r.err == nil && len(r.b) > 0 {
		c.Selection = decodeSelection(r)
	}
	if r.err == nil && len(r.b) > 0 {
		decodeStreamPre(r, &c.Streams)
	}
	if r.err == nil && len(r.b) > 0 {
		decodeStreamPosts(r, &c.Streams)
	}
	if r.err == nil && len(r.b) > 0 {
		for range r.count(16) {
			c.Geoms = append(c.Geoms, Geom{NKVHead: r.u32(), HeadDim: r.u32(), NRot: r.u32(), NFFN: r.u32()})
		}
	}
	if r.err == nil && len(r.b) > 0 {
		ps := decodePrograms(r)
		if len(ps) != 2 && r.err == nil {
			r.err = fmt.Errorf("jlm: the delta rule's statement holds %d programs", len(ps))
		}
		if len(ps) == 2 {
			c.DeltaDecay, c.DeltaGate = ps[0], ps[1]
		}
	}
	if r.err == nil && len(r.b) > 0 {
		if l := decodeDecisionLabels(r); l.MaxOptions != 0 {
			c.DecisionLabels = l
		}
	}
	if r.err == nil && len(r.b) > 0 {
		if p := decodeMarkerPrompt(r); len(p.Nodes) > 0 {
			c.MarkerPrompt = p
		}
	}
	if r.err == nil && len(r.b) > 0 {
		c.JointHead = decodeJointHead(r)
	}
	if r.err != nil {
		return nil, r.err
	}
	if err := c.olderGeoms(swa); err != nil {
		return nil, err
	}
	if err := c.geomsValid(); err != nil {
		return nil, err
	}
	if err := c.deltaValid(); err != nil {
		return nil, err
	}
	if err := exprsValid(c); err != nil {
		return nil, err
	}
	if err := c.Streams.valid(int(c.NLayer + c.NMTP)); err != nil {
		return nil, err
	}
	if err := c.Selection.valid(int(c.NLayer+c.NMTP), func(li int) bool { return li < len(c.CompKinds) && c.CompKinds[li] != CompNone }); err != nil {
		return nil, err
	}
	if c.ActQuant > ActQuantRowInt8 {
		return nil, fmt.Errorf("jlm: activation quantization %d is not one this format defines", c.ActQuant)
	}
	if n := len(c.BidirLayers); n != 0 && n != int(c.NLayer) {
		return nil, fmt.Errorf("jlm: %d layers' bidirectional statement for %d layers", n, c.NLayer)
	}
	if err := ngramValid(c); err != nil {
		return nil, err
	}
	if err := c.olderStatements(); err != nil {
		return nil, err
	}
	// One kind per layer or none at all: a short array would leave the tail
	// reading as LayerNone, and the graph would run nothing for those layers.
	if n := len(c.LayerKinds); n != 0 && uint32(n) != c.NLayer {
		return nil, fmt.Errorf("jlm: %d layer kinds for %d layers", n, c.NLayer)
	}
	for i, k := range c.LayerKinds {
		if !k.Attends() && !k.Recurrent() {
			return nil, fmt.Errorf("jlm: layer %d has kind %d, which this format does not define", i, k)
		}
	}
	// DeepSeek V4's compression kinds are one per layer too, or none.
	if n := len(c.CompKinds); n != 0 && uint32(n) != c.NLayer {
		return nil, fmt.Errorf("jlm: %d compression kinds for %d layers", n, c.NLayer)
	}
	if _, ok := decisionName(c.Decision); !ok && c.Decision != DecisionNone {
		return nil, fmt.Errorf("jlm: decision kind %d is not one this format defines", c.Decision)
	}
	for _, d := range c.DecisionTemps {
		if d.Type > QuestionNoul || !(d.T > 0) {
			return nil, fmt.Errorf("jlm: decision temperature %v for %v is not a positive temperature of a question type", d.T, d.Type)
		}
	}
	for i, k := range c.CompKinds {
		if k > CompHCA {
			return nil, fmt.Errorf("jlm: layer %d has compression kind %d, which this format does not define", i, k)
		}
	}
	if !c.Arch.Valid() {
		return nil, fmt.Errorf("jlm: architecture code %d is not one this format defines", c.Arch)
	}
	// The MLA block is all-or-nothing: a rank with no rope width would
	// attend over a key of the wrong length, fluently.
	if c.KVLoraRank != 0 {
		switch {
		case c.NRot == 0:
			return nil, fmt.Errorf("jlm: kv_lora_rank %d with no rotary head width", c.KVLoraRank)
		case c.NRot >= c.HeadDim:
			return nil, fmt.Errorf("jlm: rotary width %d leaves no un-rotated half of a %d-wide head",
				c.NRot, c.HeadDim)
		case c.HeadDimV == 0:
			return nil, fmt.Errorf("jlm: kv_lora_rank %d with no value head width", c.KVLoraRank)
		}
	}
	// A group count that does not divide the experts would leave a ragged
	// final group whose score means something different from the others'.
	if c.NExpertGroup > 1 {
		switch {
		case c.NExpert%c.NExpertGroup != 0:
			return nil, fmt.Errorf("jlm: %d experts do not divide into %d groups",
				c.NExpert, c.NExpertGroup)
		case c.NExpertGroupUsed == 0 || c.NExpertGroupUsed > c.NExpertGroup:
			return nil, fmt.Errorf("jlm: %d of %d expert groups selected",
				c.NExpertGroupUsed, c.NExpertGroup)
		}
	}
	// Two readouts at once is not a model anyone ships; refusing it keeps
	// Flags.Pooling a function rather than a precedence rule.
	if n := bitsSet(c.Flags & (FlagPoolMean | FlagPoolCLS | FlagPoolLast)); n > 1 {
		return nil, fmt.Errorf("jlm: %d pooling kinds set; an embedding has one", n)
	}
	if c.NDenseLead > c.NLayer {
		return nil, fmt.Errorf("jlm: %d leading dense blocks in a %d-block model", c.NDenseLead, c.NLayer)
	}
	// A temperature with no floor divides by zero at every position.
	if c.AttnTempScale != 0 && c.AttnTempFloor == 0 {
		return nil, fmt.Errorf("jlm: attention temperature %g with no floor", c.AttnTempScale)
	}
	// A scale is a positive finite number or zero (meaning the default); a
	// negative logit scale would flip every argmax and read as a model.
	for _, v := range []float32{c.AttnScale, c.ResidualScale, c.LogitScale, c.ClampKQV} {
		if !(v >= 0) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("jlm: scale %g in the config record", v)
		}
	}
	// A shared run that leaves no layer to share from, or a layer of one kind
	// with no earlier layer of its kind, attends to a history nobody wrote.
	if c.NKVShared != 0 {
		if c.NKVShared >= c.NLayer {
			return nil, fmt.Errorf("jlm: %d KV-sharing layers of %d", c.NKVShared, c.NLayer)
		}
		for b := int(c.NLayer - c.NKVShared); b < int(c.NLayer); b++ {
			if c.KVSource(b) < 0 {
				return nil, fmt.Errorf("jlm: KV-sharing block %d has no earlier block of its kind", b)
			}
		}
	}
	// A chunk of zero positions is a mask with no keys in it.
	if c.Flags.Has(FlagSWAChunked) && c.SWAWindow == 0 {
		return nil, fmt.Errorf("jlm: chunked attention with no chunk size")
	}
	return c, nil
}

func bitsSet(f Flags) int {
	n := 0
	for ; f != 0; f &= f - 1 {
		n++
	}
	return n
}

// --- Vision -------------------------------------------------------------

func encodeVision(v *Vision) []byte {
	if v == nil {
		return nil
	}
	w := &wbuf{}
	for _, x := range []uint32{v.NLayer, v.NEmbd, v.NHead, v.NFFN, v.ImageSize, v.PatchSize,
		v.ProjDim, v.Scale} {
		w.u32(x)
	}
	w.f32(v.Eps)
	w.u16(uint16(v.Projector))
	for i := 0; i < 3; i++ {
		w.f32(v.Mean[i])
	}
	for i := 0; i < 3; i++ {
		w.f32(v.Std[i])
	}
	w.u32(uint32(v.Flags))
	// The optional tail: a tiling preprocessor's crop count bounds (InternVL),
	// then Qwen2.5-VL's window attention, each written only when it or a field
	// after it is set, so every earlier tower's section is byte-identical.
	// The pixel bounds of a tower read at its own size come third, and
	// LFM2-VL's tile size and tolerance fourth.
	tile := v.TileSize != 0 || v.Tolerance != 0 || v.Audio != nil || v.Tower != nil
	pix := v.MinPixels != 0 || v.MaxPixels != 0 || tile
	if v.MinTiles != 0 || v.MaxTiles != 0 || v.WinPattern != 0 || v.WinSize != 0 || pix {
		w.u32(v.MinTiles)
		w.u32(v.MaxTiles)
	}
	if v.WinPattern != 0 || v.WinSize != 0 || pix {
		w.u32(v.WinPattern)
		w.u32(v.WinSize)
	}
	if pix {
		w.u32(v.MinPixels)
		w.u32(v.MaxPixels)
	}
	if tile {
		w.u32(v.TileSize)
		w.f32(v.Tolerance)
	}
	switch {
	case v.Audio != nil:
		encodeAudio(w, v.Audio)
	case v.Tower != nil:
		encodeAudio(w, &Audio{})
	}
	if v.Tower != nil {
		encodeTower(w, v.Tower)
	}
	// The audio tower after the vision tower's record, which a reader takes
	// whole: a container with no audio tower is the bytes it was.
	if a := v.Audio; a != nil && a.Tower != nil && v.Tower != nil {
		w.u32(a.MaxSamples)
		encodeTower(w, a.Tower)
	}
	// The video record last, after the audio tower's where there is one: a
	// container with no video is the bytes it was.
	if v.Tower != nil && v.Tower.Video != nil {
		encodeVideo(w, v.Tower.Video)
	}
	return w.b
}

func decodeVision(b []byte) (*Vision, error) {
	if len(b) == 0 {
		return nil, nil
	}
	r := &rbuf{b: b}
	v := &Vision{}
	for _, p := range []*uint32{&v.NLayer, &v.NEmbd, &v.NHead, &v.NFFN, &v.ImageSize, &v.PatchSize,
		&v.ProjDim, &v.Scale} {
		*p = r.u32()
	}
	v.Eps = r.f32()
	v.Projector = Projector(r.u16())
	for i := 0; i < 3; i++ {
		v.Mean[i] = r.f32()
	}
	for i := 0; i < 3; i++ {
		v.Std[i] = r.f32()
	}
	v.Flags = Flags(r.u32())
	// The optional tail; absent is a tower that takes one square and has no
	// windows.
	if r.err == nil && len(r.b) > 0 {
		v.MinTiles, v.MaxTiles = r.u32(), r.u32()
	}
	if r.err == nil && len(r.b) > 0 {
		v.WinPattern, v.WinSize = r.u32(), r.u32()
	}
	if r.err == nil && len(r.b) > 0 {
		v.MinPixels, v.MaxPixels = r.u32(), r.u32()
	}
	if r.err == nil && len(r.b) > 0 {
		v.TileSize, v.Tolerance = r.u32(), r.f32()
	}
	if r.err == nil && len(r.b) > 0 {
		if v.Audio = decodeAudio(r); v.Audio.NLayer == 0 {
			v.Audio = nil
		}
	}
	if r.err == nil && len(r.b) > 0 {
		v.Tower = decodeTower(r)
	}
	if r.err == nil && len(r.b) > 0 && v.Audio != nil {
		v.Audio.MaxSamples = r.u32()
		v.Audio.Tower = decodeTower(r)
	}
	if r.err == nil && len(r.b) > 0 && v.Tower != nil {
		v.Tower.Video = decodeVideo(r)
	}
	if r.err != nil {
		return nil, r.err
	}
	if v.Tower != nil {
		if err := v.Tower.Valid(); err != nil {
			return nil, err
		}
		if v.Tower.Video != nil {
			if err := v.Tower.Video.Valid(); err != nil {
				return nil, err
			}
		}
	}
	if v.Audio != nil && v.Audio.Tower != nil {
		if err := v.Audio.Tower.Valid(); err != nil {
			return nil, fmt.Errorf("jlm: the audio tower: %w", err)
		}
	}
	return v, nil
}

// --- Vocab --------------------------------------------------------------

func encodeVocab(v *Vocab) []byte {
	if v == nil {
		return nil
	}
	w := &wbuf{}
	w.u8(uint8(v.Kind))
	w.u32(uint32(len(v.Tokens)))
	for _, t := range v.Tokens {
		w.str(t)
	}
	w.u32(uint32(len(v.Scores)))
	for _, s := range v.Scores {
		w.f32(s)
	}
	w.u32(uint32(len(v.Kinds)))
	for _, k := range v.Kinds {
		w.u8(uint8(k))
	}
	w.u32(uint32(len(v.Merges)))
	for _, m := range v.Merges {
		w.str(m.Left)
		w.str(m.Right)
	}
	w.u32(uint32(len(v.Added)))
	for _, a := range v.Added {
		w.i32(a.ID)
		w.str(a.Content)
		w.bool(a.Special)
		w.bool(a.LStrip)
		w.bool(a.RStrip)
		w.bool(a.Normalized)
		w.bool(a.SingleWord)
	}
	for _, id := range []int32{v.BOS, v.EOS, v.Unk, v.Pad, v.Sep, v.Mask} {
		w.i32(id)
	}
	for _, f := range []bool{v.AddBOS, v.AddEOS, v.AddSpacePrefix, v.ByteFallback, v.IgnoreMerges} {
		w.bool(f)
	}
	w.u32(uint32(len(v.Pre)))
	for _, op := range v.Pre {
		w.u8(uint8(op.Kind))
		w.bool(op.Contractions)
		w.bool(op.CaseFold)
		w.bool(op.LetterMarks)
		w.u8(op.DigitGroup)
		w.bool(op.UseRegex)
		w.bool(op.AddPrefixSpace)
		w.bool(op.Individual)
		w.str(op.Behavior)
		w.bool(op.CaseRuns)
		w.bool(op.SuffixContract)
		w.bool(op.PunctSlash)
		w.bool(op.HanRuns)
	}
	w.str(v.PreName)
	w.u32(uint32(len(v.Templates)))
	for _, t := range v.Templates {
		w.str(t.Name)
		w.str(t.Body)
	}
	// The optional tails, each written only when it holds something (or a
	// later one follows), so every vocabulary without one writes the bytes it
	// wrote before. First one flag per op, then the stop ids.
	tools := len(v.Tools.Calls) > 0
	tail := len(v.Stop) > 0 || tools
	for _, op := range v.Pre {
		tail = tail || op.PunctNoNewline
	}
	if tail {
		w.u32(uint32(len(v.Pre)))
		for _, op := range v.Pre {
			w.bool(op.PunctNoNewline)
		}
	}
	if len(v.Stop) > 0 || tools {
		w.u32(uint32(len(v.Stop)))
		for _, id := range v.Stop {
			w.i32(id)
		}
	}
	// Then the tool-call form, as JSON: a record of templates that grows
	// with what forms state, read by name.
	if tools {
		// Strings, lists of them and a string map: nothing in it fails to
		// marshal.
		b, _ := json.Marshal(v.Tools)
		w.str(string(b))
	}
	return w.b
}

func decodeVocab(b []byte) (*Vocab, error) {
	if len(b) == 0 {
		return nil, nil
	}
	r := &rbuf{b: b}
	v := &Vocab{Kind: VocabKind(r.u8())}
	if n := r.count(4); n > 0 {
		v.Tokens = make([]string, n)
		for i := range v.Tokens {
			v.Tokens[i] = r.str()
		}
	}
	if n := r.count(4); n > 0 {
		v.Scores = make([]float32, n)
		for i := range v.Scores {
			v.Scores[i] = r.f32()
		}
	}
	if n := r.count(1); n > 0 {
		v.Kinds = make([]TokenKind, n)
		for i := range v.Kinds {
			v.Kinds[i] = TokenKind(r.u8())
		}
	}
	if n := r.count(8); n > 0 {
		v.Merges = make([]Merge, n)
		for i := range v.Merges {
			v.Merges[i].Left = r.str()
			v.Merges[i].Right = r.str()
		}
	}
	if n := r.count(13); n > 0 {
		v.Added = make([]AddedToken, n)
		for i := range v.Added {
			a := &v.Added[i]
			a.ID = r.i32()
			a.Content = r.str()
			a.Special, a.LStrip, a.RStrip = r.bool(), r.bool(), r.bool()
			a.Normalized, a.SingleWord = r.bool(), r.bool()
		}
	}
	for _, p := range []*int32{&v.BOS, &v.EOS, &v.Unk, &v.Pad, &v.Sep, &v.Mask} {
		*p = r.i32()
	}
	for _, p := range []*bool{&v.AddBOS, &v.AddEOS, &v.AddSpacePrefix, &v.ByteFallback, &v.IgnoreMerges} {
		*p = r.bool()
	}
	if n := r.count(12); n > 0 {
		v.Pre = make([]PreOp, n)
		for i := range v.Pre {
			op := &v.Pre[i]
			op.Kind = PreOpKind(r.u8())
			op.Contractions, op.CaseFold, op.LetterMarks = r.bool(), r.bool(), r.bool()
			op.DigitGroup = r.u8()
			op.UseRegex, op.AddPrefixSpace, op.Individual = r.bool(), r.bool(), r.bool()
			op.Behavior = r.str()
			op.CaseRuns, op.SuffixContract, op.PunctSlash = r.bool(), r.bool(), r.bool()
			op.HanRuns = r.bool()
		}
	}
	v.PreName = r.str()
	if n := r.count(8); n > 0 {
		v.Templates = make([]ChatTemplate, n)
		for i := range v.Templates {
			v.Templates[i].Name = r.str()
			v.Templates[i].Body = r.str()
		}
	}
	if r.err == nil && len(r.b) > 0 {
		if n := r.count(1); n != len(v.Pre) {
			if r.err == nil {
				return nil, fmt.Errorf("jlm: vocab tail has %d op flags for %d ops", n, len(v.Pre))
			}
		} else {
			for i := range v.Pre {
				v.Pre[i].PunctNoNewline = r.bool()
			}
		}
	}
	if r.err == nil && len(r.b) > 0 {
		if n := r.count(4); n > 0 {
			v.Stop = make([]int32, n)
			for i := range v.Stop {
				v.Stop[i] = r.i32()
			}
		}
	}
	if r.err == nil && len(r.b) > 0 {
		if s := r.str(); r.err == nil {
			if err := json.Unmarshal([]byte(s), &v.Tools); err != nil {
				return nil, fmt.Errorf("jlm: the vocabulary's tool form: %w", err)
			}
			if err := v.Tools.Valid(); err != nil {
				return nil, err
			}
		}
	}
	if r.err != nil {
		return nil, r.err
	}
	for _, id := range v.Stop {
		if id < 0 || int(id) >= len(v.Tokens) {
			return nil, fmt.Errorf("jlm: stop id %d is outside a %d-token vocabulary", id, len(v.Tokens))
		}
	}
	// The parallel arrays must be parallel, checked here rather than at the
	// first out-of-range index deep into a prompt.
	if len(v.Kinds) != 0 && len(v.Kinds) != len(v.Tokens) {
		return nil, fmt.Errorf("jlm: vocab has %d tokens and %d kinds", len(v.Tokens), len(v.Kinds))
	}
	if len(v.Scores) != 0 && len(v.Scores) != len(v.Tokens) {
		return nil, fmt.Errorf("jlm: vocab has %d tokens and %d scores", len(v.Tokens), len(v.Scores))
	}
	return v, nil
}

// ngramValid refuses an n-gram description a reader would index out of: the
// hash's constants one per gram position and one range per head, each range
// non-empty, and a row width a whole number of the table's blocks is checked
// against the table where the reader opens it.
func ngramValid(c *Config) error {
	if c.NGramBlock == 0 {
		if c.NGram != 0 || len(c.NGramMul) != 0 || len(c.NGramOff) != 0 || len(c.NGramVoc) != 0 {
			return fmt.Errorf("jlm: n-gram constants with no n-gram block")
		}
		return nil
	}
	heads := uint64(c.NGram-1) * uint64(c.NGramHeadsPer)
	switch {
	case c.NGramBlock > c.NLayer:
		return fmt.Errorf("jlm: n-gram block %d of %d", c.NGramBlock, c.NLayer)
	case c.NGram < 2 || c.NGramHeadsPer == 0 || c.NGramRowDim == 0 || c.NGramConv == 0:
		return fmt.Errorf("jlm: n-gram %d-grams, %d heads each, rows of %d, %d taps",
			c.NGram, c.NGramHeadsPer, c.NGramRowDim, c.NGramConv)
	case uint64(len(c.NGramMul)) != uint64(c.NGram):
		return fmt.Errorf("jlm: %d n-gram multipliers for %d-grams", len(c.NGramMul), c.NGram)
	case uint64(len(c.NGramOff)) != heads || uint64(len(c.NGramVoc)) != heads:
		return fmt.Errorf("jlm: %d offsets and %d vocabularies for %d n-gram heads",
			len(c.NGramOff), len(c.NGramVoc), heads)
	case c.HCMult < 2:
		return fmt.Errorf("jlm: an n-gram block with %d residual streams", c.HCMult)
	}
	for h, v := range c.NGramVoc {
		if v == 0 || c.NGramOff[h]+v < c.NGramOff[h] {
			return fmt.Errorf("jlm: n-gram head %d range [%d, +%d)", h, c.NGramOff[h], v)
		}
	}
	return nil
}

func encodeStrs(w *wbuf, ss []string) {
	w.u32(uint32(len(ss)))
	for _, x := range ss {
		w.str(x)
	}
}

func decodeStrs(r *rbuf) []string {
	var out []string
	for range r.count(4) {
		out = append(out, r.str())
	}
	return out
}

func encodeDecisionLabels(w *wbuf, l *DecisionLabels) {
	encodeStrs(w, l.Codes)
	encodeStrs(w, l.WideCodes)
	encodeStrs(w, l.Pool)
	w.u32(l.MaxOptions)
	encodeStrs(w, l.NoulTrue)
	encodeStrs(w, l.NoulFalse)
	w.u32(l.NoulRatings)
	for _, b := range []bool{l.KeyCodes, l.Spaced, l.ShowCodes, l.NoulTrueFirst, l.ScoreByKey, l.Reversed, l.SortKeys,
		l.IDInstructions, l.EntropyConfidence, l.SortedLegend} {
		w.bool(b)
	}
}

func decodeDecisionLabels(r *rbuf) *DecisionLabels {
	l := &DecisionLabels{Codes: decodeStrs(r), WideCodes: decodeStrs(r), Pool: decodeStrs(r), MaxOptions: r.u32()}
	l.NoulTrue, l.NoulFalse, l.NoulRatings = decodeStrs(r), decodeStrs(r), r.u32()
	for _, b := range []*bool{&l.KeyCodes, &l.Spaced, &l.ShowCodes, &l.NoulTrueFirst, &l.ScoreByKey, &l.Reversed,
		&l.SortKeys, &l.IDInstructions, &l.EntropyConfidence, &l.SortedLegend} {
		*b = r.bool()
	}
	return l
}
