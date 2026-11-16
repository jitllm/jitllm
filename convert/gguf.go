// Package convert turns a source model into a jlm container. It is the only
// place that knows two formats at once: the source's name vocabulary, type
// numbers, key names, architecture switch and tokenizer keys all live here,
// and jlm.Write is handed a jlm.Source in the container's own terms, so a
// .jlm is readable without the source's specification.
package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// FromGGUF writes src, a GGUF, as dst, a jlm container.
func FromGGUF(src, dst string, fp jlm.Fingerprint, opts ...Option) (*jlm.Header, error) {
	return FromGGUFs(src, "", dst, fp, opts...)
}

// FromGGUFs writes one container from a text model and, optionally, its vision
// tower. The mmproj's separate file is a property of the input format and ends
// here; the text/tower width match that libmtmd checks at runtime is checked
// at conversion, where there is a file in front of a person.
func FromGGUFs(src, mmproj, dst string, fp jlm.Fingerprint, opts ...Option) (*jlm.Header, error) {
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	// Refused rather than ignored: a GGUF's weights are already quantized.
	if o.q8 {
		return nil, fmt.Errorf("convert: WithQ8 quantizes a safetensors model; %s is a GGUF", src)
	}
	f, err := gguf.Open(src)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := headAlone(f, src); err != nil && o.head == "" {
		return nil, err
	}
	if o.head != "" {
		h, err := gguf.Open(o.head)
		if err != nil {
			return nil, err
		}
		defer h.Close()
		if f, err = withHead(f, h, src, o.head); err != nil {
			return nil, err
		}
	}
	var vf *meta.File
	if mmproj != "" {
		if vf, err = gguf.Open(mmproj); err != nil {
			return nil, err
		}
		defer vf.Close()
	}
	s, err := buildGGUF(f, vf, src, mmproj, o)
	if err != nil {
		return nil, err
	}
	// Stamp which build wrote this: most of a container's content is converter
	// decisions the format version cannot catch. See jlm.Fingerprint.Writer.
	if fp.Writer == "" {
		fp.Writer = jlm.WriterID()
	}
	return jlm.Write(dst, s, fp)
}

// buildGGUF is the container's source for a text model f and, when vf is
// not nil, its vision tower, named src and mmproj in messages.
func buildGGUF(f, vf *meta.File, src, mmproj string, o options) (*jlm.Source, error) {
	if err := o.stateAbsentKeys(f); err != nil {
		return nil, err
	}
	// A tower converted alone carries its audio half, as a joined one does.
	alone := f
	if vf == nil {
		f = withoutAudio(f)
	}
	s, err := sourceOf(f)
	if err != nil {
		return nil, err
	}
	if alone != f && s.Vision != nil {
		aud, ats, err := audioOf(alone, s.Vision.NLayer)
		if err != nil {
			return nil, err
		}
		if aud != nil {
			s.Vision.Audio = aud
			s.Tensors = append(s.Tensors, ats...)
		}
	}
	if err := o.applyModelConfig(f, s, src); err != nil {
		return nil, err
	}
	if vf != nil {
		if a := vf.Arch(); a != towerArch {
			return nil, fmt.Errorf("convert: %s is architecture %q, not a vision tower", mmproj, a)
		}
		if f.Arch() == towerArch {
			return nil, fmt.Errorf("convert: %s is itself a vision tower; the first "+
				"argument is the TEXT model", src)
		}
		af := vf
		vf = withoutAudio(vf)
		vs, err := sourceOf(vf)
		if err != nil {
			return nil, err
		}
		if vs.Vision == nil {
			return nil, fmt.Errorf("convert: %s carries no vision description", mmproj)
		}
		vp, err := projectorFor(vs.Vision.Projector)
		if err != nil {
			return nil, err
		}
		if vs.Vision.ProjDim != s.Config.NEmbd {
			return nil, fmt.Errorf("convert: %s projects to %d and %s wants %d; "+
				"these two files do not belong together",
				mmproj, vs.Vision.ProjDim, src, s.Config.NEmbd)
		}
		// Only the tower's description and its tensors come across; its Config
		// was a stand-in for a file with no language model in it.
		s.Vision = vs.Vision
		for i := range vs.Tensors {
			if !vs.Tensors[i].Role.Vision() {
				return nil, fmt.Errorf("convert: %s carries %q, which is not a tower "+
					"tensor", mmproj, vs.Tensors[i].Name)
			}
		}
		s.Tensors = append(s.Tensors, vs.Tensors...)
		// The audio half, its blocks after the vision blocks.
		aud, ats, err := audioOf(af, s.Vision.NLayer)
		if err != nil {
			return nil, err
		}
		if aud != nil {
			if aud.ProjDim != s.Config.NEmbd {
				return nil, fmt.Errorf("convert: %s's audio projects to %d and %s wants %d", mmproj, aud.ProjDim,
					src, s.Config.NEmbd)
			}
			s.Vision.Audio = aud
			s.Tensors = append(s.Tensors, ats...)
		}
		if err := transformTower(s, nil, vp, "joined"); err != nil {
			return nil, err
		}
		if err := towerFFNByShape(s); err != nil {
			return nil, err
		}
		if err := o.applyProcessorConfig(s.Vision); err != nil {
			return nil, err
		}
		if s.Vision.Tower, err = statedTower(s, o); err != nil {
			return nil, err
		}
	}
	if err := o.applyChatTemplate(s); err != nil {
		return nil, err
	}
	stateToolForm(s)
	// The blocks' graphs last: over the publisher's config where one was
	// applied, and the tower whose taps the first blocks add.
	g, err := ggufGraph(f.Arch())
	if err != nil {
		return nil, err
	}
	if g != nil {
		if err := StateBlockGraphs(&Config{Config: s.Config}, s, g, ggufKeys(f)); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// sourceOf translates a parsed source model into the container's own terms.
// towerArch is the architecture every vision or audio tower's GGUF (an
// mmproj) declares, whatever its projector.
const towerArch = "clip"

func sourceOf(f *meta.File) (*jlm.Source, error) {
	vocab, err := readTensorNames()
	if err != nil {
		return nil, err
	}
	s := &jlm.Source{}
	var c *Config
	var names *towerVocab
	var proj *projStates
	tower := f.Arch() == towerArch
	if tower {
		v, p, err := visionOf(f)
		proj = p
		if err != nil {
			return nil, err
		}
		if names, err = towerVocabOf(v, vocab); err != nil {
			return nil, err
		}
		// A tower still needs a Config, because a container always has one; its
		// language-model fields are the tower's where they mean the same thing.
		s.Vision = v
		c = &Config{Arch: ArchCLIP, Stated: &Statement{}, Config: &jlm.Config{
			NLayer: v.NLayer, NEmbd: v.NEmbd, NHead: v.NHead,
			NKVHead: v.NHead, HeadDim: v.NEmbd / max32(v.NHead, 1), NFFN: v.NFFN,
			RMSEps: v.Eps, EmbdScale: 1, AttnFactor: 1, Flags: v.Flags,
		}}
	} else {
		var err error
		if c, err = configOf(f); err != nil {
			return nil, err
		}
	}
	s.Config = c.Config
	voc, err := vocabOf(f)
	if err != nil {
		return nil, err
	}
	s.Vocab = voc

	s.Tensors = make([]jlm.Tensor, 0, len(f.Tensors))
	// Nemotron-H's one-mixer layers merge into blocks (convert/nemotronh.go).
	var nemo *mixerPlan
	if !tower && c.Stated.mergeMixers {
		if nemo, err = mixerPlanOf(f); err != nil {
			return nil, fmt.Errorf("convert: %w", err)
		}
	}
	experts := map[bankKey][]*jlm.Tensor{}
	lscales := map[int32][2][]float32{}
	clamps := map[int32]*[28]float32{}
	var alphas, fused int
	scales := map[string]float32{}
	renames := towerNames(f, proj)
	stored := func(n string) bool { _, ok := f.Get(n); return ok }
	moved, err := retargets(f.Arch(), stored)
	if err != nil {
		return nil, err
	}
	for i := range f.Tensors {
		t := &f.Tensors[i]
		if r, ok := renames[t.Name]; ok {
			tt := *t
			tt.Name = r
			t = &tt
		}
		if len(t.Dims) == 0 || len(t.Dims) > 4 {
			return nil, fmt.Errorf("convert: %s has %d dimensions", t.Name, len(t.Dims))
		}
		// A block storing beta and alpha apart (Qwen3.5's re-cut of qwen3next's
		// file) has them fused back into ssm_ba before the name table sees
		// them (fuseBetaAlpha); a block storing beta alone keeps it.
		if !tower && splitBetaAlpha(t.Name) && pairedBetaAlpha(f, t.Name) {
			if strings.HasSuffix(t.Name, ".ssm_alpha.weight") {
				alphas++
				continue // fused into its layer's ssm_beta below
			}
			b, ok := textBlock(t.Name)
			if !ok {
				return nil, fmt.Errorf("convert: %q: no block index", t.Name)
			}
			e, err := fuseBetaAlpha(f, t, c, b)
			if err != nil {
				return nil, err
			}
			fused++
			s.Tensors = append(s.Tensors, e)
			continue
		}
		if tower && names.unreadName(t.Name) {
			continue
		}
		// InternViT's layer scales have no role: they are folded into the
		// matrices they follow once every tensor is in (foldLayerScales).
		if tower {
			// Gemma 4's clipped-linear bounds are scalars with no role: they
			// are gathered into each block's RoleVClamp vector.
			if b, at, ok := clampOf(t.Name); ok {
				e := jlm.Tensor{NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name}
				copy(e.Dims[:], t.Dims)
				if e.Type, err = typeOf(t.Type); err != nil {
					return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
				}
				x, err := vecOf(&e)
				if err != nil {
					return nil, err
				}
				if len(x) != 1 {
					return nil, fmt.Errorf("convert: %s is %d values, want one bound", t.Name, len(x))
				}
				if clamps[b] == nil {
					clamps[b] = newClamps()
				}
				clamps[b][at] = x[0]
				continue
			}
			if b, w, ok := layerScaleOf(t.Name); ok {
				e := jlm.Tensor{NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name}
				copy(e.Dims[:], t.Dims)
				if e.Type, err = typeOf(t.Type); err != nil {
					return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
				}
				ls, err := vecOf(&e)
				if err != nil {
					return nil, err
				}
				pair := lscales[b]
				pair[w-1] = ls
				lscales[b] = pair
				continue
			}
		}
		// DeepSeek V4's token-id tables are I32, which is not a weight type
		// here: carried as F32 ids (hashExpertTable).
		if !tower && strings.HasSuffix(t.Name, ".ffn_gate_tid2eid.weight") {
			_, block, _, err := vocab.identify(t.Name)
			if err != nil {
				return nil, err
			}
			b, err := hashExpertTable(t, f.Bytes(t), c)
			if err != nil {
				return nil, err
			}
			e := jlm.Tensor{Role: jlm.RoleHashExperts, Block: block, Index: -1, Type: jlm.TypeF32,
				NDim: 2, Data: b, Name: t.Name}
			copy(e.Dims[:], t.Dims)
			s.Tensors = append(s.Tensors, e)
			continue
		}
		// A ternary projection's optional per-tensor scale is folded into
		// its weight's blocks (ternaryTensors).
		if w, ok := ternaryScaleOf(t.Name); ok && !tower && c.Stated.actQuant == jlm.ActQuantRowInt8 {
			e := jlm.Tensor{NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name}
			copy(e.Dims[:], t.Dims)
			if e.Type, err = typeOf(t.Type); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
			}
			x, err := vecOf(&e)
			if err != nil {
				return nil, err
			}
			if len(x) != 1 {
				return nil, fmt.Errorf("convert: %s is %d values, want one scale", t.Name, len(x))
			}
			scales[w] = x[0]
			continue
		}
		var role jlm.Role
		var block, index int32
		// The newer families' tower tensors: a fused q|k|v split, and the
		// deepstack mergers (convert/visiontowers.go).
		if tower {
			ts, ok, ferr := familyTowerTensors(f, s.Vision, proj, t)
			if ferr != nil {
				return nil, ferr
			}
			if ok {
				s.Tensors = append(s.Tensors, ts...)
				continue
			}
		}
		if tower {
			role, block, index, err = names.roleOf(t.Name)
		} else {
			role, block, index, err = vocab.identify(t.Name)
		}
		if err != nil {
			return nil, err
		}
		if !role.Vision() {
			// A second input norm in a block is a layout every block shares.
			if (role == jlm.RoleAttnNorm2 || role == jlm.RoleAttnNorm2Bias) && !stored("blk.0.attn_norm_2.weight") {
				return nil, fmt.Errorf("convert: %s: attn_norm_2 in a block when block 0 has none", t.Name)
			}
			// The text architecture's renames; a tower's roles are not its to
			// rewrite, and a merged container has both.
			if r, ok := moved[role]; ok {
				role = r
			}
			if nemo != nil {
				if role, block, err = nemo.remap(role, block); err != nil {
					return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
				}
			}
		}
		role, block = headSeam(role, block, c)
		data := f.Bytes(t)
		srcType := t.Type
		// A ternary type with no code here (TQ1_0, I2_S) is re-packed as
		// TQ2_0 or TQ2_F32, exactly (convert/bitnet.go).
		if b, ty, ok, terr := ternaryGGUF(t.Type, data, elemsOf(t.Dims)); terr != nil {
			return nil, fmt.Errorf("convert: %s (%s): %w", t.Name, t.Type, terr)
		} else if ok {
			data, srcType = b, ty
		}
		ty, err := typeOf(srcType)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
		}
		e := jlm.Tensor{
			Role: role, Block: block, Index: index, Type: ty,
			NDim: uint8(len(t.Dims)), Data: data, Name: t.Name,
		}
		copy(e.Dims[:], t.Dims)
		// A per-expert matrix (Mixtral-era blk.N.ffn_gate.E.weight) is held back
		// to be stacked into its bank: the device reads a mixture only through
		// its banks, so unstacked experts kept every such block on the host.
		if bank, ok := expertBank(role); ok && index >= 0 {
			// The index comes from a tensor name, so it is bounded by the config
			// before it sizes anything (RULE 9).
			if uint32(index) >= c.NExpert {
				return nil, fmt.Errorf("convert: %s: expert %d of %d", t.Name, index, c.NExpert)
			}
			k := bankKey{bank, block}
			if experts[k] == nil {
				experts[k] = make([]*jlm.Tensor, c.NExpert)
			}
			if experts[k][index] != nil {
				return nil, fmt.Errorf("convert: %s: expert %d twice", t.Name, index)
			}
			x := e
			experts[k][index] = &x
			continue
		}
		s.Tensors = append(s.Tensors, e)
	}
	bs, err := stackGGUFExperts(experts, c)
	if err != nil {
		return nil, err
	}
	s.Tensors = append(s.Tensors, bs...)
	if s.Tensors, err = nextnTensors(s.Tensors, c); err != nil {
		return nil, err
	}
	if err := foldPostRopeQKNorm(s); err != nil {
		return nil, err
	}
	if err := statedTensors(c, s, ggufEnv(f)); err != nil {
		return nil, err
	}
	if !tower {
		if err := scanUnitNorms(c, f, s); err != nil {
			return nil, err
		}
		if err := statedGathers(c, s, f); err != nil {
			return nil, err
		}
	}
	if err := ternaryTensors(c, s, scales); err != nil {
		return nil, err
	}
	// An alpha with no beta was skipped above and never fused: a layer whose
	// decay gate silently vanished. Every one must have been consumed.
	if alphas != fused {
		return nil, fmt.Errorf("convert: %d ssm_alpha tensors and %d ssm_beta tensors fused them; "+
			"every layer needs both", alphas, fused)
	}
	if tower {
		// The tower's two transforms: a 4-D conv kernel whose stride equals its
		// kernel is a matmul, and a verbatim F16 matrix is re-laid-out into the
		// engine's packed layout. See convert/vision.go.
		if err := clampTensors(s, clamps); err != nil {
			return nil, err
		}
		if err := foldLayerScales(s, lscales); err != nil {
			return nil, err
		}
		if err := checkDeepstack(f, s, proj); err != nil {
			return nil, err
		}
		if err := transformTower(s, f, proj, "source"); err != nil {
			return nil, err
		}
		if err := prepareTower(s, proj); err != nil {
			return nil, err
		}
	}
	// A tower's own file states its blocks' activations in its tower.
	if !tower {
		if err := StateActivations(c, s); err != nil {
			return nil, err
		}
	}
	g, err := ggufGraph(f.Arch())
	if err != nil {
		return nil, err
	}
	if err := stateGraphStreams(c.Config, s, g, ggufKeys(f)); err != nil {
		return nil, err
	}
	storeRowTables(c.Config, s)
	if err := StateSelection(c.Config, sourceStores(s)); err != nil {
		return nil, err
	}
	if err := expertsFit(c, s); err != nil {
		return nil, err
	}
	if err := valueWidthFits(c); err != nil {
		return nil, err
	}
	if err := geometryFits(c); err != nil {
		return nil, err
	}
	if err := ropeFits(c); err != nil {
		return nil, err
	}
	StateDelta(c.Config, sourceStores(s))
	if err := StateRevision(c); err != nil {
		return nil, err
	}
	if err := StateMarkerPrompt(s, int64(f.UintKey("decision.max_tokens", 0))); err != nil {
		return nil, err
	}
	return s, nil
}

// expertBank is the bank a per-expert role is stacked into.
func expertBank(r jlm.Role) (jlm.Role, bool) {
	switch r {
	case jlm.RoleExpGate:
		return jlm.RoleExpGateBank, true
	case jlm.RoleExpUp:
		return jlm.RoleExpUpBank, true
	case jlm.RoleExpDown:
		return jlm.RoleExpDownBank, true
	}
	return 0, false
}

// bankStem names a stacked bank the way a modern GGUF names it.
func bankStem(r jlm.Role) (string, bool) {
	switch r {
	case jlm.RoleExpGateBank:
		return "ffn_gate_exps", true
	case jlm.RoleExpUpBank:
		return "ffn_up_exps", true
	case jlm.RoleExpDownBank:
		return "ffn_down_exps", true
	}
	return "", false
}

func bankStemOf(r jlm.Role) string {
	s, _ := bankStem(r)
	return s
}

// stackGGUFExperts is stackBanks for a GGUF that ships its experts one matrix
// at a time. The bytes are still the source's row-major blocks, so a bank is
// the experts back to back -- the same concatenation safetensors gets, and the
// same refusal of a missing expert.
//
// A bank whose experts disagree on type is stacked from its sheets widened
// to F32 (widenedF32): a bank is one matrix of one type.
func stackGGUFExperts(experts map[bankKey][]*jlm.Tensor, c *Config) ([]jlm.Tensor, error) {
	byBlock := map[int32]map[bankKey][]*jlm.Tensor{}
	for k, v := range experts {
		if byBlock[k.block] == nil {
			byBlock[k.block] = map[bankKey][]*jlm.Tensor{}
		}
		byBlock[k.block][k] = v
	}
	blocks := make([]int32, 0, len(byBlock))
	for b := range byBlock {
		blocks = append(blocks, b)
	}
	sort.Slice(blocks, func(i, j int) bool { return blocks[i] < blocks[j] })
	var out []jlm.Tensor
	for _, b := range blocks {
		set := byBlock[b]
		for k, sheets := range set {
			if !mixedTypes(sheets) {
				continue
			}
			wide := make([]*jlm.Tensor, len(sheets))
			for e, x := range sheets {
				w, err := widenedF32(x)
				if err != nil {
					return nil, err
				}
				wide[e] = w
			}
			set[k] = wide
		}
		bs, err := stackBanks(set, c)
		if err != nil {
			return nil, err
		}
		for i := range bs {
			bs[i].Name = fmt.Sprintf("blk.%d.%s.weight", b, bankStemOf(bs[i].Role))
		}
		out = append(out, bs...)
	}
	return out, nil
}

// mixedTypes reports whether one bank's experts, every one present,
// disagree on type. Two banks may differ -- a down projection is often
// quantized wider than its gate -- and that is not mixed. A missing expert
// is not mixed either: stackBanks refuses it by name.
func mixedTypes(sheets []*jlm.Tensor) bool {
	for _, x := range sheets {
		if x == nil || sheets[0] == nil {
			return false
		}
		if x.Type != sheets[0].Type {
			return true
		}
	}
	return false
}

// widenedF32 is expert sheet x read as F32, decoded when its bytes are
// asked for: a bank holds one type, and F32 holds every type's values
// exactly.
func widenedF32(x *jlm.Tensor) (*jlm.Tensor, error) {
	if x == nil || x.Type == jlm.TypeF32 {
		return x, nil
	}
	src, ok := sourceType(x.Type)
	if !ok || x.Stored {
		return nil, fmt.Errorf("convert: %s: a sheet of %v cannot be widened", x.Name, x.Type)
	}
	w := *x
	n := int(x.Dims[0] * x.Dims[1])
	w.Type, w.Data, w.Load = jlm.TypeF32, nil, func() ([]byte, error) {
		b, err := x.Bytes()
		if err != nil {
			return nil, err
		}
		f := make([]float32, n)
		if err := quant.Dequant32(src, b, f); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", x.Name, err)
		}
		d := make([]byte, 4*n)
		for i, v := range f {
			binary.LittleEndian.PutUint32(d[4*i:], math.Float32bits(v))
		}
		return d, nil
	}
	return &w, nil
}

// prepareTower reshapes the patch embedding and quantizes every matrix the
// tower multiplies by, so that a container's vision weights are all in this
// format's packed layout.
func prepareTower(s *jlm.Source, p *projStates) error {
	v := s.Vision
	if v == nil {
		return fmt.Errorf("convert: a clip file with no vision description")
	}
	if err := foldPatchPlanes(s); err != nil {
		return err
	}
	if err := transformTower(s, nil, p, "prepare"); err != nil {
		return err
	}
	if err := padFFN(s); err != nil {
		return err
	}
	for i := range s.Tensors {
		switch s.Tensors[i].Role {
		case jlm.RoleVPatchEmbd, jlm.RoleVPatchEmbd2:
			if err := reshapePatchEmbd(&s.Tensors[i], int(v.PatchSize), int(v.NEmbd)); err != nil {
				return err
			}
		case jlm.RoleVMergeConv:
			if err := reshapeMergeConv(&s.Tensors[i], int(v.NEmbd)); err != nil {
				return err
			}
		}
	}
	if err := transformTower(s, nil, p, "finish"); err != nil {
		return err
	}
	return quantizeTower(s.Tensors)
}

func max32(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}

// f32AsBytes reinterprets a float32 slice as the little-endian bytes a
// container stores (the engine only runs on little-endian hosts).
func f32AsBytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}
