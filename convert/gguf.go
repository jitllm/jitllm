// Package convert turns a source model into a jlm container. It is the only
// place that knows two formats at once: the source's name vocabulary, type
// numbers, key names, architecture switch and tokenizer keys all live here,
// and jlm.Write is handed a jlm.Source in the container's own terms, so a
// .jlm is readable without the source's specification.
package convert

import (
	"fmt"
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
	s, err := sourceOf(f)
	if err != nil {
		return nil, err
	}
	if vf != nil {
		if a := vf.Arch(); a != "clip" {
			return nil, fmt.Errorf("convert: %s is architecture %q, not a vision tower", mmproj, a)
		}
		if s.Config.Arch == jlm.ArchCLIP {
			return nil, fmt.Errorf("convert: %s is itself a vision tower; the first "+
				"argument is the TEXT model", src)
		}
		vs, err := sourceOf(vf)
		if err != nil {
			return nil, err
		}
		if vs.Vision == nil {
			return nil, fmt.Errorf("convert: %s carries no vision description", mmproj)
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
		if s.Vision.Projector == jlm.ProjPixtral {
			if err := imgBreakRow(s); err != nil {
				return nil, err
			}
		}
	}
	if err := o.applyChatTemplate(s); err != nil {
		return nil, err
	}
	return s, nil
}

// sourceOf translates a parsed source model into the container's own terms.
func sourceOf(f *meta.File) (*jlm.Source, error) {
	s := &jlm.Source{}
	tower := f.Arch() == "clip"
	if tower {
		v, err := visionOf(f)
		if err != nil {
			return nil, err
		}
		// A tower still needs a Config, because a container always has one; its
		// language-model fields are the tower's where they mean the same thing.
		s.Vision = v
		s.Config = &jlm.Config{
			Arch: jlm.ArchCLIP, NLayer: v.NLayer, NEmbd: v.NEmbd, NHead: v.NHead,
			NKVHead: v.NHead, HeadDim: v.NEmbd / max32(v.NHead, 1), NFFN: v.NFFN,
			RMSEps: v.Eps, EmbdScale: 1, AttnFactor: 1, Flags: v.Flags,
		}
	} else {
		c, err := configOf(f)
		if err != nil {
			return nil, err
		}
		s.Config = c
	}
	voc, err := vocabOf(f)
	if err != nil {
		return nil, err
	}
	s.Vocab = voc

	s.Tensors = make([]jlm.Tensor, 0, len(f.Tensors))
	// Nemotron-H's one-mixer layers merge into blocks (convert/nemotronh.go).
	var nemo *nemotronPlan
	if !tower && s.Config.Arch == jlm.ArchNemotronH {
		if nemo, err = nemotronPlanOf(f); err != nil {
			return nil, fmt.Errorf("convert: %w", err)
		}
	}
	experts := map[bankKey][]*jlm.Tensor{}
	lscales := map[int32][2][]float32{}
	clamps := map[int32]*[28]float32{}
	var alphas, fused int
	renames := towerNames(f, s.Vision)
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
		// Qwen3.5's re-cut of qwen3next's file is undone before the name table
		// sees it: beta and alpha are fused back into ssm_ba (fuseBetaAlpha).
		if !tower && (s.Config.Arch == jlm.ArchQwen3Next || s.Config.Arch == jlm.ArchQwen4Exp) {
			if splitBetaAlpha(t.Name) && strings.HasSuffix(t.Name, ".ssm_alpha.weight") {
				alphas++
				continue // fused into its layer's ssm_beta below
			}
			if splitBetaAlpha(t.Name) {
				b, ok := textBlock(t.Name)
				if !ok {
					return nil, fmt.Errorf("convert: %q: no block index", t.Name)
				}
				e, err := fuseBetaAlpha(f, t, s.Config, b)
				if err != nil {
					return nil, err
				}
				fused++
				s.Tensors = append(s.Tensors, e)
				continue
			}
		}
		if tower && towerUnread(s.Vision, t.Name) {
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
		// here: carried as F32 ids (ds4HashTable).
		if !tower && s.Config.Arch == jlm.ArchDeepseek4 && strings.HasSuffix(t.Name, ".ffn_gate_tid2eid.weight") {
			_, block, _, err := identify(t.Name)
			if err != nil {
				return nil, err
			}
			b, err := ds4HashTable(t, f.Bytes(t), s.Config)
			if err != nil {
				return nil, err
			}
			e := jlm.Tensor{Role: jlm.RoleHashExperts, Block: block, Index: -1, Type: jlm.TypeF32,
				NDim: 2, Data: b, Name: t.Name}
			copy(e.Dims[:], t.Dims)
			s.Tensors = append(s.Tensors, e)
			continue
		}
		var role jlm.Role
		var block, index int32
		// The newer families' tower tensors: a fused q|k|v split, and the
		// deepstack mergers (convert/visiontowers.go).
		if tower {
			ts, ok, ferr := familyTowerTensors(f, s.Vision, t)
			if ferr != nil {
				return nil, ferr
			}
			if ok {
				s.Tensors = append(s.Tensors, ts...)
				continue
			}
		}
		if tower {
			role, block, index, err = towerRoleOf(s.Vision, t.Name)
			if r, ok := familyProjectorRole(s.Vision.Projector, t.Name); ok {
				role, block, index, err = r, jlm.DenseBlock, -1, nil
			}
		} else {
			role, block, index, err = identify(t.Name)
		}
		if err != nil {
			return nil, err
		}
		if !role.Vision() {
			// retarget is the text architecture's rename table; a tower's roles
			// are not its to rewrite, and a merged container has both.
			role = retarget(s.Config.Arch, role)
			if nemo != nil {
				if role, block, err = nemo.remap(role, block); err != nil {
					return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
				}
			}
			if s.Config.Arch == jlm.ArchFalcon {
				if role, err = falconNorms(f, role); err != nil {
					return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
				}
			}
		}
		ty, err := typeOf(t.Type)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
		}
		e := jlm.Tensor{
			Role: role, Block: block, Index: index, Type: ty,
			NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name,
		}
		copy(e.Dims[:], t.Dims)
		// A per-expert matrix (Mixtral-era blk.N.ffn_gate.E.weight) is held back
		// to be stacked into its bank: the device reads a mixture only through
		// its banks, so unstacked experts kept every such block on the host.
		if bank, ok := expertBank[role]; ok && index >= 0 {
			// The index comes from a tensor name, so it is bounded by the config
			// before it sizes anything (RULE 9).
			if uint32(index) >= s.Config.NExpert {
				return nil, fmt.Errorf("convert: %s: expert %d of %d", t.Name, index, s.Config.NExpert)
			}
			k := bankKey{bank, block}
			if experts[k] == nil {
				experts[k] = make([]*jlm.Tensor, s.Config.NExpert)
			}
			if experts[k][index] != nil {
				return nil, fmt.Errorf("convert: %s: expert %d twice", t.Name, index)
			}
			x := e
			experts[k][index] = &x
			continue
		}
		// A fused weight is unfused here: phi3's attn_qkv and fused ffn_up are a
		// packing choice of the file. Once packed, a row range has no byte range
		// and the device tier cannot slice it; here the bytes are still GGUF
		// row-major, so a row range is a byte range and the split is three slices.
		sub, split, uerr := unfuse(&e, s.Config)
		if uerr != nil {
			return nil, uerr
		}
		// MLA's fused up-projection likewise, except its k half is transposed per
		// head and so cannot keep the source's block layout. See unfuseMLA.
		if !split {
			if sub, split, uerr = unfuseMLA(&e, s.Config); uerr != nil {
				return nil, uerr
			}
		}
		// Mamba-2's in_proj likewise, and its convolved channels reordered to
		// the delta rule's q | k | v. See unfuseSSD.
		if !split {
			if sub, split, uerr = unfuseSSD(&e, s.Config); uerr != nil {
				return nil, uerr
			}
		}
		if !split {
			if sub, split, uerr = unfuseLFM2(&e, s.Config); uerr != nil {
				return nil, uerr
			}
		}
		if !split {
			if sub, split, uerr = unfuseMamba1(&e, s.Config); uerr != nil {
				return nil, uerr
			}
		}
		if split {
			s.Tensors = append(s.Tensors, sub...)
			continue
		}
		s.Tensors = append(s.Tensors, e)
	}
	bs, err := stackGGUFExperts(experts, s.Config)
	if err != nil {
		return nil, err
	}
	s.Tensors = append(s.Tensors, bs...)
	if s.Tensors, err = nextnTensors(s.Tensors, s.Config); err != nil {
		return nil, err
	}
	if err := foldPostRopeQKNorm(s); err != nil {
		return nil, err
	}
	if err := gemma4Tensors(s); err != nil {
		return nil, err
	}
	if err := gemma3nTensors(s); err != nil {
		return nil, err
	}
	if !tower {
		if err := mamba1UnitNorms(f, s); err != nil {
			return nil, err
		}
		if err := apertusXIELU(f, s); err != nil {
			return nil, err
		}
	}
	if err := ds4Tensors(s); err != nil {
		return nil, err
	}
	if err := kimiK3Tensors(s); err != nil {
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
		if err := checkDeepstack(f, s); err != nil {
			return nil, err
		}
		if s.Vision.Projector == jlm.ProjGemma3nV {
			mobilenetGeom(s, mobilenetPlaces(f))
			if err := prepareGemma3nV(s); err != nil {
				return nil, err
			}
		}
		if err := prepareTower(s); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// expertBank is the bank a per-expert role is stacked into.
var expertBank = map[jlm.Role]jlm.Role{
	jlm.RoleExpGate: jlm.RoleExpGateBank,
	jlm.RoleExpUp:   jlm.RoleExpUpBank,
	jlm.RoleExpDown: jlm.RoleExpDownBank,
}

// bankStem names a stacked bank the way a modern GGUF names it.
var bankStem = map[jlm.Role]string{
	jlm.RoleExpGateBank: "ffn_gate_exps", jlm.RoleExpUpBank: "ffn_up_exps",
	jlm.RoleExpDownBank: "ffn_down_exps",
}

// stackGGUFExperts is stackBanks for a GGUF that ships its experts one matrix
// at a time. The bytes are still the source's row-major blocks, so a bank is
// the experts back to back -- the same concatenation safetensors gets, and the
// same refusal of a missing expert.
//
// A block whose experts disagree on type keeps all three roles per-expert:
// a bank is one matrix of one type, and model.build decides the layout per
// block. Such a block runs on the host (model.offerRange says why).
func stackGGUFExperts(experts map[bankKey][]*jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, error) {
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
		if mixedTypes(set) {
			keys := make([]bankKey, 0, len(set))
			for k := range set {
				keys = append(keys, k)
			}
			sort.Slice(keys, func(i, j int) bool { return keys[i].role < keys[j].role })
			for _, k := range keys {
				for _, x := range set[k] {
					out = append(out, *x)
				}
			}
			continue
		}
		bs, err := stackBanks(set, c)
		if err != nil {
			return nil, err
		}
		for i := range bs {
			bs[i].Name = fmt.Sprintf("blk.%d.%s.weight", b, bankStem[bs[i].Role])
		}
		out = append(out, bs...)
	}
	return out, nil
}

// mixedTypes reports whether the experts of any one of a block's banks, every
// one present, disagree on type. Two banks may differ -- a down projection is
// often quantized wider than its gate -- and that is not mixed. A missing
// expert is not "mixed" either: stackBanks refuses it by name.
func mixedTypes(set map[bankKey][]*jlm.Tensor) bool {
	for _, sheets := range set {
		for _, x := range sheets {
			if x == nil || sheets[0] == nil {
				return false
			}
			if x.Type != sheets[0].Type {
				return true
			}
		}
	}
	return false
}

// prepareTower reshapes the patch embedding and quantizes every matrix the
// tower multiplies by, so that a container's vision weights are all in this
// format's packed layout.
func prepareTower(s *jlm.Source) error {
	v := s.Vision
	if v == nil {
		return fmt.Errorf("convert: a clip file with no vision description")
	}
	if err := foldPatchPlanes(s); err != nil {
		return err
	}
	if v.Projector == jlm.ProjGemma3 {
		if err := transposeGemma3Projection(s); err != nil {
			return err
		}
	}
	if v.Projector == jlm.ProjPixtral {
		if err := permuteMerger(s); err != nil {
			return err
		}
	}
	if v.Projector == jlm.ProjGemma4V {
		if err := halfNeoxToNeox(s); err != nil {
			return err
		}
	}
	if err := padFFN(s); err != nil {
		return err
	}
	for i := range s.Tensors {
		switch s.Tensors[i].Role {
		case jlm.RoleVPatchEmbd:
			if err := reshapePatchEmbd(&s.Tensors[i], int(v.PatchSize), int(v.NEmbd)); err != nil {
				return err
			}
		case jlm.RoleVMergeConv:
			if err := reshapeMergeConv(&s.Tensors[i], int(v.NEmbd)); err != nil {
				return err
			}
		case jlm.RoleVProj:
			if v.Projector == jlm.ProjHunyuanVL {
				if err := reshapePointConv(&s.Tensors[i]); err != nil {
					return err
				}
			}
		}
	}
	return quantizeTower(s.Tensors)
}

func max32(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}

// unfuse splits a fused weight into the tensors the container stores, or
// reports ok=false when the weight is not one.
//
// A GGUF matrix is row-major with every row a whole number of quantization
// blocks, so a row range is a contiguous byte range -- which stops being true
// once the container packs it.
func unfuse(e *jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, bool, error) {
	if c != nil && e.Role == jlm.RoleExpUpBank && e.NDim == 3 && c.NFFNExp != 0 &&
		e.Dims[1] == 2*uint64(c.NFFNExp) {
		return splitGateUpBank(e, c)
	}
	// fusesQKV decides which archs' attn_qkv is attention's q|k|v: qwen3next's
	// linear block uses the same name for a delta-rule projection that deltaGeom
	// splits, and the row-count check below refuses a wrong split. nomic-bert
	// and the C6 blocks (with a fused bias beside it) are q|k|v row ranges.
	if c == nil || !fusesQKV(c.Arch) {
		return nil, false, nil
	}
	if e.Role == jlm.RoleAttnQKVBias {
		return unfuseBias(e, c)
	}
	var parts []struct {
		role jlm.Role
		rows uint64
	}
	switch e.Role {
	case jlm.RoleAttnQKV:
		q, kv := uint64(c.NHead)*uint64(c.HeadDim), uint64(c.NKVHead)*uint64(c.HeadDim)
		parts = []struct {
			role jlm.Role
			rows uint64
		}{{jlm.RoleAttnQ, q}, {jlm.RoleAttnK, kv}, {jlm.RoleAttnV, kv}}
	case jlm.RoleFFNUp:
		// Fused only when the tensor is twice the feed-forward width; every
		// other architecture's ffn_up is exactly NFFN rows and falls through.
		if e.NDim < 2 || e.Dims[1] != 2*uint64(c.NFFN) {
			return nil, false, nil
		}
		// swiglu takes the first half as the gate.
		parts = []struct {
			role jlm.Role
			rows uint64
		}{{jlm.RoleFFNGate, uint64(c.NFFN)}, {jlm.RoleFFNUp, uint64(c.NFFN)}}
	default:
		return nil, false, nil
	}
	if e.NDim < 2 {
		return nil, false, fmt.Errorf("convert: %s is %d-dimensional", e.Name, e.NDim)
	}
	var total uint64
	for _, p := range parts {
		total += p.rows
	}
	if total != e.Dims[1] {
		return nil, false, fmt.Errorf("convert: %s has %d rows, and its parts want %d",
			e.Name, e.Dims[1], total)
	}
	src, ok := sourceType(e.Type)
	if !ok {
		return nil, false, fmt.Errorf("convert: %s is %v, which has no source type", e.Name, e.Type)
	}
	be, bb := int(src.BlockElems()), int(src.BlockBytes())
	if be == 0 || e.Dims[0]%uint64(be) != 0 {
		return nil, false, fmt.Errorf("convert: %s rows are %d elements, not a multiple of %s's %d",
			e.Name, e.Dims[0], src, be)
	}
	rowBytes := int(e.Dims[0]) / be * bb
	out := make([]jlm.Tensor, 0, len(parts))
	off := 0
	for _, p := range parts {
		n := int(p.rows) * rowBytes
		if off+n > len(e.Data) {
			return nil, false, fmt.Errorf("convert: %s: %v wants bytes [%d,%d) of %d",
				e.Name, p.role, off, off+n, len(e.Data))
		}
		t := jlm.Tensor{Role: p.role, Block: e.Block, Index: e.Index, Type: e.Type,
			NDim: 2, Data: e.Data[off : off+n], Name: e.Name + "/" + p.role.String()}
		t.Dims[0], t.Dims[1] = e.Dims[0], p.rows
		out = append(out, t)
		off += n
	}
	return out, true, nil
}

// falconNorms resolves Falcon's two input norms onto the parallel block's
// attention and FFN norms. The names read backwards: gguf-py maps Falcon-40B's
// ln_MLP to attn_norm and ln_ATTN to attn_norm_2, and llama.cpp's falcon.cpp
// feeds attention attn_norm_2 and the FFN attn_norm. With one norm (7B) the
// parallel block shares it.
func falconNorms(f *meta.File, role jlm.Role) (jlm.Role, error) {
	if _, two := f.Get("blk.0.attn_norm_2.weight"); !two {
		if role == jlm.RoleAttnNorm2 || role == jlm.RoleAttnNorm2Bias {
			return 0, fmt.Errorf("attn_norm_2 in a block when block 0 has none")
		}
		return role, nil
	}
	switch role {
	case jlm.RoleAttnNorm:
		return jlm.RoleFFNNorm, nil
	case jlm.RoleAttnNormBias:
		return jlm.RoleFFNNormBias, nil
	case jlm.RoleAttnNorm2:
		return jlm.RoleAttnNorm, nil
	case jlm.RoleAttnNorm2Bias:
		return jlm.RoleAttnNormBias, nil
	}
	return role, nil
}

// fusesQKV says which architectures' attn_qkv is attention's q|k|v rather
// than some other block's fused projection (qwen3next's delta rule).
func fusesQKV(a jlm.Arch) bool {
	switch a {
	case jlm.ArchPhi3, jlm.ArchNomicBERT, jlm.ArchModernBERT, jlm.ArchPhi2, jlm.ArchStarcoder, jlm.ArchFalcon, jlm.ArchDBRX,
		jlm.ArchBailingMoE2,
		// glm4's q, k and v are separate tensors; its gate|up is fused, which
		// this switch also admits.
		jlm.ArchGLM4:
		return true
	}
	return false
}

// unfuseBias splits a fused q|k|v bias into its three parts. It is a vector,
// so the parts are element ranges and the split is three slices of bytes.
func unfuseBias(e *jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, bool, error) {
	q, kv := uint64(c.NHead)*uint64(c.HeadDim), uint64(c.NKVHead)*uint64(c.HeadDim)
	if e.NDim != 1 || e.Dims[0] != q+2*kv {
		return nil, false, fmt.Errorf("convert: %s is %v, want one dimension of %d (q %d + k %d + v %d)",
			e.Name, e.Dims[:e.NDim], q+2*kv, q, kv, kv)
	}
	var eb uint64
	switch e.Type {
	case jlm.TypeF32:
		eb = 4
	case jlm.TypeF16, jlm.TypeBF16:
		eb = 2
	default:
		return nil, false, fmt.Errorf("convert: %s is %v; a fused bias is a float vector", e.Name, e.Type)
	}
	if uint64(len(e.Data)) != (q+2*kv)*eb {
		return nil, false, fmt.Errorf("convert: %s holds %d bytes, want %d", e.Name, len(e.Data), (q+2*kv)*eb)
	}
	parts := []struct {
		role jlm.Role
		n    uint64
	}{{jlm.RoleAttnQBias, q}, {jlm.RoleAttnKBias, kv}, {jlm.RoleAttnVBias, kv}}
	out := make([]jlm.Tensor, 0, 3)
	off := uint64(0)
	for _, p := range parts {
		t := jlm.Tensor{Role: p.role, Block: e.Block, Index: e.Index, Type: e.Type,
			NDim: 1, Data: e.Data[off*eb : (off+p.n)*eb], Name: e.Name + "/" + p.role.String()}
		t.Dims[0] = p.n
		out = append(out, t)
		off += p.n
	}
	return out, true, nil
}

// unfuseMLA splits MLA's fused up-projection attn_kv_b ({kv_lora_rank,
// n_head*(qk_nope + v_head)}, each head's W_k above its W_v), which older
// deepseek2 GGUFs ship in place of attn_k_b/attn_v_b, into the absorbed pair.
//
// attn_v_b is W_v per head, used as it stands, so it keeps the source's bytes
// exactly. attn_k_b is W_k transposed per head, because the absorb folds it
// into the query (score = (W_k^T . q_nope) . c); a transpose moves the axis the
// quantization blocks run along, so that half is dequantized and re-stored.
func unfuseMLA(e *jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, bool, error) {
	if c == nil || e.Role != jlm.RoleAttnKVB {
		return nil, false, nil
	}
	if c.KVLoraRank == 0 {
		return nil, false, fmt.Errorf("convert: %s is a fused MLA up-projection and the "+
			"config has no kv_lora_rank", e.Name)
	}
	nh := uint64(c.NHead)
	lat := uint64(c.KVLoraRank)
	nope := uint64(c.HeadDim) - uint64(c.NRot)
	vh := uint64(c.HeadDimV)
	if vh == 0 {
		vh = uint64(c.HeadDim)
	}
	per := nope + vh
	if e.NDim < 2 || e.Dims[0] != lat || e.Dims[1] != nh*per {
		return nil, false, fmt.Errorf("convert: %s is %dx%d, want %dx%d "+
			"(%d head(s) of %d nope + %d value rows over a %d-wide latent)",
			e.Name, e.Dims[0], e.Dims[1], lat, nh*per, nh, nope, vh, lat)
	}
	src, ok := sourceType(e.Type)
	if !ok {
		return nil, false, fmt.Errorf("convert: %s is %v, which has no source type", e.Name, e.Type)
	}
	be, bb := uint64(src.BlockElems()), uint64(src.BlockBytes())
	if be == 0 || lat%be != 0 {
		return nil, false, fmt.Errorf("convert: %s rows are %d elements, not a multiple of %s's %d",
			e.Name, lat, src, be)
	}
	rowBytes := lat / be * bb
	if uint64(len(e.Data)) < nh*per*rowBytes {
		return nil, false, fmt.Errorf("convert: %s holds %d bytes, want %d",
			e.Name, len(e.Data), nh*per*rowBytes)
	}

	// attn_v_b: gather each head's value rows. The file interleaves k and v
	// per head, so this is a copy, of bytes: the quantization is untouched.
	vb := make([]byte, nh*vh*rowBytes)
	for h := uint64(0); h < nh; h++ {
		srcOff := (h*per + nope) * rowBytes
		copy(vb[h*vh*rowBytes:], e.Data[srcOff:srcOff+vh*rowBytes])
	}

	// attn_k_b: dequantize each head's W_k and write its transpose.
	kb := make([]float32, nh*lat*nope)
	row := make([]float32, lat)
	for h := uint64(0); h < nh; h++ {
		for i := uint64(0); i < nope; i++ { // i indexes W_k's rows: the nope half
			off := (h*per + i) * rowBytes
			if err := quant.Dequant32(src, e.Data[off:off+rowBytes], row); err != nil {
				return nil, false, fmt.Errorf("convert: %s head %d row %d: %w", e.Name, h, i, err)
			}
			// W_k[i][j] -> W_k^T[j][i]: sheet h, row j, column i.
			for j := uint64(0); j < lat; j++ {
				kb[h*lat*nope+j*nope+i] = row[j]
			}
		}
	}

	// The k half keeps the source's precision class: stored f32, the container
	// has no packed view, so the absorb falls out of MatVecPackedGather into a
	// per-head loop on every token. A packed source becomes Q8_0 along the new
	// k (quantizeQ8Rows, ggml's recipe); a verbatim F32/F16/BF16 source, or a
	// nope that is not a whole number of Q8_0 blocks, stays F32.
	kbType, kbData := jlm.TypeF32, f32AsBytes(kb)
	if jlm.Packed(e.Type) && nope%q8Elems == 0 {
		kbType, kbData = jlm.TypeQ8, quantizeQ8Rows(kb, int(nope), int(nh*lat))
	}
	kbT := jlm.Tensor{Role: jlm.RoleAttnKB, Block: e.Block, Index: e.Index,
		Type: kbType, NDim: 3, Data: kbData,
		Name: e.Name + "/" + jlm.RoleAttnKB.String()}
	kbT.Dims[0], kbT.Dims[1], kbT.Dims[2] = nope, lat, nh
	vbT := jlm.Tensor{Role: jlm.RoleAttnVB, Block: e.Block, Index: e.Index,
		Type: e.Type, NDim: 3, Data: vb,
		Name: e.Name + "/" + jlm.RoleAttnVB.String()}
	vbT.Dims[0], vbT.Dims[1], vbT.Dims[2] = lat, vh, nh
	return []jlm.Tensor{kbT, vbT}, true, nil
}

// f32AsBytes reinterprets a float32 slice as the little-endian bytes a
// container stores (the engine only runs on little-endian hosts).
func f32AsBytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}
