package engine

import (
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// relowered has the front end lower every block's stated graph again from
// the config as it is, and again at the returned undo: a violation of what
// the lowering reads then reaches the blocks.
func relowered(m *model.Model) func() {
	fr := &m.Front
	fr.Mu.Lock()
	fr.Restated++
	fr.Mu.Unlock()
	return func() {
		fr.Mu.Lock()
		fr.Restated++
		fr.Mu.Unlock()
	}
}

// statedAgain has the front end lower the graphs the library states for m
// as a violation has left it (restatedBlocks), until the returned undo. A
// violation made of several edits may pass through a state the converter
// refuses: the blocks then state no graph, and the reason is printed, until
// an edit after it states one; a violation that ends there is refused by
// the front end.
func statedAgain(m *model.Model) func() { return statedWith(m, nil) }

// statedWith is statedAgain with edit made to the container's config before
// the library states the graphs: a violation of a field only the graphs
// carry (a block's window), which no config field of m's reaches.
func statedWith(m *model.Model, edit func(*jlm.Config)) func() {
	jc := m.Container.Config()
	was := jc.Blocks
	blocks, err := restatedBlocks(m, edit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "statedAgain: %v\n", err)
	}
	jc.Blocks = blocks
	undo := relowered(m)
	return func() {
		undo()
		jc.Blocks = was
	}
}

// withCfg has m read c as its config, over the graphs the library states
// for it (statedAgain), until the returned undo.
func withCfg(m *model.Model, c *model.Config) func() {
	was := m.Cfg
	m.Cfg = c
	undo := statedAgain(m)
	return func() {
		undo()
		m.Cfg = was
	}
}

// restatedBlocks are the graphs the library states for m as a violation has
// left it: m's source converted again (convertStated) from its container's
// config as it now is, the container's fields of every config field m.Cfg
// has changed from the one the container states (statedFields; a field with
// none is the lowering's to read), and the stored tensors its front end
// reads in place of the container's (restate). Each changed field must read
// back from the restated container as the violation set it. edit, where not
// nil, is made to the container's config after them.
func restatedBlocks(m *model.Model, edit func(*jlm.Config)) (jlm.BlockGraphs, error) {
	src, err := convertedFrom(m)
	if err != nil {
		return jlm.BlockGraphs{}, err
	}
	p, err := model.Open(m.Container.Path)
	if err != nil {
		return jlm.BlockGraphs{}, fmt.Errorf("restating %s: %w", src, err)
	}
	changed := changedFields(p.Cfg, m.Cfg)
	p.Close()
	jc := m.Container.Config()
	m.Front.Mu.Lock()
	hook := m.Front.Restate
	m.Front.Mu.Unlock()
	fields := statedFields()
	dir, err := os.MkdirTemp("", "restated")
	if err != nil {
		return jlm.BlockGraphs{}, err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "restated"+model.Ext)
	if err := convertStated(src, path, func(s *jlm.Source) {
		*s.Config = *jc
		for _, f := range changed {
			if state, ok := fields[f]; ok {
				state(s.Config, m.Cfg)
			}
		}
		if edit != nil {
			edit(s.Config)
		}
		restatedTensors(s, hook)
	}); err != nil {
		return jlm.BlockGraphs{}, fmt.Errorf("restating %s: %w", src, err)
	}
	cf, err := jlm.Open(path)
	if err != nil {
		return jlm.BlockGraphs{}, fmt.Errorf("restating %s: %w", src, err)
	}
	blocks := cf.Config().Blocks
	cf.Close()
	// A violated model may be one Open refuses (BitNet's inner norms without
	// its per-token activations); its graphs are stated all the same.
	r, err := model.Open(path)
	if err != nil {
		return blocks, nil
	}
	defer r.Close()
	for _, f := range changed {
		if _, ok := fields[f]; !ok {
			continue
		}
		if got, want := reflect.ValueOf(*r.Cfg).FieldByName(f).Interface(), reflect.ValueOf(*m.Cfg).FieldByName(f).Interface(); !sameStated(got, want) {
			return jlm.BlockGraphs{}, fmt.Errorf("restating %s: the container states %s %v where the violation set %v", src, f, got, want)
		}
	}
	return blocks, nil
}

// sameStated is whether a and b are one field's value as a container holds
// it: a float in f32, to the f32 rounding Open's arithmetic over it adds
// (YaRN's correction squared multiplies a stated score scale).
func sameStated(a, b any) bool {
	if x, ok := a.(float64); ok {
		if y, ok := b.(float64); ok {
			return math.Abs(x-y) <= 1e-6*math.Abs(y)
		}
	}
	return reflect.DeepEqual(a, b)
}

// changedFields are the exported fields of was that now differs in.
func changedFields(was, now *model.Config) []string {
	a, b := reflect.ValueOf(*was), reflect.ValueOf(*now)
	var out []string
	for i := range a.NumField() {
		f := a.Type().Field(i)
		if f.IsExported() && !reflect.DeepEqual(a.Field(i).Interface(), b.Field(i).Interface()) {
			out = append(out, f.Name)
		}
	}
	return out
}

// statedFields write each config field a violation changes into the
// container's field Open derives it from (configFrom, the other way).
func statedFields() map[string]func(c *jlm.Config, mc *model.Config) {
	flag := func(b jlm.Flags, on func(mc *model.Config) bool) func(*jlm.Config, *model.Config) {
		return func(c *jlm.Config, mc *model.Config) {
			c.Flags &^= b
			if on(mc) {
				c.Flags |= b
			}
		}
	}
	flag2 := func(b jlm.Flags2, on func(mc *model.Config) bool) func(*jlm.Config, *model.Config) {
		return func(c *jlm.Config, mc *model.Config) {
			c.Flags2 &^= b
			if on(mc) {
				c.Flags2 |= b
			}
		}
	}
	return map[string]func(*jlm.Config, *model.Config){
		"RopeNeox":       flag(jlm.FlagRopeNeox, func(mc *model.Config) bool { return mc.RopeNeox }),
		"NoExpertNorm":   flag(jlm.FlagNoExpertNorm, func(mc *model.Config) bool { return mc.NoExpertNorm }),
		"Parallel":       flag(jlm.FlagParallel, func(mc *model.Config) bool { return mc.Parallel }),
		"NoPEGlobal":     flag(jlm.FlagNoPEGlobal, func(mc *model.Config) bool { return mc.NoPEGlobal }),
		"NoPosEnc":       flag(jlm.FlagNoPosEnc, func(mc *model.Config) bool { return mc.NoPosEnc }),
		"LayerNorm":      flag(jlm.FlagLayerNorm, func(mc *model.Config) bool { return mc.LayerNorm }),
		"QKL2Norm":       flag(jlm.FlagQKL2Norm, func(mc *model.Config) bool { return mc.QKL2Norm }),
		"QKNorm":         flag(jlm.FlagQKNorm, func(mc *model.Config) bool { return mc.QKNorm }),
		"QKNormWide":     flag(jlm.FlagQKNormWide, func(mc *model.Config) bool { return mc.QKNormWide }),
		"QKNormPost":     flag(jlm.FlagQKNormPostRope, func(mc *model.Config) bool { return mc.QKNormPost }),
		"SWAChunked":     flag(jlm.FlagSWAChunked, func(mc *model.Config) bool { return mc.SWAChunked }),
		"ExpertWeightIn": flag(jlm.FlagExpertWeightIn, func(mc *model.Config) bool { return mc.ExpertWeightIn }),
		"ExpertSigmoid":  flag(jlm.FlagExpertSigmoid, func(mc *model.Config) bool { return mc.ExpertSigmoid }),
		"AttnOutGate":    flag(jlm.FlagAttnOutGate, func(mc *model.Config) bool { return mc.AttnOutGate }),
		"NonCausal":      flag(jlm.FlagNonCausal, func(mc *model.Config) bool { return mc.NonCausal }),
		"VNorm":          flag2(jlm.Flag2VNorm, func(mc *model.Config) bool { return mc.VNorm }),
		"RopeTail":       flag2(jlm.Flag2RopeTail, func(mc *model.Config) bool { return mc.RopeTail }),
		"QUnitNorm":      flag2(jlm.Flag2QUnitNorm, func(mc *model.Config) bool { return mc.QUnitNorm }),
		"OutUnrotated":   flag2(jlm.Flag2OutUnrotated, func(mc *model.Config) bool { return mc.OutUnrotated }),
		"Pooling": func(c *jlm.Config, mc *model.Config) {
			c.Flags &^= jlm.FlagPoolMean | jlm.FlagPoolCLS | jlm.FlagPoolLast
			switch mc.Pooling {
			case jlm.PoolMean:
				c.Flags |= jlm.FlagPoolMean
			case jlm.PoolCLS:
				c.Flags |= jlm.FlagPoolCLS
			case jlm.PoolLast:
				c.Flags |= jlm.FlagPoolLast
			}
		},
		"RopeSections": func(c *jlm.Config, mc *model.Config) {
			for i, v := range mc.RopeSections {
				c.RopeSections[i] = uint32(v)
			}
		},

		"NRot":             func(c *jlm.Config, mc *model.Config) { c.NRot = uint32(mc.NRot) },
		"NMTP":             func(c *jlm.Config, mc *model.Config) { c.NMTP = uint32(mc.NMTP) },
		"NFFNShExp":        func(c *jlm.Config, mc *model.Config) { c.NFFNShExp = uint32(mc.NFFNShExp) },
		"NExpertUsed":      func(c *jlm.Config, mc *model.Config) { c.NExpertUsed = uint32(mc.NExpertUsed) },
		"NExpertGroupUsed": func(c *jlm.Config, mc *model.Config) { c.NExpertGroupUsed = uint32(mc.NExpertGroupUsed) },
		"NExpertGroup":     func(c *jlm.Config, mc *model.Config) { c.NExpertGroup = uint32(mc.NExpertGroup) },
		// Open counts a marker head's blocks into NLayer.
		"NLayer":        func(c *jlm.Config, mc *model.Config) { c.NLayer = uint32(mc.NLayer) - c.HeadBlocks() },
		"NKVShared":     func(c *jlm.Config, mc *model.Config) { c.NKVShared = uint32(mc.NKVShared) },
		"PLEDim":        func(c *jlm.Config, mc *model.Config) { c.PLEDim = uint32(mc.PLEDim) },
		"AttnSoftcap":   func(c *jlm.Config, mc *model.Config) { c.AttnSoftcap = mc.AttnSoftcap },
		"FinalSoftcap":  func(c *jlm.Config, mc *model.Config) { c.FinalSoftcap = mc.FinalSoftcap },
		"ExpertScale":   func(c *jlm.Config, mc *model.Config) { c.ExpertScale = float32(mc.ExpertScale) },
		"ResidualScale": func(c *jlm.Config, mc *model.Config) { c.ResidualScale = float32(mc.ResidualScale) },
		"LogitScale":    func(c *jlm.Config, mc *model.Config) { c.LogitScale = float32(mc.LogitScale) },
		"LatentNormEps": func(c *jlm.Config, mc *model.Config) { c.LatentNormEps = float32(mc.LatentNormEps) },
		"AttnTempScale": func(c *jlm.Config, mc *model.Config) { c.AttnTempScale = float32(mc.AttnTempScale) },
		"AttnTempFloor": func(c *jlm.Config, mc *model.Config) { c.AttnTempFloor = uint32(mc.AttnTempFloor) },
		"AttnTempOffs":  func(c *jlm.Config, mc *model.Config) { c.AttnTempOffset = float32(mc.AttnTempOffs) },
		"DeltaDecay":    func(c *jlm.Config, mc *model.Config) { c.DeltaDecay = mc.DeltaDecay },
		"DeltaGate":     func(c *jlm.Config, mc *model.Config) { c.DeltaGate = mc.DeltaGate },
		// Open scales a stated score scale by YaRN's correction squared.
		"AttnScale": func(c *jlm.Config, mc *model.Config) {
			s := mc.AttnScale
			if c.YarnLogMul != 0 && c.YarnFactor > 1 {
				k := float64(c.YarnLogMul)*math.Log(float64(c.YarnFactor)) + 1
				s /= k * k
			}
			c.AttnScale = float32(s)
		},
		// A clamp's bound is the min's constant (jlm.Config.QKVAct).
		"QKVAct": func(c *jlm.Config, mc *model.Config) {
			c.ClampKQV = 0
			if mc.QKVAct != nil {
				c.ClampKQV = mc.QKVAct.Terms[1].Args[1].Value
			}
		},
	}
}

// jlmMut has m's blocks state the graphs the library states with f made to
// its container's config (statedWith), until the returned undo: the
// violation of a field the graphs carry and no config field of m's does.
func jlmMut(m *model.Model, f func(c *jlm.Config)) func() { return statedWith(m, f) }

// windowAt is block li's attention window as its graph states it, and
// whether it is an aligned chunk.
func windowAt(m *model.Model, li int) (int, bool) {
	return m.Composing().BlockGraph(li).AttendWindow()
}

// slides reports that block li's graph attends over a window.
func slides(m *model.Model, li int) bool {
	w, _ := windowAt(m, li)
	return w > 0
}

// windowOf is the window m's sliding blocks state, 0 where none slides.
func windowOf(m *model.Model) int {
	w := 0
	for li := range m.Cfg.NLayer {
		v, _ := windowAt(m, li)
		w = max(w, v)
	}
	return w
}

// quantizesRows reports that a block of m's graphs projects its rows' int8
// codes: its activations are quantized per token.
func quantizesRows(m *model.Model) bool {
	for li := range m.Cfg.NLayer {
		if m.Composing().BlockGraph(li).ReadsRowCodes() {
			return true
		}
	}
	return false
}

// expertWidthOf is the width m's routed experts read and write: the K of
// the first up bank a block's graph binds, 0 where none does.
func expertWidthOf(m *model.Model) int {
	for li := range m.Cfg.NLayer {
		if g := m.Composing().BlockGraph(li); g != nil {
			for _, o := range g.Ops {
				for _, w := range o.Weight {
					if w.Role == jlm.RoleExpUpBank {
						return w.K
					}
				}
			}
		}
	}
	return 0
}

// ropesAt reports that block li's graph turns its queries and keys by a
// rotary.
func ropesAt(m *model.Model, li int) bool {
	if g := m.Composing().BlockGraph(li); g != nil {
		for _, o := range g.Ops {
			if o.Kind == graph.OpRope {
				return true
			}
		}
	}
	return false
}

// kvSourceOf is the block whose history block li's graph reads: li where it
// reads its own.
func kvSourceOf(m *model.Model, li int) int {
	if n, err := m.Composing().BlockGraph(li).ReadsHistoryOf(); err == nil && n >= 0 {
		return n
	}
	return li
}

// attnWindow is the key range a block of window w (chunked where it is an
// aligned chunk) attends to from position pos: w0 the first key and an how
// many, so the keys are w0..w0+an-1 and the last is pos itself. A block
// that sees the whole history gets 0, pos+1; a chunked block's start is
// floor(pos/w)*w.
func attnWindow(w int, chunked bool, pos int) (w0, an int) {
	an = pos + 1
	switch {
	case w <= 0:
		return 0, an
	case chunked:
		w0 = pos / w * w
		return w0, pos + 1 - w0
	case w < an:
		return an - w, w
	}
	return 0, an
}

// restatedTensors has s's tensors read as the front end reads m's through
// restate, each block's and the model's own: a tensor the hook takes away
// goes, and one it adds or resizes stands in as zeros of its shape, the
// graphs stated over it being all that is read back.
func restatedTensors(s *jlm.Source, restate func(block int32, inv map[jlm.Role]graph.Shape)) {
	if restate == nil {
		return
	}
	text := func(t jlm.Tensor) bool { return t.Index < 0 && !t.Role.Vision() }
	invs := map[int32]map[jlm.Role]graph.Shape{}
	for _, t := range s.Tensors {
		if !text(t) {
			continue
		}
		if invs[t.Block] == nil {
			invs[t.Block] = map[jlm.Role]graph.Shape{}
		}
		sh := graph.Shape{K: int(t.Dims[0])}
		if t.NDim >= 2 {
			sh.Rows = int(t.Dims[1])
		}
		if t.NDim >= 3 {
			sh.Sheets = int(t.Dims[2])
		}
		invs[t.Block][t.Role] = sh
	}
	for _, b := range slices.Sorted(maps.Keys(invs)) {
		was := invs[b]
		inv := maps.Clone(was)
		restate(b, inv)
		s.Tensors = slices.DeleteFunc(s.Tensors, func(t jlm.Tensor) bool {
			sh, ok := inv[t.Role]
			return t.Block == b && text(t) && (!ok || sh != was[t.Role])
		})
		for _, r := range slices.Sorted(maps.Keys(inv)) {
			sh := inv[r]
			if old, ok := was[r]; ok && old == sh {
				continue
			}
			t := jlm.Tensor{Role: r, Block: b, Index: -1, Type: jlm.TypeF32, NDim: 1, Name: "restated " + r.String()}
			t.Dims[0] = uint64(sh.K)
			n := sh.K
			for _, d := range []int{sh.Rows, sh.Sheets} {
				if d > 0 {
					t.Dims[t.NDim] = uint64(d)
					t.NDim++
					n *= d
				}
			}
			t.Data = make([]byte, 4*n)
			s.Tensors = append(s.Tensors, t)
		}
	}
}

// convertStated converts the fixture at src as jlmOf converts it -- a GGUF
// with the config.json it converts with, a HuggingFace directory, its
// matrices quantized when named with q8Suffix -- to dst, with edit made to
// the source before the library states the blocks' graphs
// (convert.WithStatedSource): the graphs follow the edit.
func convertStated(src, dst string, edit func(*jlm.Source)) error {
	fp := jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}
	stated := convert.WithStatedSource(edit)
	if dir, q8 := strings.CutSuffix(src, q8Suffix); isDir(dir) {
		_, err := convert.FromSafetensors(dir, dst, fp, append(dirOptions(q8), stated)...)
		return err
	}
	_, err := convert.FromGGUFs(src, "", dst, fp, append(ggufOptions(src), stated)...)
	return err
}

// statedModel is the fixture at src converted with edit made to its source
// (convertStated) and opened: a restated model whose graphs follow the
// restatement.
func statedModel(t testing.TB, src string, edit func(*jlm.Source)) *model.Model {
	t.Helper()
	path := filepath.Join(t.TempDir(), "stated"+model.Ext)
	if err := convertStated(src, path, edit); err != nil {
		t.Fatalf("restating %s: %v", src, err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestAnUnviolatedModelIsStatedAgainAsItWas holds statedAgain to the
// converter: a model no violation has touched is stated again graph for
// graph as its container states it, and one with a feature switched off
// is not.
func TestAnUnviolatedModelIsStatedAgainAsItWas(t *testing.T) {
	for _, name := range []string{"synth-phi2.gguf", "synth-llama4.gguf", "synth-gemma3n.gguf", "synth-gemma4-e.gguf", "synth-olmo2.gguf"} {
		t.Run(name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(name))
			if !ok || !strings.HasSuffix(src, ".gguf") {
				testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", testmodels.Path(name))
			}
			m, err := model.Open(jlmOf(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			jc := m.Container.Config()
			was := jc.Blocks
			same := func() bool {
				return slices.Equal(jc.Blocks.Of, was.Of) && slices.Equal(jc.Blocks.Graphs, was.Graphs)
			}
			got, err := restatedBlocks(m, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Of, was.Of) || !slices.Equal(got.Graphs, was.Graphs) {
				t.Fatalf("stated again untouched, %d graphs for %d blocks where the container states %d for %d",
					len(got.Graphs), len(got.Of), len(was.Graphs), len(was.Of))
			}
			undo := cfgMut(m, func(c *model.Config) { c.RopeNeox = !c.RopeNeox })
			if same() {
				t.Fatal("the rotary's pairing flipped states the same graphs: statedAgain reads no violation")
			}
			undo()
			if !same() {
				t.Fatal("the undo leaves other graphs than the container's")
			}
		})
	}
}
