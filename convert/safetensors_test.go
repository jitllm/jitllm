package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
)

// A hand-built HuggingFace directory, so the refusals below are hermetic:
// every row of the refusal table is a config.json or tensor name no shipped
// model has.

type synthTensor struct {
	dtype safetensors.DType
	shape []uint64
	data  []byte
}

// synthModel is a two-layer llama with head_dim 4. Every number is small and
// every shape is distinct, so a role wired to the wrong tensor is a shape
// mismatch rather than a coincidence.
type synthModel struct {
	cfg     map[string]any
	tensors map[string]*synthTensor
	tokJSON map[string]any
	tokCfg  map[string]any
}

const (
	sNLayer = 2
	sNEmbd  = 8
	sNHead  = 2
	sNKV    = 1
	sHeadD  = 4
	sNFFN   = 16
	sNVocab = 12
)

func f32Bytes(n int, seed uint32) []byte {
	b := make([]byte, 0, n*4)
	x := seed | 1
	for i := 0; i < n; i++ {
		// A cheap LCG: deterministic, and the values span a decade so a
		// permutation applied twice cannot look like a no-op.
		x = x*1664525 + 1013904223
		v := float32(int32(x>>8)) / float32(1<<23)
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return b
}

func newSynth() *synthModel {
	m := &synthModel{
		cfg: map[string]any{
			"architectures":           []any{"LlamaForCausalLM"},
			"model_type":              "llama",
			"hidden_size":             sNEmbd,
			"intermediate_size":       sNFFN,
			"num_hidden_layers":       sNLayer,
			"num_attention_heads":     sNHead,
			"num_key_value_heads":     sNKV,
			"head_dim":                sHeadD,
			"max_position_embeddings": 64,
			"rms_norm_eps":            1e-5,
			"rope_theta":              10000,
			"vocab_size":              sNVocab,
			"hidden_act":              "silu",
			"tie_word_embeddings":     true,
			"torch_dtype":             "float32",
		},
		tensors: map[string]*synthTensor{},
	}
	add := func(name string, shape ...uint64) {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		m.tensors[name] = &synthTensor{safetensors.F32, shape, f32Bytes(n, uint32(len(name)*7+n))}
	}
	add("model.embed_tokens.weight", sNVocab, sNEmbd)
	add("model.norm.weight", sNEmbd)
	for l := 0; l < sNLayer; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"input_layernorm.weight", sNEmbd)
		add(p+"post_attention_layernorm.weight", sNEmbd)
		add(p+"self_attn.q_proj.weight", sNHead*sHeadD, sNEmbd)
		add(p+"self_attn.k_proj.weight", sNKV*sHeadD, sNEmbd)
		add(p+"self_attn.v_proj.weight", sNKV*sHeadD, sNEmbd)
		add(p+"self_attn.o_proj.weight", sNEmbd, sNHead*sHeadD)
		add(p+"mlp.gate_proj.weight", sNFFN, sNEmbd)
		add(p+"mlp.up_proj.weight", sNFFN, sNEmbd)
		add(p+"mlp.down_proj.weight", sNEmbd, sNFFN)
	}

	vocab := map[string]any{}
	for i := 0; i < sNVocab; i++ {
		vocab[fmt.Sprintf("t%d", i)] = i
	}
	m.tokJSON = map[string]any{
		"added_tokens": []any{map[string]any{
			"id": 0, "content": "t0", "special": true,
			"lstrip": false, "rstrip": false, "normalized": false, "single_word": false,
		}},
		"pre_tokenizer": map[string]any{"type": "ByteLevel", "use_regex": true, "add_prefix_space": false},
		"model": map[string]any{
			"type": "BPE", "vocab": vocab, "merges": []any{"t1 t2"},
			"ignore_merges": true, "byte_fallback": false,
		},
	}
	m.tokCfg = map[string]any{"bos_token": "t0", "eos_token": "t1"}
	return m
}

// write lays the model out as a directory, in the shape safetensors ships in.
func (m *synthModel) write(t testing.TB) string {
	t.Helper()
	dir := t.TempDir()
	put := func(name string, v any) {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if m.cfg != nil {
		put("config.json", m.cfg)
	}
	if m.tokJSON != nil {
		put("tokenizer.json", m.tokJSON)
	}
	if m.tokCfg != nil {
		put("tokenizer_config.json", m.tokCfg)
	}
	names := make([]string, 0, len(m.tensors))
	for n := range m.tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	hdr := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var data []byte
	for _, n := range names {
		tt := m.tensors[n]
		hdr[n] = map[string]any{
			"dtype": string(tt.dtype), "shape": tt.shape,
			"data_offsets": []uint64{uint64(len(data)), uint64(len(data) + len(tt.data))},
		}
		data = append(data, tt.data...)
	}
	hb, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hb)))
	out = append(out, hb...)
	out = append(out, data...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), out, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func (m *synthModel) source(t testing.TB) (*jlm.Source, error) {
	t.Helper()
	dir := m.write(t)
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return hfSourceLoaded(dir, []*safetensors.File{f})
}

// TestSynthModelConverts is the control for the refusal table below: the
// unmutated model must go all the way through, or every refusal that follows is
// refusing the fixture rather than the mutation.
func TestSynthModelConverts(t *testing.T) {
	s, err := newSynth().source(t)
	if err != nil {
		t.Fatalf("the unmutated fixture does not convert, so no refusal below proves anything: %v", err)
	}
	c := s.Config
	if c.Arch != jlm.ArchLlama || c.NLayer != sNLayer || c.NEmbd != sNEmbd ||
		c.NHead != sNHead || c.NKVHead != sNKV || c.HeadDim != sHeadD ||
		c.NFFN != sNFFN || c.NVocab != sNVocab || c.NRot != sHeadD {
		t.Fatalf("config %+v", c)
	}
	if !c.Flags.Has(jlm.FlagTiedEmbd) {
		t.Error("no lm_head.weight and FlagTiedEmbd is not set")
	}
	// 2 model-level + 9 per block.
	if want := 2 + 9*sNLayer; len(s.Tensors) != want {
		t.Fatalf("%d tensors, want %d", len(s.Tensors), want)
	}
	// post_attention_layernorm must be the FFN's norm; mapping it to
	// RolePostAttnNorm (what the name says, and what gemma2/gemma3 mean)
	// applies a norm in the wrong place and answers fluently.
	seen := map[jlm.Role]int{}
	for i := range s.Tensors {
		seen[s.Tensors[i].Role]++
	}
	if seen[jlm.RoleFFNNorm] != sNLayer || seen[jlm.RolePostFFNNorm] != 0 || seen[jlm.RolePostAttnNorm] != 0 {
		t.Errorf("post_attention_layernorm landed as %d ffn_norm, %d post_attn_norm, %d post_ffn_norm",
			seen[jlm.RoleFFNNorm], seen[jlm.RolePostAttnNorm], seen[jlm.RolePostFFNNorm])
	}
	if s.Vocab == nil || len(s.Vocab.Tokens) != sNVocab || len(s.Vocab.Merges) != 1 {
		t.Fatalf("vocab %+v", s.Vocab)
	}
	// ignore_merges is a field of tokenizer.json, not a family guess.
	if !s.Vocab.IgnoreMerges {
		t.Error("model.ignore_merges is true in the artefact and false in the container")
	}
	if s.Vocab.BOS != 0 || s.Vocab.EOS != 1 {
		t.Errorf("BOS/EOS %d/%d, want 0/1", s.Vocab.BOS, s.Vocab.EOS)
	}
	if len(s.Vocab.Pre) != 1 || s.Vocab.Pre[0].Kind != jlm.PreByteLevel {
		t.Errorf("pre-tokenizer %+v", s.Vocab.Pre)
	}
}

// TestSafetensorsRefusals is the violation table.
//
// Every row is a model that would otherwise load and be wrong: a dropped
// tensor, a mis-declared head, an unapplied rope scaling and an unsupported
// dtype all produce fluent output and no error later.
func TestSafetensorsRefusals(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*synthModel)
		want string
	}{
		{"unknown architecture", func(m *synthModel) {
			m.cfg["architectures"] = []any{"MistralForCausalLM"}
		}, "is not implemented from safetensors"},
		{"no config.json", func(m *synthModel) { m.cfg = nil },
			"hyperparameters in config.json"},
		{"no tokenizer.json", func(m *synthModel) { m.tokJSON = nil },
			"tokenizer in tokenizer.json"},
		{"mlp bias", func(m *synthModel) { m.cfg["mlp_bias"] = true },
			"no feed-forward bias role"},
		{"gelu", func(m *synthModel) { m.cfg["hidden_act"] = "gelu" },
			"feed-forward is silu-gated"},
		{"llama3 rope scaling", func(m *synthModel) {
			m.cfg["rope_scaling"] = map[string]any{"rope_type": "llama3", "factor": 32.0}
		}, `rope_scaling "llama3" is not implemented`},
		{"partial rotary", func(m *synthModel) { m.cfg["partial_rotary_factor"] = 0.5 },
			"partial_rotary_factor 0.5"},
		{"sliding window", func(m *synthModel) { m.cfg["sliding_window"] = 4096 },
			"sliding_window 4096"},
		{"no vocab_size", func(m *synthModel) { delete(m.cfg, "vocab_size") },
			"missing vocab_size"},
		// The head is the absence of lm_head, and config.json must agree.
		{"untied and headless", func(m *synthModel) { m.cfg["tie_word_embeddings"] = false },
			"output head is missing, not tied"},
		{"tied and headed", func(m *synthModel) {
			m.tensors["lm_head.weight"] = &synthTensor{safetensors.F32,
				[]uint64{sNVocab, sNEmbd}, f32Bytes(sNVocab*sNEmbd, 99)}
		}, "which one is the head"},
		// A tensor with no role is a refusal, never a skip.
		{"unknown tensor", func(m *synthModel) {
			m.tensors["model.layers.0.self_attn.rotary_scale"] = &synthTensor{
				safetensors.F32, []uint64{4}, f32Bytes(4, 3)}
		}, "has no role in this container's vocabulary"},
		{"unknown model-level tensor", func(m *synthModel) {
			m.tensors["score.weight"] = &synthTensor{safetensors.F32, []uint64{2, sNEmbd}, f32Bytes(2*sNEmbd, 5)}
		}, "has no role in this container's vocabulary"},
		// A dtype with no path is refused by name.
		{"int weights", func(m *synthModel) {
			tt := m.tensors["model.norm.weight"]
			tt.dtype, tt.data = safetensors.I64, make([]byte, sNEmbd*8)
		}, "dtype I64 has no path into this container"},
		{"f64 weights", func(m *synthModel) {
			tt := m.tensors["model.norm.weight"]
			tt.dtype, tt.data = safetensors.F64, make([]byte, sNEmbd*8)
		}, "dtype F64 has no path into this container"},
		// The shape check is the gate on the name table: more heads than the
		// weights have is the shape a wrong name mapping takes.
		{"declared heads do not match q", func(m *synthModel) {
			m.cfg["num_attention_heads"] = 4
		}, "attn_out wants [16 8]"},
		{"ffn width disagrees", func(m *synthModel) { m.cfg["intermediate_size"] = 32 },
			"ffn_down wants [32 8]"},
		{"vocab_size disagrees", func(m *synthModel) { m.cfg["vocab_size"] = 16 },
			"token_embd wants [8 16]"},
		// A token table with a hole shifts every id above it, which tokenizes
		// and decodes plausible nonsense. Both reachable shapes of a hole:
		{"id past the table", func(m *synthModel) {
			v := m.tokJSON["model"].(map[string]any)["vocab"].(map[string]any)
			delete(v, "t5")
			v["t99"] = sNVocab
		}, "so the table would have a hole and every id above it would shift"},
		{"two tokens on one id", func(m *synthModel) {
			m.tokJSON["model"].(map[string]any)["vocab"].(map[string]any)["t5"] = 4
		}, "is claimed by two tokens"},
		{"unigram tokenizer", func(m *synthModel) {
			m.tokJSON["model"].(map[string]any)["type"] = "Unigram"
		}, `tokenizer model "Unigram" is not implemented`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := newSynth()
			c.mut(m)
			_, err := m.source(t)
			if err == nil {
				t.Fatalf("accepted -- this gate proved nothing")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused for the wrong reason:\n got %v\nwant ...%s...", err, c.want)
			}
			t.Logf("refused: %v", err)
		})
	}
}

// TestBF16IsWidenedExactly checks the one place a weight's bytes change on the
// way in for a reason other than the rotary. bf16 is the top 16 bits of a
// binary32, so widening is a shift and every value, infinities and NaNs
// included, survives; the gate reads the pattern back out of the f32 written.
func TestBF16IsWidenedExactly(t *testing.T) {
	// Every interesting bit pattern: zero, denormal, one, the largest finite,
	// infinity, a NaN, and the negatives.
	pats := []uint16{0x0000, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x7F7F, 0x7F80, 0xFF80, 0x7FC0, 0x4049}
	src := make([]byte, 0, len(pats)*2)
	for _, p := range pats {
		src = binary.LittleEndian.AppendUint16(src, p)
	}
	got := widenBF16(src)
	if len(got) != len(pats)*4 {
		t.Fatalf("%d bytes for %d values", len(got), len(pats))
	}
	for i, p := range pats {
		w := binary.LittleEndian.Uint32(got[i*4:])
		if w != uint32(p)<<16 {
			t.Errorf("bf16 %04x -> f32 %08x, want %08x", p, w, uint32(p)<<16)
		}
	}
	// And it is selected: a BF16 model must arrive as TypeF32, because jit_old/cpu
	// emits a native matvec for F32 and F16 only and a BF16 weight would reach
	// the oracle. See hfTypeOf.
	m := newSynth()
	for _, n := range []string{"model.norm.weight", "model.layers.0.mlp.down_proj.weight"} {
		tt := m.tensors[n]
		half := make([]byte, len(tt.data)/2)
		for i := 0; i < len(half); i += 2 {
			// Take the high half of each f32: that is its bf16, truncated.
			copy(half[i:i+2], tt.data[i*2+2:i*2+4])
		}
		tt.dtype, tt.data = safetensors.BF16, half
	}
	s, err := m.source(t)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for i := range s.Tensors {
		e := &s.Tensors[i]
		if e.Type != jlm.TypeF32 {
			t.Fatalf("%s is %v; a container of bf16 weights runs on the oracle", e.Name, e.Type)
		}
		if e.Name == "model.layers.0.mlp.down_proj.weight" {
			n++
			if uint64(len(e.Data)) != e.Dims[0]*e.Dims[1]*4 {
				t.Errorf("%s: %d bytes for %v", e.Name, len(e.Data), e.Dims[:e.NDim])
			}
		}
	}
	if n != 1 {
		t.Fatal("the widened tensor was not examined -- this gate proved nothing")
	}
}

// TestTensorBytesAreTheFileBytes closes name -> bytes for the tensors that are
// carried verbatim: read the source file independently and compare.
//
// It is the half the cross-format gate cannot prove: that one is bounded by a
// quantisation error, this one is byte equality and fails on a single
// transposed axis or off-by-one row.
func TestTensorBytesAreTheFileBytes(t *testing.T) {
	m := newSynth()
	dir := m.write(t)
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := hfSourceLoaded(dir, []*safetensors.File{f})
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for i := range s.Tensors {
		e := &s.Tensors[i]
		raw := m.tensors[e.Name].data
		shape := m.tensors[e.Name].shape
		// The dims are the shape reversed and nothing else moves.
		if int(e.NDim) != len(shape) {
			t.Fatalf("%s: %d dims for shape %v", e.Name, e.NDim, shape)
		}
		for j, d := range shape {
			if e.Dims[len(shape)-1-j] != d {
				t.Fatalf("%s: dims %v for shape %v", e.Name, e.Dims[:e.NDim], shape)
			}
		}
		want := raw
		if e.Role == jlm.RoleAttnQ || e.Role == jlm.RoleAttnK {
			heads := sNHead
			if e.Role == jlm.RoleAttnK {
				heads = sNKV
			}
			want = permuteHeadPairs(raw, heads, sHeadD, sNEmbd*4)
			if string(want) == string(raw) {
				t.Fatalf("%s: the permutation is a no-op on this fixture, so its "+
					"assertion below proves nothing", e.Name)
			}
		}
		if string(e.Data) != string(want) {
			t.Fatalf("%s: %d bytes do not match the file", e.Name, len(e.Data))
		}
		checked++
	}
	if checked != 2+9*sNLayer {
		t.Fatalf("checked %d tensors -- this gate proved nothing", checked)
	}
}

// TestPermutationMakesConsecutiveRotationEqualRotateHalf is the semantic gate
// on the q/k row permutation: the index formula cannot be its own oracle, so
// this rotates by rotate_half (HuggingFace, pairs (i, i+hd/2)) and by
// consecutive pairs (this container) on the permuted vector and compares. It
// also asserts llama is not FlagRopeNeox, since the permutation is only right
// with that convention.
func TestPermutationMakesConsecutiveRotationEqualRotateHalf(t *testing.T) {
	const hd, heads = 8, 3
	v := make([]float32, heads*hd)
	for i := range v {
		v[i] = float32(i)*0.37 - 5
	}
	// theta_i, one per pair, distinct so a mis-paired dimension cannot cancel.
	theta := make([]float64, hd/2)
	for i := range theta {
		theta[i] = 0.3 + 0.7*float64(i)
	}
	// HuggingFace: pair (i, i+hd/2).
	half := make([]float32, len(v))
	for h := 0; h < heads; h++ {
		b := h * hd
		for i := 0; i < hd/2; i++ {
			c, s := float32(math.Cos(theta[i])), float32(math.Sin(theta[i]))
			x, y := v[b+i], v[b+i+hd/2]
			half[b+i], half[b+i+hd/2] = x*c-y*s, y*c+x*s
		}
	}
	// This container: pair (2i, 2i+1), on the permuted vector.
	asBytes := func(f []float32) []byte {
		b := make([]byte, 0, len(f)*4)
		for _, x := range f {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
		}
		return b
	}
	asFloats := func(b []byte) []float32 {
		f := make([]float32, len(b)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return f
	}
	p := asFloats(permuteHeadPairs(asBytes(v), heads, hd, 4))
	for h := 0; h < heads; h++ {
		b := h * hd
		for i := 0; i < hd/2; i++ {
			c, s := float32(math.Cos(theta[i])), float32(math.Sin(theta[i]))
			x, y := p[b+2*i], p[b+2*i+1]
			p[b+2*i], p[b+2*i+1] = x*c-y*s, y*c+x*s
		}
	}
	// Rotating the permuted vector must be the permutation of the rotated one.
	wantP := asFloats(permuteHeadPairs(asBytes(half), heads, hd, 4))
	for i := range p {
		if d := math.Abs(float64(p[i] - wantP[i])); d > 1e-5 {
			t.Fatalf("element %d: consecutive-pair rotation of the permuted vector is %g, "+
				"and the permutation of rotate_half is %g", i, p[i], wantP[i])
		}
	}
	// A no-op permutation would pass the above only if head_dim were 2; prove
	// the fixture actually moves rows.
	if string(permuteHeadPairs(asBytes(v), heads, hd, 4)) == string(asBytes(v)) {
		t.Fatal("the permutation is the identity here -- this gate proved nothing")
	}

	// And the convention the permutation is paired with.
	s, err := newSynth().source(t)
	if err != nil {
		t.Fatal(err)
	}
	if s.Config.Flags.Has(jlm.FlagRopeNeox) {
		t.Fatal("llama is FlagRopeNeox AND its q/k rows are permuted: that is rotate_half " +
			"applied to a layout already rewritten for consecutive pairs. Pick one.")
	}
}

// hfSourceLoaded is hfSourceOf with every lazy tensor read into Data, for the
// gates that compare bytes. The writer's lazy path is covered end to end by
// model.TestSafetensorsMatchTransformers.
func hfSourceLoaded(dir string, fs []*safetensors.File) (*jlm.Source, error) {
	s, err := hfSourceOf(dirFiles(dir), fs)
	if err != nil {
		return nil, err
	}
	for i := range s.Tensors {
		if s.Tensors[i].Data, err = s.Tensors[i].Bytes(); err != nil {
			return nil, err
		}
		s.Tensors[i].Load = nil
	}
	return s, nil
}
