package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

const klDir = "synth-kimilinear"

// synth-kimilinear interleaves its layer kinds (linear, full, linear) on
// purpose: a reader that treats the kind map as a prefix gives the middle
// block the wrong graph while every shape still checks out.
func kimiSource(t testing.TB) (*jlm.Source, *safetensors.File) {
	t.Helper()
	dir := testmodels.Path(klDir)
	p := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing (RULE 10). "+
			"Make it: $JITLLM_HF_PY scripts/hfgold.py %s", p, err, klDir)
	}
	f, err := safetensors.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	s, err := hfSourceLoaded(dir, []*safetensors.File{f})
	if err != nil {
		t.Fatalf("%s does not convert: %v", klDir, err)
	}
	return s, f
}

func klTensor(t testing.TB, s *jlm.Source, role jlm.Role, block int32) *jlm.Tensor {
	t.Helper()
	for i := range s.Tensors {
		if s.Tensors[i].Role == role && s.Tensors[i].Block == block {
			return &s.Tensors[i]
		}
	}
	t.Fatalf("block %d has no %v", block, role)
	return nil
}

func klRaw(t testing.TB, f *safetensors.File, name string) []byte {
	t.Helper()
	for i := range f.Tensors {
		x := &f.Tensors[i]
		if x.Name == name {
			b, err := f.ReadTensor(x)
			if err != nil {
				t.Fatal(err)
			}
			return b
		}
	}
	t.Fatalf("the fixture has no %s", name)
	return nil
}

// TestKimiLinearConverts pins the config this architecture writes, and every
// number in it is one nothing else here checks.
func TestKimiLinearConverts(t *testing.T) {
	s, _ := kimiSource(t)
	c := s.Config
	// The kind map, and it is interleaved: a prefix reader passes the length
	// check and fails here.
	want := []jlm.LayerKind{jlm.LayerLinearAttn, jlm.LayerFullAttn, jlm.LayerLinearAttn}
	if len(c.LayerKinds) != len(want) {
		t.Fatalf("%d layer kind(s), want %d", len(c.LayerKinds), len(want))
	}
	for i, k := range want {
		if c.LayerKinds[i] != k {
			t.Errorf("layer %d is %v, want %v", i, c.LayerKinds[i], k)
		}
	}
	// The recurrence. Groups == NHeadV: KDA has no grouped-query sharing on
	// the linear side. The numbers are scripts/hfgold.py's recipe (4 heads of
	// 32, kernel 4); StateSize is also the low-rank gates' rank, a multiple of
	// 32 so the device path is exercised.
	if got, want := c.SSM, (jlm.SSMConfig{ConvKernel: 4, Groups: 4, Inner: 128, StateSize: 32, NHeadV: 4}); got != want {
		t.Errorf("SSM %+v, want %+v", got, want)
	}
	// The sigmoid flag has no key behind it: KimiLinearTopkRouter hardcodes
	// router_logits.sigmoid(), and softmax would route to the wrong experts.
	if !c.Flags.Has(jlm.FlagExpertSigmoid) {
		t.Error("FlagExpertSigmoid is clear, and this router is a sigmoid with no scoring_func key to say so")
	}
	if c.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Error("FlagNoExpertNorm is set and moe_renormalize is true")
	}
	if c.NExpert != 8 || c.NExpertUsed != 2 || c.NDenseLead != 1 {
		t.Errorf("nexpert=%d used=%d denselead=%d, want 8/2/1", c.NExpert, c.NExpertUsed, c.NDenseLead)
	}
	if c.ExpertScale != 2.446 {
		t.Errorf("routed_scaling_factor %v", c.ExpertScale)
	}
	// MLA, on the full block only.
	if c.KVLoraRank != 16 || c.QLoraRank != 0 || c.HeadDim != 24 || c.HeadDimV != 16 || c.NRot != 8 {
		t.Errorf("kvlora=%d qlora=%d headdim=%d headdimv=%d nrot=%d",
			c.KVLoraRank, c.QLoraRank, c.HeadDim, c.HeadDimV, c.NRot)
	}
}

// TestKimiLinearFusesQKVInOrder is the gate the shape check cannot be.
//
// The wrong order produces exactly the right shape: deltaGeom slices the
// mixed projection by arithmetic, so q|v|k is a [NEmbd, 192] tensor whose
// every head belongs to another stream, and the model is fluent and wrong.
// Only comparing the bytes against the sources in order can say.
func TestKimiLinearFusesQKVInOrder(t *testing.T) {
	s, f := kimiSource(t)
	for _, tc := range []struct {
		role  jlm.Role
		names []string
		rows  uint64
	}{
		// 3 * linear_num_heads * linear_head_dim = 3 * 128.
		{jlm.RoleAttnQKV, []string{"q_proj.weight", "k_proj.weight", "v_proj.weight"}, 384},
		{jlm.RoleSSMConv1d, []string{"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"}, 384},
	} {
		// Block 0 and block 2 are the linear ones; block 1 must carry neither.
		for _, li := range []int32{0, 2} {
			got := klTensor(t, s, tc.role, li)
			if got.NDim != 2 || got.Dims[1] != tc.rows {
				t.Errorf("block %d %v: ndim=%d dims=%v, want 2 dims with ne1=%d",
					li, tc.role, got.NDim, got.Dims, tc.rows)
				continue
			}
			var want []byte
			for _, n := range tc.names {
				want = append(want, klRaw(t, f, "model.layers."+itoa32(li)+".self_attn."+n)...)
			}
			if !bytes.Equal(got.Data, want) {
				// Say which order would have matched.
				t.Errorf("block %d %v: the bytes are not %v concatenated in that order",
					li, tc.role, tc.names)
			}
		}
		for i := range s.Tensors {
			if s.Tensors[i].Role == tc.role && s.Tensors[i].Block == 1 {
				t.Errorf("block 1 is full attention and carries %v", tc.role)
			}
		}
	}
}

// TestKimiLinearALogBecomesAPerChannelRate pins the two transforms that turn
// A_log into what the gate kernel reads: the negated exponential, and the
// per-head value repeated across the head's channels.
//
// Both are constant, hence done at conversion, and either wrong is fluent:
// a positive rate makes the decay grow and the state diverge slowly, and a
// wrong broadcast pairs each channel with another head's decay.
func TestKimiLinearALogBecomesAPerChannelRate(t *testing.T) {
	s, f := kimiSource(t)
	c := s.Config
	heads, kDim := int(c.SSM.NHeadV), int(c.SSM.StateSize)
	for _, li := range []int32{0, 2} {
		got := klTensor(t, s, jlm.RoleSSMA, li)
		if got.NDim != 1 || got.Dims[0] != uint64(heads*kDim) {
			t.Fatalf("block %d ssm_a: ndim=%d dims=%v, want %d elements",
				li, got.NDim, got.Dims, heads*kDim)
		}
		raw := klRaw(t, f, "model.layers."+itoa32(li)+".self_attn.A_log")
		for h := 0; h < heads; h++ {
			a := math.Float32frombits(binary.LittleEndian.Uint32(raw[h*4:]))
			want := float32(-math.Exp(float64(a)))
			if want >= 0 {
				t.Fatalf("head %d: -exp(%v) = %v is not negative, so the fixture cannot "+
					"tell a negated rate from a raw one", h, a, want)
			}
			for i := 0; i < kDim; i++ {
				g := math.Float32frombits(binary.LittleEndian.Uint32(got.Data[(h*kDim+i)*4:]))
				if g != want {
					t.Fatalf("block %d ssm_a[%d] (head %d channel %d) = %v, want -exp(A_log) = %v",
						li, h*kDim+i, h, i, g, want)
				}
			}
		}
	}
}

func itoa32(v int32) string {
	if v == 0 {
		return "0"
	}
	var b [12]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// TestKimiDenseBlockTakesTheCheckpointsSpelling converts the fixture twice:
// as transformers saved it (dense block 0 under block_sparse_moe.) and with
// those names rewritten to mlp., as the published checkpoint writes them.
// Both must give the same tensors.
func TestKimiDenseBlockTakesTheCheckpointsSpelling(t *testing.T) {
	a, f := kimiSource(t)
	src := testmodels.Path(klDir)
	raw, err := os.ReadFile(f.Path)
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(raw)
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	renamed := 0
	for _, m := range []string{"gate_proj", "up_proj", "down_proj"} {
		old := "model.layers.0.block_sparse_moe." + m + ".weight"
		if v, ok := hdr[old]; ok {
			hdr["model.layers.0.mlp."+m+".weight"] = v
			delete(hdr, old)
			renamed++
		}
	}
	if renamed != 3 {
		t.Fatalf("renamed %d of 3 -- the fixture's dense block is not where this gate looks", renamed)
	}
	h, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	for len(h)%8 != 0 {
		h = append(h, ' ')
	}
	dir := t.TempDir()
	for _, e := range []string{"config.json", "tokenizer.json", "tokenizer_config.json"} {
		if b, err := os.ReadFile(filepath.Join(src, e)); err == nil {
			os.WriteFile(filepath.Join(dir, e), b, 0o644)
		}
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(append(out, h...), raw[8+n:]...)
	p := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(p, out, 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := safetensors.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	b, err := hfSourceLoaded(dir, []*safetensors.File{g})
	if err != nil {
		t.Fatalf("the checkpoint's spelling does not convert: %v", err)
	}
	klSameTensors(t, a, b)
}

// klSameTensors requires b to carry every tensor of a, by role and block, with
// the same shape and bytes.
func klSameTensors(t *testing.T, a, b *jlm.Source) {
	t.Helper()
	if len(a.Tensors) != len(b.Tensors) {
		t.Fatalf("%d tensors against %d", len(a.Tensors), len(b.Tensors))
	}
	type key struct {
		r     jlm.Role
		block int32
	}
	byKey := map[key]*jlm.Tensor{}
	for i := range b.Tensors {
		byKey[key{b.Tensors[i].Role, b.Tensors[i].Block}] = &b.Tensors[i]
	}
	for i := range a.Tensors {
		x := &a.Tensors[i]
		y := byKey[key{x.Role, x.Block}]
		if y == nil || x.Dims != y.Dims || !bytes.Equal(x.Data, y.Data) {
			t.Fatalf("%s (%v, block %d) has no identical twin", x.Name, x.Role, x.Block)
		}
	}
	if !reflect.DeepEqual(a.Config, b.Config) {
		t.Fatalf("configs differ:\n  %+v\n  %+v", *a.Config, *b.Config)
	}
}

// TestKimiReadsMoonshotsConfigForm converts the fixture with its config.json
// rewritten into Moonshot's configuration_kimi.py form (linear_attn_config
// with one-indexed kda_layers and full_attn_layers, first_k_dense_replace and
// moe_layer_freq in place of mlp_layer_types) and requires the same container.
func TestKimiReadsMoonshotsConfigForm(t *testing.T) {
	a, f := kimiSource(t)
	src := testmodels.Path(klDir)
	b, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c map[string]any
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	var kda, full []int
	for i, k := range c["layer_types"].([]any) {
		if k == "linear_attention" {
			kda = append(kda, i+1)
		} else {
			full = append(full, i+1)
		}
	}
	lead := 0
	for _, k := range c["mlp_layer_types"].([]any) {
		if k != "dense" {
			break
		}
		lead++
	}
	if len(kda) == 0 || len(full) == 0 || lead == 0 {
		t.Fatalf("the fixture has %d linear, %d full and %d dense-lead blocks; this gate needs all three", len(kda), len(full), lead)
	}
	c["linear_attn_config"] = map[string]any{
		"kda_layers": kda, "full_attn_layers": full,
		"num_heads": c["linear_num_heads"], "head_dim": c["linear_head_dim"],
		"short_conv_kernel_size": c["linear_conv_kernel_dim"],
	}
	c["first_k_dense_replace"], c["moe_layer_freq"] = lead, 1
	c["mla_use_nope"], c["moe_router_activation_func"] = true, "sigmoid"
	for _, k := range []string{"layer_types", "mlp_layer_types", "linear_num_heads", "linear_head_dim", "linear_conv_kernel_dim"} {
		delete(c, k)
	}
	dir := t.TempDir()
	out, _ := json.Marshal(c)
	os.WriteFile(filepath.Join(dir, "config.json"), out, 0o644)
	for _, e := range []string{"tokenizer.json", "tokenizer_config.json"} {
		if b, err := os.ReadFile(filepath.Join(src, e)); err == nil {
			os.WriteFile(filepath.Join(dir, e), b, 0o644)
		}
	}
	s, err := hfSourceLoaded(dir, []*safetensors.File{f})
	if err != nil {
		t.Fatalf("Moonshot's form does not convert: %v", err)
	}
	klSameTensors(t, a, s)
}
