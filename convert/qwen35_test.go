package convert

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// qwen35File builds a one-block Qwen3.5-shaped GGUF in memory, plus one
// multi-token-prediction block past the trunk, with every tensor F32 and every
// row of the two gate projections a distinct constant so the fused layout can
// be read back row by row.
func qwen35File(arch string) *meta.File {
	const (
		embd = 32
		nk   = 2 // key heads
		nv   = 4 // value heads: two per key head, so the pairing is observable
		kdim = 8
		vdim = 8
		conv = 4
	)
	chans := 2*nk*kdim + nv*vdim
	data := map[string][]byte{}
	var ts []meta.Tensor
	add := func(name string, fill func(i int) float32, dims ...uint64) {
		n := uint64(1)
		for _, d := range dims {
			n *= d
		}
		b := make([]byte, 4*n)
		for i := uint64(0); i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(fill(int(i))))
		}
		data[name] = b
		ts = append(ts, meta.Tensor{Name: name, Dims: dims, Type: quant.F32, NBytes: 4 * n})
	}
	small := func(i int) float32 { return float32(i%7) * 0.01 }
	add("token_embd.weight", small, embd, 16)
	add("output_norm.weight", small, embd)
	add("blk.0.attn_norm.weight", small, embd)
	add("blk.0.post_attention_norm.weight", small, embd)
	add("blk.0.attn_qkv.weight", small, embd, uint64(chans))
	add("blk.0.attn_gate.weight", small, embd, nv*vdim)
	// Row vh of beta is the constant 100+vh, of alpha 200+vh.
	add("blk.0.ssm_beta.weight", func(i int) float32 { return float32(100 + i/embd) }, embd, nv)
	add("blk.0.ssm_alpha.weight", func(i int) float32 { return float32(200 + i/embd) }, embd, nv)
	add("blk.0.ssm_conv1d.weight", small, conv, uint64(chans))
	add("blk.0.ssm_a", small, nv)
	add("blk.0.ssm_dt.bias", small, nv)
	add("blk.0.ssm_norm.weight", small, vdim)
	add("blk.0.ssm_out.weight", small, nv*vdim, embd)
	add("blk.0.ffn_gate.weight", small, embd, 64)
	add("blk.0.ffn_up.weight", small, embd, 64)
	add("blk.0.ffn_down.weight", small, 64, embd)
	// The MTP block: an attention block plus the tensors only a prediction
	// block has. It is kept, as block 1 past a one-block trunk.
	add("blk.1.attn_norm.weight", small, embd)
	add("blk.1.nextn.eh_proj.weight", small, 2*embd, embd)
	add("blk.1.nextn.enorm.weight", small, embd)
	add("blk.1.nextn.hnorm.weight", small, embd)
	add("blk.1.nextn.shared_head_norm.weight", small, embd)

	kv := map[string]meta.Value{
		"general.architecture":                     meta.MakeString(arch),
		arch + ".block_count":                      meta.MakeUint(2),
		arch + ".embedding_length":                 meta.MakeUint(embd),
		arch + ".feed_forward_length":              meta.MakeUint(64),
		arch + ".attention.head_count":             meta.MakeUint(2),
		arch + ".attention.head_count_kv":          meta.MakeUint(1),
		arch + ".attention.key_length":             meta.MakeUint(16),
		arch + ".attention.value_length":           meta.MakeUint(16),
		arch + ".attention.layer_norm_rms_epsilon": meta.MakeFloat(1e-6),
		arch + ".rope.dimension_count":             meta.MakeUint(8),
		arch + ".rope.dimension_sections":          meta.MakeInt32s([]int32{2, 1, 1, 0}),
		arch + ".ssm.conv_kernel":                  meta.MakeUint(conv),
		arch + ".ssm.state_size":                   meta.MakeUint(kdim),
		arch + ".ssm.group_count":                  meta.MakeUint(nk),
		arch + ".ssm.time_step_rank":               meta.MakeUint(nv),
		arch + ".ssm.inner_size":                   meta.MakeUint(nv * vdim),
		arch + ".nextn_predict_layers":             meta.MakeUint(1),
		arch + ".full_attention_interval":          meta.MakeUint(4),
	}
	f := meta.New("synth-"+arch+".gguf", kv, ts)
	f.Fetch = func(t *meta.Tensor) []byte { return data[t.Name] }
	return f
}

// TestQwen35ConvertsIntoQwen3NextsContainer is the converter half of the qwen35
// gate: the two gate projections fuse into ssm_ba in the interleave the engine
// reads, the value heads are marked tiled, the MTP block is kept as a
// prediction block past the trunk, and the M-RoPE sections are carried.
//
// The layout assertion is exact, row by row, because the wrong layouts all
// run: stacking beta above alpha, or pairing a head's beta with the next
// head's alpha, has the right shape and decays at another head's rate.
func TestQwen35ConvertsIntoQwen3NextsContainer(t *testing.T) {
	s, err := sourceOf(qwen35File("qwen35"))
	if err != nil {
		t.Fatal(err)
	}
	c := s.Config
	if c.NLayer != 1 || c.NMTP != 1 {
		t.Fatalf("NLayer %d, NMTP %d: the MTP block must be one prediction block past a one-block trunk",
			c.NLayer, c.NMTP)
	}
	mtp := map[jlm.Role]bool{}
	for i := range s.Tensors {
		if e := &s.Tensors[i]; e.Block == 1 {
			mtp[e.Role] = true
		}
	}
	for _, r := range []jlm.Role{jlm.RoleAttnNorm, jlm.RoleNextnEHProj, jlm.RoleNextnENorm,
		jlm.RoleNextnHNorm, jlm.RoleNextnHeadNorm} {
		if !mtp[r] {
			t.Fatalf("the prediction block lost its %v", r)
		}
	}
	if !c.Flags.Has(jlm.FlagDeltaKeyTiled) {
		t.Fatal("a qwen35 file with two value heads per key head was not marked tiled")
	}
	if c.RopeSections != [4]uint32{2, 1, 1, 0} {
		t.Fatalf("rope sections %v", c.RopeSections)
	}
	var ba *jlm.Tensor
	for i := range s.Tensors {
		e := &s.Tensors[i]
		if strings.Contains(e.Name, "ssm_alpha") {
			t.Fatalf("%s reached the container unfused", e.Name)
		}
		if e.Role == jlm.RoleSSMBA {
			ba = e
		}
	}
	if ba == nil {
		t.Fatal("no ssm_ba was written")
	}
	if ba.Dims[0] != 32 || ba.Dims[1] != 8 || ba.Type != jlm.TypeF32 {
		t.Fatalf("ssm_ba is %v %v, want [32, 8] F32 (same-type sources are byte copies)", ba.Dims[:2], ba.Type)
	}
	// [kh 0: beta 0, beta 1, alpha 0, alpha 1 | kh 1: beta 2, beta 3, alpha 2, alpha 3]
	want := []float32{100, 101, 200, 201, 102, 103, 202, 203}
	for r, w := range want {
		for j := 0; j < 32; j++ {
			got := math.Float32frombits(binary.LittleEndian.Uint32(ba.Data[4*(r*32+j):]))
			if got != w {
				// 100+vh is value head vh's beta, 200+vh its alpha.
				t.Fatalf("ssm_ba row %d col %d is %v, want %v", r, j, got, w)
			}
		}
	}

	// The same file under qwen3next's name is grouped: llama.cpp's qwen3next
	// converter does not reorder, so the flag is about the qwen35 converter.
	g, err := sourceOf(qwen35File("qwen3next"))
	if err != nil {
		t.Fatal(err)
	}
	if g.Config.Flags.Has(jlm.FlagDeltaKeyTiled) {
		t.Fatal("a qwen3next file was marked tiled")
	}
}

// TestQwen35RefusesAnUnpairedGate: an alpha with no beta was skipped by the
// fuse and must not vanish silently.
func TestQwen35RefusesAnUnpairedGate(t *testing.T) {
	f := qwen35File("qwen35")
	var keep []meta.Tensor
	for _, tt := range f.Tensors {
		if tt.Name != "blk.0.ssm_beta.weight" {
			keep = append(keep, tt)
		}
	}
	f.Tensors = keep
	f.Reindex()
	if _, err := sourceOf(f); err == nil || !strings.Contains(err.Error(), "ssm_alpha") {
		t.Fatalf("an alpha without its beta converted (err %v)", err)
	}
}
