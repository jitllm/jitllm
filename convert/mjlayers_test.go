package convert

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestTheLibraryStatesTheRecurrentLayers holds each recurrent and hybrid GGUF
// architecture to what its Go reader set by name before the files did: the
// flags, the recurrent geometry off its keys and each layer's kind off the
// tensors it carries.
func TestTheLibraryStatesTheRecurrentLayers(t *testing.T) {
	ssmKeys := map[string]meta.Value{"ssm.conv_kernel": meta.MakeUint(4), "ssm.group_count": meta.MakeUint(2),
		"ssm.inner_size": meta.MakeUint(64), "ssm.state_size": meta.MakeUint(16), "ssm.time_step_rank": meta.MakeUint(8)}
	ssd := jlm.SSMConfig{ConvKernel: 4, Groups: 2, Inner: 64, StateSize: 16, NHeadV: 8}
	qwen := jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagAttnOutGate
	full, lin := jlm.LayerFullAttn, jlm.LayerLinearAttn
	for _, tc := range []struct {
		arch    string
		keys    map[string]meta.Value
		tensors [][]string
		flags   jlm.Flags
		ssm     jlm.SSMConfig
		kinds   []jlm.LayerKind
	}{
		{arch: "qwen3next", keys: ssmKeys, tensors: [][]string{{"ssm_conv1d.weight"}, {"attn_q.weight"}},
			flags: qwen, ssm: ssd, kinds: []jlm.LayerKind{lin, full}},
		{arch: "qwen35", keys: ssmKeys, tensors: [][]string{{"ssm_conv1d.weight"}, {"attn_q.weight"}},
			flags: qwen | jlm.FlagDeltaKeyTiled, ssm: ssd, kinds: []jlm.LayerKind{lin, full}},
		{arch: "clef", keys: with2(ssmKeys, "ssm.time_step_rank", meta.MakeUint(2)), tensors: [][]string{{"ssm_conv1d.weight"}},
			flags: qwen, ssm: jlm.SSMConfig{ConvKernel: 4, Groups: 2, Inner: 64, StateSize: 16, NHeadV: 2}, kinds: []jlm.LayerKind{lin}},
		{arch: "mamba2", keys: ssmKeys, tensors: [][]string{{"ssm_conv1d.weight"}, {"ssm_conv1d.weight"}},
			flags: jlm.FlagNoPosEnc, ssm: ssd, kinds: []jlm.LayerKind{jlm.LayerSSD, jlm.LayerSSD}},
		{arch: "falcon-h1", keys: ssmKeys, tensors: [][]string{{"ssm_conv1d.weight", "attn_q.weight"}, {"ssm_conv1d.weight"}},
			flags: jlm.FlagRopeNeox, ssm: ssd, kinds: []jlm.LayerKind{jlm.LayerSSDAttn, jlm.LayerSSD}},
		{arch: "granitehybrid", keys: with2(ssmKeys, "ssm.group_count", nil), tensors: [][]string{{"ssm_conv1d.weight"}, {"attn_q.weight"}},
			ssm: jlm.SSMConfig{ConvKernel: 4, Groups: 1, Inner: 64, StateSize: 16, NHeadV: 8}, kinds: []jlm.LayerKind{jlm.LayerSSD, full}},
		{arch: "nemotron_h", keys: ssmKeys, ssm: ssd},
		{arch: "mamba", keys: ssmKeys, tensors: [][]string{{"ssm_conv1d.weight"}},
			flags: jlm.FlagNoPosEnc, ssm: jlm.SSMConfig{ConvKernel: 4, Groups: 1, Inner: 64, StateSize: 16, NHeadV: 64},
			kinds: []jlm.LayerKind{jlm.LayerScan}},
		{arch: "jamba", keys: ssmKeys, tensors: [][]string{{"attn_q.weight"}, {"ssm_conv1d.weight"}},
			flags: jlm.FlagNoPosEnc | jlm.FlagNoExpertNorm, ssm: jlm.SSMConfig{ConvKernel: 4, Groups: 1, Inner: 64, StateSize: 16, NHeadV: 64},
			kinds: []jlm.LayerKind{full, jlm.LayerScan}},
		{arch: "lfm2", keys: map[string]meta.Value{"shortconv.l_cache": meta.MakeUint(3), "embedding_length": meta.MakeUint(32)},
			tensors: [][]string{{"shortconv.conv.weight"}, {"attn_q.weight", "attn_q_norm.weight"}},
			flags:   jlm.FlagRopeNeox | jlm.FlagQKNorm, ssm: jlm.SSMConfig{ConvKernel: 3, Inner: 32},
			kinds: []jlm.LayerKind{jlm.LayerShortConv, full}},
		{arch: "lfm2", keys: map[string]meta.Value{"shortconv.l_cache": meta.MakeUint(3), "embedding_length": meta.MakeUint(32)},
			tensors: [][]string{{"shortconv.conv.weight"}, {"attn_q.weight"}},
			flags:   jlm.FlagRopeNeox, ssm: jlm.SSMConfig{ConvKernel: 3, Inner: 32},
			kinds: []jlm.LayerKind{jlm.LayerShortConv, full}},
	} {
		f := layerFile(tc.arch, tc.keys, tc.tensors)
		c := &Config{Config: &jlm.Config{NLayer: uint32(len(tc.tensors))}}
		if err := stateContainer(f, c); err != nil {
			t.Fatalf("%s: %v", tc.arch, err)
		}
		if err := stateLayers(f, c); err != nil {
			t.Fatalf("%s: %v", tc.arch, err)
		}
		if c.Flags != tc.flags || c.SSM != tc.ssm || !slices.Equal(c.LayerKinds, tc.kinds) {
			t.Errorf("%s: flags %v, recurrent %+v, kinds %v; want %v, %+v, %v", tc.arch, c.Flags, c.SSM, c.LayerKinds,
				tc.flags, tc.ssm, tc.kinds)
		}
	}
	for want, tc := range map[string]struct {
		arch    string
		keys    map[string]meta.Value
		tensors [][]string
	}{
		"incomplete recurrent geometry": {"mamba2", with2(ssmKeys, "ssm.state_size", nil), [][]string{{"ssm_conv1d.weight"}}},
		"two or more":                   {"lfm2", map[string]meta.Value{"shortconv.l_cache": meta.MakeUint(1), "embedding_length": meta.MakeUint(32)}, [][]string{{"shortconv.conv.weight"}}},
		"not 3 whole heads":             {"mamba2", with2(with2(ssmKeys, "ssm.time_step_rank", meta.MakeUint(3)), "ssm.group_count", meta.MakeUint(1)), [][]string{{"ssm_conv1d.weight"}}},
		"per key head": {"qwen35", with2(with2(with2(ssmKeys, "ssm.time_step_rank", meta.MakeUint(6)), "ssm.group_count", meta.MakeUint(4)),
			"ssm.inner_size", meta.MakeUint(96)), [][]string{{"ssm_conv1d.weight"}}},
		"no layer carries": {"mamba", ssmKeys, [][]string{{"attn_q.weight"}}},
	} {
		f := layerFile(tc.arch, tc.keys, tc.tensors)
		c := &Config{Config: &jlm.Config{NLayer: uint32(len(tc.tensors))}}
		err := stateContainer(f, c)
		if err == nil {
			err = stateLayers(f, c)
		}
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s with %v: %v, want an error naming %q", tc.arch, tc.keys, err, want)
		}
	}
	for _, tc := range []struct{ states, want string }{
		{`"x": {"recurrent": {"inner": "8"}}`, "convolution's taps"},
		{`"x": {"layers": [{"tensors": ["ssm_conv1d.weight"], "kind": "mamba"}]}`, "layer kind"},
		{`"x": {"layers": [{"kind": "ssd"}]}`, "layer kind"},
		{`"x": {"recurrent": {"conv_kernel": "taps"}}`, "is not the keys"},
	} {
		if _, err := parseModelJSON([]byte(`{"gguf": ["x"], "container": {`+tc.states+`}}`), "test"); err == nil ||
			!strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.states, err, tc.want)
		}
	}
}

// TestOneRuleReadsALayerCount holds the per-layer head counts to one reading:
// zeros mark the layers a count does not apply to, a count that differs by
// layer is refused unless the blocks' own geometries hold it, and a file
// stating no heads converts only where no layer attends.
func TestOneRuleReadsALayerCount(t *testing.T) {
	f := func(v meta.Value) *meta.File {
		return meta.New("x.gguf", map[string]meta.Value{"general.architecture": meta.MakeString("x"), "x.n": v}, nil)
	}
	for _, tc := range []struct {
		v    meta.Value
		n    uint32
		vary bool
	}{
		{meta.MakeUint(4), 4, false},
		{meta.MakeInt32s([]int32{0, 4, 0, 4}), 4, false},
		{meta.MakeInt32s([]int32{0, 0}), 0, false},
		{meta.MakeInt32s([]int32{2, 4}), 0, true},
	} {
		n, per, err := layerCount(f(tc.v), "n", 9)
		if err != nil || n != tc.n || (per != nil) != tc.vary {
			t.Errorf("%v: %d %v %v, want %d varying %v", tc.v.Short(), n, per, err, tc.n, tc.vary)
		}
	}
	if _, _, err := layerCount(f(meta.MakeInt32s([]int32{4, -1})), "n", 0); err == nil {
		t.Error("a negative count was read")
	}
	if _, err := uniformCount(f(meta.MakeInt32s([]int32{2, 4})), "n", 0); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("a count that differs by layer: %v", err)
	}
	heads := func(key string, v meta.Value) *meta.File {
		return layerFile("x", map[string]meta.Value{key: v}, nil)
	}
	kv := heads("attention.head_count_kv", meta.MakeInt32s([]int32{4, 2}))
	c := &Config{Config: &jlm.Config{NLayer: 2, NKVHead: 4}}
	if err := countsHeld(c, kv); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("kv heads no geometry holds: %v", err)
	}
	c.Geoms = []jlm.Geom{{NKVHead: 4}, {NKVHead: 2}}
	if err := countsHeld(c, kv); err != nil {
		t.Errorf("kv heads the geometries hold: %v", err)
	}
	c = &Config{Config: &jlm.Config{NLayer: 3, NKVHead: 8}}
	for _, key := range []string{"attention.head_count", "attention.head_count_kv"} {
		gap := heads(key, meta.MakeInt32s([]int32{8, 0, 8}))
		c.LayerKinds = nil
		if err := countsHeld(c, gap); err == nil {
			t.Errorf("%s: a zero on a layer that attends, no kinds stated", key)
		}
		c.LayerKinds = []jlm.LayerKind{jlm.LayerFullAttn, jlm.LayerFullAttn, jlm.LayerFullAttn}
		if err := countsHeld(c, gap); err == nil {
			t.Errorf("%s: a zero on a layer stated to attend", key)
		}
		c.LayerKinds[1] = jlm.LayerSSD
		if err := countsHeld(c, gap); err != nil {
			t.Errorf("%s: a zero on the layer that does not attend: %v", key, err)
		}
	}
	c = &Config{Config: &jlm.Config{NLayer: 3, NKVHead: 8}}
	c.LayerKinds = []jlm.LayerKind{jlm.LayerSSD, jlm.LayerFullAttn, jlm.LayerSSD}
	for _, tc := range []struct {
		kv []int32
		ok bool
	}{{[]int32{0, 0, 8, 0, 0}, true}, {[]int32{8, 0, 8, 0, 0}, false}, {[]int32{0, 0, 0, 0, 0}, false}} {
		merged := layerFile("x", map[string]meta.Value{"block_count": meta.MakeUint(5),
			"attention.head_count_kv": meta.MakeInt32s(tc.kv)}, nil)
		if err := countsHeld(c, merged); (err == nil) != tc.ok {
			t.Errorf("five source layers merged into three blocks, kv heads %v: %v", tc.kv, err)
		}
	}
	none := heads("attention.head_count", meta.MakeUint(0))
	c = &Config{Config: &jlm.Config{NLayer: 2, NKVHead: 1}}
	c.LayerKinds = []jlm.LayerKind{jlm.LayerScan, jlm.LayerFullAttn}
	if err := countsHeld(c, none); err == nil {
		t.Error("no heads on a model whose layer 1 attends")
	}
	c.LayerKinds[1] = jlm.LayerSSD
	if err := countsHeld(c, none); err != nil {
		t.Errorf("no heads on a model no layer of which attends: %v", err)
	}
}

func layerFile(arch string, keys map[string]meta.Value, tensors [][]string) *meta.File {
	kv := map[string]meta.Value{"general.architecture": meta.MakeString(arch)}
	for k, v := range keys {
		kv[arch+"."+k] = v
	}
	var ts []meta.Tensor
	for i, names := range tensors {
		for _, n := range names {
			ts = append(ts, meta.Tensor{Name: "blk." + itoa(i) + "." + n, Dims: []uint64{1}})
		}
	}
	return meta.New(arch+".gguf", kv, ts)
}

// with2 is m with k set to v, or deleted where v is nil.
func with2(m map[string]meta.Value, k string, v any) map[string]meta.Value {
	out := map[string]meta.Value{}
	for k, v := range m {
		out[k] = v
	}
	delete(out, k)
	if mv, ok := v.(meta.Value); ok {
		out[k] = mv
	}
	return out
}
