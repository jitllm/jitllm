package convert

import (
	"bytes"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// k3Key is a container tensor's identity across two sources.
type k3Key struct {
	role         jlm.Role
	block, index int32
}

// k3Sources converts one Kimi-K3 twice -- its HuggingFace directory through
// the safetensors path, and the GGUF llama.cpp's converter wrote from the
// same checkpoint -- and demands every Config field equal. It returns the two
// tensor sets by identity.
func k3Sources(t *testing.T, dir, gp string) (hf, gg map[k3Key]*jlm.Tensor) {
	t.Helper()
	for _, p := range []string{filepath.Join(dir, "config.json"), gp} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (docs/testing.md names where the Kimi-K3 "+
				"checkpoints and their GGUFs come from) -- this gate proved nothing", err)
		}
	}
	a, b := hfSource(t, dir), ggufSource(t, gp)
	ca, cb := reflect.ValueOf(*a.Config), reflect.ValueOf(*b.Config)
	for i := 0; i < ca.NumField(); i++ {
		if !ca.Type().Field(i).IsExported() {
			continue
		}
		if x, y := ca.Field(i).Interface(), cb.Field(i).Interface(); !reflect.DeepEqual(x, y) {
			t.Errorf("Config.%s: safetensors %v, gguf %v", ca.Type().Field(i).Name, x, y)
		}
	}
	index := func(s *jlm.Source) map[k3Key]*jlm.Tensor {
		m := map[k3Key]*jlm.Tensor{}
		for i := range s.Tensors {
			e := &s.Tensors[i]
			k := k3Key{e.Role, e.Block, e.Index}
			if m[k] != nil {
				t.Fatalf("%v block %d twice in one source", e.Role, e.Block)
			}
			m[k] = e
		}
		return m
	}
	hf, gg = index(a), index(b)
	for k := range gg {
		if hf[k] == nil {
			t.Errorf("the gguf has %v block %d and safetensors does not", k.role, k.block)
		}
	}
	for k := range hf {
		if gg[k] == nil {
			t.Errorf("safetensors has %v block %d and the gguf does not", k.role, k.block)
		}
	}
	return hf, gg
}

// k3Values decodes a tensor to f32.
func k3Values(t *testing.T, e *jlm.Tensor) []float32 {
	t.Helper()
	src, ok := sourceType(e.Type)
	if !ok {
		t.Fatalf("%s: %v has no source type", e.Name, e.Type)
	}
	n := uint64(len(e.Data)) / src.BlockBytes() * src.BlockElems()
	v := make([]float32, n)
	if err := quant.Dequant32(src, e.Data, v); err != nil {
		t.Fatalf("%s: %v", e.Name, err)
	}
	return v
}

// TestK3SafetensorsMatchesItsGGUF converts Kimi-K3 from its HuggingFace
// checkpoint and from the F32 GGUF llama.cpp's converter wrote from it, and
// demands the same container: every Config field, and every tensor byte for
// byte. Both are F32, so a role, a join order, a split, the residual scores'
// product or a dropped tensor is an inequality, not a tolerance. The GGUF
// container is the one held to Moonshot's code (TestKimiK3MatchesReference,
// TestKimiK3RealMatchesReference), so this carries that gate to the
// safetensors input. The fixture states a decay bound and the trained
// checkpoint none.
//
// One exception, measured: the KDA decay rate is -exp(A_log), which
// llama.cpp's converter takes in float32 torch and this converter in float64
// rounded once; the two may part by one ulp, and no more.
func TestK3SafetensorsMatchesItsGGUF(t *testing.T) {
	for _, c := range []struct{ name, dir, gguf string }{
		{"synth-kimik3", filepath.Join(c6HFDir(), "synth-kimik3"), testmodels.Path("synth-kimik3.gguf")},
		{"Kimi-K3-0.40B", testmodels.Path("kimik3/Kimi-K3-0.40B-hf"), testmodels.Path("kimik3/Kimi-K3-0.40B-F32.gguf")},
	} {
		t.Run(c.name, func(t *testing.T) {
			hf, gg := k3Sources(t, c.dir, c.gguf)
			same, ulp := 0, 0
			for k, e := range gg {
				x := hf[k]
				if x == nil {
					continue
				}
				switch {
				case x.Type != e.Type || x.NDim != e.NDim || x.Dims != e.Dims:
					t.Errorf("%v block %d: safetensors %v %v, gguf %v %v", k.role, k.block,
						x.Type, x.Dims[:x.NDim], e.Type, e.Dims[:e.NDim])
				case bytes.Equal(x.Data, e.Data):
					same++
				case k.role == jlm.RoleSSMA && e.Type == jlm.TypeF32:
					a, b := k3Values(t, x), k3Values(t, e)
					for i := range a {
						d := int64(math.Float32bits(a[i])) - int64(math.Float32bits(b[i]))
						if d < -1 || d > 1 {
							t.Errorf("ssm_a block %d [%d]: %g against %g, past one ulp", k.block, i, a[i], b[i])
							break
						}
					}
					ulp++
				default:
					t.Errorf("%v block %d: the bytes differ", k.role, k.block)
				}
			}
			if same == 0 {
				t.Fatal("no tensor compared -- this gate proved nothing")
			}
			t.Logf("every Config field equal; %d tensors identical, %d decay rates within one ulp", same, ulp)
		})
	}
}

// TestK3MXFP4ExpertsMatchTheGGUF converts inference-optimization's
// Kimi-K3-0.40B-MXFP4 (compressed-tensors mxfp4-pack-quantized, the release's
// format) from the checkpoint and from the MXFP4 GGUF llama.cpp's converter
// wrote from it (murillo2000/Kimi-K3-0.40B-GGUF), and demands every routed
// expert bank byte for byte: the codes and scales are moved, never
// requantized. The GGUF stores the rest at F16, so the rest is compared by
// value: the checkpoint's MXFP4 shared experts decode to exactly what the
// GGUF's F16 holds, and every float within F16's rounding.
//
// The violations are the repacks a converter could plausibly get wrong, run
// over the checkpoint's own codes: each must break the banks' equality.
func TestK3MXFP4ExpertsMatchTheGGUF(t *testing.T) {
	dir := testmodels.Path("kimik3/Kimi-K3-0.40B-MXFP4-hf")
	gp := testmodels.Path("kimik3/Kimi-K3-0.40B-MXFP4.gguf")
	hf, gg := k3Sources(t, dir, gp)
	banks, valued := 0, 0
	for k, e := range gg {
		x := hf[k]
		if x == nil || x.Dims != e.Dims {
			if x != nil {
				t.Errorf("%v block %d: safetensors %v, gguf %v", k.role, k.block, x.Dims, e.Dims)
			}
			continue
		}
		if jlm.ExpertBank(k.role) {
			if x.Type != jlm.TypeMX4 || e.Type != jlm.TypeMX4 {
				t.Errorf("%v block %d: safetensors %v, gguf %v; both should be MXFP4", k.role, k.block, x.Type, e.Type)
			} else if !bytes.Equal(x.Data, e.Data) {
				t.Errorf("%v block %d: the MXFP4 bytes differ", k.role, k.block)
			} else {
				banks++
			}
			continue
		}
		a, b := k3Values(t, x), k3Values(t, e)
		if len(a) != len(b) {
			t.Errorf("%v block %d: %d values against %d", k.role, k.block, len(a), len(b))
			continue
		}
		exact := x.Type == jlm.TypeMX4 // an F16 holds every MXFP4 value of this range
		var num, den float64
		for i := range a {
			if exact && a[i] != b[i] {
				t.Errorf("%v block %d [%d]: MXFP4 decodes to %g, the GGUF's F16 holds %g", k.role, k.block, i, a[i], b[i])
				break
			}
			d := float64(a[i] - b[i])
			num, den = num+d*d, den+float64(a[i])*float64(a[i])
		}
		if nmse := num / den; den > 0 && !(nmse < 1e-6) {
			t.Errorf("%v block %d: NMSE %.3e past F16's rounding", k.role, k.block, nmse)
		}
		valued++
	}
	if banks == 0 {
		t.Fatal("no MXFP4 bank compared -- this gate proved nothing")
	}
	t.Logf("%d MXFP4 banks identical byte for byte; %d other tensors equal by value", banks, valued)

	// The violations, over every routed expert's own codes and scales.
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, v := range []struct {
		name   string
		repack func(p, s []byte, rows, cols int) ([]byte, error)
	}{
		{"the repack itself", mxfp4Blocks},
		{"nibble order swapped", func(p, s []byte, rows, cols int) ([]byte, error) {
			q := bytes.Clone(p)
			for i := range q {
				q[i] = q[i]<<4 | q[i]>>4
			}
			return mxfp4Blocks(q, s, rows, cols)
		}},
		{"the scale off by one (bias 128 for 127)", func(p, s []byte, rows, cols int) ([]byte, error) {
			z := bytes.Clone(s)
			for i := range z {
				z[i]--
			}
			return mxfp4Blocks(p, z, rows, cols)
		}},
		{"carried as packed, no block permutation", func(p, s []byte, rows, cols int) ([]byte, error) {
			out := make([]byte, 0, len(p)+len(s))
			for b := range s {
				out = append(append(out, s[b]), p[16*b:16*b+16]...)
			}
			return out, nil
		}},
	} {
		t.Run(v.name, func(t *testing.T) {
			same, diff := 0, 0
			for k, e := range gg {
				if !jlm.ExpertBank(k.role) {
					continue
				}
				w := map[jlm.Role]string{jlm.RoleExpGateBank: "w1", jlm.RoleExpDownBank: "w2", jlm.RoleExpUpBank: "w3"}[k.role]
				sheet := len(e.Data) / int(e.Dims[2])
				for x := range int(e.Dims[2]) {
					base := k3BlockPrefix + itoa(int(k.block)) + ".block_sparse_moe.experts." + itoa(x) + "." + w
					pt, ok1 := f.Get(base + ".weight_packed")
					st, ok2 := f.Get(base + ".weight_scale")
					if !ok1 || !ok2 {
						t.Fatalf("%s: no packed weight and scale", base)
					}
					p, err := f.ReadTensor(pt)
					if err != nil {
						t.Fatal(err)
					}
					s, err := f.ReadTensor(st)
					if err != nil {
						t.Fatal(err)
					}
					got, err := v.repack(p, s, int(pt.Shape[0]), int(2*pt.Shape[1]))
					if err != nil {
						t.Fatal(err)
					}
					if bytes.Equal(got, e.Data[x*sheet:(x+1)*sheet]) {
						same++
					} else {
						diff++
					}
				}
			}
			t.Logf("%d expert matrices equal to the GGUF's, %d not", same, diff)
			if v.name == "the repack itself" {
				if diff != 0 || same == 0 {
					t.Errorf("the repack parts from the GGUF on %d of %d", diff, same+diff)
				}
			} else if same != 0 {
				t.Errorf("the gate cannot see it: %d expert matrices still equal", same)
			}
		})
	}
}

// TestMXFP4RepackIsLossless holds mxfp4Blocks to compressed-tensors' own
// reading of its bytes -- unpack_fp4_from_uint8 (element 2i the low nibble,
// kE2M1ToFloat on the magnitude, bit 3 the sign) and decompress_mx_scale
// (2^(s-127)) -- transcribed here independently of the repack, over every
// code and a sweep of scales: the GGUF blocks it writes decode
// (quant.Dequant32's MXFP4) to the same value, bit for bit, everywhere.
func TestMXFP4RepackIsLossless(t *testing.T) {
	e2m1 := [8]float64{0, 0.5, 1, 1.5, 2, 3, 4, 6}
	r := rand.New(rand.NewSource(3))
	for _, shape := range [][2]int{{1, 32}, {3, 64}, {5, 96}, {2, 512}} {
		rows, cols := shape[0], shape[1]
		p := make([]byte, rows*cols/2)
		s := make([]byte, rows*cols/32)
		r.Read(p)
		for i := range s {
			s[i] = byte(100 + r.Intn(50)) // 2^-27 .. 2^22, every value normal in f32
		}
		blocks, err := mxfp4Blocks(p, s, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, rows*cols)
		if err := quant.Dequant32(quant.MXFP4, blocks, got); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			row, col := i/cols, i%cols
			b := p[(row*cols+col)/2]
			code := b & 0xF
			if col%2 == 1 {
				code = b >> 4
			}
			want := e2m1[code&7] * math.Ldexp(1, int(s[row*cols/32+col/32])-127)
			if code&8 != 0 {
				want = -want
			}
			if float64(got[i]) != want {
				t.Fatalf("%dx%d element (%d, %d): code %#x scale %d decodes to %g, compressed-tensors reads %g",
					rows, cols, row, col, code, s[row*cols/32+col/32], got[i], want)
			}
		}
	}
	if _, err := mxfp4Blocks(make([]byte, 15), make([]byte, 1), 1, 32); err == nil {
		t.Fatal("a short code row was accepted")
	}
}

// TestCompressedTensorsRefusals: every compressed-tensors config this reader
// cannot honour is refused by name, and the release's is read.
func TestCompressedTensorsRefusals(t *testing.T) {
	release := `{"quant_method":"compressed-tensors","format":"mxfp4-pack-quantized",
		"quantization_status":"compressed","kv_cache_scheme":null,
		"config_groups":{"group_0":{"format":"mxfp4-pack-quantized","input_activations":null,
		"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,
		"dynamic":false,"group_size":32,"num_bits":4,"observer":"minmax","observer_kwargs":{},
		"scale_dtype":"torch.uint8","strategy":"group","symmetric":true,"type":"float","zp_dtype":null}}},
		"ignore":["re:.*self_attn.*"]}`
	cfg := func(q string) *hfConfig {
		b := []byte(`{"architectures":["KimiK3ForConditionalGeneration"],"text_config":{"quantization_config":` + q + `}}`)
		c := &hfConfig{path: "config.json"}
		if err := jsonUnmarshalKeys(b, c); err != nil {
			t.Fatal(err)
		}
		return c
	}
	if mx, err := compressedTensorsOf(cfg(release)); err != nil || !mx {
		t.Fatalf("the release's config: %v, %v", mx, err)
	}
	for _, c := range []struct{ name, from, to, want string }{
		{"int4 pack-quantized", `"compressed-tensors","format":"mxfp4-pack-quantized"`,
			`"compressed-tensors","format":"pack-quantized"`, `format "pack-quantized"`},
		{"not compressed", `"compressed","kv_cache`, `"frozen","kv_cache`, "quantization_status"},
		{"8-bit", `"num_bits":4`, `"num_bits":8`, "8-bit"},
		{"NVFP4's group of 16", `"group_size":32`, `"group_size":16`, "group size"},
		{"asymmetric", `"symmetric":true`, `"symmetric":false`, "asymmetric"},
		{"an FP8 scale", `"scale_dtype":"torch.uint8"`, `"scale_dtype":"torch.float8_e4m3fn"`, "scale_dtype"},
		{"a KV cache scheme", `"kv_cache_scheme":null`, `"kv_cache_scheme":{"num_bits":8}`, "kv_cache_scheme"},
		{"static input activations", `"input_activations":null`,
			`"input_activations":{"num_bits":4,"type":"float","dynamic":false}`, "static input"},
		{"output activations", `"output_activations":null`, `"output_activations":{"num_bits":8}`, "output"},
	} {
		t.Run(c.name, func(t *testing.T) {
			q := strings.Replace(release, c.from, c.to, 1)
			if q == release {
				t.Fatalf("the mutation %q matched nothing", c.from)
			}
			_, err := compressedTensorsOf(cfg(q))
			if err == nil || !strings.Contains(err.Error(), c.want) || !errors.Is(err, ErrNotImplemented) {
				t.Fatalf("got %v, want a refusal naming %q", err, c.want)
			}
		})
	}
	// Two configs that disagree.
	b := []byte(`{"quantization_config":` + strings.Replace(release, `"num_bits":4`, `"num_bits":8`, 1) +
		`,"text_config":{"quantization_config":` + release + `}}`)
	c := &hfConfig{path: "config.json"}
	if err := jsonUnmarshalKeys(b, c); err != nil {
		t.Fatal(err)
	}
	if _, err := compressedTensorsOf(c); err == nil || !strings.Contains(err.Error(), "differs") {
		t.Fatalf("two quantization_configs that disagree: %v", err)
	}
	// A packed weight with no config to read it by.
	all := map[string]hfLoc{"x.weight_packed": {t: &safetensors.Tensor{DType: safetensors.U8, Shape: []uint64{1, 16}}},
		"x.weight_scale": {t: &safetensors.Tensor{DType: safetensors.U8, Shape: []uint64{1, 1}}}}
	if err := mergeMXFP4(all, false); err == nil {
		t.Fatal("a packed weight with no compressed-tensors config was accepted")
	}
	// A scale row that does not cover the codes.
	all["x.weight_scale"] = hfLoc{t: &safetensors.Tensor{DType: safetensors.U8, Shape: []uint64{1, 2}}}
	if err := mergeMXFP4(all, true); err == nil || !strings.Contains(err.Error(), "scales") {
		t.Fatalf("a mis-shaped scale: %v", err)
	}
}

// TestK3ConfigRefusals: each Kimi-K3 graph switch the engine does not run is
// refused by its key, the class's default included, before any tensor is
// read.
func TestK3ConfigRefusals(t *testing.T) {
	dir := filepath.Join(c6HFDir(), "synth-kimik3")
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
	}
	for _, c := range []struct{ name, from, to, want string }{
		{"a SwiGLU FFN", `"hidden_act": "situ"`, `"hidden_act": "silu"`, "hidden_act"},
		{"MLA with the rotary", `"mla_use_nope": true`, `"mla_use_nope": false`, "mla_use_nope"},
		{"mla_use_nope absent (the class says false)", `"mla_use_nope": true,`, ``, "mla_use_nope"},
		{"the low-rank KDA gate", `"use_full_rank_gate": true`, `"use_full_rank_gate": false`, "use_full_rank_gate"},
		{"a softmax router", `"moe_router_activation_func": "sigmoid"`, `"moe_router_activation_func": "softmax"`, "sigmoid"},
		{"a mixture every other block", `"moe_layer_freq": 1`, `"moe_layer_freq": 2`, "moe_layer_freq"},
		{"other situ bounds", `"activation_situ_beta": 4.0`, `"activation_situ_beta": 3.0`, "situ"},
		{"a layer in both lists", `"full_attn_layers": [`, `"full_attn_layers": [1, `, "twice"},
	} {
		t.Run(c.name, func(t *testing.T) {
			b := strings.Replace(string(raw), c.from, c.to, 1)
			if b == string(raw) {
				t.Fatalf("the mutation %q matched nothing", c.from)
			}
			hc := &hfConfig{path: "config.json"}
			if err := jsonUnmarshalKeys([]byte(b), hc); err != nil {
				t.Fatal(err)
			}
			_, err := kimiK3HFConfig(hc, []string{"language_model.lm_head.weight"})
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("got %v, want a refusal naming %q", err, c.want)
			}
		})
	}
}

// TestPlanIsTheWrite holds the dry run to the conversion: PlanSafetensors,
// which reads headers only, states the header and the size FromSafetensors
// then writes, byte for byte, on Kimi-K3's two inputs (F32, and the release's
// compressed-tensors MXFP4).
func TestPlanIsTheWrite(t *testing.T) {
	for _, dir := range []string{filepath.Join(c6HFDir(), "synth-kimik3"),
		testmodels.Path("kimik3/Kimi-K3-0.40B-MXFP4-hf")} {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
			}
			p, err := PlanSafetensors(dir)
			if err != nil {
				t.Fatal(err)
			}
			// The container is written beside the models, never to /tmp.
			tmp, err := os.MkdirTemp(testmodels.Dir(), "plan-")
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(tmp)
			dst := filepath.Join(tmp, "m"+jlm.Ext)
			h, err := FromSafetensors(dir, dst, jlm.Fingerprint{})
			if err != nil {
				t.Fatal(err)
			}
			fi, err := os.Stat(dst)
			if err != nil {
				t.Fatal(err)
			}
			if uint64(fi.Size()) != p.Size {
				t.Errorf("planned %d bytes, wrote %d", p.Size, fi.Size())
			}
			if h.NTensors != p.Header.NTensors || h.NBlocks != p.Header.NBlocks || h.PageSize != p.Header.PageSize ||
				h.NExpPages != p.Header.NExpPages || h.ExpPageSize != p.Header.ExpPageSize {
				t.Errorf("planned %+v, wrote %+v", *p.Header, *h)
			}
			t.Logf("planned and wrote %d bytes, %d tensors, %d expert pages", p.Size, h.NTensors, h.NExpPages)
		})
	}
}

// TestALogPaddingIsTakenOnlyWhenZero: Kimi-K3's release stores A_log as 128
// entries for 96 heads, the last 32 zero (read off the Hub). The heads'
// entries are taken, as llama.cpp's converter takes them; a tail that is not
// zero is another geometry and is refused.
func TestALogPaddingIsTakenOnlyWhenZero(t *testing.T) {
	c := &jlm.Config{SSM: jlm.SSMConfig{NHeadV: 2, StateSize: 3}}
	e := jlm.Tensor{Type: jlm.TypeF32, NDim: 1, Dims: [4]uint64{4}, Name: "A_log"}
	padded := f32AsBytes([]float32{0.5, -1, 0, 0})
	out, err := kimiExpandALog(e, padded, &Config{Config: c})
	if err != nil {
		t.Fatal(err)
	}
	want := f32AsBytes([]float32{
		float32(-math.Exp(0.5)), float32(-math.Exp(0.5)), float32(-math.Exp(0.5)),
		float32(-math.Exp(-1)), float32(-math.Exp(-1)), float32(-math.Exp(-1))})
	if !bytes.Equal(out[0].Data, want) || out[0].Dims[0] != 6 {
		t.Fatalf("padded A_log expanded to %v", k3Values(t, &out[0]))
	}
	if _, err := kimiExpandALog(e, f32AsBytes([]float32{0.5, -1, 0, 2}), &Config{Config: c}); err == nil {
		t.Fatal("a tail that is not zero padding was taken")
	}
}

// jsonUnmarshalKeys parses config.json the way readHFConfig does.
func jsonUnmarshalKeys(b []byte, c *hfConfig) error {
	d, err := readHFConfig(hfDir{fstest.MapFS{"config.json": {Data: b}}, ""})
	if err != nil {
		return err
	}
	*c = *d
	return nil
}
