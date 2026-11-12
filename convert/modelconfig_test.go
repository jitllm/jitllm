package convert

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestModelConfigStatesWhatTheGGUFDropped converts synth-minicpmv's GGUF,
// which llama.cpp's converter wrote naming LongRoPE with none of its
// factors: refused alone, stated from the publisher's config.json with
// WithModelConfig -- the short and long factors byte for byte as float32,
// and the original context -- and refused again beside a config.json that
// states no LongRoPE.
func TestModelConfigStatesWhatTheGGUFDropped(t *testing.T) {
	src := testmodels.Path("synth-minicpmv.gguf")
	cfg := testmodels.Path("vlmb/fixtures/synth-minicpmv/config.json")
	for _, p := range []string{src, cfg} {
		if !fileExists(p) {
			testmodels.Missing(t, "MODEL MISSING: %s (scripts/minicpmvgold.py --fixture makes it) -- this gate proved nothing", p)
		}
	}
	f, err := gguf.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if kv, ok := f.Key("rope.scaling.type"); !ok {
		t.Fatal("the fixture names no rope scaling: the gate would check nothing")
	} else if v, _ := kv.String(); v != "longrope" {
		t.Fatalf("the fixture names %q, not longrope", v)
	}
	if _, err := buildGGUF(f, nil, src, "", options{}); err == nil || !strings.Contains(err.Error(), "-config") {
		t.Fatalf("converted alone: %v, want a refusal naming -config", err)
	}

	raw, err := os.ReadFile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var bare map[string]json.RawMessage
	if err := json.Unmarshal(raw, &bare); err != nil {
		t.Fatal(err)
	}
	// transformers 5 writes the scaling into rope_parameters.
	key := "rope_scaling"
	if _, ok := bare[key]; !ok {
		key = "rope_parameters"
	}
	var rs struct {
		Short   []float64 `json:"short_factor"`
		Long    []float64 `json:"long_factor"`
		OrigCtx uint32    `json:"original_max_position_embeddings"`
	}
	if err := json.Unmarshal(bare[key], &rs); err != nil {
		t.Fatalf("%s: %s: %v", cfg, key, err)
	}
	if rs.OrigCtx == 0 {
		if err := json.Unmarshal(bare["original_max_position_embeddings"], &rs.OrigCtx); err != nil {
			t.Fatalf("%s: %v", cfg, err)
		}
	}
	if len(rs.Short) == 0 || len(rs.Long) == 0 || rs.OrigCtx == 0 {
		t.Fatalf("%s states %d short and %d long factors, original context %d", cfg, len(rs.Short), len(rs.Long), rs.OrigCtx)
	}
	s, err := buildGGUF(f, nil, src, "", options{modelConfig: cfg})
	if err != nil {
		t.Fatal(err)
	}
	if s.Config.YarnOrigCtx != rs.OrigCtx {
		t.Errorf("original context %d, the config's %d", s.Config.YarnOrigCtx, rs.OrigCtx)
	}
	want := map[jlm.Role][]float64{jlm.RoleRopeFactorsShort: rs.Short, jlm.RoleRopeFactorsLong: rs.Long}
	for i := range s.Tensors {
		x := &s.Tensors[i]
		w, ok := want[x.Role]
		if !ok {
			continue
		}
		delete(want, x.Role)
		if x.Type != jlm.TypeF32 || len(x.Data) != 4*len(w) {
			t.Fatalf("%v: type %v, %d bytes for %d factors", x.Role, x.Type, len(x.Data), len(w))
		}
		for j, v := range w {
			if got := math.Float32frombits(binary.LittleEndian.Uint32(x.Data[4*j:])); got != float32(v) {
				t.Fatalf("%v[%d] = %g, the config's %g", x.Role, j, got, v)
			}
		}
	}
	if len(want) > 0 {
		t.Fatalf("the source holds no %v", want)
	}

	delete(bare, key)
	b, err := json.Marshal(bare)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := buildGGUF(f, nil, src, "", options{modelConfig: dir}); err == nil || !strings.Contains(err.Error(), "states none") {
		t.Fatalf("beside a config.json with no LongRoPE: %v, want a refusal", err)
	}
	t.Logf("%d short and %d long factors, original context %d, from %s", len(rs.Short), len(rs.Long), rs.OrigCtx, cfg)
}

// TestProcessorConfigStatesTheResample converts synth-lfm2vl beside a
// publisher's preprocessor_config.json (WithModelConfig) stating each of
// Pillow's bilinear (2) and bicubic (3): the tower's resize takes the
// config's filter whatever the mmproj names, and a config stating neither
// leaves the mmproj's.
func TestProcessorConfigStatesTheResample(t *testing.T) {
	src, proj := testmodels.Path("synth-lfm2vl.gguf"), testmodels.Path("mmproj-synth-lfm2vl.gguf")
	for _, p := range []string{src, proj} {
		if !fileExists(p) {
			testmodels.Missing(t, "MODEL MISSING: %s (scripts/lfm2visiongold.py --fixture makes it) -- this gate proved nothing", p)
		}
	}
	filter := func(body string) int32 {
		t.Helper()
		var o options
		if body != "" {
			o.modelConfig = t.TempDir()
			for name, b := range map[string]string{"config.json": "{}", "preprocessor_config.json": body} {
				if err := os.WriteFile(filepath.Join(o.modelConfig, name), []byte(b), 0o644); err != nil {
					t.Fatal(err)
				}
			}
		}
		f, err := gguf.Open(src)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		vf, err := gguf.Open(proj)
		if err != nil {
			t.Fatal(err)
		}
		defer vf.Close()
		s, err := buildGGUF(f, vf, src, proj, o)
		if err != nil {
			t.Fatal(err)
		}
		pre := s.Vision.Tower.Pre[0]
		if pre.Kind != jlm.PreTileSmart || len(pre.Ints) < 8 {
			t.Fatalf("the tower's first step is %+v, not the tiled resize", pre)
		}
		return pre.Ints[7]
	}
	file := filter("")
	bilinear, bicubic := filter(`{"resample": 2}`), filter(`{"resample": 3}`)
	if bilinear != int32(jlm.FilterPILBilinear) || bicubic != int32(jlm.FilterPILBicubic) {
		t.Fatalf("resample 2 states filter %d and 3 states %d, want %d and %d",
			bilinear, bicubic, jlm.FilterPILBilinear, jlm.FilterPILBicubic)
	}
	if other := filter(`{"resample": 0}`); other != file {
		t.Fatalf("a config stating neither filter states %d, the mmproj's %d", other, file)
	}
	t.Logf("the mmproj's filter %d; the config's bilinear %d and bicubic %d", file, bilinear, bicubic)
}
