package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/convert/safetensors"
)

// compressed-tensors' mxfp4-pack-quantized, the format Moonshot released
// Kimi-K3's routed experts in. A quantized Linear's weight is two tensors:
//
//	weight_packed  uint8 [rows, cols/2]   e2m1 codes, element 2i in the low
//	                                      nibble and 2i+1 in the high one
//	weight_scale   uint8 [rows, cols/32]  one E8M0 exponent per 32 elements of
//	                                      a row, value 2^(s-127)
//
// (compressed_tensors/compressors/mxfp4: MXFP4PackedCompressor over
// NVFP4PackedCompressor's pack_fp4_to_uint8 and unpack_fp4_from_uint8, with
// mx_utils' decompress_mx_scale.) The value is kE2M1ToFloat[code&7], negated
// when code&8, times the scale.
//
// That is OCP MXFP4, which the container stores natively (jlm.TypeMX4, GGUF's
// type 39): the same code table and the same exponent byte, so the weight is
// moved, never requantized. Only the layout differs: a GGUF block is the
// scale byte and then 16 bytes holding element j in the low nibble and
// element j+16 in the high one, where compressed-tensors packs neighbours.
// ggml's table is the e2m1 values doubled and its scale 2^(s-128), so the
// product is the same number (quant.MXFP4Values). llama.cpp's converter does
// this same repack for the GGUF repositories (repack_mxfp4_blocks).

// ctQuant is a checkpoint's compressed-tensors quantization_config, the
// subset that decides how its bytes read. Everything this converter does not
// implement is refused by name rather than read past.
type ctQuant struct {
	QuantMethod     string             `json:"quant_method"`
	Format          string             `json:"format"`
	Status          string             `json:"quantization_status"`
	ConfigGroups    map[string]ctGroup `json:"config_groups"`
	KVCacheScheme   json.RawMessage    `json:"kv_cache_scheme"`
	SparsityConfig  json.RawMessage    `json:"sparsity_config"`
	TransformConfig json.RawMessage    `json:"transform_config"`
}

type ctGroup struct {
	Format            *string         `json:"format"`
	Weights           *ctArgs         `json:"weights"`
	InputActivations  *ctArgs         `json:"input_activations"`
	OutputActivations json.RawMessage `json:"output_activations"`
}

// ctArgs is compressed-tensors' QuantizationArgs. symmetric defaults to true
// and dynamic to false there.
type ctArgs struct {
	NumBits        int             `json:"num_bits"`
	Type           string          `json:"type"`
	GroupSize      *int            `json:"group_size"`
	Strategy       string          `json:"strategy"`
	Symmetric      *bool           `json:"symmetric"`
	Dynamic        json.RawMessage `json:"dynamic"`
	ScaleDtype     *string         `json:"scale_dtype"`
	ZPDtype        *string         `json:"zp_dtype"`
	BlockStructure json.RawMessage `json:"block_structure"`
	Actorder       json.RawMessage `json:"actorder"`
}

// ctMXFP4 is the one format this converter reads.
const ctMXFP4 = "mxfp4-pack-quantized"

// empty reports whether a JSON value says nothing: absent, null, {} or [].
func empty(r json.RawMessage) bool {
	s := string(bytes.TrimSpace(r))
	return s == "" || s == "null" || s == "{}" || s == "[]"
}

// compressedTensorsOf reads the quantization_config a checkpoint states --
// at the top of config.json or, as Kimi-K3 states it, under text_config; two
// that differ are refused -- and answers whether its weights are
// compressed-tensors MXFP4. A config of another quant_method is not this
// reader's: its tensors' dtypes are refused where they are read. A
// compressed-tensors config this reader cannot honour is refused here, by
// name, before any tensor is read.
func compressedTensorsOf(c *hfConfig) (bool, error) {
	raw := c.keys["quantization_config"]
	if tc, ok := c.keys["text_config"]; ok {
		var t struct {
			Q json.RawMessage `json:"quantization_config"`
		}
		if err := json.Unmarshal(tc, &t); err != nil {
			return false, fmt.Errorf("convert: %s: text_config: %w", c.path, err)
		}
		switch {
		case empty(t.Q):
		case empty(raw):
			raw = t.Q
		default:
			var a, b any
			if json.Unmarshal(raw, &a) != nil || json.Unmarshal(t.Q, &b) != nil || !jsonEqual(a, b) {
				return false, fmt.Errorf("convert: %s: quantization_config differs between the top "+
					"level and text_config", c.path)
			}
		}
	}
	if empty(raw) {
		return false, nil
	}
	var q ctQuant
	if err := json.Unmarshal(raw, &q); err != nil {
		return false, fmt.Errorf("convert: %s: quantization_config: %w", c.path, err)
	}
	if q.QuantMethod != "compressed-tensors" {
		return false, nil
	}
	bad := func(format string, a ...any) (bool, error) {
		return false, fmt.Errorf("convert: %s: compressed-tensors "+format+": %w",
			append(append([]any{c.path}, a...), ErrNotImplemented)...)
	}
	if q.Format != ctMXFP4 {
		return bad("format %q is not implemented (%s is)", q.Format, ctMXFP4)
	}
	if q.Status != "compressed" {
		return bad("quantization_status %q: only a compressed checkpoint has packed weights to read", q.Status)
	}
	for _, x := range []struct {
		name string
		v    json.RawMessage
	}{{"kv_cache_scheme", q.KVCacheScheme}, {"sparsity_config", q.SparsityConfig},
		{"transform_config", q.TransformConfig}} {
		if !empty(x.v) {
			return bad("%s %s is not implemented", x.name, x.v)
		}
	}
	if len(q.ConfigGroups) == 0 {
		return bad("no config_groups: nothing says how the packed weights read")
	}
	for name, g := range q.ConfigGroups {
		if g.Format != nil && *g.Format != ctMXFP4 {
			return bad("config_groups.%s has format %q beside %s", name, *g.Format, ctMXFP4)
		}
		if err := mxfp4WeightArgs(g.Weights); err != nil {
			return bad("config_groups.%s.weights: %v", name, err)
		}
		if !empty(g.OutputActivations) {
			return bad("config_groups.%s quantizes output activations", name)
		}
		// Input activations quantized dynamically need no tensor and change
		// nothing in the weights; the engine runs the weights' activations at
		// its own precision, as for every MXFP4 model. A static scheme carries
		// an input_global_scale this reader does not read, and is refused.
		// (Kimi-K3 states none: its experts are W4A16.)
		if a := g.InputActivations; a != nil && !isTrue(a.Dynamic) {
			return bad("config_groups.%s has static input activations (an input_global_scale per "+
				"weight)", name)
		}
	}
	return true, nil
}

// mxfp4WeightArgs is the one weight scheme mxfp4-pack-quantized means.
func mxfp4WeightArgs(w *ctArgs) error {
	switch {
	case w == nil:
		return fmt.Errorf("absent")
	case w.NumBits != 4 || w.Type != "float":
		return fmt.Errorf("%d-bit %s, not MXFP4's 4-bit float", w.NumBits, w.Type)
	case w.GroupSize == nil || *w.GroupSize != 32 || w.Strategy != "group":
		return fmt.Errorf("strategy %q, group size %v: MXFP4 scales every 32 elements of a row", w.Strategy, w.GroupSize)
	case w.Symmetric != nil && !*w.Symmetric:
		return fmt.Errorf("asymmetric: a zero point MXFP4 does not have")
	case isTrue(w.Dynamic):
		return fmt.Errorf("dynamic weights")
	case w.ScaleDtype != nil && *w.ScaleDtype != "torch.uint8":
		return fmt.Errorf("scale_dtype %s, not the E8M0 byte", *w.ScaleDtype)
	case w.ZPDtype != nil:
		return fmt.Errorf("zp_dtype %s", *w.ZPDtype)
	case !empty(w.BlockStructure) || !empty(w.Actorder):
		return fmt.Errorf("block_structure %s, actorder %s", w.BlockStructure, w.Actorder)
	}
	return nil
}

func isTrue(r json.RawMessage) bool {
	var v any
	return json.Unmarshal(r, &v) == nil && v != nil && v != false
}

func jsonEqual(a, b any) bool {
	x, err1 := json.Marshal(a)
	y, err2 := json.Marshal(b)
	return err1 == nil && err2 == nil && bytes.Equal(x, y)
}

// mxfp4Suffixes are the two tensors one MXFP4 weight is stored as.
const (
	ctPacked = ".weight_packed"
	ctScale  = ".weight_scale"
)

// mergeMXFP4 replaces each X.weight_packed and X.weight_scale in all with
// one X.weight that reads as MXFP4, its shape checked against both now, from
// the headers, so a malformed pair fails before the writer has laid anything
// out. With mx false any packed tensor is refused: its codes would otherwise
// reach identify as a weight nobody can read. A lone weight_scale is only
// compressed-tensors' when mx is true; otherwise it is the architecture's
// (BitNet's per-tensor scale) or reaches identify and is refused there.
func mergeMXFP4(all map[string]hfLoc, mx bool) error {
	for name, loc := range all {
		base, ok := strings.CutSuffix(name, ctPacked)
		if !ok {
			if s, isScale := strings.CutSuffix(name, ctScale); isScale && mx {
				if _, paired := all[s+ctPacked]; !paired {
					return fmt.Errorf("convert: %s has no %s beside it", name, s+ctPacked)
				}
			}
			continue
		}
		if !mx {
			return fmt.Errorf("convert: %s is a compressed-tensors packed weight and the config "+
				"states no compressed-tensors MXFP4 quantization", name)
		}
		w := base + ".weight"
		if _, dup := all[w]; dup {
			return fmt.Errorf("convert: %s and %s both carry one weight", w, name)
		}
		sc, ok := all[base+ctScale]
		if !ok {
			return fmt.Errorf("convert: %s has no %s beside it", name, base+ctScale)
		}
		p, s := loc.t, sc.t
		if p.DType != safetensors.U8 || s.DType != safetensors.U8 || len(p.Shape) != 2 || len(s.Shape) != 2 {
			return fmt.Errorf("convert: %s is %s%v and %s %s%v; MXFP4 is two uint8 matrices",
				name, p.DType, p.Shape, base+ctScale, s.DType, s.Shape)
		}
		rows, cols := p.Shape[0], 2*p.Shape[1]
		if cols%32 != 0 || s.Shape[0] != rows || s.Shape[1] != cols/32 {
			return fmt.Errorf("convert: %s codes %d x %d values, and its scales are %v; MXFP4 wants "+
				"[%d, %d]", name, rows, cols, s.Shape, rows, cols/32)
		}
		all[w] = hfLoc{f: loc.f, t: loc.t, sf: sc.f, st: sc.t}
		delete(all, name)
		delete(all, base+ctScale)
	}
	return nil
}

// readMXFP4 reads one weight's codes and scales and returns them as GGUF's
// MXFP4 blocks.
func (l hfLoc) readMXFP4() ([]byte, error) {
	p, err := l.f.ReadTensor(l.t)
	if err != nil {
		return nil, err
	}
	s, err := l.sf.ReadTensor(l.st)
	if err != nil {
		return nil, err
	}
	return mxfp4Blocks(p, s, int(l.t.Shape[0]), int(2*l.t.Shape[1]))
}

// mxfp4Blocks repacks compressed-tensors' rows (codes in element order, two
// to a byte, and a scale row) into GGUF's block_mxfp4: per 32 elements of a
// row, the scale byte and then 16 bytes, byte j holding element j in the low
// nibble and element j+16 in the high one. No bit of a code or a scale
// changes.
func mxfp4Blocks(packed, scale []byte, rows, cols int) ([]byte, error) {
	nb := cols / 32
	if cols%32 != 0 || len(packed) != rows*cols/2 || len(scale) != rows*nb {
		return nil, fmt.Errorf("convert: MXFP4 %d x %d wants %d code bytes and %d scales, have %d and %d",
			rows, cols, rows*cols/2, rows*nb, len(packed), len(scale))
	}
	out := make([]byte, rows*nb*17)
	for blk := range rows * nb {
		src := packed[blk*16 : blk*16+16]
		dst := out[blk*17 : blk*17+17]
		dst[0] = scale[blk]
		for j := range 16 {
			// Element j is the nibble j&1 of byte j>>1; element j+16 the same
			// nibble of byte 8+j>>1.
			sh := uint(4 * (j & 1))
			lo := src[j>>1] >> sh & 0xF
			hi := src[8+j>>1] >> sh & 0xF
			dst[1+j] = lo | hi<<4
		}
	}
	return out, nil
}
