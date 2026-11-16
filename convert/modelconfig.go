package convert

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// applyModelConfig states what a GGUF dropped from the publisher's
// config.json (WithModelConfig): a LongRoPE's factors as the
// rope_factors_short/long tensors, the original context the long ones turn
// past and the magnitude. A GGUF that carries its factors keeps them. One
// that names LongRoPE and carries none is refused without the option: what
// it would run is not the model's rotary.
func (o *options) applyModelConfig(f *meta.File, s *jlm.Source, src string) error {
	named := false
	if kv, ok := f.Key("rope.scaling.type"); ok {
		v, _ := kv.String()
		named = v == "longrope" || v == "su"
	}
	for i := range s.Tensors {
		if s.Tensors[i].Role == jlm.RoleRopeFactorsShort {
			return nil
		}
	}
	if o.modelConfig == "" {
		if named {
			return fmt.Errorf("convert: %s names LongRoPE and carries none of its factors (llama.cpp's converter "+
				"writes none for this architecture): give the publisher's config.json (WithModelConfig, "+
				"jitllm convert -config)", src)
		}
		return nil
	}
	c, err := modelConfigOf(o.modelConfig)
	if err != nil {
		return err
	}
	lr, err := hfLongRopeConsume(c)
	if err != nil {
		return err
	}
	if lr == nil {
		if named {
			return fmt.Errorf("convert: %s names LongRoPE and %s states none", src, c.path)
		}
		return nil
	}
	ts, err := lr.state(c.path, &Config{Config: s.Config})
	if err != nil {
		return err
	}
	s.Tensors = append(s.Tensors, ts...)
	return nil
}

// modelConfigOf reads path, a config.json or the directory holding one; a
// vision-language model's text model is its text_config where the top level
// states no rotary scaling.
func modelConfigOf(path string) (*hfConfig, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("convert: model config: %w", err)
	}
	dir := path
	if !fi.IsDir() {
		if filepath.Base(path) != "config.json" {
			return nil, fmt.Errorf("convert: model config: %s is not a config.json", path)
		}
		dir = filepath.Dir(path)
	}
	c, err := readHFConfig(dirFiles(dir))
	if err != nil {
		return nil, err
	}
	if len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null" {
		return c, nil
	}
	raw, ok := c.keys["text_config"]
	if !ok {
		return c, nil
	}
	t := &hfConfig{path: c.path + ":text_config"}
	if err := json.Unmarshal(raw, t); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", t.path, err)
	}
	if err := json.Unmarshal(raw, &t.keys); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", t.path, err)
	}
	return t, nil
}

// applyProcessorConfig states the tower's resampling filter from the
// publisher's preprocessor_config.json beside its config.json
// (WithModelConfig), where the mmproj names none: LiquidAI's LFM2-VL files
// carry no image_resize_algo, and the processor resamples as its
// preprocessor_config says, not as its class defaults. Pillow's 2 is
// bilinear and 3 bicubic; a config stating neither leaves the file's. Its
// size.longest_edge is the longer side a tiling processor resizes a
// picture to before it cuts it (Idefics3's), kept for the stater's step
// (statedTower); its max_image_size.longest_edge, the tile, is the
// tower's square or the config is refused.
func (o *options) applyProcessorConfig(v *jlm.Vision) error {
	if o.modelConfig == "" || v == nil {
		return nil
	}
	dir := o.modelConfig
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		dir = filepath.Dir(dir)
	}
	path := filepath.Join(dir, "preprocessor_config.json")
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("convert: %w", err)
	}
	type edge struct {
		LongestEdge int32 `json:"longest_edge"`
	}
	var p struct {
		Resample *int  `json:"resample"`
		Size     *edge `json:"size"`
		MaxImage *edge `json:"max_image_size"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("convert: %s: %w", path, err)
	}
	if p.Size != nil && p.Size.LongestEdge > 0 {
		if p.MaxImage != nil && p.MaxImage.LongestEdge != int32(v.ImageSize) {
			return fmt.Errorf("convert: %s cuts tiles of %d and the tower reads squares of %d", path,
				p.MaxImage.LongestEdge, v.ImageSize)
		}
		o.longestEdge = p.Size.LongestEdge
	}
	if p.Resample == nil {
		return nil
	}
	switch *p.Resample {
	case 2:
		v.Flags &^= jlm.FlagVisionBicubic
	case 3:
		v.Flags |= jlm.FlagVisionBicubic
	}
	return nil
}

// stateAbsentKeys states each key a GGUF f does not carry from the
// publisher's config.json (WithModelConfig), as the library file of its
// class maps the config onto the keys (its keys and its sliding window):
// what the converter that wrote f did not state. A key the config does not
// state either stays absent.
func (o *options) stateAbsentKeys(f *meta.File) error {
	if o.modelConfig == "" {
		return nil
	}
	c, err := modelConfigOf(o.modelConfig)
	if err != nil {
		return err
	}
	lib, err := modelLibrary()
	if err != nil {
		return err
	}
	var mj *modelJSON
	for _, n := range c.Architectures {
		if mj = lib[n]; mj != nil {
			break
		}
	}
	if mj == nil || mj.Config.Arch != f.Arch() {
		return nil
	}
	env, err := mj.env(c)
	if err != nil {
		return err
	}
	eval := func(what, src string) (any, error) {
		n, err := parseExpr(src)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %w", mj.name, what, err)
		}
		v, err := evalMJ(n, env)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %s: %w", c.path, mj.name, what, err)
		}
		return v, nil
	}
	ns := f.Arch() + "."
	for _, k := range slices.Sorted(maps.Keys(mj.Config.Keys)) {
		if _, ok := f.KV[ns+k]; ok {
			continue
		}
		spec := mj.Config.Keys[k]
		if v, err := eval(k, spec.Uint+spec.Float+spec.Int32s+spec.Bool); err != nil || v == nil {
			continue
		}
		v, err := mj.keyValue(c.path, k, eval)
		if err != nil {
			return err
		}
		f.KV[ns+k] = v
	}
	if _, ok := f.KV[ns+"attention.sliding_window"]; ok || mj.Config.Window == nil {
		return nil
	}
	w, err := mj.windowSize(c.path, eval)
	if err != nil {
		return err
	}
	f.KV[ns+"attention.sliding_window"] = meta.MakeUint(w)
	return nil
}
