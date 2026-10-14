package model

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Pictures in a decision request (d1's): the systemone template writes one
// marker per picture at the head of the user turn (its `images` list), and
// each marker becomes the picture's run of the prompt as the model's tower
// lays it out (Tower.pictureRun: LFM2-VL's tiles, thumbnail and markers).
// The pictures are cut once per request; every question's prompt carries
// the same Picture spans, so the second question's prefill finds their rows
// in the model's image cache.

// MaxDecisionImages is the most pictures one request carries: llama.cpp's
// DECISION_MAX_IMAGES.
const MaxDecisionImages = 8

// decisionMarker stands for a picture in the rendered prompt. Request text
// that contains it is given a space in its place first, as llama.cpp's
// server does with its media marker, so the count of markers is the count of
// pictures.
const decisionMarker = "<__jitllm_decision_image__>"

// d1MaxPixels is d1's runner's bound on a picture before the processor
// (runner.py VISION_MAX_PIXELS): a bigger one is resized to it with Pillow's
// BICUBIC. llama.cpp's server hands mtmd the picture as it came, a
// divergence (RULE 7m): d1's own code is what its answers were calibrated on.
const d1MaxPixels = 1024 * 1024

// d1Cap is runner.cap_pixels: w*h at most d1MaxPixels, each side scaled by
// sqrt(max/(w*h)) and truncated, at least 1, Pillow BICUBIC.
func d1Cap(img image.Image) image.Image {
	p := toRGB8(img)
	if p.w*p.h <= d1MaxPixels {
		return img
	}
	scale := math.Sqrt(float64(d1MaxPixels) / float64(p.w*p.h))
	nw, nh := max(1, int(float64(p.w)*scale)), max(1, int(float64(p.h)*scale))
	return p.resizePIL(nw, nh, aaBicubic).image()
}

// replaceText is v with old replaced by repl in every string it holds, keys
// aside.
func replaceText(v jinja.Value, old, repl string) jinja.Value {
	switch {
	case v.IsString():
		if strings.Contains(v.AsString(), old) {
			return jinja.NewString(strings.ReplaceAll(v.AsString(), old, repl))
		}
		return v
	case v.IsList():
		items := v.AsList().Items
		out := make([]jinja.Value, len(items))
		for i, x := range items {
			out[i] = replaceText(x, old, repl)
		}
		return jinja.NewList(out)
	case v.IsDict():
		d := v.AsDict()
		out := jinja.NewDict()
		for _, k := range d.Keys {
			out.AsDict().Set(k, replaceText(d.Data[k], old, repl))
		}
		return out
	}
	return v
}

// checkImages refuses pictures a readout cannot read.
func (d *Decider) checkImages(n int) error {
	if n == 0 {
		return nil
	}
	if n > MaxDecisionImages {
		return fmt.Errorf("too many images, the maximum is %d", MaxDecisionImages)
	}
	if d.kind != jlm.DecisionLFM2D1 {
		return fmt.Errorf("this model reads text only; a request with images needs a vision decision model")
	}
	if d.m.Tower() == nil {
		return fmt.Errorf("this container carries no vision tower; convert the model with its mmproj " +
			"(jitllm convert model.gguf mmproj.gguf out.jlm)")
	}
	return nil
}

// cutPictures is each picture's run of the prompt, d1's bound applied first.
func (d *Decider) cutPictures(imgs []image.Image) ([][]Span, error) {
	tw := d.m.Tower()
	out := make([][]Span, len(imgs))
	for i, img := range imgs {
		if img == nil {
			return nil, fmt.Errorf("images[%d] is empty", i)
		}
		sp, err := tw.pictureRun(d.m, i, d1Cap(img))
		if err != nil {
			return nil, fmt.Errorf("images[%d]: %w", i, err)
		}
		out[i] = sp
	}
	return out, nil
}

// pictureSpans is the rendered prompt as spans: the text between the
// markers encoded as the whole prompt would be, and each picture's run where
// its marker stood.
func (d *Decider) pictureSpans(prompt string, pics [][]Span) ([]Span, error) {
	segs := strings.Split(prompt, decisionMarker)
	if len(segs)-1 != len(pics) {
		return nil, fmt.Errorf("the systemone template wrote %d image marker(s) for %d image(s)", len(segs)-1, len(pics))
	}
	var out []Span
	for i, seg := range segs {
		if ids := d.m.Vocab.EncodeSpecial(seg, false); len(ids) > 0 {
			out = append(out, Span{Tokens: ids})
		}
		if i < len(pics) {
			out = append(out, pics[i]...)
		}
	}
	return mergeTokenSpans(out), nil
}

// image is p as an opaque image.RGBA, for a caller that takes an image.Image
// (a resize before the processor's own).
func (p rgb8) image() image.Image {
	out := image.NewRGBA(image.Rect(0, 0, p.w, p.h))
	for i := 0; i < p.w*p.h; i++ {
		copy(out.Pix[4*i:4*i+3], p.px[3*i:3*i+3])
		out.Pix[4*i+3] = 255
	}
	return out
}

// DecisionMedia takes a /v1/systemone request's pictures out of it, as
// llama.cpp's server does (server-decision.cpp parse_state): the URLs of its
// "files" and "images" arrays (one is an alias of the other), then every
// image_url part of a state that is chat messages -- a list of them, or an
// object holding one under "messages" -- in message order, each part taken
// out of its message's content. The URLs must be data: URLs; "videos" is
// refused. The state comes back without the parts.
func DecisionMedia(req *jinja.Dict, state jinja.Value) (jinja.Value, []string, error) {
	if v, ok := req.Get("videos"); ok && !v.IsNone() && !(v.IsList() && v.AsList().Len() == 0) {
		return state, nil, fmt.Errorf(`"videos" is not supported`)
	}
	var urls []string
	add := func(u jinja.Value) error {
		if !u.IsString() || !strings.HasPrefix(u.AsString(), "data:") {
			return fmt.Errorf("images must be data URLs (data:image/...;base64,...)")
		}
		if len(urls) >= MaxDecisionImages {
			return fmt.Errorf("too many images, the maximum is %d", MaxDecisionImages)
		}
		urls = append(urls, u.AsString())
		return nil
	}
	for _, key := range []string{"files", "images"} {
		v, ok := req.Get(key)
		if !ok || v.IsNone() {
			continue
		}
		if !v.IsList() {
			return state, nil, fmt.Errorf("%q must be an array", key)
		}
		for _, u := range v.AsList().Items {
			if err := add(u); err != nil {
				return state, nil, err
			}
		}
	}
	wrapped := false
	msgs := state
	if state.IsDict() {
		if m, ok := state.AsDict().Get("messages"); ok {
			msgs, wrapped = m, true
		}
	}
	if !msgs.IsList() {
		return state, urls, nil
	}
	out := make([]jinja.Value, 0, msgs.AsList().Len())
	for _, msg := range msgs.AsList().Items {
		var content jinja.Value
		ok := false
		if msg.IsDict() {
			content, ok = msg.AsDict().Get("content")
		}
		if !ok || !content.IsList() {
			out = append(out, msg)
			continue
		}
		var kept []jinja.Value
		for _, part := range content.AsList().Items {
			if part.IsDict() {
				pd := part.AsDict()
				typ, _ := pd.Get("type")
				iu, has := pd.Get("image_url")
				if typ.IsString() && typ.AsString() == "image_url" && has {
					if iu.IsDict() {
						if u, ok := iu.AsDict().Get("url"); ok {
							iu = u
						}
					}
					if err := add(iu); err != nil {
						return state, nil, err
					}
					continue
				}
			}
			kept = append(kept, part)
		}
		m := jinja.NewDict()
		for _, k := range msg.AsDict().Keys {
			m.AsDict().Set(k, msg.AsDict().Data[k])
		}
		m.AsDict().Set("content", jinja.NewList(kept))
		out = append(out, m)
	}
	if !wrapped {
		return jinja.NewList(out), urls, nil
	}
	s := jinja.NewDict()
	for _, k := range state.AsDict().Keys {
		s.AsDict().Set(k, state.AsDict().Data[k])
	}
	s.AsDict().Set("messages", jinja.NewList(out))
	return s, urls, nil
}

// DecodeImageDataURL decodes a data:<type>;base64,<data> picture, PNG or
// JPEG: `jitllm decide`'s reader. The server decodes its requests' pictures
// under its own bounds (server.ImagePolicy).
func DecodeImageDataURL(u string) (image.Image, error) {
	rest, ok := strings.CutPrefix(u, "data:")
	head, data, found := strings.Cut(rest, ",")
	if !ok || !found || !strings.HasSuffix(head, ";base64") {
		return nil, fmt.Errorf("not a base64 data: URL")
	}
	b, err := base64.StdEncoding.DecodeString(strings.TrimSpace(data))
	if err != nil {
		return nil, fmt.Errorf("the data URL's base64: %w", err)
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	return img, err
}
