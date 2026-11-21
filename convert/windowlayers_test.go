package convert

import (
	"slices"
	"strings"
	"testing"
)

// TestWindowLayersBecomeAPeriod: a window's layer statement -- a list of
// layer types, a period, a start, or every layer -- is the slides it states,
// and the slides the container's period: the last (or first) of each period
// global, every layer past the last, none 0; a pattern no period describes is
// refused.
func TestWindowLayersBecomeAPeriod(t *testing.T) {
	env := mjEnv(map[string]any{
		"layer_types": []any{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
		"pattern":     4.0, "start": 2.0,
	})
	eval := func(what, src string) (any, error) {
		n, err := parseExpr(src)
		if err != nil {
			return nil, err
		}
		return evalMJ(n, env)
	}
	s, f := true, false
	for _, r := range []struct {
		name string
		l    mjWindowLayers
		want []bool
	}{
		{"a list", mjWindowLayers{List: "layer_types", Slides: "sliding_attention"}, []bool{s, s, s, f}},
		{"no list, a period", mjWindowLayers{List: "missing", Slides: "sliding_attention", Period: "pattern"}, []bool{s, s, s, f}},
		{"a start", mjWindowLayers{From: "start"}, []bool{f, f, s, s}},
		{"every layer", mjWindowLayers{}, []bool{s, s, s, s}},
	} {
		got, err := mjWindowSlides(&r.l, eval, 4)
		if err != nil || !slices.Equal(got, r.want) {
			t.Errorf("%s: %v %v, want %v", r.name, got, err, r.want)
		}
	}
	if _, err := mjWindowSlides(&mjWindowLayers{List: "layer_types", Slides: "x"}, eval, 3); err == nil {
		t.Error("a list of 4 for 3 layers was accepted")
	}
	for _, r := range []struct {
		slides []bool
		p      uint32
		first  bool
		err    string
	}{
		{[]bool{s, s, s, f}, 4, false, ""},
		{[]bool{s, s, s, f, s, s, s, f}, 4, false, ""},
		{[]bool{f, s, s, s, f, s, s, s}, 4, true, ""},
		{[]bool{s, s, s, s}, 5, false, ""},
		{[]bool{f, f, f}, 0, false, ""},
		{[]bool{f, f, s, s}, 0, false, "no period"},
	} {
		p, first, err := windowPeriod(r.slides)
		switch {
		case r.err != "":
			if err == nil || !strings.Contains(err.Error(), r.err) {
				t.Errorf("%v: %v, want an error saying %q", r.slides, err, r.err)
			}
		case err != nil || p != r.p || first != r.first:
			t.Errorf("%v: period %d first %v %v, want %d %v", r.slides, p, first, err, r.p, r.first)
		}
	}
}
