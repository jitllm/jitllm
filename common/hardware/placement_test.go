package hardware

import "testing"

func TestValidatePlacement(t *testing.T) {
	for _, c := range []struct {
		place, spec string
		ok          bool
	}{
		{"", "auto", true},
		{"0-15=cuda:0,16-31=vulkan:0", "cuda:0,vulkan:0", true},
		{"0-3=cuda:0,*=host,head=cuda:0!", "cuda:0,vulkan:0", true},
		{"0-15=cuda:0,16-31=vulkan:0", "cuda:0", false}, // vulkan:0 is not opened
		{"0-15=cuda:0", "all", false},                   // all names no device
		{"0-15", "cuda:0", false},                       // not SEL=WHERE
		{"9-3=cuda:0", "cuda:0", false},                 // empty range
	} {
		if err := ValidatePlacement(c.place, c.spec); (err == nil) != c.ok {
			t.Errorf("ValidatePlacement(%q, %q) = %v, want ok=%v", c.place, c.spec, err, c.ok)
		}
	}
}
