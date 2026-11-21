package cpu

import "testing"

// TestSpecOptionsStateTheVariants holds each cpu device spec variant to the
// Target it makes, and refuses one that names no variant.
func TestSpecOptionsStateTheVariants(t *testing.T) {
	plain := New()
	for _, c := range []struct {
		sel  string
		want func(*Target) bool
	}{
		{"", func(tg *Target) bool { return tg.ISA() == plain.ISA() }},
		{"novnni", func(tg *Target) bool { f := tg.Features(); return !f.AVXVNNI && !f.AVX512VNNI }},
		{"nodotprod", func(tg *Target) bool { return tg.ISA()&ISADotProd == 0 }},
		{"perfmap", func(tg *Target) bool { return tg.ISA() == plain.ISA() }},
		{"novnni+nodotprod", func(tg *Target) bool { return !tg.Features().AVXVNNI && tg.ISA()&ISADotProd == 0 }},
	} {
		o, err := SpecOptions(c.sel)
		if err != nil {
			t.Fatalf("cpu:%s: %v", c.sel, err)
		}
		tg := New(o...)
		if !c.want(tg) {
			t.Errorf("cpu:%s made a Target of ISA %v", c.sel, tg.ISA())
		}
		tg.Close()
	}
	if plain.Tier() == TierAVX2 {
		o, err := SpecOptions("sse")
		if err != nil {
			t.Fatal(err)
		}
		if tg := New(o...); tg.Tier() != TierSSE || tg.ISA()&ISAAVX2 != 0 {
			t.Errorf("cpu:sse made a Target at %v, ISA %v", tg.Tier(), tg.ISA())
		}
	}
	for _, bad := range []string{"avx512", "sse+", "+novnni", "SSE"} {
		if _, err := SpecOptions(bad); err == nil {
			t.Errorf("cpu:%s was accepted", bad)
		}
	}
}
