package kernels

import "testing"

// TestDecodePlansCoverTheKeys holds the plan functions to the kernels'
// contracts at every key count a variant may be compiled for: FlashDecodeKV's
// splits times its chunk, and the staged path's splits times its chunk, cover
// the keys; and pins the choices the measurements were taken at, so a change
// to a rule shows up here before it shows up in a rate.
func TestDecodePlansCoverTheKeys(t *testing.T) {
	shapes := []FlashShape{
		{Heads: 32, KVHeads: 8, Dim: 64, Rows: 1, Scale: .125, Page: 256},  // Llama-3.2-1B
		{Heads: 8, KVHeads: 2, Dim: 256, Rows: 1, Scale: .06, Page: 256},   // Qwen3.5-0.8B
		{Heads: 24, KVHeads: 8, Dim: 128, Rows: 1, Scale: .088, Page: 256}, // Phi-4-mini
		{Heads: 16, KVHeads: 8, Dim: 256, Rows: 4, Scale: .06, Page: 256, F16: true, F16K: true},
	}
	for _, s := range shapes {
		for _, slots := range []int{0, 30720, 163840} {
			for keys := 1; keys <= 1<<17; keys = keys*3/2 + 1 {
				kv := FlashKVPlan(s, keys, slots)
				c := kv
				c.Splits = 1
				if kv.Splits*FlashKVChunk(c) < keys {
					t.Fatalf("%+v at %d keys, %d slots: %d splits of %d keys", s, keys, slots, kv.Splits, FlashKVChunk(c))
				}
				if _, err := FlashDecodeKV(kv); err != nil {
					t.Fatalf("FlashKVPlan's shape does not build: %v", err)
				}
				st := PagedStagedPlan(s, keys, slots)
				if st.Splits*st.Chunk < keys || st.Chunk%32 != 0 || st.Chunk < 32 {
					t.Fatalf("%+v at %d keys: staged %d x %d", s, keys, st.Splits, st.Chunk)
				}
				if _, err := PagedStagedAcc(st); err != nil {
					t.Fatalf("PagedStagedPlan's shape does not build: %v", err)
				}
			}
		}
	}
	// The measured points (docs/engineering-history/gpu-kernels.md).
	for _, c := range []struct {
		s                 FlashShape
		keys, slots       int
		kvSplits, kvWarps int
		stSplits, stChunk int
	}{
		{shapes[0], 512, 30720, 2, 0, 8, 64},       // a 20-SM part: the cap rounded up
		{shapes[0], 4096, 30720, 3, 0, 16, 256},    //
		{shapes[0], 512, 163840, 8, 0, 16, 32},     // an 80-SM part
		{shapes[0], 4096, 163840, 8, 0, 64, 64},    //
		{shapes[0], 16384, 163840, 9, 0, 128, 128}, //
		{shapes[1], 4096, 30720, 3, 4, 16, 256},    // four warps past 128 dims at 4096 keys
		{shapes[1], 16384, 30720, 9, 4, 32, 512},   //
		{shapes[1], 4096, 163840, 16, 4, 64, 64},   //
		{shapes[1], 16384, 163840, 16, 4, 128, 128},
	} {
		kv := FlashKVPlan(c.s, c.keys, c.slots)
		st := PagedStagedPlan(c.s, c.keys, c.slots)
		if kv.Splits != c.kvSplits || kv.Warps != c.kvWarps || st.Splits != c.stSplits || st.Chunk != c.stChunk {
			t.Errorf("h%d d%d at %d keys, %d slots: kv %d splits %d warps, staged %d x %d; want %d, %d, %d x %d",
				c.s.Heads, c.s.Dim, c.keys, c.slots, kv.Splits, kv.Warps, st.Splits, st.Chunk, c.kvSplits, c.kvWarps, c.stSplits, c.stChunk)
		}
	}
	// The paths the measurements chose (FlashDecodeKV where they did not
	// separate).
	for _, c := range []struct {
		api   string
		s     FlashShape
		keys  int
		slots int
		want  DecodePath
	}{
		{"ptx", shapes[0], 4096, 30720, PathFlashKV},
		{"ptx", shapes[0], 16384, 30720, PathStaged},
		{"ptx", shapes[1], 512, 30720, PathFlashKV},
		{"ptx", shapes[1], 4096, 30720, PathStaged},
		{"ptx", shapes[1], 4096, 163840, PathFlashKV},
		{"ptx", shapes[1], 16384, 163840, PathStaged},
		{"ptx", shapes[0], 16384, 163840, PathFlashKV},
		{"spirv", shapes[1], 4096, 0, PathFlashKV},
		{"spirv", shapes[0], 16384, 0, PathStaged},
		{"msl", shapes[1], 16384, 0, PathStaged},
		{"msl", shapes[1], 512, 0, PathFlashKV},
		{"msl", shapes[0], 4096, 0, PathFlashKV},
		{"msl", shapes[0], 16384, 0, PathStaged},
		{"msl", FlashShape{Heads: 16, KVHeads: 1, Dim: 576, MLA: 512, Rows: 1, Page: 256}, 512, 0, PathStaged},
	} {
		p := ChooseDecodePlan(c.api, c.s, c.keys, c.slots)
		if p.Path != c.want {
			t.Errorf("%s h%d d%d at %d keys, %d slots: %v, want %v", c.api, c.s.Heads, c.s.Dim, c.keys, c.slots, p.Path, c.want)
		}
		if p.Path == PathFlashKV && ((c.api == "spirv" && c.keys < 16384) || c.api == "msl") && p.Shape.Warps != flashKVWarps(c.s) {
			t.Errorf("%s at %d keys took %d warps", c.api, c.keys, p.Shape.Warps)
		}
	}
}
