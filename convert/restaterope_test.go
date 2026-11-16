package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRestatedRotaryTableIsAFreshConversion holds a container that states
// its rotary sections and their layout's bits alone, as a writer before the
// rotary table laid it, to what `jitllm upgrade` makes of it: byte for byte
// the container a fresh conversion writes, for each layout (Qwen2-VL's
// contiguous sections, Qwen3-VL's interleaved, HunyuanVL's over the
// elements). The violation states each with another layout's bits and must
// not reach the fresh container.
func TestRestatedRotaryTableIsAFreshConversion(t *testing.T) {
	fp := jlm.Fingerprint{Host: "test", Writer: "test"}
	for _, tc := range []struct {
		name        string
		bits, wrong jlm.Flags
	}{
		{"synth-qwen2vl.gguf", 0, legacyRopeInterleaved},
		{"synth-qwen3vl.gguf", legacyRopeInterleaved, 0},
		{"synth-hunyuanvl.gguf", legacyRopeXD, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := testmodels.Path(tc.name)
			if _, err := os.Stat(src); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", src)
			}
			dir := t.TempDir()
			fresh := filepath.Join(dir, "fresh"+jlm.Ext)
			if _, err := FromGGUF(src, fresh, fp); err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(fresh)
			if err != nil {
				t.Fatal(err)
			}
			f, err := gguf.Open(src)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			s, err := buildGGUF(f, nil, src, "", options{})
			if err != nil {
				t.Fatal(err)
			}
			if s.Config.RopeSections == [4]uint32{} || len(s.Config.RopeTable) == 0 {
				t.Fatalf("the fixture states sections %v and a table of %d: it tests nothing", s.Config.RopeSections, len(s.Config.RopeTable))
			}
			s.Config.RopeTable, s.Config.PictureRope = nil, jlm.PictureRope{}
			flags := s.Config.Flags
			for _, arm := range []struct {
				bits  jlm.Flags
				fresh bool
			}{{tc.bits, true}, {tc.wrong, false}} {
				s.Config.Flags = flags | arm.bits
				old := filepath.Join(dir, "old"+jlm.Ext)
				if _, err := jlm.Write(old, s, fp); err != nil {
					t.Fatal(err)
				}
				if c, err := jlm.Open(old); err != nil {
					t.Fatal(err)
				} else {
					_, err := c.Config().RopePairs()
					c.Close()
					if err == nil {
						t.Fatal("the container written without its table reads one")
					}
				}
				if changed, err := jlm.Restate(old, StateStored); err != nil || !changed {
					t.Fatalf("restating with bits %#x: changed %v, %v", arm.bits, changed, err)
				}
				got, err := os.ReadFile(old)
				if err != nil {
					t.Fatal(err)
				}
				if bytes.Equal(got, want) != arm.fresh {
					t.Fatalf("bits %#x: the restated container is the fresh one: %v, want %v", arm.bits, !arm.fresh, arm.fresh)
				}
				if arm.fresh {
					if changed, err := jlm.Restate(old, StateStored); err != nil || changed {
						t.Fatalf("restating a container that states its table: changed %v, %v", changed, err)
					}
				}
			}
		})
	}
}
