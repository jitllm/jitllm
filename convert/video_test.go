package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestQwen3VLStatesItsVideo converts Qwen3-VL's fixture and reads the video
// its container states: Qwen3-VL's published sampling and bounds, the entry
// over two frames binding the second patch plane, and that plane in the
// container beside the planes' sum -- its own matrix in the sum's layout, not
// the sum again.
func TestQwen3VLStatesItsVideo(t *testing.T) {
	src := testmodels.Path("synth-qwen3vl.gguf")
	mm := testmodels.Path("mmproj-synth-qwen3vl.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py synth-qwen3vl) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "qwen3vl.jlm")
	if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v := f.Vision()
	if v == nil || v.Tower == nil || v.Tower.Video == nil {
		t.Fatal("the container states no video")
	}
	vid := v.Tower.Video
	if err := vid.Valid(); err != nil {
		t.Fatal(err)
	}
	if vid.Fps != 2 || vid.DefaultFps != 24 || vid.MinFrames != 4 || vid.MaxFrames != 768 {
		t.Fatalf("the video samples %+v", vid)
	}
	if got := vid.Pre[0].Ints; got[1] != 4096 || got[2] != 25165824 || jlm.Precision(got[4]) != jlm.PrecTorch {
		t.Fatalf("the video resizes with %v", got)
	}
	var binds []jlm.Role
	for _, op := range vid.Entry.Ops {
		binds = append(binds, op.Bind...)
	}
	if !slices.Contains(binds, jlm.RoleVPatchEmbd2) || !slices.Contains(binds, jlm.RoleVPatchEmbd) {
		t.Fatalf("the video's entry binds %v", binds)
	}
	sum, ok1 := f.Find(jlm.RoleVPatchEmbd, jlm.DenseBlock, -1)
	second, ok2 := f.Find(jlm.RoleVPatchEmbd2, jlm.DenseBlock, -1)
	if !ok1 || !ok2 {
		t.Fatalf("the container holds the sum %v and the second plane %v", ok1, ok2)
	}
	if sum.Type != second.Type || sum.Dims != second.Dims {
		t.Fatalf("the second plane is %v %v and the sum %v %v", second.Type, second.Dims, sum.Type, sum.Dims)
	}
	a, err := f.ReadPayload(sum)
	if err != nil {
		t.Fatal(err)
	}
	b, err := f.ReadPayload(second)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(a, b) {
		t.Fatal("the second plane holds the sum")
	}
	if _, ok := f.Find(jlm.RoleVPatchEmbd, jlm.DenseBlock, 1); ok {
		t.Fatal("the second plane is held twice")
	}
}
