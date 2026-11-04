package convert

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok"
)

// TestSmolVLMStatesItsTower converts SmolVLM-256M's two files and holds the
// container's stated tower (jlm.Tower) to what the converter knows of
// idefics3: it validates, every weight its programs bind is a tensor of the
// container, its square and patch grid are the tower's, its shuffle the
// file's, and its markers are mtmd's overview pieces. The record read back
// from the file is the one the conversion stated.
func TestSmolVLMStatesItsTower(t *testing.T) {
	src := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")
	mm := testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "smolvlm.jlm")
	if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v := f.Vision()
	if v == nil || v.Tower == nil {
		t.Fatal("the container states no tower")
	}
	tw := v.Tower
	if err := tw.Valid(); err != nil {
		t.Fatal(err)
	}
	for _, r := range tw.Roles() {
		if !f.Has(r, jlm.DenseBlock, -1) {
			t.Errorf("the tower binds %v, which the container does not hold", r)
		}
	}
	if got := tw.Pre[0].Ints[:2]; got[0] != int32(v.ImageSize) || got[1] != int32(v.ImageSize) {
		t.Errorf("the picture is resized to %v, want the tower's %d square", got, v.ImageSize)
	}
	if got := tw.Entry.Ops[0].Span; got != int32(v.PatchSize) {
		t.Errorf("patches of %d, want the tower's %d", got, v.PatchSize)
	}
	if got := tw.Head.Ops[1]; got.Code != jlm.TOpRegroup || got.Span != int32(v.Scale) {
		t.Errorf("the head's second op is %+v, want the file's shuffle of %d", got, v.Scale)
	}
	voc, err := tok.New(f.Vocab())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		ids  []int32
		want []string
	}{
		{tw.Span.Before, []string{"\n\n", "<fake_token_around_image>", "<global-img>"}},
		{tw.Span.After, []string{"<fake_token_around_image>"}},
	} {
		// A special token decodes to nothing, so each marker is held to the
		// id its piece looks up.
		var want []int32
		for _, p := range c.want {
			id, ok := voc.ID(p)
			if !ok {
				t.Fatalf("%q is no token of SmolVLM's vocabulary", p)
			}
			want = append(want, id)
		}
		if !reflect.DeepEqual(c.ids, want) {
			t.Errorf("the span's markers are %v, want mtmd's %q at %v", c.ids, c.want, want)
		}
	}
	t.Logf("%d steps, %d entry ops, %d head ops, %d weights bound, markers %v / %v",
		len(tw.Pre), len(tw.Entry.Ops), len(tw.Head.Ops), len(tw.Roles()), tw.Span.Before, tw.Span.After)
}

// TestGemma3StatesItsTower converts Gemma 3 4B's two files and holds the
// stated tower to Gemma3ImageProcessor and the projector: Pillow's BILINEAR
// resize to the tower's square, SigLIP's patches, the post-norm, a pool of
// the file's scale, the soft-embedding RMS norm and the input projection;
// every bound weight a tensor of the container; the picture written as
// Gemma3Processor writes it, its rows attending both ways.
func TestGemma3StatesItsTower(t *testing.T) {
	src := testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")
	mm := testmodels.Path("vlm/mmproj-gemma-3-4b-it-f16.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "gemma3.jlm")
	if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v := f.Vision()
	if v == nil || v.Tower == nil {
		t.Fatal("the container states no tower")
	}
	tw := v.Tower
	if err := tw.Valid(); err != nil {
		t.Fatal(err)
	}
	for _, r := range tw.Roles() {
		if !f.Has(r, jlm.DenseBlock, -1) {
			t.Errorf("the tower binds %v, which the container does not hold", r)
		}
	}
	if got := tw.Pre[0].Ints; got[0] != int32(v.ImageSize) || got[1] != int32(v.ImageSize) ||
		jlm.Filter(got[2]) != jlm.FilterPILBilinear || jlm.Precision(got[3]) != jlm.PrecPIL {
		t.Errorf("the picture is resized by %v, want Pillow's bilinear to the tower's %d square", got, v.ImageSize)
	}
	var codes []jlm.OpCode
	for _, op := range tw.Head.Ops {
		codes = append(codes, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpLayerNorm, jlm.TOpPool, jlm.TOpRMSNorm, jlm.TOpLinear}; !reflect.DeepEqual(codes, want) {
		t.Errorf("the head is %v, want %v", codes, want)
	}
	if got := tw.Head.Ops[1].Span; got != int32(v.Scale) || v.Scale != 4 {
		t.Errorf("a pool of %d over the file's %d, want 4", got, v.Scale)
	}
	if s := tw.Span; s.BeforeText != "\n\n<start_of_image>" || s.AfterText != "<end_of_image>\n\n" || !s.Bidir ||
		len(s.Before) != 0 || len(s.After) != 0 {
		t.Errorf("the picture's span is %+v, want Gemma3Processor's text and rows attending both ways", s)
	}
	if v.ImageSize != 896 || v.PatchSize != 14 || v.ProjDim != 2560 {
		t.Errorf("a %d square of %d patches into %d, want 896, 14 and 2560", v.ImageSize, v.PatchSize, v.ProjDim)
	}
}
