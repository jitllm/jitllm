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
			testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
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
			testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
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

// TestLlavaStatesItsTower converts llava-phi-3-mini's two files and holds the
// stated tower to CLIPImageProcessor and LlavaForConditionalGeneration:
// Pillow's BICUBIC resize of the shorter side to the tower's square and its
// centre cropped, CLIP's patches after a class row and the pre-norm, the
// file's 23 blocks, the class row dropped and the two-layer projector with
// transformers' exact GELU; every bound weight a tensor of the container;
// the picture the processor's "<image>" then a newline.
func TestLlavaStatesItsTower(t *testing.T) {
	src := testmodels.Path("llava-phi-3-mini-int4.gguf")
	mm := testmodels.Path("llava-phi-3-mini-mmproj-f16.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "llava.jlm")
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
	if len(tw.Pre) != 3 || tw.Pre[0].Kind != jlm.PreResizeShort || tw.Pre[1].Kind != jlm.PreCenterCrop ||
		tw.Pre[0].Ints[0] != 336 || jlm.Filter(tw.Pre[0].Ints[1]) != jlm.FilterPILBicubic ||
		tw.Pre[1].Ints[0] != 336 || tw.Pre[1].Ints[1] != 336 {
		t.Errorf("the picture's steps are %+v, want Pillow's bicubic of the shorter side to 336 and its 336 centre", tw.Pre)
	}
	var entry, head []jlm.OpCode
	for _, op := range tw.Entry.Ops {
		entry = append(entry, op.Code)
	}
	for _, op := range tw.Head.Ops {
		head = append(head, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpPatchify, jlm.TOpLinear, jlm.TOpLeadRow, jlm.TOpPosTable, jlm.TOpAdd, jlm.TOpLayerNorm}; !reflect.DeepEqual(entry, want) {
		t.Errorf("the entry is %v, want %v", entry, want)
	}
	if want := []jlm.OpCode{jlm.TOpDropRows, jlm.TOpLinear, jlm.TOpStated, jlm.TOpLinear}; !reflect.DeepEqual(head, want) {
		t.Errorf("the head is %v, want %v", head, want)
	}
	if tw.Entry.Ops[0].Offset != 1 || tw.Head.Ops[0].Offset != 1 || tw.Head.Ops[2].Width != 3072 {
		t.Errorf("a class row of %d dropped by %d, a projector %d wide; want 1, 1 and 3072",
			tw.Entry.Ops[0].Offset, tw.Head.Ops[0].Offset, tw.Head.Ops[2].Width)
	}
	if v.NLayer != 23 || !v.Flags.Has(jlm.FlagQuickGELU) {
		t.Errorf("%d blocks, quick-GELU %v; want the file's 23 under CLIP's quick-GELU", v.NLayer, v.Flags.Has(jlm.FlagQuickGELU))
	}
	if s := tw.Span; s.Mark != "<image>\n" || s.AfterText != "\n" || s.Bidir {
		t.Errorf("the picture's span is %+v, want the processor's \"<image>\" and its newline", s)
	}
}

// TestInternVLStatesItsTower converts InternVL3-1B's two files and holds the
// stated tower to the reference's load_image and InternVLChatModel: between
// 1 and 12 tiles and a thumbnail, Pillow's bicubic to the tower's square;
// InternViT's patches after a class row (the class embedding read as the
// vector it is); the class row dropped, the 2x2 shuffle, mlp1's LayerNorm
// at nn.LayerNorm's epsilon, its two matrices with the exact GELU, which the
// blocks use too; every bound weight a tensor of the container; the picture
// between <img> and </img>.
func TestInternVLStatesItsTower(t *testing.T) {
	src := testmodels.Path("vlm/InternVL3-1B-Instruct-Q8_0.gguf")
	mm := testmodels.Path("vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "internvl.jlm")
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
	if len(tw.Pre) != 3 || tw.Pre[0].Kind != jlm.PreTile || !reflect.DeepEqual(tw.Pre[0].Ints, []int32{1, 12, 1}) ||
		tw.Pre[1].Ints[0] != 448 || jlm.Filter(tw.Pre[1].Ints[2]) != jlm.FilterPILBicubic {
		t.Errorf("the picture's steps are %+v, want 1 to 12 tiles and a thumbnail, Pillow's bicubic to 448", tw.Pre)
	}
	var head []jlm.OpCode
	for _, op := range tw.Head.Ops {
		head = append(head, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpDropRows, jlm.TOpRegroup, jlm.TOpLayerNorm, jlm.TOpLinear, jlm.TOpStated, jlm.TOpLinear}; !reflect.DeepEqual(head, want) {
		t.Errorf("the head is %v, want %v", head, want)
	}
	if op := tw.Head.Ops[1]; op.Span != 2 || tw.Head.Ops[2].Eps != 1e-5 || tw.Head.Ops[4].Width != 896 {
		t.Errorf("a shuffle by %d, a norm at %v, a projector %d wide; want 2, 1e-5 and 896", op.Span, tw.Head.Ops[2].Eps, tw.Head.Ops[4].Width)
	}
	var erf jlm.ExprBuilder
	erf.GELUErf(jlm.Operand(0))
	if !reflect.DeepEqual(tw.Block.Act.Terms, erf.T) {
		t.Errorf("the blocks' activation is %+v, want the exact GELU", tw.Block.Act)
	}
	voc, err := tok.New(f.Vocab())
	if err != nil {
		t.Fatal(err)
	}
	img, _ := voc.ID("<img>")
	end, _ := voc.ID("</img>")
	if s := tw.Span; !reflect.DeepEqual(s.Before, []int32{img}) || !reflect.DeepEqual(s.After, []int32{end}) ||
		s.Mark != "<image>\n" || s.AfterText != "\n" {
		t.Errorf("the picture's span is %+v, want <img> (%d) and </img> (%d) for the chat's \"<image>\"", s, img, end)
	}
	if vid := tw.Video; vid == nil || !vid.EveryFrame || !vid.Whole || !vid.StampIndex || vid.StampBefore != "Frame" ||
		vid.StampAfter != ": " || vid.Between != "\n" || len(vid.Before) != 0 || len(vid.After) != 0 {
		t.Errorf("the video reads %+v, want every frame whole, \"Frame{n}: \" before each and a line between", vid)
	}
}

// TestQwen2VLStatesItsTower converts synth-qwen2vl's two files
// (scripts/qwenvlgold.py) and holds the stated tower to Qwen2-VL's: a resize
// to the picture's own size in multiples of a merge window, within the
// processor class's bounds; the patches in the windows' order through the
// one matrix; the blocks' rotary over the grid, half a head's pairs a
// patch's row and half its column; the merger's LayerNorm, the fold of each
// window, the two matrices with the exact GELU between; and the picture
// between the vision markers, its grid the text model's over the merge.
func TestQwen2VLStatesItsTower(t *testing.T) {
	src := testmodels.Path("synth-qwen2vl.gguf")
	mm := testmodels.Path("mmproj-synth-qwen2vl.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py synth-qwen2vl) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "qwen2vl.jlm")
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
	if len(tw.Pre) != 2 || tw.Pre[0].Kind != jlm.PreSmartResize ||
		!reflect.DeepEqual(tw.Pre[0].Ints, []int32{28, 56 * 56, 28 * 28 * 1280, int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}) {
		t.Errorf("the picture's steps are %+v, want a size of its own in 28s within 56^2 and 28^2*1280, Pillow's bicubic", tw.Pre)
	}
	if op := tw.Entry.Ops[0]; op.Code != jlm.TOpPatchify || op.Span != 14 || op.Width != 2 || op.Offset != 0 {
		t.Errorf("the patches are %+v, want 14 a side in 2x2 windows' order", op)
	}
	hd := v.NEmbd / v.NHead
	if r := tw.Block.Rope; r.Base != 10000 || r.Dim != int32(hd/2) || !r.Neox ||
		!reflect.DeepEqual(r.Runs, []jlm.TowerRopeRun{{Pairs: int32(hd / 4)}, {Pairs: int32(hd / 4), Axis: 1}}) {
		t.Errorf("the blocks' rotary is %+v, want NEOX at 10000 over %d, %d pairs the row's and %d the column's", r, hd/2, hd/4, hd/4)
	}
	var head []jlm.OpCode
	for _, op := range tw.Head.Ops {
		head = append(head, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpLayerNorm, jlm.TOpFold, jlm.TOpLinear, jlm.TOpStated, jlm.TOpLinear}; !reflect.DeepEqual(head, want) {
		t.Errorf("the head is %v, want %v", head, want)
	}
	if op := tw.Head.Ops[1]; op.Span != 4 {
		t.Errorf("a fold of %d rows, want a 2x2 window's 4", op.Span)
	}
	var quick jlm.ExprBuilder
	quick.QuickGELU(jlm.Operand(0))
	if !reflect.DeepEqual(tw.Block.Act.Terms, quick.T) {
		t.Errorf("the blocks' activation is %+v, want quick_gelu", tw.Block.Act)
	}
	voc, err := tok.New(f.Vocab())
	if err != nil {
		t.Fatal(err)
	}
	vs, _ := voc.ID("<|vision_start|>")
	ve, _ := voc.ID("<|vision_end|>")
	if s := tw.Span; !reflect.DeepEqual(s.Before, []int32{vs}) || !reflect.DeepEqual(s.After, []int32{ve}) ||
		s.Mark != "<|vision_start|><|image_pad|><|vision_end|>" || s.GridDiv != 2 || s.Bidir {
		t.Errorf("the picture's span is %+v, want <|vision_start|> (%d) and <|vision_end|> (%d), a grid over 2", s, vs, ve)
	}
}

// TestQwen25VLStatesItsTower converts synth-qwen25vl's two files and holds
// the stated tower to Qwen2.5-VL's on top of Qwen2-VL's: RMSNorm blocks with
// a gated SiLU MLP, windows of window_size pixels (112, 8 patches) that
// every block but each n_wa_pattern'th attends inside, the merger's
// RMSNorm, and the head's rows put back in raster order.
func TestQwen25VLStatesItsTower(t *testing.T) {
	src := testmodels.Path("synth-qwen25vl.gguf")
	mm := testmodels.Path("mmproj-synth-qwen25vl.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py synth-qwen25vl) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "qwen25vl.jlm")
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
	if b := tw.Block; b.Norm != jlm.NormRMS || b.Window != 8 || !reflect.DeepEqual(b.Full, []int32{1, 3}) {
		t.Errorf("the blocks are norm %v, windows of %d, whole-picture blocks %v; want RMSNorm, 8 and [1 3]", b.Norm, b.Window, b.Full)
	}
	var silu jlm.ExprBuilder
	silu.Fn(jlm.FnMul, silu.SiLU(jlm.Operand(0)), jlm.Operand(1))
	if !reflect.DeepEqual(tw.Block.Act.Terms, silu.T) {
		t.Errorf("the blocks' activation is %+v, want the gated SiLU", tw.Block.Act)
	}
	var head []jlm.OpCode
	for _, op := range tw.Head.Ops {
		head = append(head, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpRMSNorm, jlm.TOpFold, jlm.TOpLinear, jlm.TOpStated, jlm.TOpLinear, jlm.TOpUntile}; !reflect.DeepEqual(head, want) {
		t.Errorf("the head is %v, want %v", head, want)
	}
	if op := tw.Entry.Ops[0]; op.Code != jlm.TOpPatchify || op.Width != 2 {
		t.Errorf("the patches are %+v, want 2x2 windows' order", op)
	}
}

// TestQwen35VLStatesItsTower converts Qwen3.5-VL's fixture, whose tower is
// Qwen3-VL's with no deepstack taps, and reads its statement back: the
// published processor's bounds, the patches through their biased matrix
// plus the 8x8 learned table resampled to the patch grid in the merge
// windows' order, LayerNorm blocks with the ungated GELU-tanh MLP, and
// Qwen2-VL's merger.
func TestQwen35VLStatesItsTower(t *testing.T) {
	src := testmodels.Path("synth-qwen35vl.gguf")
	mm := testmodels.Path("mmproj-synth-qwen35vl.gguf")
	for _, p := range []string{src, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (run scripts/qwenvlgold.py synth-qwen35vl) -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "qwen35vl.jlm")
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
	if pre := tw.Pre[0]; pre.Kind != jlm.PreSmartResize || pre.Ints[0] != 32 || pre.Ints[1] != 65536 || pre.Ints[2] != 16777216 {
		t.Errorf("the resize is %+v, want merge windows of 32 pixels within [65536, 16777216]", pre)
	}
	var entry []jlm.OpCode
	for _, op := range tw.Entry.Ops {
		entry = append(entry, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpPatchify, jlm.TOpLinear, jlm.TOpPosResample, jlm.TOpAdd}; !reflect.DeepEqual(entry, want) {
		t.Fatalf("the entry is %v, want %v", entry, want)
	}
	if op := tw.Entry.Ops[2]; op.Span != 8 || op.Width != 2 || op.Chain != jlm.ResampleAligned || op.Filter != jlm.TapsLinear {
		t.Errorf("the table's resample is %+v, want 8 a side, aligned, linear, in 2x2 windows' order", op)
	}
	if op := tw.Entry.Ops[1]; len(op.Bind) != 2 {
		t.Errorf("the patch matrix binds %v, want its bias too", op.Bind)
	}
	var gelu jlm.ExprBuilder
	gelu.GELUTanh(jlm.Operand(0))
	if b := tw.Block; b.Norm != jlm.NormLayer || b.Window != 0 || !reflect.DeepEqual(b.Act.Terms, gelu.T) {
		t.Errorf("the blocks are norm %v, windows of %d, activation %+v; want LayerNorm, none, the ungated GELU-tanh", b.Norm, b.Window, b.Act)
	}
	var head []jlm.OpCode
	for _, op := range tw.Head.Ops {
		head = append(head, op.Code)
	}
	if want := []jlm.OpCode{jlm.TOpLayerNorm, jlm.TOpFold, jlm.TOpLinear, jlm.TOpStated, jlm.TOpLinear}; !reflect.DeepEqual(head, want) {
		t.Errorf("the head is %v, want %v", head, want)
	}
}

// TestQwen3VLStatesItsTaps converts Qwen3-VL's fixture, whose tower taps
// blocks 0 and 2 (clip.vision.is_deepstack_layers), and reads the taps back:
// one a listed block, in order, each its block's rows through its own merger
// -- the merge window's patches one row, its LayerNorm, two matrices with
// the exact GELU between -- whose tensors the container holds at the
// block's index.
func TestQwen3VLStatesItsTaps(t *testing.T) {
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
	if v == nil || v.Tower == nil {
		t.Fatal("the container states no tower")
	}
	tw := v.Tower
	if err := tw.Valid(); err != nil {
		t.Fatal(err)
	}
	var blocks []int32
	for _, tp := range tw.Taps {
		blocks = append(blocks, tp.Block)
	}
	if !reflect.DeepEqual(blocks, []int32{0, 2}) {
		t.Fatalf("the taps read blocks %v, want [0 2]", blocks)
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	for _, tp := range tw.Taps {
		var codes []jlm.OpCode
		for _, op := range tp.Prog.Ops {
			codes = append(codes, op.Code)
			for _, r := range op.Bind {
				if !f.Has(r, jlm.DenseBlock, tp.Block) {
					t.Errorf("tap %d binds %v, which the container does not hold at its index", tp.Block, r)
				}
			}
		}
		if want := []jlm.OpCode{jlm.TOpFold, jlm.TOpLayerNorm, jlm.TOpLinear, jlm.TOpStated, jlm.TOpLinear}; !reflect.DeepEqual(codes, want) {
			t.Errorf("tap %d is %v, want %v", tp.Block, codes, want)
		}
		if op := tp.Prog.Ops[3]; !reflect.DeepEqual(op.Expr.Terms, gelu.T) || op.Width != 256 {
			t.Errorf("tap %d's activation is %+v over %d, want the exact GELU over 256", tp.Block, op.Expr, op.Width)
		}
	}
	if op := tw.Entry.Ops[2]; op.Code != jlm.TOpPosResample || op.Span != 8 {
		t.Errorf("the entry's third op is %+v, want the 8x8 table's resample", op)
	}
}

// TestGELUTowersStateTheExactForm converts Janus-Pro's and HunyuanVL's
// towers and reads back the erf GELU their classes run ("gelu", ACT2FN's
// GELUActivation), where the file states only GELU and llama.cpp runs its
// tanh form; and a file whose tower is not GELU is refused, not read as
// one.
func TestGELUTowersStateTheExactForm(t *testing.T) {
	var erf jlm.ExprBuilder
	erf.GELUErf(jlm.Operand(0))
	want := jlm.Expr{Terms: erf.T}
	for _, c := range []struct{ text, mmproj string }{
		{"minicpmv/Janus-Pro-1B-Q4_K_M.gguf", "minicpmv/mmproj-Janus-Pro-1B-Q8_0.gguf"},
		{"synth-hunyuanvl.gguf", "mmproj-synth-hunyuanvl.gguf"},
	} {
		src, mm := testmodels.Path(c.text), testmodels.Path(c.mmproj)
		for _, p := range []string{src, mm} {
			if _, err := os.Stat(p); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
			}
		}
		dst := filepath.Join(t.TempDir(), "tower.jlm")
		if _, err := FromGGUFs(src, mm, dst, jlm.Fingerprint{Writer: "gate"}); err != nil {
			t.Fatal(err)
		}
		f, err := jlm.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		v := f.Vision()
		if v == nil || v.Tower == nil || !reflect.DeepEqual(v.Tower.Block.Act, want) {
			f.Close()
			t.Fatalf("%s's tower blocks run %+v, the exact GELU is %+v", c.mmproj, v.Tower.Block.Act, want)
		}
		f.Close()
	}
	if _, err := erfGELU(&jlm.Vision{Flags: jlm.FlagQuickGELU}, "x"); err == nil {
		t.Fatal("a tower of QuickGELU was stated as the exact GELU")
	}
	if _, err := erfGELU(&jlm.Vision{}, "x"); err == nil {
		t.Fatal("a tower of SiLU was stated as the exact GELU")
	}
}
