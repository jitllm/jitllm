//go:build linux

package engine

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// visionModel is the smallest of the three projector families.
var visionModel = testmodels.Path("SmolVLM-256M-Instruct-Q8_0-vlm.jlm")

// A picture attached in the app must reach the model as the picture.
//
// It runs the engine as the screen drives it (session.ChatMessages carrying a
// path, the worker encoding with the tower and prefilling spans), which
// model.TestChatSpansDescribeAnImage does not cover. A blank picture runs
// beside it, because a tower wired to nothing still answers fluently.
func TestAnAttachedPictureReachesTheModel(t *testing.T) {
	fresh.Must(t, visionModel)
	if _, err := os.Stat(visionModel); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	pic := filepath.Join("..", "..", "engine", "model", "testdata", "quad-and-disc.png")
	if _, err := os.Stat(pic); err != nil {
		t.Fatalf("IMAGE MISSING: %v", err)
	}
	// A private cache dir: a prefix an earlier run left on disk restores the
	// picture's rows whole, the image cache is never asked, and the history
	// half below reads 0 hits.
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	blank := filepath.Join(t.TempDir(), "blank.png")
	writeGrey(t, blank, 384)
	// A private cache folder: the prefix cache outlives the process, and pages
	// an earlier run left restore this test's prompts whole, so neither the
	// tower nor the image cache runs and the follow-up's assertion reads 0 hits.
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sh := newTestShell()
	st := sh.Store
	e := New(sh, sh.state())
	defer e.Close()
	e.Load(visionModel)
	pump(t, sh, 180*time.Second, "the model to load", func() bool { return st.Loaded.Get() })
	if !st.Vision.Get() {
		t.Fatal("a VLM loaded and Store.Vision is false: the attach control stays disabled")
	}

	ask := func(chat bool, hist []session.Turn, prompt string, imgs ...string) string {
		t.Helper()
		reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
		e.Send(session.ChatRequest{
			Chat: chat, Prompt: prompt, Reply: reply, MaxTokens: 32,
			Messages: session.ChatMessages("", hist, prompt, imgs...),
		})
		pump(t, sh, 120*time.Second, "generation to finish", func() bool { return !st.Busy.Get() })
		return strings.ToLower(st.Turn(reply).Text)
	}

	// Completion mode, as the app defaults: the picture still goes through the
	// template, which is what the screen builds Messages for.
	//
	// The question does not name the answer: a leading question got "a circle"
	// for a blank square too.
	const describe = "Describe this image in one sentence."
	got := ask(false, nil, describe, pic)
	none := ask(false, nil, describe, blank)
	t.Logf("picture %q", got)
	t.Logf("blank   %q", none)
	if !strings.Contains(got, "yellow") {
		t.Errorf("the description does not name the yellow disc: %q", got)
	}
	if strings.Contains(none, "yellow") {
		t.Errorf("a BLANK picture is yellow too (%q): the picture is not what answered", none)
	}

	// A follow-up turn carries the picture in its history, encoded once and
	// taken from the cache. The question is leading, so this half proves the
	// history path and the cache, not sight.
	_, miss0 := e.m.ImageCacheStats()
	hist := []session.Turn{
		{Role: session.RoleUser, Text: "What shape is in the middle?", Images: []string{pic}},
		{Role: session.RoleAssistant, Text: "The shape in the middle is a circle."},
	}
	follow := ask(true, hist, "What colour is the circle?")
	t.Logf("follow  %q", follow)
	if !strings.Contains(follow, "yellow") {
		t.Errorf("the follow-up does not see the picture from the history: %q", follow)
	}
	if hits, miss1 := e.m.ImageCacheStats(); miss1 != miss0 || hits == 0 {
		t.Errorf("the history's picture ran the tower again (%d misses, then %d; %d hits)", miss0, miss1, hits)
	}
}

func writeGrey(t *testing.T, path string, n int) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, n, n))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 128, 128, 128, 255
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}
