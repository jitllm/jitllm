package screen

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/ui/app"
)

// A dropped picture is attached to the next message and the window stays on
// the conversation; a dropped model still goes to the catalog.
func TestDropFilesRoutesPicturesToTheSession(t *testing.T) {
	sh := dropShell(t)
	sh.Store.Vision.Set(true)
	sh.Store.Tab.Set(app.TabSession)

	DropFiles(sh, []string{"/pics/cat.JPG", "/pics/notes.txt"})

	if got := sh.Store.Attach.Get(); !slices.Equal(got, []string{"/pics/cat.JPG"}) {
		t.Errorf("attached %v, want the one picture", got)
	}
	if sh.Store.Tab.Get() != app.TabSession {
		t.Error("a picture drop left the conversation")
	}

	DropFiles(sh, []string{"/models/tinyllama.gguf"})
	if sh.Store.Tab.Get() != app.TabConvert {
		t.Error("a GGUF drop no longer reaches the Convert tab")
	}
}

// Without a vision tower a picture is refused by name rather than attached to
// a message that will then fail.
func TestDropFilesRefusesAPictureWithoutAVisionModel(t *testing.T) {
	sh := dropShell(t)
	sh.Store.Vision.Set(false)

	DropFiles(sh, []string{"/pics/cat.png"})

	if got := sh.Store.Attach.Get(); len(got) != 0 {
		t.Errorf("attached %v to a model that cannot see", got)
	}
	if got := sh.Store.Status.Get(); !strings.Contains(got, "vision") {
		t.Errorf("status = %q; the refusal must say why", got)
	}
}

// A picture sent in completion mode still reaches the engine as a templated
// turn, and the transcript keeps it.
func TestSendCarriesThePicture(t *testing.T) {
	sh := dropShell(t)
	st := sh.Store
	st.Loaded.Set(true)
	st.Vision.Set(true)
	st.Chat.Set(false)
	st.Draft.Set("what is this?")
	st.Attach.Set([]string{"/pics/cat.png"})
	var got []ChatRequest
	Attach(sh, Deps{Engine: &fakeEngine{send: func(r ChatRequest) { got = append(got, r) }}})

	sessionSend(sh, SessionOptions{MaxTokens: 8})

	if len(got) != 1 {
		t.Fatalf("%d request(s) sent", len(got))
	}
	r := got[0]
	if len(r.Messages) != 1 || !slices.Equal(r.Messages[0].Images, []string{"/pics/cat.png"}) {
		t.Errorf("completion-mode request carries %+v: the picture never reaches the engine", r.Messages)
	}
	if u := st.Turn(0); !slices.Equal(u.Images, []string{"/pics/cat.png"}) {
		t.Errorf("the user turn kept %v; the transcript cannot draw the picture", u.Images)
	}
	if len(st.Attach.Get()) != 0 {
		t.Error("the attachment survived the send and would go with the next message too")
	}
}

// "Prompt file..." takes text, not whatever file it is handed.
func TestPromptFileRefusesAPictureAndBinary(t *testing.T) {
	dir := t.TempDir()
	png := filepath.Join("..", "..", "engine", "testdata", "quad-and-disc.png")
	bin := filepath.Join(dir, "blob.dat")
	if err := os.WriteFile(bin, []byte{'a', 0, 'b'}, 0o644); err != nil {
		t.Fatal(err)
	}
	txt := filepath.Join(dir, "p.txt")
	if err := os.WriteFile(txt, []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := sessionReadPrompt(png); err == nil {
		t.Error("a PNG was read into the prompt box")
	}
	if _, err := sessionReadPrompt(bin); err == nil {
		t.Error("a file with a NUL byte was read into the prompt box")
	}
	if got, err := sessionReadPrompt(txt); err != nil || got != "hello" {
		t.Errorf("a text file read as %q, %v", got, err)
	}
}
