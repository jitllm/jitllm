package hf

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestParseAcceptsTheFormsItDocuments is the table behind the usage text.
//
// It needs no network and must not grow one: a gate that only runs when the
// Hub is reachable is a gate that skips.
func TestParseAcceptsTheFormsItDocuments(t *testing.T) {
	const file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
	for _, c := range []struct {
		in     string
		remote bool
		want   Ref
	}{
		// The short form, which is what this command's own messages print back.
		{"hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/" + file, true,
			Ref{"TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "main", file}},
		// A revision, pinned on the repo segment so the whole thing stays one word.
		{"hf://bartowski/Qwen3-1.7B-GGUF@v1.2/Qwen3-1.7B-Q4_K_M.gguf", true,
			Ref{"bartowski/Qwen3-1.7B-GGUF", "v1.2", "Qwen3-1.7B-Q4_K_M.gguf"}},
		{"hf://o/r@abc123", true, Ref{"o/r", "abc123", ""}},
		// A repository, file unchosen: Resolve answers or refuses.
		{"hf://example-org/models", true, Ref{"example-org/models", "main", ""}},
		{"hf://example-org/models/", true, Ref{"example-org/models", "main", ""}},
		// A path inside the repo survives whole.
		{"hf://example-org/models/tinyllamas/stories15M-q4_0.gguf", true,
			Ref{"example-org/models", "main", "tinyllamas/stories15M-q4_0.gguf"}},
		// The download button.
		{"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/" + file, true,
			Ref{"TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "main", file}},
		// ...with the query string the Hub appends to it.
		{"https://huggingface.co/o/r/resolve/main/m.gguf?download=true", true,
			Ref{"o/r", "main", "m.gguf"}},
		// The address bar, which is the URL a person actually has.
		{"https://huggingface.co/o/r/blob/main/m.gguf", true, Ref{"o/r", "main", "m.gguf"}},
		{"https://huggingface.co/o/r/raw/main/m.gguf", true, Ref{"o/r", "main", "m.gguf"}},
		// A commit sha, and a nested path, through the URL form.
		{"https://hf.co/o/r/resolve/9a0c1f2/sub/dir/m.gguf", true, Ref{"o/r", "9a0c1f2", "sub/dir/m.gguf"}},
		{"https://huggingface.co/o/r", true, Ref{"o/r", "main", ""}},
		{"http://huggingface.co/o/r/resolve/main/m.gguf", true, Ref{"o/r", "main", "m.gguf"}},

		// Local paths, which must pass through untouched. The engine's own
		// models are named like the last two.
		{"models/gemma-2b.gguf", false, Ref{}},
		{"/mnt/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", false, Ref{}},
		{"./out.jlm", false, Ref{}},
		{"", false, Ref{}},
	} {
		got, remote, err := Parse(c.in)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.in, err)
			continue
		}
		if remote != c.remote {
			t.Errorf("Parse(%q): remote %v, want %v", c.in, remote, c.remote)
			continue
		}
		if got != c.want {
			t.Errorf("Parse(%q) = %+v, want %+v", c.in, got, c.want)
		}
	}
}

// TestParseRefusesWhatItCannotName is the other half, and the more important
// one: a reference that is wrong must not come back as a local path, because the
// error a caller then prints is "no such file or directory" about a URL.
func TestParseRefusesWhatItCannotName(t *testing.T) {
	for _, in := range []string{
		"hf://",
		"hf://owner",                               // no repository
		"hf://owner/",                              // ditto, with the slash
		"hf:///repo/m.gguf",                        // no owner
		"hf://o/r/../../etc/passwd",                // a write outside the directory
		"hf://o/r/sub//m.gguf",                     // an empty segment
		"hf://o/@rev/m.gguf",                       // no repository name, only a revision
		"hf://o/r@/m.gguf",                         // an empty revision
		"https://example.com/a/b.gguf",             // not the Hub
		"https://huggingface.co/",                  // no repository
		"https://huggingface.co/owner",             // ditto
		"https://huggingface.co/o/r/tree/main/sub", // a directory listing
		"https://huggingface.co/o/r/resolve/main",  // a revision and no file
		"https://huggingface.co/o/r/discussions/3", // not a file at all
	} {
		_, remote, err := Parse(in)
		if err == nil {
			t.Errorf("Parse(%q) was accepted; it names no fetchable file", in)
			continue
		}
		if !remote {
			t.Errorf("Parse(%q) reported a LOCAL path for a malformed reference: %v", in, err)
		}
		t.Logf("%s -> %s", in, strings.SplitN(err.Error(), "\n", 2)[0])
	}
}

// TestDestNamesAFileUnderTheDirectory is the destination-path logic, and the
// reason it is its own gate is the traversal: File comes off a URL, and a
// basename is the only part of it that may reach the filesystem.
func TestDestNamesAFileUnderTheDirectory(t *testing.T) {
	dir := filepath.Join("/media", "models")
	for _, c := range []struct {
		r    Ref
		want string
	}{
		{Ref{"o/r", "main", "m.gguf"}, filepath.Join(dir, "m.gguf")},
		// A nested path flattens: the repo's directory layout is not ours.
		{Ref{"o/r", "main", "sub/dir/m.gguf"}, filepath.Join(dir, "m.gguf")},
		// The revision does not enter the name, which is the trade Ref.Dest
		// states: the file is named what everyone already calls it.
		{Ref{"o/r", "9a0c1f2", "m.gguf"}, filepath.Join(dir, "m.gguf")},
	} {
		got, err := c.r.Dest(dir)
		if err != nil {
			t.Errorf("%s.Dest: %v", c.r, err)
			continue
		}
		if got != c.want {
			t.Errorf("%s.Dest = %q, want %q", c.r, got, c.want)
		}
	}
	for _, r := range []Ref{
		{"o/r", "main", ""},     // a repository names no file
		{"o/r", "main", ".."},   // the parent of the model directory
		{"o/r", "main", "a/.."}, // ...spelled so path.Base has to do the work
		{"o/r", "main", "sub/"}, // a directory
		{"o/r", "main", "."},
		{"o/r", "main", "/"},
	} {
		got, err := r.Dest(dir)
		if err == nil {
			t.Errorf("Dest(%q) = %q; it must refuse a File that is not a filename", r.File, got)
		}
	}
}

// TestURLsAreBuiltFromTheRefAndNotFromTheInput: whatever form came in, the
// request goes to /resolve/, escaped. A blob:// URL that was pasted straight
// through would fetch an HTML page and convert it.
func TestURLsAreBuiltFromTheRefAndNotFromTheInput(t *testing.T) {
	r, _, err := Parse("https://huggingface.co/o/r/blob/main/sub/a b+c.gguf")
	if err != nil {
		t.Fatal(err)
	}
	const want = "https://huggingface.co/o/r/resolve/main/sub/a%20b+c.gguf"
	if got := r.URL(""); got != want {
		t.Errorf("URL = %q, want %q", got, want)
	}
	if got, want := r.APIURL("http://127.0.0.1:1/"), "http://127.0.0.1:1/api/models/o/r/revision/main"; got != want {
		t.Errorf("APIURL = %q, want %q", got, want)
	}
}
