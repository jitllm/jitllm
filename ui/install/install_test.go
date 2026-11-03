package install

import (
	"encoding/xml"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeApp lays out jitllm.app with its executable and the bundled programs,
// and returns the executable and the bin directory.
func fakeApp(t *testing.T, root string) (string, string) {
	t.Helper()
	exe := filepath.Join(root, "jitllm.app", "Contents", "MacOS", "jitllm-desktop")
	bin := filepath.Join(root, "jitllm.app", "Contents", "Resources", "bin")
	for _, p := range append([]string{exe}, filepath.Join(bin, "jitllm"), filepath.Join(bin, "jitllmd")) {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return exe, bin
}

func TestBundledFindsThePrograms(t *testing.T) {
	root := t.TempDir()
	exe, bin := fakeApp(t, root)
	if got, ok := bundledFrom(exe); !ok || got != bin {
		t.Fatalf("bundledFrom = %q, %v; want %q", got, ok, bin)
	}
	if _, ok := bundledFrom(filepath.Join(root, "jitllm-desktop")); ok {
		t.Error("a bare binary was taken for a bundle")
	}
	if err := os.Remove(filepath.Join(bin, "jitllmd")); err != nil {
		t.Fatal(err)
	}
	if _, ok := bundledFrom(exe); ok {
		t.Error("a bundle missing jitllmd was taken as carrying the programs")
	}
}

func TestLinkUnlinkRoundTrip(t *testing.T) {
	root := t.TempDir()
	_, bin := fakeApp(t, root)
	dirs := []string{filepath.Join(root, "usr-local-bin"), filepath.Join(root, "home", ".local", "bin")}
	if d := Linked(bin, dirs); d != "" {
		t.Fatalf("linked in %s before Link", d)
	}
	if err := Link(bin, dirs[1]); err != nil {
		t.Fatal(err)
	}
	if d := Linked(bin, dirs); d != dirs[1] {
		t.Fatalf("Linked = %q, want %q", d, dirs[1])
	}
	// Again: an existing link is this app's to replace.
	if err := Link(bin, dirs[1]); err != nil {
		t.Fatalf("relinking: %v", err)
	}
	// A link into another copy of the app is not this one's.
	_, other := fakeApp(t, filepath.Join(root, "other"))
	if d := Linked(other, dirs); d != "" {
		t.Errorf("links into %s counted for %s", bin, other)
	}
	if err := Unlink(other, dirs[1]); err != nil {
		t.Fatal(err)
	}
	if Linked(bin, dirs) == "" {
		t.Fatal("Unlink removed links into another bundle")
	}
	if err := Unlink(bin, dirs[1]); err != nil {
		t.Fatal(err)
	}
	for _, p := range Programs() {
		if _, err := os.Lstat(filepath.Join(dirs[1], p)); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("%s still there after Unlink: %v", p, err)
		}
	}
}

// TestLinkRefusesAFileItDidNotMake: a jitllm installed by the install
// script is a regular file, and Link leaves it -- and every other name --
// untouched.
func TestLinkRefusesAFileItDidNotMake(t *testing.T) {
	root := t.TempDir()
	_, bin := fakeApp(t, root)
	dir := filepath.Join(root, "bin")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "jitllmd"), []byte("mine"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Link(bin, dir); !errors.Is(err, ErrNotALink) {
		t.Fatalf("Link over a regular file: %v, want ErrNotALink", err)
	}
	if _, err := os.Lstat(filepath.Join(dir, "jitllm")); !errors.Is(err, os.ErrNotExist) {
		t.Error("a refused Link wrote a link anyway")
	}
	if b, err := os.ReadFile(filepath.Join(dir, "jitllmd")); err != nil || string(b) != "mine" {
		t.Errorf("the file was changed: %q %v", b, err)
	}
}

// TestAgentPlistIsWellFormed decodes the property list and reads back the
// arguments, escaped paths included.
func TestAgentPlistIsWellFormed(t *testing.T) {
	a := Agent{Home: "/srv/h&y"}
	p := a.PlistFor("/Applications/jitllm.app/Contents/Resources/bin/jitllmd", "/srv/h&y/<models>")
	d := xml.NewDecoder(strings.NewReader(p))
	d.Strict = false
	var strs []string
	inString := false
	for {
		tok, err := d.Token()
		if err != nil {
			break
		}
		switch v := tok.(type) {
		case xml.StartElement:
			inString = v.Name.Local == "string"
		case xml.CharData:
			if inString {
				strs = append(strs, string(v))
			}
		case xml.EndElement:
			inString = false
		}
	}
	want := []string{AgentLabel,
		"/Applications/jitllm.app/Contents/Resources/bin/jitllmd", "serve", "-addr", AgentAddr, "-models", "/srv/h&y/<models>",
		"Background", a.Log(), a.Log()}
	if strings.Join(strs, "|") != strings.Join(want, "|") {
		t.Errorf("strings\n got %q\nwant %q", strs, want)
	}
	if !strings.HasPrefix(a.Plist(), "/srv/h&y/Library/LaunchAgents/") {
		t.Errorf("plist at %s", a.Plist())
	}
}

func TestQuoting(t *testing.T) {
	if got := shQuote("/a b/it's"); got != `'/a b/it'\''s'` {
		t.Errorf("shQuote = %s", got)
	}
	if got := asQuote(`say "hi" \ there`); got != `"say \"hi\" \\ there"` {
		t.Errorf("asQuote = %s", got)
	}
}
