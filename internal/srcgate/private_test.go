package srcgate

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoPrivateValuesInTrackedFiles keeps the repository publishable: no
// tracked file may carry a machine's address, a person's home or mount path,
// or a private host, key or account name,
// and no tracked path may be a local-only file (CLAUDE.md, local-*), which
// .gitignore keeps out and this keeps out of an index that was forced.
//
// What is allowed is what documentation needs: loopback and the unspecified
// address, the RFC 5737 documentation ranges, placeholder users (/home/u),
// and hardware model names, which are public facts. SVG path data is skipped
// for addresses, because "3.5.7.7" there is four coordinates.
//
// The set of files is `git ls-files`; without git (a tarball, a shipped test
// binary) it is a walk of the tree that skips dot-directories and the model
// directory, which is what .gitignore would leave.
func TestNoPrivateValuesInTrackedFiles(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	files := trackedFiles(t, root)
	if len(files) < 500 {
		t.Fatalf("found %d files under %s: the tree was not found, so this gate proved nothing", len(files), root)
	}
	self := filepath.ToSlash(filepath.Join("internal", "srcgate", "private_test.go"))
	privateNames := localPrivateNames(t, root)

	var (
		mu    sync.Mutex
		found []string
		wg    sync.WaitGroup
		next  = make(chan string)
	)
	report := func(s string) {
		mu.Lock()
		found = append(found, s)
		mu.Unlock()
	}
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rel := range next {
				for _, f := range privateValuesIn(root, rel, rel == self, privateNames) {
					report(f)
				}
			}
		}()
	}
	for _, rel := range files {
		next <- rel
	}
	close(next)
	wg.Wait()

	sort.Strings(found)
	for _, f := range found {
		t.Error(f)
	}
}

// trackedFiles lists the repository's files relative to root, with forward
// slashes.
func trackedFiles(t *testing.T, root string) []string {
	t.Helper()
	cmd := exec.Command("git", "ls-files", "-z")
	cmd.Dir = root
	if out, err := cmd.Output(); err == nil {
		var files []string
		for _, f := range bytes.Split(out, []byte{0}) {
			if len(f) > 0 {
				files = append(files, string(f))
			}
		}
		return files
	}
	var files []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != root && (strings.HasPrefix(d.Name(), ".") || d.Name() == "models" || d.Name() == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		// A worktree's .git is a file naming the main checkout's path; it is
		// git's, not the tree's, like the .git directory skipped above, and a
		// models link names the model disk, like the models directory.
		if d.Name() == ".git" || d.Name() == "models" && d.Type()&fs.ModeSymlink != 0 {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		files = append(files, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

// localPrivateNames reads root/local-private-names: strings that identify a
// person, a host or a key and have no business in a public tree, whatever the
// context. The specific ones cannot be spelled here -- this file is public,
// and the list would publish what it guards -- so they live one per line at
// the root of a checkout, which .gitignore keeps out of the tree like every
// local-* file. A checkout without that file runs the generic checks alone.
// Blank lines and #-comments are ignored, names compared case-insensitively.
func localPrivateNames(t *testing.T, root string) []string {
	b, err := os.ReadFile(filepath.Join(root, "local-private-names"))
	if os.IsNotExist(err) {
		t.Log("no local-private-names in this checkout: only the generic checks run")
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, l := range strings.Split(string(b), "\n") {
		l = strings.ToLower(strings.TrimSpace(l))
		if l != "" && !strings.HasPrefix(l, "#") {
			names = append(names, l)
		}
	}
	return names
}

func homePath() *regexp.Regexp {
	return regexp.MustCompile(`/(home|media|Users)/([A-Za-z0-9_.$<>{}-]+)`)
}

func svgPath() *regexp.Regexp { return regexp.MustCompile(`\bd=\\?"[^"]*"`) }

// placeholderUser is a user name documentation uses for "somebody".
func placeholderUser(name string) bool {
	switch name {
	case "u", "user", "you", "me", "USER", "$USER", "${USER}", "<user>", "example":
		return true
	}
	return false
}

// privateValuesIn returns one line per finding in the file at rel. The whole
// file is scanned at once and a line number is worked out only for a finding,
// because most of the bytes are numeric goldens that contain nothing.
func privateValuesIn(root, rel string, self bool, privateNames []string) []string {
	var found []string
	base := filepath.Base(rel)
	if base == "CLAUDE.md" || strings.HasPrefix(base, "local-") {
		found = append(found, fmt.Sprintf("%s: a local-only file is tracked (.gitignore keeps CLAUDE.md and local-* out)", rel))
	}
	if self {
		return found
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		if os.IsNotExist(err) {
			return found // deleted in the working tree, not yet in the index
		}
		return append(found, fmt.Sprintf("%s: %v", rel, err))
	}
	if bytes.IndexByte(b[:min(len(b), 8000)], 0) >= 0 {
		return found // binary
	}
	line := func(off int) int { return bytes.Count(b[:off], []byte{'\n'}) + 1 }

	lower := bytes.ToLower(b)
	for _, name := range privateNames {
		for off, i := 0, 0; ; off += i + len(name) {
			i = bytes.Index(lower[off:], []byte(name))
			if i < 0 {
				break
			}
			found = append(found, fmt.Sprintf("%s:%d: private name %q", rel, line(off+i), name))
		}
	}
	if bytes.Contains(b, []byte("/home/")) || bytes.Contains(b, []byte("/media/")) || bytes.Contains(b, []byte("/Users/")) {
		for _, m := range homePath().FindAllSubmatchIndex(b, -1) {
			if user := string(b[m[4]:m[5]]); !placeholderUser(user) {
				found = append(found, fmt.Sprintf("%s:%d: personal path %q", rel, line(m[0]), b[m[0]:m[1]]))
			}
		}
	}
	text := b
	if bytes.Contains(b, []byte(`d="`)) || bytes.Contains(b, []byte(`d=\"`)) {
		// Blank SVG path data in place, so offsets still name the line.
		text = svgPath().ReplaceAllFunc(bytes.Clone(b), func(m []byte) []byte {
			return bytes.Repeat([]byte{' '}, len(m))
		})
	}
	for _, at := range ipv4s(text) {
		a := string(text[at[0]:at[1]])
		if !allowedAddress(a) {
			found = append(found, fmt.Sprintf("%s:%d: address %s", rel, line(at[0]), a))
		}
	}
	return found
}

// ipv4s returns the [start, end) of every dotted quad of decimal octets (no
// leading zero, each at most 255) that is not part of a longer dotted or
// numeric run. It is a hand scanner rather than a regexp because it runs over
// every golden in the tree.
func ipv4s(b []byte) [][2]int {
	digit := func(c byte) bool { return c >= '0' && c <= '9' }
	joined := func(c byte) bool {
		return digit(c) || c == '.' || c == '-' || c == '_' || (c|0x20 >= 'a' && c|0x20 <= 'z')
	}
	var out [][2]int
	for i := 0; i < len(b); i++ {
		if !digit(b[i]) || (i > 0 && joined(b[i-1])) {
			continue // inside a number, a version (v1.2.3.4) or a coordinate run
		}
		j, ok := i, true
		for g := 0; g < 4 && ok; g++ {
			k := j
			for k < len(b) && digit(b[k]) {
				k++
			}
			n := k - j
			if n == 0 || n > 3 || (n > 1 && b[j] == '0') {
				ok = false
				break
			}
			if v, _ := strconv.Atoi(string(b[j:k])); v > 255 {
				ok = false
				break
			}
			j = k
			if g < 3 {
				if j >= len(b) || b[j] != '.' {
					ok = false
					break
				}
				j++
			}
		}
		if !ok {
			continue
		}
		if j < len(b) && (digit(b[j]) || b[j] == '.') {
			continue // a fifth field or a version tail: not an address
		}
		out = append(out, [2]int{i, j})
		i = j
	}
	return out
}

// allowedAddress is loopback, the unspecified address and the RFC 5737
// documentation ranges.
func allowedAddress(a string) bool {
	switch {
	case a == "0.0.0.0", strings.HasPrefix(a, "127."):
		return true
	case strings.HasPrefix(a, "192.0.2."), strings.HasPrefix(a, "198.51.100."), strings.HasPrefix(a, "203.0.113."):
		return true
	}
	return false
}
