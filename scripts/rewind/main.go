// Command rewind fails a pull request whose commits would rewind work that
// other pull requests landed on main.
//
//	go run ./scripts/rewind -base origin/main -head <sha> -event $GITHUB_EVENT_PATH
//
// A commit written against an old main but committed on a newer parent
// carries old copies of files, and its diff deletes what the pull requests
// merged in between added; every test those pull requests added goes with
// their code, so CI stays green. This program reads the PR's own diff (merge
// base to head), blames each deleted line at the merge base, and calls a line
// foreign when the commit that added it landed after the PR's first commit
// was written: the PR cannot have known about it. A hunk of at least
// -min foreign lines, or a deleted file with any, is a rewind, reported with
// the file, the lost line ranges and the commit (and merge) that added them.
//
// Intent is stated in the PR description, one line each:
//
//	Removes: <path> (reason)
//
// A path ending in / covers everything under it.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type finding struct {
	Path   string
	Ranges []string
	Lines  int
	Commit string // "sha subject"
	Merge  string // the merge that brought Commit to the base, if any
}

type checker struct {
	dir   string
	base  string
	since int64 // author time of the PR's first commit
	times map[string]int64
}

func (c *checker) git(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = c.dir
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, ee.Stderr)
		}
		return "", err
	}
	return string(out), nil
}

func hunkRe() *regexp.Regexp {
	return regexp.MustCompile(`^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@`)
}
func blameRe() *regexp.Regexp {
	return regexp.MustCompile(`^([0-9a-f]{40}) \d+ (\d+)`)
}

// blame maps each line of path at rev to the commit that added it.
func (c *checker) blame(rev, path string) (map[int]string, error) {
	out, err := c.git("blame", "--porcelain", rev, "--", path)
	if err != nil {
		return nil, err
	}
	m := map[int]string{}
	blameRe := blameRe()
	for _, l := range strings.Split(out, "\n") {
		if g := blameRe.FindStringSubmatch(l); g != nil {
			n, _ := strconv.Atoi(g[2])
			m[n] = g[1]
		}
	}
	return m, nil
}

func (c *checker) commitTime(sha string) int64 {
	if t, ok := c.times[sha]; ok {
		return t
	}
	out, err := c.git("show", "-s", "--format=%ct", sha)
	if err != nil {
		return 0
	}
	t, _ := strconv.ParseInt(strings.TrimSpace(out), 10, 64)
	c.times[sha] = t
	return t
}

func (c *checker) describe(sha string) (commit, merge string) {
	out, _ := c.git("show", "-s", "--format=%h %s", sha)
	commit = strings.TrimSpace(out)
	m, _ := c.git("log", "--first-parent", "--merges", "--ancestry-path", "--reverse", "-1", "--format=%h %s", sha+".."+c.base)
	return commit, strings.TrimSpace(m)
}

func check(dir, base, head, body string, min int) ([]finding, error) {
	c := &checker{dir: dir, base: base, times: map[string]int64{}}
	mbOut, err := c.git("merge-base", base, head)
	if err != nil {
		return nil, err
	}
	mb := strings.TrimSpace(mbOut)
	times, err := c.git("log", "--format=%at", mb+".."+head)
	if err != nil {
		return nil, err
	}
	for _, f := range strings.Fields(times) {
		t, _ := strconv.ParseInt(f, 10, 64)
		if c.since == 0 || t < c.since {
			c.since = t
		}
	}
	if c.since == 0 {
		return nil, nil
	}
	diff, err := c.git("diff", "-M", "-U0", "--no-color", mb, head)
	if err != nil {
		return nil, err
	}
	allowed := removes(body)

	type key struct{ path, commit string }
	got := map[key]*finding{}
	var oldPath string
	deletedFile := false
	blames := map[string]map[int]string{}
	hunkRe := hunkRe()
	for _, l := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(l, "diff --git "):
			oldPath, deletedFile = "", false
		case strings.HasPrefix(l, "--- a/"):
			oldPath = l[len("--- a/"):]
		case l == "+++ /dev/null":
			deletedFile = true
		default:
			g := hunkRe.FindStringSubmatch(l)
			if g == nil || oldPath == "" {
				continue
			}
			start, _ := strconv.Atoi(g[1])
			count := 1
			if g[2] != "" {
				count, _ = strconv.Atoi(g[2])
			}
			if count == 0 || covered(allowed, oldPath) {
				continue
			}
			bl, ok := blames[oldPath]
			if !ok {
				if bl, err = c.blame(mb, oldPath); err != nil {
					return nil, err
				}
				blames[oldPath] = bl
			}
			per := map[string]int{}
			for n := start; n < start+count; n++ {
				if sha := bl[n]; sha != "" && c.commitTime(sha) > c.since {
					per[sha]++
				}
			}
			for sha, n := range per {
				if n < min && !deletedFile {
					continue
				}
				k := key{oldPath, sha}
				f := got[k]
				if f == nil {
					f = &finding{Path: oldPath}
					f.Commit, f.Merge = c.describe(sha)
					got[k] = f
				}
				f.Lines += n
				f.Ranges = append(f.Ranges, fmt.Sprintf("%d-%d", start, start+count-1))
			}
		}
	}
	var out []finding
	for _, f := range got {
		out = append(out, *f)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Commit < out[j].Commit
	})
	return out, nil
}

func removesRe() *regexp.Regexp {
	return regexp.MustCompile(`(?m)^Removes:[ \t]*(\S+)`)
}

func removes(body string) []string {
	var out []string
	for _, g := range removesRe().FindAllStringSubmatch(body, -1) {
		out = append(out, g[1])
	}
	return out
}

func covered(allowed []string, path string) bool {
	for _, a := range allowed {
		if a == path || (strings.HasSuffix(a, "/") && strings.HasPrefix(path, a)) {
			return true
		}
	}
	return false
}

func main() {
	dir := flag.String("C", ".", "repository")
	base := flag.String("base", "origin/main", "the branch the PR merges into")
	head := flag.String("head", "HEAD", "the PR's head commit")
	event := flag.String("event", "", "GitHub event payload (JSON); its pull_request.body carries the Removes: lines")
	min := flag.Int("min", 3, "foreign lines in one hunk that make it a rewind")
	flag.Parse()

	body := ""
	if *event != "" {
		b, err := os.ReadFile(*event)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		var ev struct {
			PR struct {
				Body string `json:"body"`
			} `json:"pull_request"`
		}
		if err := json.Unmarshal(b, &ev); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		body = ev.PR.Body
	}
	fs, err := check(*dir, *base, *head, body, *min)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if len(fs) == 0 {
		fmt.Println("rewind: no change of another PR is removed")
		return
	}
	for _, f := range fs {
		fmt.Printf("::error file=%s::rewinds %d lines (old lines %s) added by %s", f.Path, f.Lines, strings.Join(f.Ranges, ", "), f.Commit)
		if f.Merge != "" {
			fmt.Printf(" via %s", f.Merge)
		}
		fmt.Println()
	}
	fmt.Println("If the removal is intended, put `Removes: <path> (reason)` in the PR description; otherwise rebuild the commit on current main.")
	os.Exit(1)
}
