// Package hf acquires GGUF weights from HuggingFace.
//
// It is acquisition and nothing else, not an HF client library: one verb (make
// this file present locally, resumably, and say where) plus the queries that
// verb needs, on net/http alone.
//
// It reads no environment: the token, endpoint and destination are fields, and
// cmd/jitllm reads HF_TOKEN and hands it over.
package hf

import (
	"fmt"
	"net/url"
	"path"
	"path/filepath"
	"strings"
)

// DefaultEndpoint is the Hub. A mirror or a private Hub deployment replaces it
// through Client.Endpoint; it is also what makes the download path testable
// against an httptest.Server rather than against the internet.
const DefaultEndpoint = "https://huggingface.co"

// Ref names one file inside a HuggingFace model repository.
//
// An empty File is a reference to a repository, not a defaulted file:
// Client.Resolve names the file when the repo holds exactly one GGUF and
// refuses when it holds several, since the wrong quant runs fine and is not the
// model asked for.
type Ref struct {
	Repo string // "owner/name"
	Rev  string // a branch, a tag or a commit sha; "main" when the caller named none
	File string // the path inside the repo; "" means the file is not chosen yet
}

// hosts lists the Hub's hosts. A URL on any other host is an error, not a
// local path. hf.co is what the Hub's share button copies.
func isHubHost(h string) bool {
	switch h {
	case "huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co":
		return true
	}
	return false
}

// Parse reports whether s names a file on HuggingFace, and what it names.
//
// remote is true when s carries a scheme this package owns, whether or not it
// parsed: a malformed hf:// reference is never a local file. Callers check err
// first.
//
// A bare "owner/repo/file.gguf" is deliberately not accepted: it is also a
// relative path, and the scheme is the user saying which one they mean.
func Parse(s string) (r Ref, remote bool, err error) {
	switch {
	case strings.HasPrefix(s, "hf://"):
		r, err = parseShort(s)
		return r, true, err
	case strings.HasPrefix(s, "https://"), strings.HasPrefix(s, "http://"):
		r, err = parseURL(s)
		return r, true, err
	}
	return Ref{}, false, nil
}

// parseShort reads hf://OWNER/REPO[@REV][/PATH/TO/FILE.gguf].
func parseShort(s string) (Ref, error) {
	body := strings.TrimSuffix(strings.TrimPrefix(s, "hf://"), "/")
	if body == "" {
		return Ref{}, fmt.Errorf("%q names no repository: hf://OWNER/REPO[@REV]/FILE.gguf", s)
	}
	seg := strings.Split(body, "/")
	if len(seg) < 2 {
		return Ref{}, fmt.Errorf("%q is missing the repository name: hf://OWNER/REPO[@REV]/FILE.gguf", s)
	}
	name, rev := seg[1], "main"
	// The revision rides on the repo segment, not on a query parameter, so the
	// whole reference stays one shell word with nothing to quote.
	if i := strings.IndexByte(name, '@'); i >= 0 {
		name, rev = name[:i], name[i+1:]
	}
	r := Ref{Repo: seg[0] + "/" + name, Rev: rev, File: strings.Join(seg[2:], "/")}
	return r, r.Validate(s)
}

// parseURL reads the two URLs the Hub's own UI hands out: the address bar
// (/blob/) and the download button (/resolve/), which name the same bytes.
func parseURL(s string) (Ref, error) {
	u, err := url.Parse(s)
	if err != nil {
		return Ref{}, fmt.Errorf("%q is not a URL: %w", s, err)
	}
	if !isHubHost(strings.ToLower(u.Hostname())) {
		return Ref{}, fmt.Errorf("%q is on %s: this fetches from huggingface.co and hf.co only.\n"+
			"  For anything else, download it yourself and convert the local file",
			s, u.Hostname())
	}
	seg := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(seg) < 2 || seg[0] == "" || seg[1] == "" {
		return Ref{}, fmt.Errorf("%q names no repository: "+
			"https://huggingface.co/OWNER/REPO/resolve/REV/FILE.gguf", s)
	}
	r := Ref{Repo: seg[0] + "/" + seg[1], Rev: "main"}
	if len(seg) > 2 {
		switch seg[2] {
		case "resolve", "blob", "raw":
			if len(seg) < 5 {
				return Ref{}, fmt.Errorf("%q stops at the revision and names no file: "+
					"https://huggingface.co/OWNER/REPO/resolve/REV/FILE.gguf", s)
			}
			r.Rev, r.File = seg[3], strings.Join(seg[4:], "/")
		case "tree":
			return Ref{}, fmt.Errorf("%q is a directory listing, not a file: open the file on "+
				"the Hub and copy ITS address, or drop everything after the repository name "+
				"to be shown what the repo holds", s)
		default:
			return Ref{}, fmt.Errorf("%q is not a file in a repository: "+
				"https://huggingface.co/OWNER/REPO/resolve/REV/FILE.gguf", s)
		}
	}
	return r, r.Validate(s)
}

// Validate refuses a reference that cannot be turned into a request and a
// filename. orig is quoted back so the message names what the caller typed.
//
// The "." and ".." refusals matter: File's basename becomes a path under a
// directory the caller chose, and ".." would write outside it.
func (r Ref) Validate(orig string) error {
	if orig == "" {
		orig = "hf://" + r.Repo
	}
	owner, name, ok := strings.Cut(r.Repo, "/")
	if !ok || owner == "" || name == "" || strings.Contains(name, "/") {
		return fmt.Errorf("%q: the repository is OWNER/NAME, and %q is not", orig, r.Repo)
	}
	for _, s := range []string{owner, name, r.Rev} {
		if s == "" || s == "." || s == ".." || strings.ContainsAny(s, "/\\ \t\n") {
			return fmt.Errorf("%q: %q cannot be part of a repository reference", orig, s)
		}
	}
	if r.File == "" {
		return nil
	}
	for _, s := range strings.Split(r.File, "/") {
		if s == "" || s == "." || s == ".." || strings.ContainsAny(s, "\\\t\n") {
			return fmt.Errorf("%q: %q cannot be part of a path inside the repository", orig, s)
		}
	}
	return nil
}

// String renders the reference in the short form, which is the form to paste
// back into a command line.
func (r Ref) String() string {
	s := "hf://" + r.Repo
	if r.Rev != "" && r.Rev != "main" {
		s += "@" + r.Rev
	}
	if r.File != "" {
		s += "/" + r.File
	}
	return s
}

// URL is where the bytes are. endpoint may be empty for the Hub.
func (r Ref) URL(endpoint string) string {
	return base(endpoint) + "/" + esc(r.Repo) + "/resolve/" + esc(r.Rev) + "/" + esc(r.File)
}

// APIURL is where the file list is; the Hub gives it a different shape from URL.
func (r Ref) APIURL(endpoint string) string {
	return base(endpoint) + "/api/models/" + esc(r.Repo) + "/revision/" + esc(r.Rev)
}

func base(endpoint string) string {
	if endpoint == "" {
		return DefaultEndpoint
	}
	return strings.TrimSuffix(endpoint, "/")
}

// esc escapes each segment and leaves the separators, so a filename with a
// space or a plus survives the round trip.
func esc(p string) string {
	seg := strings.Split(p, "/")
	for i, s := range seg {
		seg[i] = url.PathEscape(s)
	}
	return strings.Join(seg, "/")
}

// Dest is where r's file lands under dir.
//
// The name is the basename, not the Hub cache's hashed tree, so a download
// lands beside existing models under the name they are known by. The cost is a
// possible collision between repos, which Fetch's size and digest checks turn
// into an error.
func (r Ref) Dest(dir string) (string, error) {
	if r.File == "" {
		return "", fmt.Errorf("%s names a repository and not a file", r)
	}
	// path.Base eats a trailing separator (Base("sub/") is "sub"), so a
	// directory reference would otherwise look like a filename.
	b := path.Base(r.File)
	if strings.HasSuffix(r.File, "/") || b == "" || b == "." || b == ".." || b == "/" || strings.ContainsAny(b, `/\`) {
		return "", fmt.Errorf("%s: %q is not a filename", r, b)
	}
	return filepath.Join(dir, b), nil
}
