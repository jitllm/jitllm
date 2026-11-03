// Package localserver is the one place every front end (the CLI, the window,
// the terminal app and jitllmd itself) learns where the local service is and
// where its models live: the default address, the per-system models folder,
// and the check "does a jitllmd answer here, and which version".
//
// It reads no environment: each command reads JITLLM_MODELS, XDG_DATA_HOME
// and its own address flag and hands them in.
package localserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// DefaultAddr is where the service listens unless told otherwise: localhost,
// which a server binds as 127.0.0.1 and [::1] and nothing else (see
// server.Listen), so installing it opens nothing to the network. A client
// dials it through NewTransport, which resolves "localhost" to loopback
// addresses only.
const DefaultAddr = "localhost:4242"

// LegacyAddr is where releases before DefaultAddr moved installed the
// service. An installed unit keeps the address it was written with, so a
// client that finds nothing on DefaultAddr looks here too.
const LegacyAddr = "127.0.0.1:8080"

// LegacyNotice is the one line a client says when it used the service on
// LegacyAddr.
const LegacyNotice = "the jitllmd service runs on " + LegacyAddr + ", the old default port; " +
	"`jitllmd service install` moves it to " + DefaultAddr

// LegacyFor is the address to try when nothing answers at addr: LegacyAddr
// when addr is DefaultAddr (or empty), and none otherwise, since an address
// the caller typed is never second-guessed. WithDefaults changes the pair.
func LegacyFor(addr string, opts ...Option) (string, bool) {
	o := optionsOf(opts)
	if addr == "" || addr == o.primary || addr == "http://"+o.primary {
		return o.legacy, true
	}
	return "", false
}

// ModelsDir is the models folder: models ($JITLLM_MODELS) when set, else the
// per-system default (ModelsDirFor with this host's directories).
func ModelsDir(models, dataHome string) string {
	if models != "" {
		return models
	}
	home, _ := os.UserHomeDir()
	local := ""
	if runtime.GOOS == "windows" {
		local, _ = os.UserCacheDir() // %LOCALAPPDATA%
	}
	return ModelsDirFor(runtime.GOOS, dataHome, home, local)
}

// ModelsDirFor is the default models folder on goos: XDG's data home on Linux
// and the other Unixes, Application Support on macOS, the local AppData on
// Windows (where the installer creates it), each with jitllm/models under it.
//
// The path follows goos's rules, not this host's: slash-separated for every
// system but Windows, backslash-separated there.
func ModelsDirFor(goos, dataHome, home, localAppData string) string {
	switch goos {
	case "darwin":
		return path.Join(home, "Library", "Application Support", "jitllm", "models")
	case "windows":
		if localAppData == "" {
			localAppData = winJoin(home, "AppData", "Local")
		}
		return winJoin(localAppData, "jitllm", "models")
	}
	if !path.IsAbs(dataHome) {
		// The XDG spec: an empty or relative XDG_DATA_HOME is invalid and
		// ignored.
		dataHome = path.Join(home, ".local", "share")
	}
	return path.Join(dataHome, "jitllm", "models")
}

// winJoin joins Windows path elements with backslashes, on any host.
func winJoin(elem ...string) string {
	out := strings.TrimRight(elem[0], `\/`)
	for _, e := range elem[1:] {
		out += `\` + strings.Trim(e, `\/`)
	}
	return out
}

// LegacyDir is where releases before the per-system folder looked: ./models
// in the working directory. It is searched, never written, so nothing moves
// silently.
const LegacyDir = "models"

// SearchDirs is primary, then the legacy folder when it exists and is not the
// same directory.
func SearchDirs(primary string) []string {
	out := []string{primary}
	if fi, err := os.Stat(LegacyDir); err == nil && fi.IsDir() {
		a, _ := filepath.Abs(LegacyDir)
		b, _ := filepath.Abs(primary)
		if a != b {
			out = append(out, LegacyDir)
		}
	}
	return out
}

// APIVersion is the control plane's compatibility version, reported by
// jitllmd (ServerInfo.api_version, and server_api in /v1/capabilities). It
// changes only when a field changes meaning or goes away; adding one does not.
// A client refuses a server whose number differs from the one it was built
// with.
const APIVersion = 1

// Info is what a probe learned about a running service.
type Info struct {
	Addr       string
	Version    string
	APIVersion int
	// Legacy is a service found on LegacyAddr when DefaultAddr was asked.
	Legacy bool
}

// NewTransport is the transport a client of the local service uses: the
// default one, except that the name "localhost" is dialled at its loopback
// addresses only (127.0.0.1 and ::1, whichever the resolver lists first and
// answers), whatever else the resolver says it is. Any other host is dialled
// as usual.
func NewTransport() http.RoundTripper {
	t := http.DefaultTransport.(*http.Transport).Clone()
	d := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}
	t.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil || !strings.EqualFold(host, "localhost") {
			return d.DialContext(ctx, network, addr)
		}
		var ips []string
		if found, err := net.DefaultResolver.LookupIPAddr(ctx, host); err == nil {
			for _, ip := range found {
				if ip.IP.IsLoopback() {
					ips = append(ips, ip.IP.String())
				}
			}
		}
		if len(ips) == 0 {
			ips = []string{"127.0.0.1", "::1"}
		}
		var last error
		for _, ip := range ips {
			c, err := d.DialContext(ctx, network, net.JoinHostPort(ip, port))
			if err == nil {
				return c, nil
			}
			last = err
		}
		return nil, last
	}
	return t
}

// NewClient is an http.Client over a [NewTransport], for the one-off requests.
// Close its idle connections when done.
func NewClient() *http.Client { return &http.Client{Transport: NewTransport()} }

type sentinel string

func (e sentinel) Error() string { return string(e) }

// ErrUnauthorized is a jitllmd that answers and wants an API key this client
// has none of (or the wrong one) for.
const ErrUnauthorized sentinel = "the jitllmd needs an API key"

// Option configures how a client reaches a service.
type Option func(*options)

type options struct{ key, primary, legacy string }

func optionsOf(opts []Option) options {
	o := options{primary: DefaultAddr, legacy: LegacyAddr}
	for _, f := range opts {
		f(&o)
	}
	return o
}

// WithAPIKey makes a client send key as `Authorization: Bearer key`.
func WithAPIKey(key string) Option { return func(o *options) { o.key = key } }

// WithDefaults changes the default address and the old one a client falls
// back to from DefaultAddr and LegacyAddr, for a front end that serves on
// others (and for tests, which must not depend on what runs on this box).
func WithDefaults(primary, legacy string) Option {
	return func(o *options) { o.primary, o.legacy = primary, legacy }
}

// APIKey is the key the options carry.
func APIKey(opts ...Option) string { return optionsOf(opts).key }

// Authorize sets the request's bearer key, when there is one.
func Authorize(r *http.Request, key string) {
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
}

// KeyTransport is rt (the default transport when nil) sending key on every
// request; an empty key returns rt unchanged.
func KeyTransport(key string, rt http.RoundTripper) http.RoundTripper {
	if rt == nil {
		rt = NewTransport()
	}
	if key == "" {
		return rt
	}
	return keyTransport{key, rt}
}

type keyTransport struct {
	key string
	rt  http.RoundTripper
}

func (t keyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	Authorize(r, t.key)
	return t.rt.RoundTrip(r)
}

// ErrIncompatible is a jitllmd that answers but speaks another API version. A
// client says so and stops rather than starting a second engine beside it.
const ErrIncompatible sentinel = "incompatible jitllmd"

// Compatible is nil when a server reporting api may be used by this build,
// and otherwise says what to do about it. Zero is a server from before the
// number existed, which is read as 1.
func Compatible(api int) error {
	if api == 0 {
		api = 1
	}
	if api != APIVersion {
		return fmt.Errorf("%w: the running jitllmd speaks API version %d and this program %d; "+
			"upgrade both to the same release", ErrIncompatible, api, APIVersion)
	}
	return nil
}

// BaseURL turns an address (host:port, :port, or a URL) into the URL a client
// dials.
func BaseURL(addr string) string {
	if strings.HasPrefix(addr, "http://") || strings.HasPrefix(addr, "https://") {
		return strings.TrimRight(addr, "/")
	}
	if strings.HasPrefix(addr, ":") {
		addr = "127.0.0.1" + addr
	}
	return "http://" + addr
}

// Probe asks addr whether a jitllmd answers there, through its capability
// manifest (GET /v1/capabilities, which carries the build's version). A
// refused connection, a timeout, or anything that is not that manifest is an
// error, and the caller runs embedded.
func Probe(ctx context.Context, addr string, opts ...Option) (Info, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, BaseURL(addr)+"/v1/capabilities", nil)
	if err != nil {
		return Info{}, err
	}
	Authorize(req, APIKey(opts...))
	c := NewClient()
	defer c.CloseIdleConnections()
	resp, err := c.Do(req)
	if err != nil {
		return Info{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return Info{Addr: addr}, fmt.Errorf("%w at %s: give it the key (-api-key, or JITLLM_API_KEY)", ErrUnauthorized, addr)
	}
	if resp.StatusCode != http.StatusOK {
		return Info{}, fmt.Errorf("%s: /v1/capabilities answered %s, not a jitllmd", addr, resp.Status)
	}
	var m struct {
		Schema    int `json:"schema"`
		ServerAPI int `json:"server_api"`
		Build     *struct {
			Version string `json:"version"`
		} `json:"build"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil || m.Schema == 0 {
		return Info{}, fmt.Errorf("%s: /v1/capabilities is not a jitllm manifest", addr)
	}
	in := Info{Addr: addr, APIVersion: m.ServerAPI}
	if m.Build != nil {
		in.Version = m.Build.Version
	}
	return in, Compatible(m.ServerAPI)
}

// Locate is Probe that looks further when addr is DefaultAddr: a service
// installed before the default moved still listens on LegacyAddr, and a
// client that finds nothing on the new port asks the old one. What answers
// there must be a jitllmd (the capability manifest, as Probe checks), not
// just an open port. The Info says where it answered and whether that was
// LegacyAddr.
func Locate(ctx context.Context, addr string, opts ...Option) (Info, error) {
	if addr == "" {
		addr = optionsOf(opts).primary
	}
	in, err := Probe(ctx, addr, opts...)
	legacy, ok := LegacyFor(addr, opts...)
	if err == nil || !ok || errors.Is(err, ErrIncompatible) || errors.Is(err, ErrUnauthorized) {
		return in, err
	}
	old, lerr := Probe(ctx, legacy, opts...)
	if lerr == nil || errors.Is(lerr, ErrIncompatible) || errors.Is(lerr, ErrUnauthorized) {
		old.Legacy = true
		return old, lerr
	}
	return in, err
}
