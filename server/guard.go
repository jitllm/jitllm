package server

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// Guard is what stands between the network and Handler: an optional API key,
// and the two checks that keep a web page from reaching a loopback server.
// The server package reads no environment, so jitllmd (or a front end that
// serves the API) hands it in.
//
// The zero Guard checks nothing about keys and, with no Addr, nothing about
// the Host header. Origin is always checked: a request without one is not a
// browser's and passes.
type Guard struct {
	// APIKey, when set, is required on every route but /healthz, /health, /v1/health and the web UI's /ui/ files,
	// as `Authorization: Bearer KEY` (and `x-api-key: KEY` on the Anthropic
	// API's /v1/messages routes).
	APIKey string
	// Addr is the address the server listens on, as the listener was asked
	// for it. When it is a loopback address a request must name this host (or
	// localhost, 127.0.0.1, [::1]) in its Host header: a DNS name that an
	// attacker's page pointed at 127.0.0.1 is refused.
	Addr string
	// AllowOrigins are the browser origins (scheme://host[:port]) that may
	// call the server cross-origin, beside the loopback ones. They get CORS
	// headers; no other origin does.
	AllowOrigins []string
}

// Loopback reports whether addr (host:port) is bound to loopback only. A
// missing host (":4242") is every interface, so it is not.
func Loopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	return host != "" && loopbackHost(host)
}

func loopbackHost(host string) bool {
	host = strings.Trim(host, "[]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Wrap returns h behind the guard.
func (g Guard) Wrap(h http.Handler) http.Handler {
	// The keys are compared as MACs under a key made for this process: equal
	// length whatever the key's, so the comparison leaks neither. This is not
	// password storage; nothing is kept but the one configured key.
	macKey := make([]byte, 32)
	if _, err := rand.Read(macKey); err != nil {
		panic(err)
	}
	mac := func(s string) []byte {
		h := hmac.New(sha256.New, macKey)
		h.Write([]byte(s))
		return h.Sum(nil)
	}
	keyHash := mac(g.APIKey)
	loop := Loopback(g.Addr)
	allow := map[string]bool{}
	for _, o := range g.AllowOrigins {
		if o = strings.TrimRight(strings.TrimSpace(o), "/"); o != "" {
			allow[strings.ToLower(o)] = true
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if loop && !g.hostOK(r.Host) {
			guardFail(w, r, http.StatusForbidden, "permission_denied",
				"the Host header "+r.Host+" is not this server's address; "+
					"a server bound to loopback answers only localhost, 127.0.0.1 and [::1]")
			return
		}
		if o := r.Header.Get("Origin"); o != "" {
			listed := allow[strings.ToLower(strings.TrimRight(o, "/"))]
			if !listed && !loopbackOrigin(o) {
				guardFail(w, r, http.StatusForbidden, "permission_denied",
					"cross-origin request from "+o+" refused; start the server with -allow-origin "+o+" to permit it")
				return
			}
			if listed {
				corsHeaders(w, r, o)
				if r.Method == http.MethodOptions {
					w.WriteHeader(http.StatusNoContent)
					return
				}
			}
		}
		if g.APIKey != "" && !openRoute(r.URL.Path) && !keyOK(r, keyHash, mac) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="jitllmd"`)
			guardFail(w, r, http.StatusUnauthorized, "unauthenticated",
				"missing or wrong API key; send Authorization: Bearer <key>")
			return
		}
		h.ServeHTTP(w, r)
	})
}

// hostOK reports whether a Host header names this server on loopback.
func (g Guard) hostOK(hostport string) bool {
	if hostport == "" {
		// HTTP/1.0 without a Host: not a browser.
		return true
	}
	if hostport == g.Addr {
		return true
	}
	host, _, err := net.SplitHostPort(hostport)
	if err != nil {
		host = hostport
	}
	if loopbackHost(host) {
		return true
	}
	ah, _, err := net.SplitHostPort(g.Addr)
	return err == nil && strings.EqualFold(ah, strings.Trim(host, "[]"))
}

// loopbackOrigin is an http(s) origin on localhost, 127.0.0.0/8 or [::1].
func loopbackOrigin(o string) bool {
	u, err := url.Parse(o)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return false
	}
	return loopbackHost(u.Hostname())
}

func corsHeaders(w http.ResponseWriter, r *http.Request, origin string) {
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", origin)
	h.Add("Vary", "Origin")
	if r.Method == http.MethodOptions {
		h.Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		if want := r.Header.Get("Access-Control-Request-Headers"); want != "" {
			h.Set("Access-Control-Allow-Headers", want)
		} else {
			h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key")
		}
		h.Set("Access-Control-Max-Age", "600")
	}
}

// openRoute is a route that needs no key: the liveness probes and the web UI's
// static files. The page is public code that holds no data; it has to load
// before it can ask for the key, and every RPC it makes still carries one.
func openRoute(path string) bool {
	return path == "/healthz" || path == "/health" || path == "/v1/health" || path == "/ui" || strings.HasPrefix(path, "/ui/")
}

func anthropicRoute(path string) bool {
	return path == "/v1/messages" || strings.HasPrefix(path, "/v1/messages/")
}

// keyOK compares the key a request carries with the configured one in
// constant time (over their hashes, so the length does not leak either).
func keyOK(r *http.Request, want []byte, mac func(string) []byte) bool {
	var got string
	if a := r.Header.Get("Authorization"); len(a) > 7 && strings.EqualFold(a[:7], "Bearer ") {
		got = strings.TrimSpace(a[7:])
	} else if anthropicRoute(r.URL.Path) {
		got = r.Header.Get("x-api-key")
	}
	return got != "" && subtle.ConstantTimeCompare(mac(got), want) == 1
}

// guardFail answers in the error shape of the API the path belongs to, so a
// client library parses a refusal as it parses any other. connectCode is the
// Connect error code for the control plane's routes.
func guardFail(w http.ResponseWriter, r *http.Request, status int, connectCode, msg string) {
	p := r.URL.Path
	switch {
	case strings.HasPrefix(p, "/jitllm."):
		writeJSON(w, status, map[string]string{"code": connectCode, "message": msg})
	case anthropicRoute(p):
		typ := "authentication_error"
		if status == http.StatusForbidden {
			typ = "permission_error"
		}
		anFail(w, status, typ, msg)
	default:
		code := "invalid_api_key"
		if status == http.StatusForbidden {
			code = "forbidden"
		}
		writeJSON(w, status, oaError{Error: oaErrorBody{Message: msg, Type: "invalid_request_error", Code: code}})
	}
}
