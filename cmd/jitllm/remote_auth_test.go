package main

import (
	"bytes"
	"context"
	"flag"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/localserver"
)

// keyedService is fakeService behind a bearer key.
func keyedService(t *testing.T, key string, got *map[string]any) *httptest.Server {
	t.Helper()
	inner := fakeService(t, localserver.APIVersion, got)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+key {
			http.Error(w, `{"error":{"message":"missing or wrong API key"}}`, http.StatusUnauthorized)
			return
		}
		inner.Config.Handler.ServeHTTP(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

// TestRunSendsTheAPIKey: a service that wants a key is used when the run has
// it, on the probe and on the request, and refused by name when it has not.
func TestRunSendsTheAPIKey(t *testing.T) {
	var got map[string]any
	s := keyedService(t, "kk", &got)

	_, _, err := runTarget(context.Background(), "qwen3", s.URL, "", false, nil, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("no key: %v", err)
	}
	svc, _, err := runTarget(context.Background(), "qwen3", s.URL, "kk", false, nil, &bytes.Buffer{})
	if err != nil || svc == nil {
		t.Fatalf("with the key: %v %v", svc, err)
	}
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	var out bytes.Buffer
	r := newRemoteRun(s.URL, "kk", "qwen3", 5, "", fs, samplingFlags{temp: 0.5})
	if err := r.once(false, "hello", &out); err != nil || out.String() != "from qwen3\n" {
		t.Fatalf("request with the key: %q %v", out.String(), err)
	}
	r = newRemoteRun(s.URL, "", "qwen3", 5, "", fs, samplingFlags{temp: 0.5})
	if err := r.once(false, "hello", &out); err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("request without the key: %v", err)
	}
	if keyOf("flag") != "flag" {
		t.Error("the flag does not win")
	}
	t.Setenv("JITLLM_API_KEY", "env")
	if keyOf("") != "env" {
		t.Error("JITLLM_API_KEY is not read")
	}
}

// TestRunFindsAServiceOnTheOldPort: nothing on the default address, a jitllmd
// on the old one: the run goes through it and says where it runs.
func TestRunFindsAServiceOnTheOldPort(t *testing.T) {
	var got map[string]any
	old := fakeService(t, localserver.APIVersion, &got)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	primary := l.Addr().String()
	l.Close()
	addrOpts := []localserver.Option{localserver.WithDefaults(primary, strings.TrimPrefix(old.URL, "http://"))}

	var note bytes.Buffer
	svc, _, err := runTarget(context.Background(), "qwen3", primary, "", false, nil, &note, addrOpts...)
	if err != nil || svc == nil || svc.Addr != strings.TrimPrefix(old.URL, "http://") {
		t.Fatalf("runTarget: %+v %v", svc, err)
	}
	if strings.Count(note.String(), localserver.LegacyNotice) != 1 {
		t.Errorf("stderr: %q", note.String())
	}
}
