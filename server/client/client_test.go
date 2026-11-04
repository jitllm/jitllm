package client_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/internal/capabilities"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// TestDiscoverReadsServerInfo: a real engine's ServerInfo carries the API
// version, the platform, the models folder and a manifest that decodes, and
// the root module's probe agrees with it over /v1/capabilities.
func TestDiscoverReadsServerInfo(t *testing.T) {
	dir := t.TempDir()
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: dir, Version: "v9.9.9"})
	defer e.Close()
	s := httptest.NewServer(e.Handler())
	defer s.Close()

	c, err := client.Discover(context.Background(), s.URL)
	if err != nil {
		t.Fatal(err)
	}
	i := c.Info
	if i.GetApiVersion() != localserver.APIVersion || i.GetVersion() != "v9.9.9" ||
		i.GetModelDirectory() != dir || i.GetOs() == "" || i.GetArch() == "" || i.GetRevision() == "" {
		t.Fatalf("ServerInfo is missing fields: %v", i)
	}
	if len(i.GetDevices()) == 0 {
		t.Errorf("ServerInfo lists no devices; the host is always one")
	}
	if !c.Has("server_info") {
		t.Errorf("features %v lack server_info", i.GetFeatures())
	}
	var m capabilities.Manifest
	err = c.Capabilities(&m)
	if err != nil || m.Schema == 0 || len(m.Architectures) == 0 || m.ServerAPI != localserver.APIVersion {
		t.Fatalf("capabilities: %v, schema %d, %d architectures, api %d", err, m.Schema, len(m.Architectures), m.ServerAPI)
	}

	p, err := localserver.Probe(context.Background(), s.URL)
	if err != nil || p.Version != "v9.9.9" || p.APIVersion != localserver.APIVersion {
		t.Fatalf("Probe: %+v, %v", p, err)
	}
}

type oldServer struct {
	jitllmv1connect.UnimplementedTelemetryServiceHandler
}

func (oldServer) GetServerInfo(context.Context, *connect.Request[v1.GetServerInfoRequest]) (*connect.Response[v1.GetServerInfoResponse], error) {
	return connect.NewResponse(&v1.GetServerInfoResponse{Info: &v1.ServerInfo{ApiVersion: localserver.APIVersion + 1}}), nil
}

// TestDiscoverRefusesAnotherAPIVersion: a server speaking another API version
// is reported as incompatible, not as absent, so a front end says so instead
// of starting a second engine.
func TestDiscoverRefusesAnotherAPIVersion(t *testing.T) {
	mux := http.NewServeMux()
	mux.Handle(jitllmv1connect.NewTelemetryServiceHandler(oldServer{}))
	s := httptest.NewServer(mux)
	defer s.Close()
	_, err := client.Discover(context.Background(), s.URL)
	if !errors.Is(err, localserver.ErrIncompatible) {
		t.Fatalf("Discover against API %d: %v, want ErrIncompatible", localserver.APIVersion+1, err)
	}
	if _, err := client.Discover(context.Background(), "127.0.0.1:1"); err == nil || errors.Is(err, localserver.ErrIncompatible) {
		t.Fatalf("nothing listening: %v, want a plain error", err)
	}
}
