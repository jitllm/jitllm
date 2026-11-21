package server

import (
	"net/http"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

func chatMessage(role, content string) model.ChatMessage {
	return model.ChatMessage{Role: role, Content: content}
}

// Handler mounts every surface on one mux:
//
//	/jitllm.v1.*/...        the ConnectRPC control plane (Connect, gRPC, gRPC-Web)
//	/v1/chat/completions    OpenAI-compatible
//	/v1/completions         OpenAI-compatible
//	/v1/models              OpenAI-compatible
//	/v1/embeddings          OpenAI-compatible
//	/v1/messages            Anthropic-compatible
//	/healthz, /health, /metrics
//	/v1/capabilities        the capability manifest
//
// connect-go mounts each service at /<package>.<Service>/, so the /v1 routes
// cannot collide with it.
func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()

	mount := func(path string, h http.Handler) { mux.Handle(path, h) }
	mount(jitllmv1connect.NewModelServiceHandler(&ModelService{E: e}))
	mount(jitllmv1connect.NewDeviceServiceHandler(&DeviceService{E: e}))
	mount(jitllmv1connect.NewPlacementServiceHandler(&PlacementService{E: e}))
	mount(jitllmv1connect.NewSessionServiceHandler(&SessionService{E: e}))
	mount(jitllmv1connect.NewInferenceServiceHandler(&InferenceService{B: e, Images: e.cfg.Images}))
	mount(jitllmv1connect.NewTelemetryServiceHandler(&TelemetryService{E: e}))

	mux.Handle("/", CompatHandlerWith(e, e.cfg.Images))

	// /metrics, /healthz, /health and /v1/capabilities: engineRoutes
	// (capabilities.go), the table the manifest reads.
	for _, r := range engineRoutes() {
		h := r.handle
		mux.HandleFunc(r.path, func(w http.ResponseWriter, q *http.Request) { h(e, w, q) })
	}

	// h2c, because plain gRPC needs HTTP/2 and would otherwise fail over
	// cleartext while Connect works. Behind TLS, ALPN takes over and this
	// wrapper is inert.
	return h2c.NewHandler(mux, &http2.Server{})
}
