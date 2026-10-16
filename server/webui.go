package server

import (
	"embed"
	"io/fs"
	"net/http"
)

// webuiFS is the built web UI (server/webui: `npm ci && npm run build`).
// It is served from memory, and only when the caller asks (jitllmd -webui).
//
//go:embed all:webui/dist
var webuiFS embed.FS

// WithWebUI serves the embedded web UI under /ui/ and sends every other path
// to next. The UI speaks to the same server over its ConnectRPC services, so
// it needs no route of its own beyond the static files.
func WithWebUI(next http.Handler) http.Handler {
	dist, err := fs.Sub(webuiFS, "webui/dist")
	if err != nil {
		panic(err) // the embed pattern names this directory
	}
	files := http.StripPrefix("/ui/", http.FileServerFS(dist))
	mux := http.NewServeMux()
	mux.Handle("/ui/", files)
	mux.HandleFunc("/ui", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/ui/", http.StatusTemporaryRedirect)
	})
	mux.Handle("/", next)
	return mux
}
