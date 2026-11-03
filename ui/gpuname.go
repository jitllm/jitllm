package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/jitllm/jitllm/common/crash"
)

// adapterTap passes every log record on, and names the GPU in the crash
// reports from the one gogpu logs when it picks an adapter ("adapter
// selected", with its name, backend and type). gogpu says which backend it
// chose there and nowhere a caller can ask.
type adapterTap struct {
	slog.Handler
	guard *crash.Guard
}

func (t adapterTap) Handle(ctx context.Context, r slog.Record) error {
	if r.Message == "adapter selected" {
		var name, backend, typ string
		r.Attrs(func(a slog.Attr) bool {
			switch a.Key {
			case "name":
				name = a.Value.String()
			case "backend":
				backend = a.Value.String()
			case "type":
				typ = a.Value.String()
			}
			return true
		})
		t.guard.SetGPU(fmt.Sprintf("%s (%s, %s)", name, backend, typ))
	}
	return t.Handler.Handle(ctx, r)
}

func (t adapterTap) WithAttrs(as []slog.Attr) slog.Handler {
	return adapterTap{t.Handler.WithAttrs(as), t.guard}
}

func (t adapterTap) WithGroup(g string) slog.Handler {
	return adapterTap{t.Handler.WithGroup(g), t.guard}
}

// tapAdapter installs the tap on the default logger. The log goes where it
// went: to os.Stderr, which logToFileWithoutAConsole may have pointed at the
// log file.
func tapAdapter(guard *crash.Guard) {
	slog.SetDefault(slog.New(adapterTap{slog.NewTextHandler(os.Stderr, nil), guard}))
}
