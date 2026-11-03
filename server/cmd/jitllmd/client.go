package main

// The client half of jitllmd. The same binary that serves the engine also
// calls one, over the same ConnectRPC control plane, through the same
// generated stubs the server is built from.
//
// It lives here, not in cmd/jitllm, so the root module keeps its single
// dependency: connect and protobuf are already in this module's go.mod. Every
// request and response is a generated message from server/gen/jitllm/v1.

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/localserver/service"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// defaultAddr keeps the common case short: a daemon on this host, on the
// address `jitllmd serve` itself defaults to.
const defaultAddr = localserver.DefaultAddr

// cli is where a verb writes. It is a struct rather than os.Stdout/os.Stderr
// so a gate can drive a verb in process and watch stdout arrive, telling a
// streaming client from one that accumulates and dumps.
type cli struct {
	out io.Writer
	err io.Writer

	// addrOpts lets a gate stand in other default and old-default addresses
	// (localserver.WithDefaults), so it does not depend on what runs on this
	// box.
	addrOpts []localserver.Option
	// thisService replaces the person's service on this system with one a
	// gate roots in a temporary home that records its commands.
	thisService func() (service.Service, error)
	// waitUp is how long install and start wait for the service to answer;
	// zero is fifteen seconds.
	waitUp time.Duration
}

func stdio() cli { return cli{out: os.Stdout, err: os.Stderr} }

// errHandled means the verb has already printed its own explanation -- a flag
// parse error, or -h. main exits without saying anything a second time.
const errHandled verbError = "jitllmd: bad flags"

type verbError string

func (e verbError) Error() string { return string(e) }

// usageErr is a command line that was wrong, as opposed to an operation that
// failed. It exits 2 ("retrying will not help", as Go's flag package does for
// an unknown flag); a failed operation exits 1.
type usageErr struct{ error }

func usagef(format string, a ...any) error { return usageErr{fmt.Errorf(format, a...)} }

// ---------------------------------------------------------------- dialling

// conn holds one generated client per service. They share one *http.Client,
// so they share one connection pool.
type conn struct {
	base string

	Model     jitllmv1connect.ModelServiceClient
	Device    jitllmv1connect.DeviceServiceClient
	Placement jitllmv1connect.PlacementServiceClient
	Session   jitllmv1connect.SessionServiceClient
	Inference jitllmv1connect.InferenceServiceClient
	Telemetry jitllmv1connect.TelemetryServiceClient
}

// dial builds the six clients against one base URL.
//
// The http.Client has no Timeout: it would bound the whole exchange including
// the body, and an over-committed model can stream healthily for minutes. Nor
// is there a ResponseHeaderTimeout: the wait for the first byte is the device
// queue, which queue_timeout_millis already controls. A generate is bounded by
// the context, cancelled on ^C.
func dial(addr, key string) (*conn, error) {
	base, err := baseURL(addr)
	if err != nil {
		return nil, err
	}
	h := &http.Client{Transport: localserver.KeyTransport(key, nil)}
	return &conn{
		base:      base,
		Model:     jitllmv1connect.NewModelServiceClient(h, base),
		Device:    jitllmv1connect.NewDeviceServiceClient(h, base),
		Placement: jitllmv1connect.NewPlacementServiceClient(h, base),
		Session:   jitllmv1connect.NewSessionServiceClient(h, base),
		Inference: jitllmv1connect.NewInferenceServiceClient(h, base),
		Telemetry: jitllmv1connect.NewTelemetryServiceClient(h, base),
	}, nil
}

// target is dial for a verb: it sends the verb's -api-key (else
// $JITLLM_API_KEY), and when the address is the default and no jitllmd
// answers there it uses one on the old default port, saying so on stderr.
func target(ctx context.Context, c cli, fs *flag.FlagSet, addr string) (*conn, error) {
	key := keyOf(fs)
	a := strings.TrimSpace(addr)
	opts := append([]localserver.Option{localserver.WithAPIKey(key)}, c.addrOpts...)
	if _, ok := localserver.LegacyFor(a, opts...); ok {
		if in, _ := localserver.Locate(ctx, a, opts...); in.Legacy {
			fmt.Fprintf(c.err, "jitllmd: %s\n", localserver.LegacyNotice)
			a = in.Addr
		}
	}
	return dial(a, key)
}

// keyOf is the API key a verb sends: its -api-key flag, else $JITLLM_API_KEY.
func keyOf(fs *flag.FlagSet) string {
	if f := fs.Lookup("api-key"); f != nil && f.Value.String() != "" {
		return f.Value.String()
	}
	return os.Getenv("JITLLM_API_KEY")
}

// baseURL turns what a person types into what connect-go wants. `-addr :4242`
// means "this host", as it does for `jitllmd serve`.
func baseURL(addr string) (string, error) {
	a := strings.TrimSpace(addr)
	if a == "" {
		a = defaultAddr
	}
	if !strings.Contains(a, "://") {
		if strings.HasPrefix(a, ":") {
			a = "localhost" + a
		}
		if _, _, err := splitHostPort(a); err != nil {
			return "", err
		}
		a = "http://" + a
	}
	u, err := url.Parse(a)
	if err != nil {
		return "", usagef("-addr %q: %v", addr, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", usagef("-addr %q: scheme %q is not http or https", addr, u.Scheme)
	}
	if u.Host == "" {
		return "", usagef("-addr %q: no host", addr)
	}
	return strings.TrimSuffix(u.Scheme+"://"+u.Host+u.Path, "/"), nil
}

// splitHostPort adds the default port to a bare host and refuses a port that
// is not a number.
func splitHostPort(a string) (string, string, error) {
	i := strings.LastIndexByte(a, ':')
	if i < 0 || strings.Contains(a[i+1:], "]") {
		return a, "", usagef("-addr %q: no port; the daemon defaults to %s", a, defaultAddr)
	}
	if _, err := strconv.Atoi(a[i+1:]); err != nil {
		return "", "", usagef("-addr %q: %q is not a port", a, a[i+1:])
	}
	return a[:i], a[i+1:], nil
}

// ---------------------------------------------------------------- errors

// clientError renders a server error as the server's own sentence, with the
// code beside it. The engine's refusals carry the fix (a GGUF's convert
// command, the reason a placement was declined), so they are never flattened.
func clientError(what string, base string, err error) error {
	var ce *connect.Error
	if !errors.As(err, &ce) {
		return fmt.Errorf("%s: %v", what, err)
	}
	msg := fmt.Sprintf("%s: %s: %s", what, ce.Code(), ce.Message())
	if ce.Code() == connect.CodeUnauthenticated {
		msg += "\n  this jitllmd wants an API key -- pass -api-key KEY, or set JITLLM_API_KEY"
	}
	if ce.Code() == connect.CodeUnavailable {
		// The transport reached nothing, so the client says which address it
		// tried.
		msg += fmt.Sprintf("\n  no jitllmd answered at %s -- start one with `jitllmd serve`, "+
			"or point -addr at the one that is running", base)
	}
	return errors.New(msg)
}

// ---------------------------------------------------------------- flags

func addrFlag(fs *flag.FlagSet) *string {
	fs.String("api-key", "", "the API key the jitllmd wants, if it has one (default: $JITLLM_API_KEY)")
	return fs.String("addr", defaultAddr,
		"the jitllmd to talk to: host:port, :port, or a full http(s) URL")
}

// parse parses a verb's flags and applies the misplacedFlag guard to every
// verb: Go's flag package stops at the first positional, so a flag after an id
// or a prompt would otherwise be silently ignored.
func parse(c cli, fs *flag.FlagSet, args []string) error {
	fs.SetOutput(c.err)
	if err := fs.Parse(args); err != nil {
		// flag has already written the reason and the usage to c.err.
		return errHandled
	}
	if err := misplacedFlag(fs); err != nil {
		return usageErr{err}
	}
	return nil
}

// wasSet reports whether the caller actually typed a flag, so a proto3
// `optional` is sent only when given: unset max_device_blocks means "as many
// as fit" where 0 means "none on the device".
func wasSet(fs *flag.FlagSet, name string) bool {
	found := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}

// csv splits a comma-separated flag, dropping empties. It is the same grammar
// `jitllm run -devices cuda:0,cpu` uses, so one habit serves both binaries.
func csv(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// ---------------------------------------------------------------- printing

// flusher is implemented by a buffered stdout. os.Stdout is unbuffered and
// needs none of this; a test writer, a pipe wrapper or a future pager does.
type flusher interface{ Flush() error }

// emit writes and flushes in one step. Every token goes through it.
func emit(w io.Writer, s string) error {
	if _, err := io.WriteString(w, s); err != nil {
		return err
	}
	if f, ok := w.(flusher); ok {
		return f.Flush()
	}
	return nil
}

// bytesOf reads a ByteSize's derived human form, so the client never formats
// a byte count itself.
func bytesOf(b *v1.ByteSize) string {
	if b == nil {
		return "0 B"
	}
	if h := b.GetHuman(); h != "" {
		return h
	}
	return fmt.Sprintf("%d B", b.GetBytes())
}

func finishName(r v1.FinishReason) string {
	switch r {
	case v1.FinishReason_FINISH_REASON_STOP:
		return "stop"
	case v1.FinishReason_FINISH_REASON_MAX_TOKENS:
		return "max-tokens"
	case v1.FinishReason_FINISH_REASON_EOS:
		return "eos"
	case v1.FinishReason_FINISH_REASON_CANCELLED:
		return "cancelled"
	case v1.FinishReason_FINISH_REASON_ERROR:
		return "error"
	}
	return "unspecified"
}

func kindName(k v1.BlockKind) string {
	switch k {
	case v1.BlockKind_BLOCK_KIND_ATTENTION:
		return "attention"
	case v1.BlockKind_BLOCK_KIND_LINEAR:
		return "linear"
	case v1.BlockKind_BLOCK_KIND_VISION:
		return "vision"
	}
	return "block"
}

func deviceKindName(k v1.DeviceKind) string {
	switch k {
	case v1.DeviceKind_DEVICE_KIND_HOST:
		return "host"
	case v1.DeviceKind_DEVICE_KIND_DISCRETE:
		return "discrete"
	case v1.DeviceKind_DEVICE_KIND_INTEGRATED:
		return "integrated"
	}
	return "unknown"
}

func execName(e v1.ExecutionMode) string {
	switch e {
	case v1.ExecutionMode_EXECUTION_MODE_PARALLEL:
		return "parallel"
	case v1.ExecutionMode_EXECUTION_MODE_SERIALISED:
		return "serialised"
	}
	return "unspecified"
}

// commas groups an integer for reading: 816,010,912 rather than 816010912.
func commas(n uint64) string {
	s := strconv.FormatUint(n, 10)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
	}
	for i := pre; i < len(s); i += 3 {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(s[i : i+3])
	}
	return b.String()
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}
