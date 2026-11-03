package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/localserver/service"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// service is the person's service on this system.
func (c cli) service() (service.Service, error) {
	if c.thisService != nil {
		return c.thisService()
	}
	return service.ForThisUser(runtime.GOOS, os.Getenv("XDG_CONFIG_HOME"), os.Getenv("LOCALAPPDATA"))
}

// wait is how long install and start wait for the service to answer.
func (c cli) wait() time.Duration {
	if c.waitUp > 0 {
		return c.waitUp
	}
	return 15 * time.Second
}

// cmdService installs and drives jitllmd as the person's background service:
// a systemd user unit on Linux, a launchd agent on macOS, the login entry
// the Windows installer writes on Windows.
func cmdService(ctx context.Context, c cli, args []string) error {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return usagef("service: give install, uninstall, status, start or stop")
	}
	what, args := args[0], args[1:]
	fs := flag.NewFlagSet("jitllmd service "+what, flag.ContinueOnError)
	addr := fs.String("addr", defaultAddr, "the address the service listens on; loopback by default")
	models := fs.String("models", defaultModels(), "the models folder the service serves (JITLLM_MODELS sets the default)")
	exe := fs.String("jitllmd", "", "the jitllmd the service runs (default: this one)")
	fs.String("api-key", "", "install: the API key the service requires (default: $JITLLM_API_KEY); "+
		"the other actions: the key to ask the service with")
	if err := parse(c, fs, args); err != nil {
		return err
	}
	s, err := c.service()
	if err != nil {
		return err
	}
	switch what {
	case "install":
		bin := *exe
		if bin == "" {
			if bin, err = os.Executable(); err != nil {
				return err
			}
			if bin, err = filepath.EvalSymlinks(bin); err != nil {
				return err
			}
		}
		dir, err := filepath.Abs(*models)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("the models folder %s: %w", dir, err)
		}
		key := keyOf(fs)
		var extra []string
		if key != "" {
			extra = []string{"-api-key", key}
		}
		if err := s.Install(service.Args(bin, dir, *addr, extra...)); err != nil {
			return err
		}
		fmt.Fprintf(c.out, "installed %s (%s)\n", s.Manager(), s.File())
		fmt.Fprintf(c.out, "models    %s\nlog       %s\n", dir, s.Log())
		return waitFor(ctx, c, *addr, keyOf(fs), false)
	case "uninstall":
		if !s.Installed() {
			fmt.Fprintf(c.out, "not installed (%s)\n", s.File())
			return nil
		}
		if err := s.Uninstall(); err != nil {
			return err
		}
		fmt.Fprintf(c.out, "removed %s\n", s.Manager())
		return nil
	case "start":
		if err := s.Start(); err != nil {
			return err
		}
		// A unit installed before the default moved still listens on the old
		// port, and start brings it back as it was.
		return waitFor(ctx, c, *addr, keyOf(fs), true)
	case "stop":
		if err := s.Stop(); err != nil {
			return err
		}
		fmt.Fprintf(c.out, "stopped %s\n", s.Manager())
		return nil
	case "status":
		return serviceStatus(ctx, c, s, *addr, keyOf(fs))
	}
	return usagef("service: unknown action %q: give install, uninstall, status, start or stop", what)
}

// waitFor waits for a jitllmd to answer on addr and says so.
// With legacy it also accepts a service on the old default port, which is
// where a unit written before the default moved listens.
func waitFor(ctx context.Context, c cli, addr, key string, legacy bool) error {
	end := time.Now().Add(c.wait())
	probe := localserver.Probe
	if legacy {
		probe = localserver.Locate
	}
	for {
		in, err := probe(ctx, addr, localserver.WithAPIKey(key))
		if err == nil {
			fmt.Fprintf(c.out, "running   http://%s (jitllmd %s)\n", hostOf(in.Addr), in.Version)
			if in.Legacy {
				fmt.Fprintf(c.out, "note      %s\n", localserver.LegacyNotice)
			}
			return nil
		}
		if errors.Is(err, localserver.ErrIncompatible) || errors.Is(err, localserver.ErrUnauthorized) || time.Now().After(end) {
			return fmt.Errorf("the service did not answer on %s: %w", addr, err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// serviceStatus prints whether the service is installed, whether a jitllmd
// answers on addr, and what it serves.
func serviceStatus(ctx context.Context, c cli, s service.Service, addr, key string) error {
	inst := "not installed"
	if s.Installed() {
		inst = "installed (" + s.File() + ")"
	}
	fmt.Fprintf(c.out, "service   %s: %s\nlog       %s\n", s.Manager(), inst, s.Log())
	cn, err := client.Discover(ctx, addr, localserver.WithAPIKey(key))
	if errors.Is(err, localserver.ErrUnauthorized) {
		fmt.Fprintf(c.out, "address   %s: answers, and wants an API key (-api-key, or JITLLM_API_KEY)\n", addr)
		return nil
	}
	if err != nil && !errors.Is(err, localserver.ErrIncompatible) {
		fmt.Fprintf(c.out, "address   %s: nothing answers\n", addr)
		return nil
	}
	if cn.Legacy {
		addr = strings.TrimPrefix(cn.Base, "http://")
		fmt.Fprintf(c.out, "note      %s\n", localserver.LegacyNotice)
	}
	i := cn.Info
	fmt.Fprintf(c.out, "address   http://%s\nversion   jitllmd %s (%s, api %d)\n", hostOf(addr), i.GetVersion(), i.GetRevision(), i.GetApiVersion())
	if err != nil {
		return err
	}
	fmt.Fprintf(c.out, "models    %s\n", i.GetModelDirectory())
	r, err := cn.Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{LoadedOnly: true}))
	if err != nil {
		return clientError("listing models", cn.Base, err)
	}
	var loaded []string
	for _, m := range r.Msg.GetLoaded() {
		loaded = append(loaded, m.GetModelId())
	}
	if len(loaded) == 0 {
		loaded = []string{"none"}
	}
	fmt.Fprintf(c.out, "loaded    %s\n", strings.Join(loaded, ", "))
	return nil
}
