package app

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/api"
	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
)

// The window's badge follows the API it actually serves, not the setting.
func TestTheBadgeFollowsTheAPI(t *testing.T) {
	sh := NewShell(nil, DefaultConfig(Env{}))
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	badge := sh.Badge()
	if got := badge.Get(); got != "Embedded · API off" {
		t.Fatalf("badge %q", got)
	}
	sh.Store.APIAddr.Set("127.0.0.1:0")
	stop := sh.ServeAPI(e)
	defer stop()
	sh.Store.API.Set(true)
	if got := badge.Get(); !strings.HasPrefix(got, "Embedded · API on :") {
		t.Fatalf("API on, badge %q", got)
	}
	sh.Store.API.Set(false)
	if got := badge.Get(); got != "Embedded · API off" {
		t.Fatalf("API off, badge %q", got)
	}
	sh.Store.Engine.Set(core.Where{Service: true, Addr: "localhost:4242", Version: "1.0"})
	if got := badge.Get(); got != "Service · localhost:4242 (jitllmd 1.0)" {
		t.Fatalf("connected, badge %q", got)
	}
}

// Turning the API on where a jitllmd answers is refused: nothing listens, the
// setting goes back off, and the window offers to connect to the service.
func TestTheAPIToggleIsRefusedOnAHeldAddress(t *testing.T) {
	svc := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir(), Version: "7.7.7"})
	defer svc.Close()
	held, err := api.Start("127.0.0.1:0", svc)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()

	sh := NewShell(nil, DefaultConfig(Env{}))
	var connected string
	sh.ConnectTo = func(addr string) { connected = addr }
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	sh.Store.APIAddr.Set(held.Addr())
	stop := sh.ServeAPI(e)
	defer stop()
	sh.Store.API.Set(true)
	sh.DrainQueue(0)

	if sh.Store.API.Get() {
		t.Error("the API setting stayed on")
	}
	if got := sh.Store.APIServing.Get(); got != "" {
		t.Errorf("an API serves on %s beside the service", got)
	}
	d := sh.Store.Dialog.Get()
	if d.Kind != DialogConfirm || d.OK != "Connect" || !strings.Contains(d.Title, "7.7.7") {
		t.Fatalf("no offer to connect: %+v", d)
	}
	d.OnOK()
	if connected != held.Addr() {
		t.Errorf("Connect went to %q, want %q", connected, held.Addr())
	}
}
