package main

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/charmbracelet/huh"
	"github.com/jitllm/jitllm/common/api"
	"github.com/jitllm/jitllm/common/config"
)

// keysView is what the key groups put on screen for a draft.
func keysView(d *draft) string {
	// Groups are hidden or shown as the form moves to them, so start on one
	// before them, as the settings form does.
	f := huh.NewForm(huh.NewGroup(huh.NewNote().Title("before")), keyGroup(d, true), keyGroup(d, false)).
		WithWidth(100).WithHeight(30)
	f.Init()
	f.NextGroup()
	return stripANSI(f.View())
}

// TestAPIKeysAreMaskedUntilShown: the key fields never print a key until the
// person asks, and then print it.
func TestAPIKeysAreMaskedUntilShown(t *testing.T) {
	d := &draft{serviceKey: "svc-secret-1", apiKey: "api-secret-2"}
	v := keysView(d)
	if !strings.Contains(v, "Key for the service") || !strings.Contains(v, "Key for this app's API") {
		t.Fatalf("the key fields are not on screen:\n%s", v)
	}
	if strings.Contains(v, "secret") {
		t.Errorf("a masked key is on screen:\n%s", v)
	}
	d.showKeys = true
	v = keysView(d)
	if !strings.Contains(v, "svc-secret-1") || !strings.Contains(v, "api-secret-2") {
		t.Errorf("shown keys are not on screen:\n%s", v)
	}
}

// TestSettingsSaveTheKeysPrivately: the keys go to the settings file, which is
// private, and a blank field is no key. The default address is not written.
func TestSettingsSaveTheKeysPrivately(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
	r := newTestRoot(t)
	dirs := r.env.cfg.ModelDirs
	r.env.cfg = config.LoadConfig()
	r.env.cfg.ModelDirs = dirs
	p := newSettingsPage(r.env)
	if p.d.apiAddr != "" {
		t.Fatalf("the draft starts with the default spelled out: %q", p.d.apiAddr)
	}
	p.d.serviceKey, p.d.apiKey = " svc-key ", "api-key"
	p.apply()
	cfg := r.env.cfg
	if cfg.ServiceKey != "svc-key" || cfg.APIKey != "api-key" || cfg.APIAddr != "" {
		t.Fatalf("applied: service %q api %q addr %q", cfg.ServiceKey, cfg.APIKey, cfg.APIAddr)
	}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(cfg.Path())
	if err != nil || !strings.Contains(string(b), "svc-key") || strings.Contains(string(b), api.DefaultAddr) {
		t.Fatalf("settings file: %v\n%s", err, b)
	}
	if fi, _ := os.Stat(cfg.Path()); runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600 {
		t.Errorf("settings file mode %v", fi.Mode())
	}
	// Blank is no key.
	p = newSettingsPage(r.env)
	p.d.serviceKey, p.d.apiKey = "  ", ""
	p.apply()
	if cfg.ServiceKey != "" || cfg.APIKey != "" {
		t.Errorf("blank fields left keys %q %q", cfg.ServiceKey, cfg.APIKey)
	}
	// An address that was typed is kept.
	p = newSettingsPage(r.env)
	p.d.apiAddr = "127.0.0.1:9000"
	p.apply()
	if cfg.APIAddr != "127.0.0.1:9000" {
		t.Errorf("a typed address was dropped: %q", cfg.APIAddr)
	}
}

// The block placement typed in Settings reaches the config and the engine's
// state, and clearing it leaves the engine to place.
func TestSettingsCarryTheBlockPlacement(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
	r := newTestRoot(t)
	p := newSettingsPage(r.env)
	p.d.spec, p.d.placement = "cuda:0,vulkan:0", " 0-15=cuda:0,16-31=vulkan:0 "
	p.apply()
	const want = "0-15=cuda:0,16-31=vulkan:0"
	if got := r.env.cfg.Placement; got != want {
		t.Fatalf("config placement %q, want %q", got, want)
	}
	if got := r.env.f.st.Placement.Get(); got != want {
		t.Fatalf("engine state placement %q, want %q", got, want)
	}
	p.d.placement = ""
	p.apply()
	if r.env.cfg.Placement != "" || r.env.f.st.Placement.Get() != "" {
		t.Fatal("clearing the field left a placement behind")
	}
}
