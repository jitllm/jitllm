package app

import (
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestKeysAreSavedPrivatelyAndTheDefaultAddressIsNot: the two key fields reach
// the settings file (mode 0600) and the config at once, blank is none, and
// nothing writes the default address into the file.
func TestKeysAreSavedPrivatelyAndTheDefaultAddressIsNot(t *testing.T) {
	isolateConfig(t)
	cfg := LoadConfig(Env{})
	if cfg.APIAddr != "" || cfg.ServiceAddr != "" {
		t.Fatalf("a first run holds %q and %q", cfg.APIAddr, cfg.ServiceAddr)
	}
	sh := NewShell(nil, cfg)
	sh.Store.ServiceKey.Set(" svc-key ")
	sh.Store.APIKey.Set("api-key")
	if cfg.ServiceKey != "svc-key" || cfg.APIKey != "api-key" {
		t.Fatalf("the config lags the fields: %q %q", cfg.ServiceKey, cfg.APIKey)
	}

	var saved Config
	deadline := time.Now().Add(5 * time.Second)
	for {
		sh.DrainQueue(0)
		b, err := os.ReadFile(cfg.Path())
		if err == nil && json.Unmarshal(b, &saved) == nil && saved.ServiceKey == "svc-key" {
			if saved.APIKey != "api-key" || saved.APIAddr != "" || strings.Contains(string(b), "4242") {
				t.Fatalf("settings file:\n%s", b)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the keys never reached %s", cfg.Path())
		}
		time.Sleep(20 * time.Millisecond)
	}
	if fi, err := os.Stat(cfg.Path()); err != nil || (runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600) {
		t.Errorf("settings file mode %v, %v", fi.Mode(), err)
	}

	// Blank is no key, and the file says so.
	sh.Store.ServiceKey.Set("")
	sh.Store.APIKey.Set("  ")
	if cfg.ServiceKey != "" || cfg.APIKey != "" {
		t.Errorf("blank fields left keys %q %q", cfg.ServiceKey, cfg.APIKey)
	}
}
