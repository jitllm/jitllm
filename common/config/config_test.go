package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestKeysFallBackToTheEnvironment(t *testing.T) {
	defer UseEnv(Env{})
	UseEnv(Env{APIKey: "from-env"})
	c := &Config{}
	if c.ServingKey() != "from-env" || c.ServiceAPIKey() != "from-env" {
		t.Errorf("env: %q %q", c.ServingKey(), c.ServiceAPIKey())
	}
	c = &Config{APIKey: "a", ServiceKey: "s"}
	if c.ServingKey() != "a" || c.ServiceAPIKey() != "s" {
		t.Errorf("file: %q %q", c.ServingKey(), c.ServiceAPIKey())
	}
	var none *Config
	if none.ServingKey() != "from-env" {
		t.Errorf("a nil config: %q", none.ServingKey())
	}
	UseEnv(Env{})
	if (&Config{}).ServingKey() != "" {
		t.Error("a key out of nothing")
	}
}

// TestASavedLegacyAPIAddressFollowsTheDefault: a settings file written before
// the default moved holds the old default, which the app wrote and the person
// never chose; any other address is theirs.
func TestASavedLegacyAPIAddressFollowsTheDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
	p := ConfigPath()
	if p == "" {
		t.Fatal("this system has no config folder, so LoadConfig reads nothing and the gate proves nothing")
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	for saved, want := range map[string]string{
		"127.0.0.1:8080": "",
		"localhost:4242": "",
		"127.0.0.1:9000": "127.0.0.1:9000",
	} {
		if err := os.WriteFile(p, []byte(`{"api_addr":"`+saved+`"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := LoadConfig().APIAddr; got != want {
			t.Errorf("saved %q loads as %q, want %q", saved, got, want)
		}
	}
}

// TestTheDefaultAddressIsNotSaved: a first run holds "use the default" as an
// empty address, the file records only what was typed, and it is private.
func TestTheDefaultAddressIsNotSaved(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
	c := LoadConfig()
	if c.APIAddr != "" || c.ServiceAddr != "" {
		t.Fatalf("a first run holds %q and %q, want both empty", c.APIAddr, c.ServiceAddr)
	}
	c.APIKey, c.ServiceKey = "k1", "k2"
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(c.Path())
	if err != nil {
		t.Fatal(err)
	}
	// The address fields themselves: a substring search over the file also
	// read the temporary directory's random name in the model folder, which
	// held "4242" once on macOS.
	var saved struct {
		APIAddr     string `json:"api_addr"`
		ServiceAddr string `json:"service_addr"`
	}
	if err := json.Unmarshal(b, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.APIAddr != "" || saved.ServiceAddr != "" {
		t.Errorf("an address the person never typed was written: %q, %q", saved.APIAddr, saved.ServiceAddr)
	}
	fi, err := os.Stat(c.Path())
	if err != nil || (runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600) {
		t.Errorf("settings file mode %v, %v", fi.Mode(), err)
	}
	// An older, world-readable file is tightened by the next save.
	if err := os.Chmod(c.Path(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	if fi, _ = os.Stat(c.Path()); runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600 {
		t.Errorf("an old file stayed %v", fi.Mode())
	}
	c = LoadConfig()
	if c.APIKey != "k1" || c.ServiceKey != "k2" {
		t.Errorf("keys did not round-trip: %q %q", c.APIKey, c.ServiceKey)
	}
	c.APIAddr = "0.0.0.0:9000"
	c.Save()
	if LoadConfig().APIAddr != "0.0.0.0:9000" {
		t.Error("an address the person typed was lost")
	}
}
