package app

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

// A change is written to the settings file without waiting for the window to
// close, to the system's config folder, and the chats beside it.
func TestSettingsAreSavedAsTheyChange(t *testing.T) {
	isolateConfig(t)
	cfg := LoadConfig(Env{})
	if cfg.Path() == "" {
		t.Fatal("no settings path on this system")
	}
	sh := NewShell(nil, cfg)
	sh.Store.MaxSeq.Set(1234)
	sh.Store.AppendTurn(Turn{Role: RoleUser, Text: "kept across a crash"})

	var saved Config
	deadline := time.Now().Add(5 * time.Second)
	for {
		sh.DrainQueue(0)
		if b, err := os.ReadFile(cfg.Path()); err == nil && json.Unmarshal(b, &saved) == nil && saved.MaxSeq == 1234 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the change never reached %s", cfg.Path())
		}
		time.Sleep(20 * time.Millisecond)
	}
	r := NewStore()
	r.LoadChats(cfg.ChatsPath())
	if r.Turn(0).Text != "kept across a crash" {
		t.Fatalf("the chat was not saved beside the settings: %q", r.Turn(0).Text)
	}
}

// A config that was not read from disk is never written: a test would
// otherwise overwrite the person's own settings.
func TestAConfigWithNoFileIsNeverSaved(t *testing.T) {
	isolateConfig(t)
	if err := DefaultConfig(Env{}).Save(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(ConfigPath()); !os.IsNotExist(err) {
		t.Fatalf("a default config wrote %s", ConfigPath())
	}
}

// isolateConfig points os.UserConfigDir at a temporary folder on every OS:
// XDG_CONFIG_HOME on Linux, HOME on macOS (~/Library/Application Support),
// AppData on Windows. Setting only the first let the macOS tests write the
// developer's own settings file.
func isolateConfig(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	t.Setenv("AppData", dir)
}
