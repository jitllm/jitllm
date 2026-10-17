// Package config is the settings both front ends read: the window and the
// terminal share one file, so a model folder added in one is there in the
// other.
package config

import (
	"github.com/jitllm/jitllm/common/api"
	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/localserver"

	"encoding/json"
	"os"
	"path/filepath"
)

// Env is what the app takes from the process environment. A library reads no
// environment, so each command (ui/main.go, tui/main.go) reads these and hands them over
// with UseEnv before LoadConfig.
type Env struct {
	// Models is $JITLLM_MODELS, the model folder the engine's tests read.
	Models string
	// DataHome is $XDG_DATA_HOME, Linux's place for application data.
	DataHome string
	// APIKey is $JITLLM_API_KEY: the key the app's own API requires and the
	// key it asks a jitllmd with, when the settings file names none.
	APIKey string
}

var env Env

// UseEnv sets what the app takes from the environment.
func UseEnv(e Env) { env = e }

// DefaultModelDir is where a first run looks for models and puts downloads;
// see [catalog.DefaultDir].
func DefaultModelDir() string { return catalog.DefaultDir(env.Models, env.DataHome) }

// Config is what survives a restart. Window size is persisted and position is
// not: neither library reports or restores a window position.
type Config struct {
	WindowW int  `json:"window_w"`
	WindowH int  `json:"window_h"`
	Light   bool `json:"light"`
	// HideTuning is the sampling panel closed beside the chat.
	HideTuning bool     `json:"hide_tuning"`
	ModelDirs  []string `json:"model_dirs"`

	LastModel  string `json:"last_model"`
	DeviceSpec string `json:"device_spec"`
	// Placement pins blocks to the devices DeviceSpec names, in the CLI's
	// -placement grammar ("0-15=cuda:0,16-31=vulkan:0"); empty leaves every
	// block to the engine.
	Placement string `json:"placement,omitempty"`
	// MaxMem is the host weight budget as the CLI spells it ("24G"), or empty
	// for sched.MemBudget().
	MaxMem string `json:"max_mem"`
	MaxSeq int    `json:"max_seq"`
	Chat   bool   `json:"chat"`
	System string `json:"system"`

	Sampling session.Sampling `json:"sampling"`

	// NoKVCache turns off the prompt cache kept on disk between runs.
	NoKVCache bool `json:"no_kv_cache"`

	// API serves jitllm's API (common/api) while the app runs, on APIAddr.
	//
	// APIAddr is empty for "use the default" (api.DefaultAddr) and holds only
	// an address the person typed, so a later change of the default cannot
	// move a choice they made.
	API     bool   `json:"api"`
	APIAddr string `json:"api_addr"`
	// APIKey, when set, is required on the app's API (as a bearer token).
	// Empty falls back to $JITLLM_API_KEY, else the API needs no key.
	APIKey string `json:"api_key"`
	// ServiceKey is the API key a jitllmd at ServiceAddr wants of this app.
	// Empty falls back to $JITLLM_API_KEY.
	ServiceKey string `json:"service_key"`

	// Engine is which engine the app runs: auto (a jitllmd that answers at
	// ServiceAddr, else its own), embedded or service (engine.Mode*). Empty
	// is auto. ServiceAddr empty is the login service's address.
	Engine      string `json:"engine"`
	ServiceAddr string `json:"service_addr"`

	// CLIOffered is set once the macOS app has offered to put its
	// command-line programs on PATH, so the offer is made on the first
	// launch only; the settings page keeps the choice reachable after.
	CLIOffered bool `json:"cli_offered"`

	path string
}

// ServingKey is the key the app's own API requires, "" for none.
func (c *Config) ServingKey() string {
	if c != nil && c.APIKey != "" {
		return c.APIKey
	}
	return env.APIKey
}

// ServiceAPIKey is the key the app sends a jitllmd, "" for none.
func (c *Config) ServiceAPIKey() string {
	if c != nil && c.ServiceKey != "" {
		return c.ServiceKey
	}
	return env.APIKey
}

// DefaultConfig is the configuration a first run gets.
func DefaultConfig() *Config {
	return &Config{
		WindowW:    1440,
		WindowH:    900,
		ModelDirs:  []string{DefaultModelDir()},
		DeviceSpec: "auto",
		MaxSeq:     4096,
		Sampling:   session.DefaultSampling(),
	}
}

// ConfigPath is where the settings file lives: this system's config folder
// (~/.config on Linux, Application Support on macOS, AppData on Windows).
func ConfigPath() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "jitllm", "ui.json")
}

// chatsPath is where the conversations are kept, beside the settings. A
// config that was not read from disk -- a test's, the shots' -- keeps none.
func (c *Config) ChatsPath() string {
	if c.path == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(c.path), "chats.json")
}

// Path is the settings file this config is saved to, or "" for one that is
// never saved.
func (c *Config) Path() string { return c.path }

// LoadConfig reads the settings file, falling back to [DefaultConfig] for a
// missing or unreadable one. A corrupt file is not an error worth refusing to
// start over.
func LoadConfig() *Config {
	c := DefaultConfig()
	c.path = ConfigPath()
	if c.path == "" {
		return c
	}
	b, err := os.ReadFile(c.path)
	if err != nil {
		return c
	}
	saved := DefaultConfig()
	if err := json.Unmarshal(b, saved); err != nil {
		return c
	}
	saved.path = c.path
	// The API address an earlier release wrote was the default the app
	// filled in, not a choice: it becomes "use the default", so it follows the
	// default from here on.
	if saved.APIAddr == localserver.LegacyAddr || saved.APIAddr == api.DefaultAddr {
		saved.APIAddr = ""
	}
	if len(saved.ModelDirs) == 0 {
		saved.ModelDirs = c.ModelDirs
	}
	if saved.WindowW <= 0 || saved.WindowH <= 0 {
		saved.WindowW, saved.WindowH = c.WindowW, c.WindowH
	}
	return saved
}

// Save writes the settings file, creating its directory. A config that was not
// read from disk -- a test's, the shots' -- is never written, or a test would
// overwrite the person's own settings.
func (c *Config) Save() error {
	if c.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	// 0600: the file may hold API keys. WriteFile keeps the mode of a file
	// that exists, so an older, world-readable one is tightened too.
	if err := os.WriteFile(c.path, b, 0o600); err != nil {
		return err
	}
	return os.Chmod(c.path, 0o600)
}

// AddModelDir adds a directory if it is not already configured.
func (c *Config) AddModelDir(dir string) bool {
	for _, d := range c.ModelDirs {
		if d == dir {
			return false
		}
	}
	c.ModelDirs = append(c.ModelDirs, dir)
	return true
}
