package app

import (
	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/session"
)

// The settings live in common/config, shared with the terminal; these are
// re-exported so the window keeps one import.
type (
	Env    = config.Env
	Config = config.Config
)

// DefaultConfig is re-exported; see [config.DefaultConfig].
func DefaultConfig(e Env) *Config { return config.DefaultConfig(e) }

// ConfigPath is re-exported; see [config.ConfigPath].
func ConfigPath() string { return config.ConfigPath() }

// LoadConfig is re-exported; see [config.LoadConfig].
func LoadConfig(e Env) *Config { return config.LoadConfig(e) }

// KVCacheDir is re-exported; see [session.KVCacheDir].
func KVCacheDir() string { return session.KVCacheDir() }
