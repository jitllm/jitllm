package cpudev

import "github.com/jitllm/jitllm/nn/history"

// The history's storage is no device's: these name it where cpudev did.
type (
	View     = history.View
	Table    = history.Table
	Pages    = history.Pages
	PagePool = history.PagePool
	Arena    = history.Arena
	Arenas   = history.Arenas
)
