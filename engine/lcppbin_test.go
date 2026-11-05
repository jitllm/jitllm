package engine_test

import (
	"path/filepath"
	"runtime"
)

// lcppBin is a llama.cpp program in a build directory: name.exe on Windows,
// where a shipped test binary runs these gates too, and name elsewhere.
func lcppBin(dir, name string) string {
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(dir, name)
}
