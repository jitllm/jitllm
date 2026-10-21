package main

import (
	"os"
	"path/filepath"
	"runtime"
)

// rocmDir is the ROCm library directory to name to the device tier, or "" for
// its default search: JITLLM_ROCM, else on Windows the HIP SDK's bin directory
// under HIP_PATH, which the SDK's installer sets. jit_old/gpu/hip reads no
// environment, so this is where both are read.
func rocmDir() string {
	if v := os.Getenv("JITLLM_ROCM"); v != "" {
		return v
	}
	if runtime.GOOS == "windows" {
		if v := os.Getenv("HIP_PATH"); v != "" {
			return filepath.Join(v, "bin")
		}
	}
	return ""
}
