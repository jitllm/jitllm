package gpu

import "github.com/jitllm/jitllm/backend"

// closeKern closes a kernel the device compiled, nil being none.
func closeKern(k backend.Kernel) {
	if k != nil {
		k.Close()
	}
}
