package tier

import (
	"encoding/hex"

	"github.com/jitllm/jitllm/backend/cuda"
)

// cudaUUIDForTest is the identity of CUDA device ord WITHOUT opening it, in the
// same "uuid:<hex>" form backend.Identity uses.
//
// It lets a gate ask "does CUDA hold this card" without paying for a driver
// context, a locked thread and an owner goroutine.
func cudaUUIDForTest(ord int) (string, bool) {
	u, ok := cuda.DeviceUUID(ord)
	if !ok {
		return "", false
	}
	return "uuid:" + hex.EncodeToString(u[:]), true
}
