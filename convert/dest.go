package convert

import (
	"os"
	"path/filepath"
	"strings"
)

// DestFor is the container a source converts to when the caller names none:
// the source's name with .jlm, placed in dir -- the model directory -- or
// beside the source when dir is "".
//
// It defaults to the model directory rather than the source's directory, so a
// container does not land on the main disk beside another tool's model
// store or a download. The CLI and the app both derive the default here.
func DestFor(src, dir string) string {
	// '/' too: Windows takes both separators, and a directory typed or
	// dropped as C:/models/qwen/ named ".jlm" with only '\' trimmed.
	s := strings.TrimRight(strings.TrimSpace(src), "/"+string(os.PathSeparator))
	if s == "" {
		return ""
	}
	dst := strings.TrimSuffix(strings.TrimSuffix(s, ".gguf"), ".safetensors") + ".jlm"
	if dir == "" {
		return dst
	}
	return filepath.Join(dir, filepath.Base(dst))
}
