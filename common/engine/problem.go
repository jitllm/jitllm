package engine

import (
	"errors"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/engine/model"
)

// The stages an engine error can come from, which decide how it is explained.
const (
	stageOpen   = "open"   // model.Open
	stageDevice = "device" // opening the device tier for a session
	stagePrompt = "prompt" // rendering and tokenizing the request
	stageReply  = "reply"  // prefill and decode
)

// fix is what a problem's button does.
type fix int

const (
	fixNone      fix = iota
	fixConvert       // fill the convert form with the file
	fixReconvert     // reconvert a container this build cannot read
	fixDevices       // Settings, where the device is chosen
)

// explain turns an engine error into what a person reads: what happened, why,
// and which fix applies.
//
// The cases are told apart structurally, not by matching the message: a GGUF
// is a *model.NotConvertedError, and a stale container is detected from its
// version word with catalog.ReadVersion. The original error stays in the
// detail wherever it adds anything.
func explain(stage, path string, err error) (title, detail string, f fix) {
	name := short(path)
	var nc *model.NotConvertedError
	switch {
	case errors.As(err, &nc):
		return "This file needs converting first",
			name + " is an input format. jitllm converts it once into a .jlm, which is what it runs.",
			fixConvert
	case stage == stageOpen:
		if v, verr := catalog.ReadVersion(path); verr == nil && v > catalog.CurrentVersion {
			return "This model was made by a newer jitllm",
				fmt.Sprintf("%s is container format %d and this build reads %d. Update the app to load it.",
					name, v, catalog.CurrentVersion),
				fixNone
		}
		if v, verr := catalog.ReadVersion(path); verr == nil && v != catalog.CurrentVersion && !catalog.Readable(v) {
			return "This model was converted by an older jitllm",
				fmt.Sprintf("%s is container format %d and this build reads %d. Reconvert it from the file it came from.",
					name, v, catalog.CurrentVersion),
				fixReconvert
		}
		return "Couldn't open " + name, err.Error(), fixNone
	case stage == stageDevice:
		return "The device setting doesn't fit this machine",
			err.Error() + ". Choose another device, or auto, in Settings.",
			fixDevices
	case stage == stagePrompt && strings.Contains(err.Error(), "chat template"):
		// The template's own words say what it refused; what to do about it is
		// the part it cannot say.
		return "This model's chat format refused the conversation",
			err.Error() + ". Start a new chat, or send the message as Completion.",
			fixNone
	case stage == stagePrompt:
		return "Couldn't build the prompt", err.Error(), fixNone
	}
	return "The reply stopped with an error", err.Error(), fixNone
}
