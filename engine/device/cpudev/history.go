package cpudev

import "github.com/jitllm/jitllm/nn"

// A session's history contract is nn's; these name it where cpudev did.
type (
	HistoryBinding = nn.HistoryBinding
	HistTable      = nn.HistTable
	History        = nn.History
	HistoryOf      = nn.HistoryOf
	Prompt         = nn.Prompt
)
