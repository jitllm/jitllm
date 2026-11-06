package engine

import "github.com/jitllm/jitllm/engine/device/cpudev"

// cpuOf is the CPU device s's lanes run on, for a gate that reads its
// target, forms or kernels.
func cpuOf(s *Instance) *cpudev.CPU { return s.eng.dev.(*cpudev.Execs).CPU() }

// execOf is s's execution on the CPU device, for a gate that binds a form's
// weights itself.
func execOf(s *Instance) *cpudev.Exec { return s.places[routeLanes].(*cpudev.Exec) }
