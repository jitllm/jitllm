// Package engine runs a model. An instance composes the model's blocks for
// the devices it places them on, the CPU among them; a session holds one
// conversation's history; the scheduler leases a session an instance for a
// phase. The model (package model) is weights and graphs; the engine holds a
// pointer to it.
package engine
