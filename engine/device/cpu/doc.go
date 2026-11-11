// Package cpu is the host machine beneath the CPU device: the instruction
// set tier and target a kernel is generated for, the executable memory
// generated code is mapped into, the trampolines that call it (the only
// assembly in the project) and the worker pool. It knows no model.
//
// The device, which composes forms and runs blocks, is cpudev. It is a
// package of its own because it imports composer, while composer's tests
// run the code they generate through this one.
package cpu
