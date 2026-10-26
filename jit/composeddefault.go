//go:build !jitllmcomposed

package jit

// ComposedByDefault is whether this build serves models through the composer
// with no option set: not outside a jitllmcomposed build.
const ComposedByDefault = false
