//go:build jitllmcomposed

package jit

// ComposedByDefault is whether this build serves models through the composer
// with no option set: every tier serves the blocks and the head through it in
// a jitllmcomposed build.
const ComposedByDefault = true
