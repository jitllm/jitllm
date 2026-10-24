//go:build jitllmcomposed

package tier

// composedDefault is Config.Composed's default in a jitllmcomposed build:
// every tier opened serves the blocks and the head through the composer, so
// the device gates run it without each naming it.
const composedDefault = ComposedBlock | ComposedHead
