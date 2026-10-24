//go:build !jitllmcomposed

package tier

// composedDefault is Config.Composed's default: nothing, outside a
// jitllmcomposed build.
const composedDefault Parts = 0
