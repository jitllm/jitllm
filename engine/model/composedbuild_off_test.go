//go:build !jitllmcomposed

package model

// composedByDefault says every tier these tests open serves its blocks and
// head through the composer (the tier's composedDefault).
const composedByDefault = false
