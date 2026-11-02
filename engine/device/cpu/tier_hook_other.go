//go:build !amd64 && jitllmtest

package cpu

// ForceTierForTest mirrors the amd64 hook so a test in another package
// compiles on every architecture without a build tag of its own (the lesson
// prevnni_hook_other.go records). There is no second tier here to force, so it
// changes nothing and reports the host's own tier; a caller that must SEE the
// forced tier asks HostTier afterwards and skips.
func ForceTierForTest(Tier) Tier { return HostTier() }
