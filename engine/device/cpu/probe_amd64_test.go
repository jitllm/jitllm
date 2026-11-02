//go:build amd64

package cpu

import "sync"

// reprobeHostFeatures re-runs the CPUID probe so a test that has just written
// forceNoVNNI gets an answer that reflects it. There is one probe, so a force
// set before it fires would otherwise be cached for the whole binary;
// resetting on the way in and out keeps the force scoped.
func reprobeHostFeatures() {
	featuresOnce = sync.Once{}
	features = Features{}
	CPU()
}
