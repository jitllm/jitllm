//go:build !jitllmfault && !jitllmtest

package tier

// The release build: no fault or violation state. hooks is empty, so a Config
// pays nothing for it, and every check is a constant the compiler folds away.

type hooks struct{}

func (*Config) pagedFaulted(pagedFaultID) bool { return false }
func (*Config) recFaulted(recFaultID) bool     { return false }
func (*Config) rowsFaulted(rowsFaultID) bool   { return false }
func (*Config) roomFaulted(roomFaultID) bool   { return false }
func (*Config) moeFaulted(moeFaultID) bool     { return false }
func (*Config) peerFaulted(peerFaultID) bool   { return false }
func (*Config) blockFaulted() bool             { return false }
func (*Config) addNormBroken() bool            { return false }
func (*Config) normGateBroken() bool           { return false }
func (*Config) qkvGemmSwapped() bool           { return false }
func (*Config) gateUpSwapped() bool            { return false }
func (*Config) gateUp2Swapped() bool           { return false }
func (*Config) allocFault()                    {}
func (*Config) applyEnv()                      {}

// WithFault and WithViolation are no-ops in a release build.
func WithFault(family, name string) Option { return func(*openOpts) {} }
func WithViolation(name string) Option     { return func(*openOpts) {} }
func (*GPU) SetFault(family, name string)  {}
func (*GPU) SetViolation(name string, on bool) bool {
	return false
}
