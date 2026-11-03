package app

// The tab indices, named so a screen can switch to another one without
// counting. They must match the order main registers the screens in, which is
// asserted by TestTabOrderMatchesTheNames.
const (
	TabSession = iota
	TabModels
	TabDownload
	TabConvert
	TabMachine
	TabSettings
)

// TabNames() is the registration order. main ranges over it.
func TabNames() []string {
	return []string{"Chat", "Models", "Discover", "Convert", "Machine", "Settings"}
}

// GoSession switches to the conversation. Safe from any goroutine -- the
// engine calls it after a load so the user lands where the model now is.
func (s *Shell) GoSession() { s.SelectTab(TabSession) }

// GoModels switches to the catalog. Safe from any goroutine.
func (s *Shell) GoModels() { s.SelectTab(TabModels) }

// GoConvert switches to the converter. Safe from any goroutine.
func (s *Shell) GoConvert() { s.SelectTab(TabConvert) }

// GoMachine switches to the hardware page. Safe from any goroutine.
func (s *Shell) GoMachine() { s.SelectTab(TabMachine) }

// GoSettings switches to the settings page. Safe from any goroutine.
func (s *Shell) GoSettings() { s.SelectTab(TabSettings) }
