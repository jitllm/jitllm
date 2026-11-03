package screen

// fakeEngine is an Engine whose methods do what the test sets and nothing
// otherwise. Package mock is the full stand-in; this package's own tests
// cannot import it (mock imports screen).
type fakeEngine struct {
	load   func(string)
	unload func(string)
	run    func(func()) bool
	send   func(ChatRequest)
	stop   func()
}

func (f *fakeEngine) Load(p string) {
	if f.load != nil {
		f.load(p)
	}
}
func (f *fakeEngine) Use(string) {}
func (f *fakeEngine) Unload(p string) {
	if f.unload != nil {
		f.unload(p)
	}
}
func (f *fakeEngine) SetPriority(bool) {}
func (f *fakeEngine) Relocate(int)     {}
func (f *fakeEngine) Run(job func()) bool {
	if f.run != nil {
		return f.run(job)
	}
	go job()
	return true
}
func (f *fakeEngine) Send(r ChatRequest) {
	if f.send != nil {
		f.send(r)
	}
}
func (f *fakeEngine) Stop() {
	if f.stop != nil {
		f.stop()
	}
}
