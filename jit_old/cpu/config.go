package cpu

// perfMapOn makes every generated region appear in /tmp/perf-<pid>.map so a
// profiler can name generated code (exec_linux.go). It is process-wide because
// the map is one file named for the process: there is no per-instance form of
// it. cmd/jitllm turns it on once at startup; the library reads no environment.
var perfMapOn bool

// EnablePerfMap turns the profiler's perf map on for the process. Call it
// before any code is generated.
func EnablePerfMap() { perfMapOn = true }
