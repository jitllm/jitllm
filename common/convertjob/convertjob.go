// Package convertjob turns a GGUF or safetensors model into the .jlm container
// the engine runs, for a front end: the request, its refusals, the progress of
// a running conversion and the decision of what a stale container needs. It
// exists because model.Open refuses a GGUF: the engine reads one weight format
// and GGUF is the converter's input.
//
// Progress is estimated by polling the size of the file being written, since
// convert.FromGGUFs and jlm.Write have no progress hook, and there is no
// cancel because the conversion cannot be stopped.
package convertjob

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

// Request is one conversion: a source model, optionally its vision tower, and
// the container to write. The tower is an input; one container carries both
// text and vision blocks.
type Request struct {
	Src    string
	MMProj string
	Dst    string
	// Dir is the model directory a derived Dst goes to; "" puts it beside the
	// source.
	Dir string
}

// DestFor is the container a source converts to when the caller names none:
// in the model directory dir, whatever disk the source is on. It is
// convert.DestFor, which `jitllm convert` derives with too, so a file
// converted here and a file converted there land in the same place.
func DestFor(src, dir string) string { return convert.DestFor(src, dir) }

// PrimaryDir is the model directory conversions write to: the first
// configured one, which defaults to the model disk (config.DefaultModelDir).
func PrimaryDir(dirs []string) string {
	if len(dirs) == 0 {
		return ""
	}
	return dirs[0]
}

// Plan resolves a request and refuses the ones that cannot work, with the
// reason a person can act on. It is separate from [Run] so the refusals
// (missing source, destination equal to source, an mmproj handed to a
// HuggingFace directory, which carries its own tower) are testable without
// converting anything.
func Plan(req Request) (Request, error) {
	out := Request{
		Src:    strings.TrimSpace(req.Src),
		MMProj: strings.TrimSpace(req.MMProj),
		Dst:    strings.TrimSpace(req.Dst),
		Dir:    strings.TrimSpace(req.Dir),
	}
	if out.Src == "" {
		return out, errors.New("choose a .gguf or a safetensors model to convert")
	}
	fi, err := os.Stat(out.Src)
	if err != nil {
		return out, fmt.Errorf("source: %w", err)
	}

	st := convert.IsSafetensors(out.Src)
	if st && out.MMProj != "" {
		return out, fmt.Errorf(
			"%s is a safetensors model and %s is an mmproj: the two-file contract "+
				"is GGUF's, and a HuggingFace tower lives in the model directory",
			out.Src, out.MMProj)
	}
	if !st && fi.IsDir() {
		return out, fmt.Errorf("%s is a directory and holds no safetensors weight file", out.Src)
	}
	if out.MMProj != "" {
		if _, err := os.Stat(out.MMProj); err != nil {
			return out, fmt.Errorf("vision tower: %w", err)
		}
	}

	if out.Dst == "" {
		out.Dst = DestFor(out.Src, out.Dir)
	}
	if out.Dst == out.Src {
		return out, fmt.Errorf("refusing to write %s over itself", out.Src)
	}
	if out.MMProj != "" && out.Dst == out.MMProj {
		return out, fmt.Errorf("refusing to write %s over the vision tower", out.Dst)
	}
	return out, nil
}

// SourceBytes is how many bytes of model the input holds: the file's size for
// a GGUF or a single shard, and the sum of the weight files for a HuggingFace
// directory (a directory entry's own size would make the progress meaningless).
func SourceBytes(src string) (int64, error) {
	fi, err := os.Stat(src)
	if err != nil {
		return 0, err
	}
	if !fi.IsDir() {
		return fi.Size(), nil
	}
	shards, err := filepath.Glob(filepath.Join(src, "*.safetensors"))
	if err != nil {
		return 0, err
	}
	var n int64
	for _, p := range shards {
		if s, err := os.Stat(p); err == nil {
			n += s.Size()
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("%s holds no weight file", src)
	}
	return n, nil
}

// Fraction is the progress a partly-written destination represents. It
// clamps at 0.99 because a container can be larger than its source, so the
// ratio can overshoot; the label beside the bar carries the real bytes.
func Fraction(dstBytes, srcBytes int64) float64 {
	if srcBytes <= 0 || dstBytes <= 0 {
		return 0
	}
	f := float64(dstBytes) / float64(srcBytes)
	if f > 0.99 {
		return 0.99
	}
	return f
}

// Summary is the line a finished conversion leaves behind: what went in, what
// came out, how much bigger it got, and which build wrote it. The writer
// matters because a container from a stale converter has a current format
// version and is otherwise indistinguishable.
func Summary(h *jlm.Header, dst string, srcBytes, dstBytes int64, el time.Duration) string {
	grew := ""
	if srcBytes > 0 {
		grew = fmt.Sprintf(" (%+.0f%%)", 100*(float64(dstBytes)/float64(srcBytes)-1))
	}
	rate := ""
	if s := el.Seconds(); s > 0 && srcBytes > 0 {
		rate = fmt.Sprintf(", %.2f GB/s", float64(srcBytes)/s/1e9)
	}
	line := fmt.Sprintf("%s: %s -> %s%s in %s%s",
		filepath.Base(dst), session.Bytes(uint64(srcBytes)), session.Bytes(uint64(dstBytes)),
		grew, session.Dur(el), rate)
	if h != nil {
		line += fmt.Sprintf("; %d block(s) of %s, %d tensors",
			h.NBlocks, session.Bytes(h.PageSize), h.NTensors)
		if h.NVisBlocks > 0 {
			line += fmt.Sprintf("; %d vision block(s) of %s",
				h.NVisBlocks, session.Bytes(h.VisPageSize))
		}
	}
	return line + "; written by " + jlm.WriterID()
}

// Stage is where a running conversion is.
type Stage uint8

// The stages a [Progress] hears, in order: Started once, Writing on every
// poll, then one of Done or Failed.
const (
	Started Stage = iota
	Writing
	Done
	Failed
)

// Update is one progress report.
type Update struct {
	Stage Stage
	// Fraction is the bar's estimate, 0 to 0.99 (see [Fraction]); 0 on Done
	// and Failed, when the bar goes away.
	Fraction float64
	// Written and Source are the bytes on disk so far and the input's size.
	Written, Source int64
	// Label is the line beside the bar: what is being converted and how far
	// it is, the summary on Done, and empty on Failed.
	Label string
	// Err is why a Failed conversion failed.
	Err error
}

// Progress hears a running conversion. It is called from the polling
// goroutine and from Run's, never two at once.
type Progress func(Update)

// pollEvery is how often the bar re-reads the file being written.
const pollEvery = 250 * time.Millisecond

// Run converts one model and reports through progress, which may be nil. It
// returns the summary line. It blocks; the caller decides where it runs, and
// that should be the engine worker when there is one, because a conversion
// beside a decode ruins both.
func Run(req Request, progress Progress) (string, error) {
	if progress == nil {
		progress = func(Update) {}
	}
	plan, err := Plan(req)
	if err != nil {
		return "", err
	}
	srcBytes, err := SourceBytes(plan.Src)
	if err != nil {
		return "", err
	}

	stop := make(chan struct{})
	polled := make(chan struct{})
	go poll(progress, plan.Dst, srcBytes, stop, polled)
	stopPoll := func() {
		close(stop)
		<-polled
	}

	host, _ := os.Hostname()
	fp := jlm.Fingerprint{Host: host}

	t0 := time.Now()
	var h *jlm.Header
	if convert.IsSafetensors(plan.Src) {
		h, err = convert.FromSafetensors(plan.Src, plan.Dst, fp)
	} else {
		h, err = convert.FromGGUFs(plan.Src, plan.MMProj, plan.Dst, fp)
	}
	stopPoll()
	if err != nil {
		// Nothing to clean up: jlm.Write renames PartPath(Dst) over Dst only
		// when complete, so on failure Dst is whatever was there before (on a
		// reconvert, the user's container). Removing it would delete a good file.
		progress(Update{Stage: Failed, Source: srcBytes, Err: err})
		return "", err
	}
	el := time.Since(t0)

	var dstBytes int64
	if fi, err := os.Stat(plan.Dst); err == nil {
		dstBytes = fi.Size()
	}
	line := Summary(h, plan.Dst, srcBytes, dstBytes, el)
	progress(Update{Stage: Done, Written: dstBytes, Source: srcBytes, Label: line})
	return line, nil
}

// PartPath is the file jlm.Write grows while it converts, renamed over dst
// when it is done. It mirrors the name in format/jlm/write.go.
func PartPath(dst string) string { return dst + ".part" }

// poll is the progress bar, polling because the writer has no hook. It stats
// PartPath(dst), the file actually being written, four times a second.
func poll(progress Progress, dst string, srcBytes int64, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)

	base := filepath.Base(dst)
	part := PartPath(dst)
	tick := time.NewTicker(pollEvery)
	defer tick.Stop()

	progress(Update{Stage: Started, Source: srcBytes, Label: "converting " + base})

	for {
		select {
		case <-stop:
			return
		case <-tick.C:
			var n int64
			if fi, err := os.Stat(part); err == nil {
				n = written(fi)
			}
			progress(Update{Stage: Writing, Fraction: Fraction(n, srcBytes), Written: n, Source: srcBytes,
				Label: fmt.Sprintf("converting %s: %s written, source %s",
					base, session.Bytes(uint64(n)), session.Bytes(uint64(srcBytes)))})
		}
	}
}

// ReconvertSource is where a container's GGUF is looked for: same name,
// .gguf, same folder, because a container does not record its source.
func ReconvertSource(container string) string {
	return strings.TrimSuffix(container, ".jlm") + ".gguf"
}

// LoadableAt reports whether path holds a container this build can open, from
// its version word -- the same structural read the catalog makes.
func LoadableAt(path string) bool {
	v, err := catalog.ReadVersion(path)
	return err == nil && catalog.Readable(v)
}

// ReplaceNote is what converting into dst would replace, as one sentence for
// the form, or "" when nothing is there.
func ReplaceNote(dst string) string {
	switch v, err := catalog.ReadVersion(dst); {
	case err != nil:
		return ""
	case catalog.Readable(v):
		return filepath.Base(dst) + " already exists and loads. Converting again replaces it."
	default:
		return filepath.Base(dst) + " exists but will not load in this version of the app. Converting replaces it."
	}
}

// Reconvert is what rebuilding a container takes.
type Reconvert struct {
	// Req is the form, filled: the guessed source, no tower, the container as
	// the destination.
	Req Request
	// Note is the sentence that says what is missing, or "" when Start.
	Note string
	// Status is a status-bar line, when there is one to show.
	Status string
	// Start is true when the conversion may start at once, with no look at
	// the form first.
	Start bool
}

// PlanReconvert decides how the container at path is rebuilt from the GGUF
// beside it: in one step when that is safe, and otherwise with the form
// filled and a note saying what is missing. It is one step only when nothing
// loadable is at path; replacing a loadable container gets a look at the form
// first. busy is a conversion already running: two at once would fight over
// one progress bar.
func PlanReconvert(path string, busy bool) Reconvert {
	guess := ReconvertSource(path)
	r := Reconvert{Req: Request{Src: guess, Dst: path}}
	base := filepath.Base(path)

	if _, err := os.Stat(guess); err != nil {
		r.Note = filepath.Base(guess) + " is not beside this model. Pick the GGUF it was made from, then press Convert."
		if tower, _, _ := catalog.HasTower(path); tower {
			r.Note += " Add its mmproj file as the vision tower too."
		}
		r.Status = "pick the source for " + base
		return r
	}
	if busy {
		r.Note = "Another conversion is running. Press Convert when it finishes."
		return r
	}
	if LoadableAt(path) {
		r.Note = "Press Convert to rebuild " + base + " from " + filepath.Base(guess) + "."
		return r
	}
	// Never a one-step downgrade: a file from a newer build is stale too, and
	// rebuilding it would write an older format over it.
	if v, err := catalog.ReadVersion(path); err == nil && v > catalog.CurrentVersion {
		r.Note = base + " was made by a newer version of the app. Update the app to load it, or press Convert to rebuild it for this one."
		return r
	}
	// Nor a vision model rebuilt without its tower: nothing records which
	// mmproj a container came from, so a one-step rebuild would be text-only.
	switch tower, known, _ := catalog.HasTower(path); {
	case !known:
		r.Note = "If " + base + " could see pictures, add its mmproj file as the vision tower. Then press Convert."
		return r
	case tower:
		r.Note = base + " can see pictures: add its mmproj file as the vision tower, then press Convert."
		return r
	}
	r.Start = true
	return r
}

// SourceFacts is a source file's line of facts: its format, quantization,
// size, and whether it is a vision tower.
func SourceFacts(e catalog.Entry) string {
	kind := "GGUF"
	if e.Kind == catalog.KindSafetensors {
		kind = "safetensors"
	}
	parts := []string{kind}
	if e.Quant != "" {
		parts = append(parts, e.Quant)
	}
	parts = append(parts, session.Bytes(uint64(max(e.Size, 0))))
	if e.Tower {
		parts = append(parts, "a vision tower")
	}
	return strings.Join(parts, " · ")
}

// SourceState is a source file's state in words: whether its container exists
// and loads, or where a vision tower goes.
func SourceState(e catalog.Entry) string {
	switch SourceContainer(e) {
	case "--":
		return "Goes with its model"
	case "none":
		return "Not converted"
	case "converted":
		return "Converted"
	case "newer":
		return "Made by a newer app"
	}
	return "Needs reconverting"
}

// SourceContainer says whether the file's container exists and loads: "--"
// for a tower, "none", "converted", "newer" or "outdated".
func SourceContainer(e catalog.Entry) string {
	switch {
	case e.Tower:
		return "--" // a tower goes inside its model's container
	case e.TargetVersion == 0:
		return "none"
	case catalog.Readable(e.TargetVersion):
		return "converted"
	case e.TargetVersion > catalog.CurrentVersion:
		return "newer"
	}
	return "outdated"
}
