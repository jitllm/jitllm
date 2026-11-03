// Package install is what the macOS app does to the system it is installed
// on, at the person's request: link the command-line programs it carries
// (jitllm.app/Contents/Resources/bin) into PATH, and start jitllmd at login
// through a launchd user agent. It has no GUI; the settings screen and the
// first-launch offer call it.
//
// The Windows installer (packaging/windows) does the same two things itself,
// so on Windows and Linux Bundled reports no bundle and nothing here runs.
package install

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Programs() are the command-line programs the bundle carries, by file name.
func Programs() []string {
	return []string{"jitllm", "jitllmd"}
}

// Bundled is the directory holding the bundle's command-line programs when
// this process is a macOS app's executable (Contents/MacOS/<exe>) and the
// bundle carries them, and false otherwise: a bare binary, a go run, or a
// bundle built without -cli.
func Bundled() (string, bool) {
	if runtime.GOOS != "darwin" {
		return "", false
	}
	exe, err := os.Executable()
	if err != nil {
		return "", false
	}
	if exe, err = filepath.EvalSymlinks(exe); err != nil {
		return "", false
	}
	return bundledFrom(exe)
}

// bundledFrom is Bundled for the executable at exe, on any system.
func bundledFrom(exe string) (string, bool) {
	macos := filepath.Dir(exe)
	if filepath.Base(macos) != "MacOS" {
		return "", false
	}
	bin := filepath.Join(filepath.Dir(macos), "Resources", "bin")
	for _, p := range Programs() {
		if fi, err := os.Stat(filepath.Join(bin, p)); err != nil || fi.IsDir() {
			return "", false
		}
	}
	return bin, true
}

// LinkDirs are the directories a link is looked for in, in the order Link
// prefers them: /usr/local/bin is on every macOS shell's PATH, Homebrew's
// prefix on Apple silicon is where a cask's links go, and ~/.local/bin needs
// no administrator.
func LinkDirs(home string) []string {
	return []string{"/usr/local/bin", "/opt/homebrew/bin", filepath.Join(home, ".local", "bin")}
}

// Linked is the directory where every program is already a link into bin, or
// "" when none holds all of them. A copy installed some other way (the
// install script, a formula) is not a link into this bundle and does not
// count: running it would not run this app's version.
func Linked(bin string, dirs []string) string {
	for _, d := range dirs {
		all := true
		for _, p := range Programs() {
			if !linksTo(filepath.Join(d, p), filepath.Join(bin, p)) {
				all = false
				break
			}
		}
		if all {
			return d
		}
	}
	return ""
}

// linksTo reports whether link is a symbolic link resolving to target.
func linksTo(link, target string) bool {
	fi, err := os.Lstat(link)
	if err != nil || fi.Mode()&os.ModeSymlink == 0 {
		return false
	}
	got, err1 := filepath.EvalSymlinks(link)
	want, err2 := filepath.EvalSymlinks(target)
	return err1 == nil && err2 == nil && got == want
}

// ErrNotALink is returned when a program's name in the target directory is
// taken by something that is not a link: a file this code did not make,
// which it does not replace.
const ErrNotALink installError = "exists and is not a link"

type installError string

func (e installError) Error() string { return string(e) }

// Link makes each program in dir a symbolic link into bin, creating dir. A
// link already there is replaced, since it is a link this app (or an earlier
// copy of it) made; anything else is refused with ErrNotALink, before any
// link is written.
func Link(bin, dir string) error {
	for _, p := range Programs() {
		fi, err := os.Lstat(filepath.Join(dir, p))
		if err == nil && fi.Mode()&os.ModeSymlink == 0 {
			return fmt.Errorf("%s: %w", filepath.Join(dir, p), ErrNotALink)
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, p := range Programs() {
		dst := filepath.Join(dir, p)
		if err := os.Remove(dst); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err := os.Symlink(filepath.Join(bin, p), dst); err != nil {
			return err
		}
	}
	return nil
}

// LinkAsAdmin is Link into dir through macOS's administrator prompt, for a
// directory the person cannot write (/usr/local/bin is root's on a fresh
// Mac). It refuses a name that is not a link as Link does, before prompting.
func LinkAsAdmin(bin, dir string) error {
	for _, p := range Programs() {
		fi, err := os.Lstat(filepath.Join(dir, p))
		if err == nil && fi.Mode()&os.ModeSymlink == 0 {
			return fmt.Errorf("%s: %w", filepath.Join(dir, p), ErrNotALink)
		}
	}
	script := "mkdir -p " + shQuote(dir)
	for _, p := range Programs() {
		script += " && ln -sfn " + shQuote(filepath.Join(bin, p)) + " " + shQuote(filepath.Join(dir, p))
	}
	out, err := exec.Command("osascript", "-e",
		"do shell script "+asQuote(script)+" with administrator privileges").CombinedOutput()
	if err != nil {
		return fmt.Errorf("osascript: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// Unlink removes each program's link in dir that points into bin, and
// nothing else.
func Unlink(bin, dir string) error {
	for _, p := range Programs() {
		dst := filepath.Join(dir, p)
		if linksTo(dst, filepath.Join(bin, p)) {
			if err := os.Remove(dst); err != nil {
				return err
			}
		}
	}
	return nil
}

// shQuote quotes s for /bin/sh.
func shQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'" }

// asQuote quotes s as an AppleScript string literal.
func asQuote(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s) + `"`
}
