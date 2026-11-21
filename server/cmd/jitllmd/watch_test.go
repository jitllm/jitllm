package main

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/server"
)

// TestWatchStatsPrintsEachSnapshotAsItArrives is the second streaming gate.
// WatchStats has no last snapshot, so a buffering client would print nothing
// ever. ^C is the normal way out and must exit clean.
//
// VIOLATION SIGNATURE. Collect the snapshots into a slice and print them after
// the Receive loop and this fails with
//
//	no snapshot reached stdout in 5s: a watch that prints at the end prints
//	nothing, because the stream only ends when the caller stops it
//
// Drop the `&& ctx.Err() == nil` guard on st.Err() and it fails with
//
//	the caller's own ^C was reported as a failure: watching stats: canceled
func TestWatchStatsPrintsEachSnapshotAsItArrives(t *testing.T) {
	addr, _ := engineServer(t, server.Config{Version: "watch-1"})

	w := newNotifyWriter(16)
	var e syncBuffer
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- cmdStats(ctx, cli{out: w, err: &e},
			[]string{"-addr", addr, "-watch", "-interval", "250"})
	}()

	// Two snapshots must arrive while the verb runs. Wait for snapshots, not
	// any write: the server-info header is printed before the stream opens.
	for seen := 0; seen < 2; {
		select {
		case s := <-w.ch:
			if strings.Contains(s, "----") {
				seen++
			}
		case <-time.After(5 * time.Second):
			cancel()
			<-done
			t.Fatalf("only %d snapshot(s) reached stdout in 5s: a watch that prints at the end "+
				"prints nothing, because the stream only ends when the caller stops it", seen)
		}
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the caller's own ^C was reported as a failure: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the watch did not stop when its context was cancelled")
	}

	if n := strings.Count(w.String(), "----"); n < 2 {
		t.Fatalf("only %d snapshot(s) were printed:\n%s", n, w.String())
	}
	if !strings.Contains(w.String(), "jitllmd watch-1") {
		t.Fatalf("the watch did not print the server info header:\n%s", w.String())
	}
}

// TestEveryMutationRefusesAnUnknownIdWithTheServersWords gates the refusal
// half of the mutating verbs (the success half needs a loaded model): each
// reaches the right RPC with its id and relays the server's answer.
func TestEveryMutationRefusesAnUnknownIdWithTheServersWords(t *testing.T) {
	addr, _ := engineServer(t, server.Config{})

	cases := []struct {
		name string
		fn   func(context.Context, cli, []string) error
		args []string
		want string
	}{
		{"models -unload", cmdModels, []string{"-addr", addr, "-unload", "ghost"}, `model "ghost"`},
		{"models <id>", cmdModels, []string{"-addr", addr, "ghost"}, `model "ghost"`},
		{"sessions -new", cmdSessions, []string{"-addr", addr, "-new", "-model", "ghost"}, `"ghost"`},
		{"sessions -close", cmdSessions, []string{"-addr", addr, "-close", "ghost"}, `session "ghost"`},
		{"sessions -reset", cmdSessions, []string{"-addr", addr, "-reset", "ghost"}, `session "ghost"`},
		{"sessions <id>", cmdSessions, []string{"-addr", addr, "ghost"}, `session "ghost"`},
		{"place -session", cmdPlace, []string{"-addr", addr, "-session", "ghost"}, `session "ghost"`},
		{"place -model", cmdPlace, []string{"-addr", addr, "-model", "ghost"}, `"ghost"`},
		{"place -move", cmdPlace, []string{"-addr", addr, "-session", "ghost", "-move", "0:3", "-to", "cpu"}, `"ghost"`},
		{"place -tune-seam", cmdPlace, []string{"-addr", addr, "-session", "ghost", "-tune-seam"}, `"ghost"`},
		{"devices <id>", cmdDevices, []string{"-addr", addr, "ghost:9"}, "ghost:9"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var o, e syncBuffer
			err := runVerb(t, c.fn, cli{out: &o, err: &e}, c.args, 20*time.Second)
			if err == nil {
				t.Fatalf("an unknown id was accepted; stdout:\n%s", o.String())
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("the refusal does not name %s: %v", c.want, err)
			}
			if !strings.Contains(err.Error(), "not_found") {
				t.Fatalf("an unknown id came back as something other than not_found, so a "+
					"caller cannot tell a typo from a server fault: %v", err)
			}
			t.Logf("%s", err)
		})
	}
}

// TestPlaceRefusesToGuessWhichHalfOfTheServiceYouMeant. Placement is per
// session and the pager is per model; a verb that silently picked one would act
// on something the caller did not name.
func TestPlaceRefusesToGuessWhichHalfOfTheServiceYouMeant(t *testing.T) {
	for _, args := range [][]string{
		{"-addr", "127.0.0.1:1"},
		{"-addr", "127.0.0.1:1", "-session", "s", "-model", "m"},
	} {
		var o, e syncBuffer
		err := runVerb(t, cmdPlace, cli{out: &o, err: &e}, args, 10*time.Second)
		if err == nil {
			t.Fatalf("%v was accepted", args)
		}
		var ue usageErr
		if !errors.As(err, &ue) {
			t.Fatalf("%v is not a usage error: %v", args, err)
		}
	}
}

// TestABlockRangeIsParsedOrRefused. `-move 0:11` is the seam's own vocabulary
// and a silently mis-parsed range would move the wrong blocks.
func TestABlockRangeIsParsedOrRefused(t *testing.T) {
	for _, c := range []struct {
		in          string
		first, last int32
	}{
		{"0:11", 0, 11},
		{"7", 7, 7},
		{" 3 : 5 ", 3, 5},
	} {
		f, l, err := blockRange(c.in)
		if err != nil || f != c.first || l != c.last {
			t.Errorf("blockRange(%q) = (%d,%d,%v), want (%d,%d,nil)", c.in, f, l, err, c.first, c.last)
		}
	}
	for _, bad := range []string{"11:0", "a:3", "3:b", ""} {
		if f, l, err := blockRange(bad); err == nil {
			t.Errorf("blockRange(%q) = (%d,%d), want a refusal", bad, f, l)
		}
	}
}
