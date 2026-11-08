package game

import (
	"encoding/json"
	"testing"
)

// playing is a fresh game past its READY pause.
func playing(pacmanAI, ghostsAI bool) *Game {
	g := New(pacmanAI, ghostsAI)
	g.Step(ReadySeconds+Substep, GreedyControls{})
	if g.Status != Playing {
		panic("game did not start")
	}
	return g
}

// noGhosts sends every ghost home for good, so a test sees Pac-Man alone.
func noGhosts(g *Game) {
	for id := Blinky; id < NActors; id++ {
		g.Actors[id].State = House
	}
	g.LifeTime = -1e9
}

func TestMazeShape(t *testing.T) {
	m := NewMaze(Layout)
	if m.W != 28 || m.H != 31 {
		t.Fatalf("maze is %dx%d, want 28x31", m.W, m.H)
	}
	if m.Left != 244 {
		t.Fatalf("%d pellets, want the arcade's 244 (240 and 4 power)", m.Left)
	}
	if !m.Walkable(Tile{-1, 14}) || m.Next(Tile{0, 14}, Left) != (Tile{27, 14}) {
		t.Fatal("the tunnel does not wrap")
	}
	if m.Walkable(Tile{13, 12}) {
		t.Fatal("the ghost-house door is walkable")
	}
	d := m.DistTo(PacmanStart)
	if d[m.Key(Tile{1, 1})] <= 0 {
		t.Fatal("the top-left corner is unreachable from Pac-Man's start")
	}
}

func TestPacmanMovesAtHisSpeed(t *testing.T) {
	g := playing(false, false)
	noGhosts(g)
	x0, _ := g.Actors[Pacman].Position()
	for i := 0; i < 60; i++ {
		g.Tick(Substep, GreedyControls{})
	}
	x1, _ := g.Actors[Pacman].Position()
	if got := x0 - x1; got < SpeedPacman-0.6 || got > SpeedPacman+0.6 {
		t.Fatalf("Pac-Man moved %.2f tiles left in a second, want about %.1f", got, SpeedPacman)
	}
}

func TestKeyboardSteersAndWallsStop(t *testing.T) {
	g := playing(false, false)
	noGhosts(g)
	g.Key = Up // a wall above the start: he keeps going left
	for i := 0; i < 3; i++ {
		g.Tick(Substep, GreedyControls{})
	}
	if g.Actors[Pacman].Dir != Left {
		t.Fatalf("Pac-Man turned %v into a wall", g.Actors[Pacman].Dir)
	}
	// (12,22) is open: he turns up on reaching (12,23).
	for g.Actors[Pacman].Tile.X > 12 {
		g.Tick(Substep, GreedyControls{})
	}
	if g.Actors[Pacman].Dir != Up {
		t.Fatalf("at an opening Pac-Man heads %v, want the up he asked for", g.Actors[Pacman].Dir)
	}
	g.Tick(Substep, GreedyControls{})
	g.Key = Down
	g.Tick(Substep, GreedyControls{})
	if g.Actors[Pacman].Dir != Down {
		t.Fatalf("reversing mid-corridor: heading %v, want down", g.Actors[Pacman].Dir)
	}
}

func TestEatingScores(t *testing.T) {
	g := playing(false, false)
	noGhosts(g)
	m := g.Maze
	k := m.Key(Tile{12, 23})
	if !m.Pellet[k] {
		t.Fatal("no pellet beside the start")
	}
	for g.Actors[Pacman].Tile != (Tile{12, 23}) {
		g.Tick(Substep, GreedyControls{})
	}
	if g.Score != 10 || m.Pellet[k] || m.Left != 243 {
		t.Fatalf("after one pellet: score %d, pellet still there %v, %d left", g.Score, m.Pellet[k], m.Left)
	}
	g.eat(Tile{1, 3})
	if g.Score != 60 || g.FrightLeft != FrightSeconds(1) {
		t.Fatalf("power pellet: score %d, fright %.1f", g.Score, g.FrightLeft)
	}
}

func TestPowerPelletFrightensAndGhostsScoreInARow(t *testing.T) {
	g := playing(false, false)
	g.eat(Tile{1, 3})
	if g.Actors[Blinky].State != Frightened {
		t.Fatalf("Blinky is %v after a power pellet", g.Actors[Blinky].State)
	}
	if g.Actors[Pinky].State != House {
		t.Fatal("a ghost still in the house was frightened")
	}
	score := g.Score
	for i, want := range []int{200, 400} {
		id := Blinky + i
		a := &g.Actors[id]
		a.State, a.Tile, a.Progress, a.Dir = Frightened, g.Actors[Pacman].Tile, 0, Left
		g.collisions()
		if a.State != Eaten || g.Score-score != want {
			t.Fatalf("ghost %d eaten: state %v, scored %d, want %d", i, a.State, g.Score-score, want)
		}
		score = g.Score
	}
}

func TestCaughtCostsALifeAndTheGameEnds(t *testing.T) {
	g := playing(false, false)
	for life := 3; life >= 1; life-- {
		b := &g.Actors[Blinky]
		b.State, b.Tile, b.Progress = Normal, g.Actors[Pacman].Tile, 0
		g.collisions()
		if g.Status != Dying || g.CaughtBy != Blinky {
			t.Fatalf("touching Blinky: status %v caught by %d", g.Status, g.CaughtBy)
		}
		g.Step(DyingSeconds+Substep, GreedyControls{})
		if g.Lives != life-1 {
			t.Fatalf("lives %d, want %d", g.Lives, life-1)
		}
		if life > 1 {
			g.Step(ReadySeconds+Substep, GreedyControls{})
		}
	}
	if g.Status != GameOver {
		t.Fatalf("status %v after the last life", g.Status)
	}
}

func TestLevelClears(t *testing.T) {
	g := playing(false, false)
	m := g.Maze
	for k := range m.Pellet {
		m.Pellet[k], m.Power[k] = false, false
	}
	m.Left = 1
	m.Pellet[m.Key(Tile{12, 23})] = true
	noGhosts(g)
	for g.Status == Playing {
		g.Tick(Substep, GreedyControls{})
	}
	if g.Status != LevelClear {
		t.Fatalf("status %v after the last pellet", g.Status)
	}
	g.Step(LevelClearSeconds+Substep, GreedyControls{})
	if g.Level != 2 || m.Left != 244 {
		t.Fatalf("level %d with %d pellets after clearing", g.Level, m.Left)
	}
}

func TestModeSchedule(t *testing.T) {
	for _, c := range []struct {
		s    float64
		want Mode
	}{{0, Scatter}, {6.9, Scatter}, {7.1, Chase}, {27.1, Scatter}, {1000, Chase}} {
		if got := ModeAt(c.s); got != c.want {
			t.Errorf("ModeAt(%v) = %v, want %v", c.s, got, c.want)
		}
	}
}

// recorder answers with the greedy rule and keeps every question.
type recorder struct{ ps []Point }

func (r *recorder) Decide(g *Game, p *Point) Dir {
	r.ps = append(r.ps, *p)
	return g.Greedy(p, g.Options(p))
}

func TestDrivenActorsAreAskedAndTheRequestIsJevmans(t *testing.T) {
	g := playing(true, true)
	r := &recorder{}
	for i := 0; i < 600 && g.Status == Playing; i++ {
		g.Tick(Substep, r)
	}
	asked := map[int]bool{}
	for _, p := range r.ps {
		asked[p.Actor] = true
		if p.N < 1 {
			t.Fatalf("a question with no options: %+v", p)
		}
		if p.Actor != Pacman && !p.Escape {
			for _, d := range p.Options[:p.N] {
				if d == p.Heading.Reverse() {
					t.Fatalf("ghost %d was offered to reverse", p.Actor)
				}
			}
		}
	}
	if !asked[Pacman] || !asked[Blinky] {
		t.Fatalf("asked %v, want Pac-Man and Blinky at least", asked)
	}
	ps := g.Questions(Pacman, nil)
	ps = g.Questions(Blinky, ps)
	if len(ps) == 0 {
		t.Skip("no open question at this instant")
	}
	body := g.Request(nil, ps)
	var req struct {
		State struct {
			Maze   []string                  `json:"maze"`
			Legend string                    `json:"legend"`
			Pacman map[string]any            `json:"pacman"`
			Ghosts map[string]map[string]any `json:"ghosts"`
			Score  int                       `json:"score"`
		} `json:"state"`
		Questions map[string]struct {
			Type         string            `json:"type"`
			Instructions string            `json:"instructions"`
			Criteria     map[string]string `json:"criteria"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("the request is not JSON: %v\n%s", err, body)
	}
	if len(req.State.Maze) != 31 || len(req.State.Maze[0]) != 28 || len(req.State.Ghosts) != 4 || req.State.Legend == "" {
		t.Fatalf("state: %d rows, %d ghosts", len(req.State.Maze), len(req.State.Ghosts))
	}
	for id, q := range req.Questions {
		if q.Type != "choice" || q.Instructions == "" || len(q.Criteria) == 0 {
			t.Fatalf("question %s: %+v", id, q)
		}
		for k := range q.Criteria {
			if DirOf(k) == NoDir {
				t.Fatalf("question %s offers %q", id, k)
			}
		}
	}
}

func TestGreedyPlaysAGame(t *testing.T) {
	g := playing(true, false)
	for i := 0; i < 60*120 && g.Status != GameOver; i++ {
		g.Step(Substep, GreedyControls{})
	}
	if g.Score < 500 {
		t.Fatalf("the greedy rule scored %d in two minutes", g.Score)
	}
}

func TestAWarmStepDoesNotAllocate(t *testing.T) {
	g := playing(true, false)
	for i := 0; i < 600; i++ {
		g.Step(Substep, GreedyControls{})
	}
	var buf []byte
	ps := make([]Point, 0, 8)
	allocs := testing.AllocsPerRun(200, func() {
		if g.Status == GameOver {
			return
		}
		g.Step(Substep, GreedyControls{})
		ps = g.Questions(Pacman, ps[:0])
		buf = g.Request(buf[:0], ps)
	})
	if allocs > 0 {
		t.Fatalf("a warm step and request made %.1f allocations", allocs)
	}
}
