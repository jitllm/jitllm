package game

// Controls answers the game's questions: which way an actor leaves a junction,
// and whether Pac-Man turns back in a corridor. NoDir means "not yet": the
// actor waits at the junction (an escape question just keeps him going).
type Controls interface {
	Decide(g *Game, p *Point) Dir
}

// Key names one question in one situation. Any move of the actor's epoch, its
// mood, or (for a ghost) the phase makes an older answer stale. It is a
// comparable value, so answers are kept in a map without building strings.
type Key struct {
	Actor   int8
	Escape  bool
	X, Y    int16
	Heading Dir
	Mood    uint8
	Mode    Mode
	Threats uint8
	Epoch   int32
}

// Point is one question: where, heading which way, the options in offer order.
type Point struct {
	Actor   int
	Tile    Tile
	Heading Dir
	Options [4]Dir
	N       int
	Key     Key
	// Distance is the steps the actor still walks to Tile (0 when there).
	Distance int
	// Escape is the mid-corridor "keep going or turn back" question; its
	// options are [keep going, turn back].
	Escape bool
	// Threats is a bit per ghost id in the corridor ahead or able to block
	// its end (escape questions).
	Threats uint8
}

// Offers says whether d is one of p's options.
func (p *Point) Offers(d Dir) bool {
	for _, o := range p.Options[:p.N] {
		if o == d {
			return true
		}
	}
	return false
}

// QuestionID is p's id in a decision request.
func (p *Point) QuestionID() string {
	if p.Escape {
		return "pacman_escape"
	}
	return IDs[p.Actor]
}

func (g *Game) mood(id int) uint8 {
	if id == Pacman {
		if g.FrightLeft > 0 {
			return 1
		}
		return 0
	}
	return uint8(g.Actors[id].State)
}

// pointAt is the question at junction t arriving with heading h: ghosts may
// not reverse, Pac-Man may.
func (g *Game) pointAt(id int, t Tile, h Dir) Point {
	p := Point{Actor: id, Tile: t, Heading: h}
	var open [4]Dir
	n := g.Maze.Open(t, &open)
	for _, d := range open[:n] {
		if id != Pacman && d == h.Reverse() {
			continue
		}
		p.Options[p.N] = d
		p.N++
	}
	p.Key = Key{Actor: int8(id), X: int16(t.X), Y: int16(t.Y), Heading: h, Mood: g.mood(id), Epoch: int32(g.Actors[id].Epoch)}
	if id != Pacman {
		p.Key.Mode = g.Mode
	}
	return p
}

// NextPoint is the next junction where a driven actor needs a decision,
// walking forward from where it is, so the question can be asked before it
// gets there.
func (g *Game) NextPoint(id int) (Point, bool) {
	if !g.Driven(id) {
		return Point{}, false
	}
	a := &g.Actors[id]
	if a.Waiting {
		if g.isJunction(a.Tile, a.Dir) {
			return g.pointAt(id, a.Tile, a.Dir), true
		}
		return Point{}, false
	}
	t, h := g.Maze.Next(a.Tile, a.Dir), a.Dir
	for i := 0; i < g.Maze.W*g.Maze.H; i++ {
		if g.isJunction(t, h) {
			p := g.pointAt(id, t, h)
			p.Distance = i + 1
			return p, true
		}
		h = g.forced(t, h)
		if h == NoDir {
			return Point{}, false
		}
		t = g.Maze.Next(t, h)
	}
	return Point{}, false
}

// EscapePoint is Pac-Man's mid-corridor question when a dangerous ghost is in
// the corridor ahead or can reach its far junction no later than he can.
func (g *Game) EscapePoint() (Point, bool) {
	a := &g.Actors[Pacman]
	if g.Status != Playing || !g.PacmanAI || a.Waiting || !g.canTurnBack(a) {
		return Point{}, false
	}
	ahead := g.f.ahead[:0]
	t, h := g.Maze.Next(a.Tile, a.Dir), a.Dir
	for i := 0; i < corridorLimit; i++ {
		ahead = append(ahead, t)
		n := g.forced(t, h)
		if n == NoDir || g.isJunction(t, h) {
			break
		}
		h = n
		t = g.Maze.Next(t, h)
	}
	g.f.ahead = ahead
	j := ahead[len(ahead)-1]
	toJ := g.Maze.BFS(j, g.Maze.Key(a.Tile), g.f.scratch)
	g.f.scratch = toJ
	var threats uint8
	for id := Blinky; id < NActors; id++ {
		gh := &g.Actors[id]
		if gh.State != Normal {
			continue
		}
		at := g.Occupied(gh)
		d := toJ[g.Maze.Key(at)]
		in := false
		for _, x := range ahead {
			if x == at {
				in = true
				break
			}
		}
		if in || (d >= 0 && int(d) <= len(ahead)) {
			threats |= 1 << id
		}
	}
	if threats == 0 {
		return Point{}, false
	}
	p := Point{Actor: Pacman, Tile: a.Tile, Heading: a.Dir, Escape: true, Threats: threats, N: 2}
	p.Options[0], p.Options[1] = a.Dir, a.Dir.Reverse()
	p.Key = Key{Actor: Pacman, Escape: true, X: int16(j.X), Y: int16(j.Y), Mood: g.mood(Pacman), Threats: threats, Epoch: int32(a.Epoch)}
	return p, true
}

// Questions appends every open question of a driven actor: Pac-Man's escape
// question first, then the next junction.
func (g *Game) Questions(id int, dst []Point) []Point {
	if id == Pacman {
		if p, ok := g.EscapePoint(); ok {
			dst = append(dst, p)
		}
	}
	if p, ok := g.NextPoint(id); ok {
		dst = append(dst, p)
	}
	return dst
}
