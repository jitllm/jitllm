package game

import "math"

// Actor ids: Pac-Man and the four ghosts.
const (
	Pacman = iota
	Blinky
	Pinky
	Inky
	Clyde
	NActors
)

// Names are the actors' display names; IDs their question ids.
var (
	Names = [NActors]string{"Pac-Man", "Blinky", "Pinky", "Inky", "Clyde"}
	IDs   = [NActors]string{"pacman", "blinky", "pinky", "inky", "clyde"}
)

// Status is where the game is between lives and levels.
type Status uint8

// The statuses.
const (
	Ready Status = iota
	Playing
	Dying
	LevelClear
	GameOver
)

// GhostState is what a ghost is doing.
type GhostState uint8

// The ghost states. Pac-Man is always Normal.
const (
	House GhostState = iota
	Normal
	Frightened
	Eaten
)

var ghostStateNames = [4]string{"house", "normal", "frightened", "eaten"}

func (s GhostState) String() string { return ghostStateNames[s] }

// Mode is the ghosts' scatter/chase phase.
type Mode uint8

// The modes.
const (
	Scatter Mode = iota
	Chase
)

func (m Mode) String() string {
	if m == Scatter {
		return "scatter"
	}
	return "chase"
}

// Speeds in tiles per second, and the fixed timings of the arcade game.
const (
	SpeedPacman     = 7.5
	SpeedGhost      = 7.0
	SpeedFrightened = 4.5
	SpeedEaten      = 14.0

	ReadySeconds      = 1.5
	DyingSeconds      = 1.5
	LevelClearSeconds = 2.0
	FruitSeconds      = 9.5

	// Substep is one tick of the simulation: every movement and collision is
	// resolved at 60 Hz whatever the frame rate.
	Substep = 1.0 / 60

	collide       = 0.6
	corridorLimit = 40
	eps           = 1e-9
)

var (
	releaseAt      = [NActors]float64{0, 0, 1, 4, 7}
	fruitThreshold = [2]int{70, 170}
	schedule       = [...]struct {
		m Mode
		s float64
	}{{Scatter, 7}, {Chase, 20}, {Scatter, 7}, {Chase, 20}, {Scatter, 5}, {Chase, 20}, {Scatter, 5}, {Chase, math.Inf(1)}}
	fruits = [...]struct {
		upTo   int
		kind   string
		points int
	}{{1, "cherry", 100}, {2, "strawberry", 300}, {4, "orange", 500}, {6, "apple", 700},
		{8, "melon", 1000}, {10, "galaxian", 2000}, {12, "bell", 3000}, {math.MaxInt, "key", 5000}}
)

// ModeAt is the ghosts' phase this many seconds into a life.
func ModeAt(s float64) Mode {
	for _, p := range schedule {
		if s < p.s {
			return p.m
		}
		s -= p.s
	}
	return Chase
}

// FrightSeconds is how long a power pellet lasts on a level.
func FrightSeconds(level int) float64 { return max(2, 7-float64(level)) }

// FruitFor is the bonus fruit of a level.
func FruitFor(level int) (string, int) {
	for _, f := range fruits {
		if level <= f.upTo {
			return f.kind, f.points
		}
	}
	return "key", 5000
}

// Actor is Pac-Man or a ghost: the tile it last stood on the centre of, the
// way it is going, and how far it is towards the next tile.
type Actor struct {
	ID       int
	Tile     Tile
	Dir      Dir
	Progress float64
	// Waiting is an actor stopped at a junction for its decision.
	Waiting bool
	// Epoch moves whenever the actor is put somewhere, so an answer to a
	// question about where it was is recognised as stale.
	Epoch int
	State GhostState
}

// Fruit is the bonus on the board.
type Fruit struct {
	On     bool
	Kind   string
	Points int
	Tile   Tile
	Left   float64
}

// Popup is a score floating where it was won.
type Popup struct {
	Tile   Tile
	Points int
	TTL    float64
}

// Game is the whole state of one game.
type Game struct {
	Maze   *Maze
	Actors [NActors]Actor
	// PacmanAI is Pac-Man steered by Controls; false is the keyboard.
	PacmanAI bool
	// GhostsAI is the ghosts played by Controls; false is the classic
	// scripted ghosts.
	GhostsAI bool
	// Key is the direction the player last asked for.
	Key Dir

	Status       Status
	StatusTimer  float64
	Mode         Mode
	ModeTime     float64
	FrightLeft   float64
	FrightChain  int
	Score        int
	Lives        int
	Level        int
	PelletsEaten int
	Fruit        Fruit
	FruitsShown  int
	LifeTime     float64
	Popups       [8]Popup
	CaughtBy     int
	// Ticks is the substeps played.
	Ticks int

	f features
	// q is the question being asked, kept here so handing it to Controls
	// does not move it to the heap.
	q Point
}

// New starts a game: three lives, level 1, everyone at home.
func New(pacmanAI, ghostsAI bool) *Game {
	g := &Game{Maze: NewMaze(Layout), PacmanAI: pacmanAI, GhostsAI: ghostsAI, Lives: 3, Level: 1, Key: NoDir}
	for i := range g.Actors {
		g.Actors[i].ID = i
	}
	g.f.init(g.Maze)
	g.resetPositions()
	return g
}

func (g *Game) resetPositions() {
	p := &g.Actors[Pacman]
	p.Tile, p.Dir, p.Progress, p.Waiting, p.State = PacmanStart, Left, 0, false, Normal
	p.Epoch++
	for id := Blinky; id < NActors; id++ {
		a := &g.Actors[id]
		a.Tile, a.Progress, a.Waiting = HouseTiles[id], 0, false
		a.Epoch++
		if id == Blinky {
			a.Dir, a.State = Left, Normal
		} else {
			a.Dir, a.State = Up, House
		}
	}
	g.Status, g.StatusTimer = Ready, ReadySeconds
	g.Mode, g.ModeTime, g.FrightLeft, g.FrightChain, g.LifeTime = Scatter, 0, 0, 0, 0
	g.Key, g.CaughtBy = NoDir, -1
}

// Driven says whether Controls decides for actor id right now: Pac-Man when
// the AI steers him, a ghost when the AI plays the ghosts and it is out and
// not eaten (eyes find their own way home).
func (g *Game) Driven(id int) bool {
	if id == Pacman {
		return g.PacmanAI
	}
	s := g.Actors[id].State
	return g.GhostsAI && (s == Normal || s == Frightened)
}

// Position is a's continuous position in tiles.
func (a *Actor) Position() (float64, float64) {
	dx, dy := a.Dir.Vec()
	return float64(a.Tile.X) + float64(dx)*a.Progress, float64(a.Tile.Y) + float64(dy)*a.Progress
}

// Occupied is the tile a is mostly on.
func (g *Game) Occupied(a *Actor) Tile {
	if a.Progress >= 0.5 {
		return g.Maze.Next(a.Tile, a.Dir)
	}
	return a.Tile
}

// forced is the only sensible way on from t arriving with heading h, or NoDir
// at a real junction.
func (g *Game) forced(t Tile, h Dir) Dir {
	var open [4]Dir
	n := g.Maze.Open(t, &open)
	ahead, first := 0, NoDir
	for _, d := range open[:n] {
		if d != h.Reverse() {
			ahead++
			if first == NoDir {
				first = d
			}
		}
	}
	switch {
	case ahead == 1:
		return first
	case ahead == 0 && n > 0:
		return open[0]
	case ahead == 0:
		return NoDir
	}
	return NoDir
}

func (g *Game) isJunction(t Tile, h Dir) bool {
	var open [4]Dir
	n := g.Maze.Open(t, &open)
	ahead := 0
	for _, d := range open[:n] {
		if d != h.Reverse() {
			ahead++
		}
	}
	return ahead >= 2
}

// Step advances the game by dt seconds, in substeps.
func (g *Game) Step(dt float64, c Controls) {
	for i := range g.Popups {
		g.Popups[i].TTL -= dt
	}
	if g.Status == GameOver {
		return
	}
	if g.Status != Playing {
		g.StatusTimer -= dt
		if g.StatusTimer > eps {
			return
		}
		switch g.Status {
		case Ready:
			g.Status = Playing
		case Dying:
			g.Lives--
			if g.Lives <= 0 {
				g.Status = GameOver
			} else {
				g.resetPositions()
			}
		case LevelClear:
			g.Level++
			g.Maze.Refill()
			g.PelletsEaten, g.Fruit.On, g.FruitsShown = 0, false, 0
			g.resetPositions()
		}
		return
	}
	for left := dt; left > eps && g.Status == Playing; left -= Substep {
		g.Tick(min(left, Substep), c)
	}
}

// Tick is one substep of play.
func (g *Game) Tick(h float64, c Controls) {
	g.Ticks++
	g.LifeTime += h
	if g.FrightLeft > 0 {
		g.FrightLeft = max(0, g.FrightLeft-h)
		if g.FrightLeft == 0 {
			for id := Blinky; id < NActors; id++ {
				if g.Actors[id].State == Frightened {
					g.Actors[id].State = Normal
				}
			}
		}
	} else {
		g.ModeTime += h
		g.Mode = ModeAt(g.ModeTime)
	}
	if g.Fruit.On {
		g.Fruit.Left -= h
		if g.Fruit.Left <= 0 {
			g.Fruit.On = false
		}
	}
	for id := Blinky; id < NActors; id++ {
		a := &g.Actors[id]
		if a.State == House && g.LifeTime >= releaseAt[id] {
			a.State, a.Tile, a.Dir, a.Progress, a.Waiting = Normal, DoorExit, Left, 0, false
			a.Epoch++
		}
	}
	g.move(&g.Actors[Pacman], h, c)
	if g.Status != Playing {
		return
	}
	for id := Blinky; id < NActors; id++ {
		if g.Actors[id].State != House {
			g.move(&g.Actors[id], h, c)
		}
	}
	g.collisions()
}

func (g *Game) speed(a *Actor) float64 {
	if a.ID == Pacman {
		return SpeedPacman
	}
	switch a.State {
	case Eaten:
		return SpeedEaten
	case Frightened:
		return SpeedFrightened
	}
	return SpeedGhost
}

func (g *Game) canTurnBack(a *Actor) bool {
	return a.Progress > 0 || g.Maze.IsOpen(a.Tile, a.Dir.Reverse())
}

func (g *Game) turnBack(a *Actor) {
	if !g.canTurnBack(a) {
		return
	}
	if a.Progress > 0 {
		a.Tile = g.Maze.Next(a.Tile, a.Dir)
		a.Progress = 1 - a.Progress
	}
	a.Dir = a.Dir.Reverse()
}

func (g *Game) move(a *Actor, h float64, c Controls) {
	if a.Waiting && !g.choose(a, c) {
		return
	}
	if a.ID == Pacman && !g.PacmanAI && g.Key == a.Dir.Reverse() && a.Progress > 0 {
		g.turnBack(a)
	}
	if a.ID == Pacman && g.PacmanAI {
		var ok bool
		if g.q, ok = g.EscapePoint(); ok && c.Decide(g, &g.q) == a.Dir.Reverse() {
			g.turnBack(a)
		}
	}
	a.Progress += g.speed(a) * h
	for a.Progress >= 1 {
		a.Tile = g.Maze.Next(a.Tile, a.Dir)
		a.Progress--
		g.arrive(a)
		if !g.choose(a, c) {
			a.Progress = 0
			return
		}
	}
}

func (g *Game) choose(a *Actor, c Controls) bool {
	d := g.nextDir(a, c)
	if d == NoDir {
		a.Waiting = true
		return false
	}
	a.Dir, a.Waiting = d, false
	return true
}

func (g *Game) nextDir(a *Actor, c Controls) Dir {
	m := g.Maze
	if a.ID == Pacman && !g.PacmanAI {
		if g.Key != NoDir && m.IsOpen(a.Tile, g.Key) {
			return g.Key
		}
		if m.IsOpen(a.Tile, a.Dir) {
			return a.Dir
		}
		return NoDir
	}
	if a.ID != Pacman && a.State == Eaten {
		home := m.DistTo(DoorExit)
		var open [4]Dir
		n := m.Open(a.Tile, &open)
		best := open[0]
		for _, d := range open[1:n] {
			if home[m.Key(m.Next(a.Tile, d))] < home[m.Key(m.Next(a.Tile, best))] {
				best = d
			}
		}
		return best
	}
	if f := g.forced(a.Tile, a.Dir); f != NoDir || !g.isJunction(a.Tile, a.Dir) {
		return f
	}
	g.q = g.pointAt(a.ID, a.Tile, a.Dir)
	p := &g.q
	if a.ID != Pacman && !g.GhostsAI {
		return g.Classic(p)
	}
	d := c.Decide(g, p)
	if d != NoDir && !p.Offers(d) {
		return NoDir
	}
	return d
}

func (g *Game) arrive(a *Actor) {
	if a.ID == Pacman {
		g.eat(a.Tile)
		return
	}
	if a.State == Eaten && a.Tile == DoorExit {
		a.State = Normal
		a.Epoch++
	}
}

func (g *Game) eat(t Tile) {
	m := g.Maze
	k := m.Key(t)
	switch {
	case m.Pellet[k]:
		m.Pellet[k] = false
		m.Left--
		g.Score += 10
		g.pellet()
	case m.Power[k]:
		m.Power[k] = false
		m.Left--
		g.Score += 50
		g.pellet()
		g.FrightLeft, g.FrightChain = FrightSeconds(g.Level), 0
		for id := Blinky; id < NActors; id++ {
			if g.Actors[id].State == Normal {
				g.Actors[id].State = Frightened
			}
		}
	}
	if g.Fruit.On && t == g.Fruit.Tile {
		g.Score += g.Fruit.Points
		g.popup(t, g.Fruit.Points)
		g.Fruit.On = false
	}
	if m.Left == 0 {
		g.Status, g.StatusTimer = LevelClear, LevelClearSeconds
	}
}

func (g *Game) pellet() {
	g.PelletsEaten++
	if g.FruitsShown < len(fruitThreshold) && g.PelletsEaten == fruitThreshold[g.FruitsShown] {
		kind, pts := FruitFor(g.Level)
		g.Fruit = Fruit{On: true, Kind: kind, Points: pts, Tile: FruitTile, Left: FruitSeconds}
		g.FruitsShown++
	}
}

func (g *Game) popup(t Tile, pts int) {
	oldest := 0
	for i := range g.Popups {
		if g.Popups[i].TTL < g.Popups[oldest].TTL {
			oldest = i
		}
	}
	g.Popups[oldest] = Popup{Tile: t, Points: pts, TTL: 1.5}
}

func (g *Game) collisions() {
	px, py := g.Actors[Pacman].Position()
	w := float64(g.Maze.W)
	for id := Blinky; id < NActors; id++ {
		a := &g.Actors[id]
		if a.State == House || a.State == Eaten {
			continue
		}
		qx, qy := a.Position()
		dx := math.Abs(px - qx)
		dx = min(dx, w-dx)
		dy := py - qy
		if dx*dx+dy*dy >= collide*collide {
			continue
		}
		if a.State == Frightened {
			pts := 200 << g.FrightChain
			g.FrightChain++
			g.Score += pts
			a.State = Eaten
			g.popup(g.Occupied(a), pts)
			continue
		}
		g.Status, g.StatusTimer, g.CaughtBy = Dying, DyingSeconds, id
		return
	}
}
