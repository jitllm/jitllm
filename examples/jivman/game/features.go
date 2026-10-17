package game

import "math"

// Unknown stands for "not reachable this way" in a feature.
const Unknown = -1

// Option is what the game knows about leaving a junction one way: the facts a
// model is told, and the facts the greedy rule decides on.
type Option struct {
	Dir Dir
	// Goal is the steps to the actor's current goal this way.
	Goal int
	// ToPacman is a ghost's steps to Pac-Man this way.
	ToPacman int
	Pellet   int
	// CorridorPellets is the pellets up to and including the next junction.
	CorridorPellets int
	Danger          int
	// DangerIn is a bit per normal ghost standing in the corridor.
	DangerIn    uint8
	Frightened  int
	Fruit       int
	Power       int
	Approaching bool
	Nearby      int
	// JunctionSteps is the walk to the end of this way's corridor; JunctionGhost
	// the dangerous ghost that gets there first (-1 for none) in GhostSteps.
	JunctionSteps int
	JunctionGhost int
	GhostSteps    int
}

// Trap says a ghost reaches the end of this way's corridor no later than
// Pac-Man.
func (o *Option) Trap() bool { return o.JunctionGhost >= 0 && o.GhostSteps <= o.JunctionSteps }

// GoalKind is what an actor is after.
type GoalKind uint8

// The goals.
const (
	GoalEat GoalKind = iota
	GoalHunt
	GoalPacman
	GoalAmbush
	GoalFlank
	GoalCorner
	GoalFlee
)

var goalLabels = [...]string{"pellet", "frightened ghost", "Pac-Man", "ambush point ahead of Pac-Man", "flanking point", "home corner", "Pac-Man"}

func (k GoalKind) String() string { return goalLabels[k] }

const (
	shyDistance  = 8
	nearbySteps  = 8
	pacmanPerSec = 7
)

// features is the scratch the option facts are computed in, so a question
// allocates nothing once the game is warm.
type features struct {
	dist    []int16
	scratch []int16
	ahead   []Tile
	corr    []Tile
	opts    [4]Option
}

func (f *features) init(m *Maze) {
	f.dist = make([]int16, m.W*m.H)
	f.scratch = make([]int16, m.W*m.H)
	f.ahead = make([]Tile, 0, corridorLimit)
	f.corr = make([]Tile, 0, corridorLimit)
}

// Goal is what actor id is after, and the tile, when it has one.
func (g *Game) Goal(id int) (GoalKind, Tile, bool) {
	if id == Pacman {
		if g.FrightLeft > 0 {
			return GoalHunt, Tile{}, false
		}
		return GoalEat, Tile{}, false
	}
	m := g.Maze
	gh := &g.Actors[id]
	pac := g.Occupied(&g.Actors[Pacman])
	if gh.State == Frightened {
		return GoalFlee, pac, true
	}
	if g.Mode == Scatter {
		return GoalCorner, Corners[id], true
	}
	ahead := func(n int) Tile {
		dx, dy := g.Actors[Pacman].Dir.Vec()
		return Tile{pac.X + dx*n, pac.Y + dy*n}
	}
	switch id {
	case Pinky:
		return GoalAmbush, m.NearestWalkable(ahead(4)), true
	case Inky:
		pv := ahead(2)
		b := g.Occupied(&g.Actors[Blinky])
		return GoalFlank, m.NearestWalkable(Tile{2*pv.X - b.X, 2*pv.Y - b.Y}), true
	case Clyde:
		d := m.DistTo(pac)[m.Key(g.Occupied(gh))]
		if d >= 0 && d <= shyDistance {
			return GoalCorner, Corners[id], true
		}
	}
	return GoalPacman, pac, true
}

// Options computes the facts of each of p's options. The slice is the game's
// scratch, valid until the next call.
func (g *Game) Options(p *Point) []Option {
	m := g.Maze
	kind, target, hasTarget := g.Goal(p.Actor)
	pac := g.Occupied(&g.Actors[Pacman])
	out := g.f.opts[:p.N]
	for i, d := range p.Options[:p.N] {
		start, origin := m.Next(p.Tile, d), p.Tile
		if p.Escape && d != p.Heading && g.Actors[Pacman].Progress > 0 {
			start, origin = p.Tile, m.Next(p.Tile, p.Heading)
		}
		dist := m.BFS(start, m.Key(origin), g.f.dist)
		g.f.dist = dist
		steps := func(t Tile) int {
			v := dist[m.Key(t)]
			if v < 0 {
				return Unknown
			}
			return int(v) + 1
		}
		o := Option{Dir: d, Goal: Unknown, ToPacman: Unknown, Pellet: Unknown, Danger: Unknown, Frightened: Unknown,
			Fruit: Unknown, Power: Unknown, JunctionGhost: -1}
		for k := range m.Pellet {
			if !m.Pellet[k] && !m.Power[k] {
				continue
			}
			s := dist[k]
			if s < 0 {
				continue
			}
			o.Pellet = minKnown(o.Pellet, int(s)+1)
			if m.Power[k] {
				o.Power = minKnown(o.Power, int(s)+1)
			}
		}
		corr := g.corridor(start, d)
		j := corr[len(corr)-1]
		toJ := m.DistTo(j)
		for _, t := range corr {
			if k := m.Key(t); m.Pellet[k] || m.Power[k] {
				o.CorridorPellets++
			}
		}
		closest, closestID := Unknown, -1
		for id := Blinky; id < NActors; id++ {
			if id == p.Actor {
				continue
			}
			gh := &g.Actors[id]
			at := g.Occupied(gh)
			switch gh.State {
			case Normal:
				if jd := toJ[m.Key(at)]; jd >= 0 && (o.JunctionGhost < 0 || int(jd) < o.GhostSteps) {
					o.JunctionGhost, o.GhostSteps = id, int(jd)
				}
				s := steps(at)
				if s == Unknown {
					continue
				}
				if closest == Unknown || s < closest {
					closest, closestID = s, id
				}
				if s <= nearbySteps {
					o.Nearby++
				}
				for _, t := range corr {
					if t == at {
						o.DangerIn |= 1 << id
						break
					}
				}
			case Frightened:
				if s := steps(at); s != Unknown {
					o.Frightened = minKnown(o.Frightened, s)
				}
			}
		}
		o.Danger = closest
		if closestID >= 0 {
			o.Approaching = g.movingToward(&g.Actors[closestID], origin)
		}
		if g.Fruit.On {
			o.Fruit = steps(g.Fruit.Tile)
		}
		if p.Actor != Pacman {
			o.ToPacman = steps(pac)
		}
		switch {
		case hasTarget:
			o.Goal = steps(target)
		case kind == GoalHunt && o.Frightened != Unknown:
			o.Goal = o.Frightened
		default:
			o.Goal = o.Pellet
		}
		o.JunctionSteps = p.Distance + len(corr)
		out[i] = o
	}
	return out
}

func minKnown(a, b int) int {
	if a == Unknown || b < a {
		return b
	}
	return a
}

func (g *Game) corridor(start Tile, h Dir) []Tile {
	c := g.f.corr[:0]
	t := start
	for i := 0; i < corridorLimit; i++ {
		c = append(c, t)
		if g.isJunction(t, h) {
			break
		}
		n := g.forced(t, h)
		if n == NoDir {
			break
		}
		h = n
		t = g.Maze.Next(t, h)
	}
	g.f.corr = c
	return c
}

func (g *Game) movingToward(gh *Actor, to Tile) bool {
	m := g.Maze
	dist := m.DistTo(to)
	here := dist[m.Key(gh.Tile)]
	if !gh.Waiting {
		n := dist[m.Key(m.Next(gh.Tile, gh.Dir))]
		return n >= 0 && n < here
	}
	var open [4]Dir
	n := m.Open(gh.Tile, &open)
	for _, d := range open[:n] {
		if d == gh.Dir.Reverse() {
			continue
		}
		if v := dist[m.Key(m.Next(gh.Tile, d))]; v >= 0 && v < here {
			return true
		}
	}
	return false
}

// huntReach is the steps Pac-Man still covers before the fright ends.
func (g *Game) huntReach(p *Point) int {
	return int(math.Floor(g.FrightLeft*pacmanPerSec)) - p.Distance
}

func safe(o *Option) bool {
	return o.DangerIn == 0 && (o.Danger == Unknown || o.Danger > 2) && !o.Trap()
}

// FruitRoute is the fastest way to the fruit that gets there in time with no
// ghost in its corridor, close by, or able to block its junction; NoDir when
// there is none.
func (g *Game) FruitRoute(p *Point, opts []Option) Dir {
	if !g.Fruit.On {
		return NoDir
	}
	reach := int(math.Floor(g.Fruit.Left * pacmanPerSec))
	hr := g.huntReach(p)
	best := -1
	for i := range opts {
		o := &opts[i]
		if o.Fruit == Unknown || p.Distance+o.Fruit > reach || !safe(o) {
			continue
		}
		if o.Frightened != Unknown && o.Frightened <= o.Fruit && o.Frightened > hr {
			continue
		}
		if best < 0 || o.Fruit < opts[best].Fruit {
			best = i
		}
	}
	if best < 0 {
		return NoDir
	}
	return opts[best].Dir
}

func lowest(opts []Option, keep func(*Option) bool, score func(*Option) int) Dir {
	best, bs := NoDir, 0
	for i := range opts {
		o := &opts[i]
		if keep != nil && !keep(o) {
			continue
		}
		s := score(o)
		if best == NoDir || s < bs {
			best, bs = o.Dir, s
		}
	}
	return best
}

func or999(v int) int {
	if v == Unknown {
		return 999
	}
	return v
}

// Greedy is the fixed rule used when a model cannot answer in time, and the
// classic ghosts' rule: ghosts take the shortest way to their goal (or the
// longest from Pac-Man when frightened); Pac-Man keeps to safe ways, hunts a
// frightened ghost he can reach, takes a safe fruit, else the nearest pellet.
func (g *Game) Greedy(p *Point, opts []Option) Dir {
	if p.Actor != Pacman {
		if k, _, _ := g.Goal(p.Actor); k == GoalFlee {
			return lowest(opts, nil, func(o *Option) int { return -or999(o.ToPacman) })
		}
		return lowest(opts, nil, func(o *Option) int { return or999(o.Goal) })
	}
	anySafe := false
	for i := range opts {
		anySafe = anySafe || safe(&opts[i])
	}
	pool := func(o *Option) bool { return !anySafe || safe(o) }
	hr := g.huntReach(p)
	if d := lowest(opts, func(o *Option) bool { return pool(o) && o.Frightened != Unknown && o.Frightened <= hr },
		func(o *Option) int { return o.Frightened }); d != NoDir {
		return d
	}
	// The fruit route is taken from the same pool.
	var sub []Option
	var buf [4]Option
	if anySafe {
		n := 0
		for i := range opts {
			if safe(&opts[i]) {
				buf[n] = opts[i]
				n++
			}
		}
		sub = buf[:n]
	} else {
		sub = opts
	}
	if d := g.FruitRoute(p, sub); d != NoDir {
		return d
	}
	return lowest(opts, pool, func(o *Option) int { return or999(o.Pellet) })
}

// GreedyControls answers every question with Greedy: jevman's stand-in for a
// model that does not answer, and the classic ghosts' rule.
type GreedyControls struct{}

// Decide implements Controls.
func (GreedyControls) Decide(g *Game, p *Point) Dir { return g.Greedy(p, g.Options(p)) }

// Classic is the scripted ghosts' choice at a junction.
func (g *Game) Classic(p *Point) Dir { return g.Greedy(p, g.Options(p)) }
