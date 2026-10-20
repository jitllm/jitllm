// Package game is jivman's Pac-Man: the maze, the rules, the ghosts' classic
// behaviour, and the questions a decision model is asked at a junction. It
// knows nothing of models or windows; a Controls answers its questions.
package game

// Dir is a direction of travel. The order is the order options are offered in.
type Dir int8

// The directions, and NoDir for "not decided yet".
const (
	Up Dir = iota
	Left
	Down
	Right
	NoDir Dir = -1
)

// Dirs is every direction in offer order.
var Dirs = [4]Dir{Up, Left, Down, Right}

var dirNames = [4]string{"up", "left", "down", "right"}

func (d Dir) String() string {
	if d < 0 || d > 3 {
		return "none"
	}
	return dirNames[d]
}

// Reverse is the opposite direction.
func (d Dir) Reverse() Dir { return (d + 2) % 4 }

// DirOf parses a direction's name.
func DirOf(s string) Dir {
	for i, n := range dirNames {
		if n == s {
			return Dir(i)
		}
	}
	return NoDir
}

var vec = [4][2]int{{0, -1}, {-1, 0}, {0, 1}, {1, 0}}

// Vec is d's unit step.
func (d Dir) Vec() (int, int) { return vec[d][0], vec[d][1] }

// Tile is a maze cell.
type Tile struct{ X, Y int }

// Layout is the arcade maze: '#' wall, '-' the ghost-house door, '_' inside
// the house (not walkable), '.' a pellet, 'o' a power pellet, ' ' an empty
// corridor. Row 14 wraps through the side tunnel.
var Layout = []string{
	"############################",
	"#............##............#",
	"#.####.#####.##.#####.####.#",
	"#o####.#####.##.#####.####o#",
	"#.####.#####.##.#####.####.#",
	"#..........................#",
	"#.####.##.########.##.####.#",
	"#.####.##.########.##.####.#",
	"#......##....##....##......#",
	"######.##### ## #####.######",
	"######.##### ## #####.######",
	"######.##          ##.######",
	"######.## ###--### ##.######",
	"######.## #______# ##.######",
	"      .   #______#   .      ",
	"######.## #______# ##.######",
	"######.## ######## ##.######",
	"######.##          ##.######",
	"######.## ######## ##.######",
	"######.## ######## ##.######",
	"#............##............#",
	"#.####.#####.##.#####.####.#",
	"#.####.#####.##.#####.####.#",
	"#o..##.......  .......##..o#",
	"###.##.##.########.##.##.###",
	"###.##.##.########.##.##.###",
	"#......##....##....##......#",
	"#.##########.##.##########.#",
	"#.##########.##.##########.#",
	"#..........................#",
	"############################",
}

// Fixed places of the arcade maze.
var (
	PacmanStart = Tile{13, 23}
	DoorExit    = Tile{13, 11}
	FruitTile   = Tile{13, 17}
	HouseTiles  = [5]Tile{{}, {13, 11}, {13, 14}, {11, 14}, {15, 14}}
	Corners     = [5]Tile{{}, {26, 1}, {1, 1}, {26, 29}, {1, 29}}
)

// Maze is the board: walls, and the pellets still on it.
type Maze struct {
	W, H   int
	Rows   []string
	wall   []bool
	Pellet []bool
	Power  []bool
	// Left is the pellets and power pellets still on the board.
	Left int

	dist  map[int][]int16
	queue []int32
}

// NewMaze builds a maze from rows of equal length.
func NewMaze(rows []string) *Maze {
	m := &Maze{H: len(rows), W: len(rows[0]), Rows: rows, dist: map[int][]int16{}}
	n := m.W * m.H
	m.wall = make([]bool, n)
	m.Pellet = make([]bool, n)
	m.Power = make([]bool, n)
	m.queue = make([]int32, 0, n)
	for y, r := range rows {
		if len(r) != m.W {
			panic("game: ragged maze")
		}
		for x := 0; x < m.W; x++ {
			c := r[x]
			m.wall[y*m.W+x] = c == '#' || c == '-' || c == '_'
		}
	}
	m.Refill()
	return m
}

// Refill puts every pellet back, for a new level.
func (m *Maze) Refill() {
	m.Left = 0
	for y, r := range m.Rows {
		for x := 0; x < m.W; x++ {
			k := y*m.W + x
			m.Pellet[k] = r[x] == '.'
			m.Power[k] = r[x] == 'o'
			if m.Pellet[k] || m.Power[k] {
				m.Left++
			}
		}
	}
}

// Key is t's index.
func (m *Maze) Key(t Tile) int { return t.Y*m.W + t.X }

// At is the tile of index k.
func (m *Maze) At(k int) Tile { return Tile{k % m.W, k / m.W} }

// Walkable says whether t is a corridor, wrapping x through the tunnel.
func (m *Maze) Walkable(t Tile) bool {
	if t.Y < 0 || t.Y >= m.H {
		return false
	}
	x := ((t.X % m.W) + m.W) % m.W
	return !m.wall[t.Y*m.W+x]
}

// Next is the tile beside t in d, wrapping horizontally.
func (m *Maze) Next(t Tile, d Dir) Tile {
	dx, dy := d.Vec()
	return Tile{(t.X + dx + m.W) % m.W, t.Y + dy}
}

// Open writes the open directions from t into out, in offer order, and
// returns how many.
func (m *Maze) Open(t Tile, out *[4]Dir) int {
	n := 0
	for _, d := range Dirs {
		if m.Walkable(m.Next(t, d)) {
			out[n] = d
			n++
		}
	}
	return n
}

// IsOpen says whether d leads somewhere from t.
func (m *Maze) IsOpen(t Tile, d Dir) bool { return m.Walkable(m.Next(t, d)) }

// BFS writes into dst the step count from `from` to every tile (-1 for
// unreachable), never passing through the tile of index blocked (-1 for
// none), and returns dst.
func (m *Maze) BFS(from Tile, blocked int, dst []int16) []int16 {
	if cap(dst) < m.W*m.H {
		dst = make([]int16, m.W*m.H)
	}
	dst = dst[:m.W*m.H]
	for i := range dst {
		dst[i] = -1
	}
	if !m.Walkable(from) {
		return dst
	}
	q := m.queue[:0]
	s := m.Key(from)
	dst[s] = 0
	q = append(q, int32(s))
	for i := 0; i < len(q); i++ {
		k := int(q[i])
		t := m.At(k)
		for _, d := range Dirs {
			n := m.Next(t, d)
			if !m.Walkable(n) {
				continue
			}
			nk := m.Key(n)
			if nk == blocked || dst[nk] != -1 {
				continue
			}
			dst[nk] = dst[k] + 1
			q = append(q, int32(nk))
		}
	}
	m.queue = q
	return dst
}

// DistTo is the cached distance map to t (the maze is undirected, so it is
// also the map from t). The walls never change, so neither does a map.
func (m *Maze) DistTo(t Tile) []int16 {
	k := m.Key(t)
	if d, ok := m.dist[k]; ok {
		return d
	}
	d := m.BFS(t, -1, nil)
	m.dist[k] = d
	return d
}

// NearestWalkable is the corridor tile closest to t, t clamped into the grid
// and then searched in growing diamonds.
func (m *Maze) NearestWalkable(t Tile) Tile {
	c := Tile{min(m.W-1, max(0, t.X)), min(m.H-1, max(0, t.Y))}
	if m.Walkable(c) {
		return c
	}
	for r := 1; r < m.W+m.H; r++ {
		for dx := -r; dx <= r; dx++ {
			dy := r - abs(dx)
			for _, n := range [2]Tile{{c.X + dx, c.Y + dy}, {c.X + dx, c.Y - dy}} {
				if n.X >= 0 && n.X < m.W && m.Walkable(n) {
					return n
				}
			}
		}
	}
	return c
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
