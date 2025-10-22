import React, { useEffect, useRef, useState } from "react"

/**
 * Particle Life backend + editable ruleset (A matrix) with toolbar.
 * - World in [-1,1]^2; optional wrap.
 * - Pairwise accelerator with rMin (Particle-Life style).
 * - Second-order dynamics with drag and vMax clamp.
 * - Uniform-grid neighbor search.
 * - Toolbar shows a K×K clickable grid to edit A.
 *   * Left-click cycles value: -1 → -0.5 → 0 → +0.5 → +1 → …
 *   * Right-click cycles the other direction.
 *   * Values are saved to localStorage and applied live.
 * - “Ring preset”: each type attracts itself and its next color (i→i, i→i+1).
 */

// ========================= Spec =========================
type Spec = {
  N: number
  K: number
  seed: number

  A: number[][] // K×K entries in [-1,1]
  rMin: number
  R: number

  dt: number
  drag: number
  vMax: number

  wrap: boolean
  cellSize: number

  pixelScale: number
  genMatrix: boolean

  overlays: { showVel: boolean; showGrid: boolean }
}

const TYPE_COLORS = [
  "#FF5A5A", // red
  "#FF8A3C", // orange
  "#FFD23B", // yellow
  "#7DDE3B", // yellow-green
  "#34C759", // green
  "#30D7A9", // aquamarine
  "#4AA8FF", // blue
  "#6E6BFF", // indigo
  "#A15BFF", // violet
  "#FF5BD1", // magenta
  "#FF7F50", // coral
  "#00CED1", // dark turquoise
  "#708090", // slate gray
  "#7FFF00", // chartreuse
  "#CC79A7", // pink (okabe-ito)
  "#000000", // black
]

const SPEC: Spec = {
  N: 500,
  K: 8,
  seed: 1337,

  A: [
    [0.8, 0.6],
    [0.0, 0.8],
  ], // replaced in init
  rMin: 0.1,
  R: 0.9,

  dt: 0.05,
  drag: 5,
  vMax: 10,

  wrap: false,
  cellSize: 0.12,

  pixelScale: 900,
  genMatrix: true,

  overlays: { showVel: false, showGrid: false },
}

// ========================= RNG =========================
function mulberry32(seed: number) {
  let t = seed >>> 0
  return function () {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}
function randRange(rng: () => number, a: number, b: number) {
  return a + (b - a) * rng()
}
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

// ========================= World helpers =========================
const WORLD_SIZE = 2.0 // [-1,1]

function wrapCoord(x: number) {
  if (x < -1) return x + WORLD_SIZE
  if (x > 1) return x - WORLD_SIZE
  return x
}
function torusDelta(a: number, b: number) {
  let d = b - a
  if (d > 1) d -= WORLD_SIZE
  else if (d < -1) d += WORLD_SIZE
  return d
}

// ========================= Particle Life accelerator =========================
function accelMag(a: number, r: number, rMin: number): number {
  if (r <= 0) return 0
  if (r < rMin) return r / rMin - 1 // repulsion
  const denom = Math.max(1e-6, 1 - rMin)
  return a * (1 - Math.abs(1 + rMin - 2 * r) / denom)
}

// ========================= A matrix helpers =========================
function genRandomMatrix(K: number, rng: () => number) {
  const A: number[][] = Array.from({ length: K }, () => Array(K).fill(0))
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) {
      A[i][j] = i === j ? randRange(rng, 0.5, 0.9) : randRange(rng, -1, 1)
    }
  }
  return A
}

/** Ring preset: each type attracts itself and its next color (i→i, i→i+1). */
function genRingPreset(K: number, self = 0.9, next = 0.6, others = 0.0) {
  const A: number[][] = Array.from({ length: K }, () => Array(K).fill(others))
  for (let i = 0; i < K; i++) {
    A[i][i] = self
    A[i][(i + 1) % K] = next
  }
  return A
}

const LS_KEY = "pl_rules_v1"
function saveMatrixToLS(A: number[][], K: number) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ K, A }))
  } catch {}
}
function loadMatrixFromLS(K: number): number[][] | null {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (!raw) return null
    const obj = JSON.parse(raw)
    if (!obj || obj.K !== K) return null
    const A = obj.A
    if (
      !Array.isArray(A) ||
      A.length !== K ||
      A.some((r: any) => !Array.isArray(r) || r.length !== K)
    ) {
      return null
    }
    return A
  } catch {
    return null
  }
}

// ========================= Simulation state =========================
type Sim = {
  spec: Spec
  K: number
  x: Float32Array
  y: Float32Array
  vx: Float32Array
  vy: Float32Array
  type: Uint16Array

  fx: Float32Array
  fy: Float32Array

  gridDim: number
  cellHead: Int32Array
  next: Int32Array

  rng: () => number

  frame: number
  lastMaxSpeed: number

  A: number[][]
}

function initSim(spec: Spec, seedOverride?: number): Sim {
  const rng = mulberry32(seedOverride ?? spec.seed)
  const K = spec.K

  // choose matrix: localStorage → preset (ring) → random
  let A: number[][] | null = loadMatrixFromLS(K)
  if (!A) A = spec.genMatrix ? genRingPreset(K) : spec.A

  const x = new Float32Array(spec.N)
  const y = new Float32Array(spec.N)
  const vx = new Float32Array(spec.N)
  const vy = new Float32Array(spec.N)
  const type = new Uint16Array(spec.N)

  for (let i = 0; i < spec.N; i++) {
    x[i] = randRange(rng, -1, 1)
    y[i] = randRange(rng, -1, 1)
    vx[i] = randRange(rng, -0.05, 0.05)
    vy[i] = randRange(rng, -0.05, 0.05)
    type[i] = Math.floor(rng() * K)
  }

  const gridDim = Math.max(1, Math.ceil(WORLD_SIZE / spec.cellSize))
  const cellHead = new Int32Array(gridDim * gridDim)
  const next = new Int32Array(spec.N)

  return {
    spec,
    K,
    x,
    y,
    vx,
    vy,
    type,
    fx: new Float32Array(spec.N),
    fy: new Float32Array(spec.N),
    gridDim,
    cellHead,
    next,
    rng,
    frame: 0,
    lastMaxSpeed: 0,
    A: A!,
  }
}

// ========================= Neighbors =========================
function cellIndexOf(x: number, y: number, gridDim: number): number {
  const gx = clamp(Math.floor((x + 1) * 0.5 * gridDim), 0, gridDim - 1)
  const gy = clamp(Math.floor((y + 1) * 0.5 * gridDim), 0, gridDim - 1)
  return gx + gy * gridDim
}
function rebuildGrid(sim: Sim) {
  sim.cellHead.fill(-1)
  const N = sim.spec.N
  const gdim = sim.gridDim
  for (let i = 0; i < N; i++) {
    const idx = cellIndexOf(sim.x[i], sim.y[i], gdim)
    sim.next[i] = sim.cellHead[idx]
    sim.cellHead[idx] = i
  }
}
function forEachNeighbor(sim: Sim, i: number, fn: (j: number) => void) {
  const gdim = sim.gridDim
  const cx = clamp(Math.floor((sim.x[i] + 1) * 0.5 * gdim), 0, gdim - 1)
  const cy = clamp(Math.floor((sim.y[i] + 1) * 0.5 * gdim), 0, gdim - 1)

  // how many grid cells we must span to cover radius R
  const reach = Math.max(1, Math.ceil(sim.spec.R / sim.spec.cellSize))

  for (let dy = -reach; dy <= reach; dy++) {
    for (let dx = -reach; dx <= reach; dx++) {
      let nx = cx + dx
      let ny = cy + dy

      if (sim.spec.wrap) {
        nx = ((nx % gdim) + gdim) % gdim
        ny = ((ny % gdim) + gdim) % gdim
      } else {
        if (nx < 0 || ny < 0 || nx >= gdim || ny >= gdim) continue
      }

      let j = sim.cellHead[nx + ny * gdim]
      while (j !== -1) {
        if (j !== i) fn(j)
        j = sim.next[j]
      }
    }
  }
}

// ========================= Physics step =========================
function step(sim: Sim) {
  const sp = sim.spec
  const { N, R, rMin, dt } = { N: sp.N, R: sp.R, rMin: sp.rMin, dt: sp.dt }

  rebuildGrid(sim)
  sim.fx.fill(0)
  sim.fy.fill(0)

  // symmetric accumulation
  for (let i = 0; i < N; i++) {
    const ti = sim.type[i] | 0
    forEachNeighbor(sim, i, (j) => {
      if (j <= i) return
      let dx = sp.wrap ? torusDelta(sim.x[i], sim.x[j]) : sim.x[j] - sim.x[i]
      let dy = sp.wrap ? torusDelta(sim.y[i], sim.y[j]) : sim.y[j] - sim.y[i]
      const r2 = dx * dx + dy * dy
      if (r2 === 0) return
      const r = Math.sqrt(r2)
      if (r > R) return
      const tj = sim.type[j] | 0
      const aij = sim.A[ti][tj]
      if (aij === 0) return

      const invr = 1 / r
      const mag = accelMag(aij, r, rMin)
      const fx = mag * dx * invr
      const fy = mag * dy * invr
      sim.fx[i] += fx
      sim.fy[i] += fy
      sim.fx[j] -= fx
      sim.fy[j] -= fy
    })
  }

  // integrate
  let maxSpeed = 0
  const dragFactor = Math.max(0, 1 - sp.drag * dt)
  const vMax2 = sp.vMax * sp.vMax

  for (let i = 0; i < N; i++) {
    sim.vx[i] = (sim.vx[i] + dt * sim.fx[i]) * dragFactor
    sim.vy[i] = (sim.vy[i] + dt * sim.fy[i]) * dragFactor

    const v2 = sim.vx[i] * sim.vx[i] + sim.vy[i] * sim.vy[i]
    if (v2 > vMax2) {
      const s = sp.vMax / Math.sqrt(v2)
      sim.vx[i] *= s
      sim.vy[i] *= s
    }
    const speed = Math.sqrt(sim.vx[i] * sim.vx[i] + sim.vy[i] * sim.vy[i])
    if (speed > maxSpeed) maxSpeed = speed

    let nx = sim.x[i] + dt * sim.vx[i]
    let ny = sim.y[i] + dt * sim.vy[i]
    if (sp.wrap) {
      nx = wrapCoord(nx)
      ny = wrapCoord(ny)
    } else {
      if (nx < -1) {
        nx = -1 + (-1 - nx)
        sim.vx[i] = Math.abs(sim.vx[i])
      }
      if (nx > +1) {
        nx = +1 - (nx - 1)
        sim.vx[i] = -Math.abs(sim.vx[i])
      }
      if (ny < -1) {
        ny = -1 + (-1 - ny)
        sim.vy[i] = Math.abs(sim.vy[i])
      }
      if (ny > +1) {
        ny = +1 - (ny - 1)
        sim.vy[i] = -Math.abs(sim.vy[i])
      }
    }
    sim.x[i] = nx
    sim.y[i] = ny
  }
  sim.lastMaxSpeed = maxSpeed
  sim.frame++
}

// ========================= Rendering =========================
type Renderer = {
  ctx: CanvasRenderingContext2D
  width: number
  height: number
  scale: number
  dpr: number
}
function setupCanvas(canvas: HTMLCanvasElement, pixels: number): Renderer {
  const dpr = Math.max(1, (window.devicePixelRatio as number) || 1)
  const size = pixels
  canvas.style.width = `${size}px`
  canvas.style.height = `${size}px`
  canvas.width = Math.floor(size * dpr)
  canvas.height = Math.floor(size * dpr)
  const ctx = canvas.getContext("2d")!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  const scale = size / WORLD_SIZE
  return { ctx, width: size, height: size, scale, dpr }
}
function worldToScreen(x: number, y: number, r: Renderer) {
  const sx = (x + 1) * 0.5 * r.width
  const sy = (y + 1) * 0.5 * r.height
  return [sx, sy] as const
}
function draw(sim: Sim, r: Renderer) {
  const { ctx, width, height } = r
  const { showVel, showGrid } = sim.spec.overlays

  ctx.fillStyle = "#0a0a0a"
  ctx.fillRect(0, 0, width, height)

  if (showGrid) {
    ctx.strokeStyle = "rgba(255,255,255,0.06)"
    ctx.lineWidth = 1
    const step = (sim.spec.cellSize * r.width) / WORLD_SIZE
    for (let x = 0; x <= width; x += step) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    for (let y = 0; y <= height; y += step) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }

  const radiusPx = 2
  const N = sim.spec.N
  for (let i = 0; i < N; i++) {
    const t = sim.type[i] | 0
    const [px, py] = worldToScreen(sim.x[i], sim.y[i], r)
    ctx.fillStyle = TYPE_COLORS[t % TYPE_COLORS.length]
    ctx.beginPath()
    ctx.arc(px, py, radiusPx, 0, Math.PI * 2)
    ctx.fill()

    if (showVel) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)"
      ctx.beginPath()
      ctx.moveTo(px, py)
      ctx.lineTo(
        px + sim.vx[i] * r.scale * 0.15,
        py + sim.vy[i] * r.scale * 0.15
      )
      ctx.stroke()
    }
  }

  ctx.fillStyle = "#FFFFFF"
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
  const A0 = sim.A[0]
    ?.slice(0, Math.min(6, sim.K))
    .map((v) => v.toFixed(2))
    .join(", ")
  const lines = [
    `N=${sim.spec.N}  K=${sim.K}  frame=${sim.frame}`,
    `dt=${sim.spec.dt.toFixed(3)}  vMax=${sim.spec.vMax.toFixed(
      2
    )}  drag=${sim.spec.drag.toFixed(2)}  wrap=${sim.spec.wrap ? 1 : 0}`,
    `rMin=${sim.spec.rMin.toFixed(2)}  R=${sim.spec.R.toFixed(
      2
    )}  cell=${sim.spec.cellSize.toFixed(2)}`,
    `A[0,*]=[${A0 ?? ""}]`,
  ]
  let y = 16
  for (const ln of lines) {
    ctx.fillText(ln, 8, y)
    y += 14
  }
}

// ========================= Matrix Toolbar =========================
type MatrixToolbarProps = {
  K: number
  A: number[][]
  onChange: (A: number[][]) => void
  onRingPreset: () => void
  colors: string[]
}

function valueToSwatch(v: number): string {
  // map -1..1 to color: repulsion blue→neutral dark→attraction green/yellow
  const t = (v + 1) / 2 // 0..1
  const h = 220 * (1 - t) + 80 * t // 220 (blue) to 80 (yellow-green)
  const s = 70
  const l = 35 + 25 * Math.abs(v)
  return `hsl(${h.toFixed(0)} ${s}% ${l}%)`
}
function cycleValue(v: number, dir: number) {
  const steps = [-1, -0.5, 0, 0.5, 1]
  const idx = steps.findIndex((x) => Math.abs(x - v) < 1e-6)
  if (idx < 0) return 0
  const ni = (idx + dir + steps.length) % steps.length
  return steps[ni]
}
const cellBtn: React.CSSProperties = {
  width: 22,
  height: 22,
  borderRadius: 4,
  border: "1px solid #222",
  cursor: "pointer",
  display: "inline-block",
}

function MatrixToolbar({
  K,
  A,
  onChange,
  onRingPreset,
  colors,
}: MatrixToolbarProps) {
  function updateCell(i: number, j: number, v: number) {
    const nv = clamp(v, -1, 1)
    const NA = A.map((row, ri) =>
      ri === i ? row.map((x, cj) => (cj === j ? nv : x)) : row
    )
    onChange(NA)
  }
  function handleClick(i: number, j: number, e: React.MouseEvent) {
    e.preventDefault()
    const dir = e.type === "contextmenu" || e.button === 2 ? -1 : +1
    updateCell(i, j, cycleValue(A[i][j], dir))
  }

  return (
    <div
      style={{
        position: "fixed",
        right: 12,
        top: 64,
        background: "#121212",
        border: "1px solid #222",
        borderRadius: 8,
        padding: 12,
        color: "#e5e7eb",
        maxHeight: "80vh",
        overflow: "auto",
        boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 8,
        }}
      >
        <strong>Ruleset</strong>
        <button
          onClick={onRingPreset}
          style={{ ...chipStyle, padding: "4px 8px" }}
          title="Each type attracts itself and its next color"
        >
          Ring preset
        </button>
        <button
          onClick={() => {
            onChange(genRandomMatrix(K, mulberry32(Date.now())))
          }}
          style={{ ...chipStyle, padding: "4px 8px" }}
        >
          Randomize
        </button>
        <button
          onClick={() => {
            onChange(Array.from({ length: K }, () => Array(K).fill(0)))
          }}
          style={{ ...chipStyle, padding: "4px 8px" }}
        >
          Clear
        </button>
      </div>

      {/* legend */}
      <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 8 }}>
        Left-click: cycle up. Right-click: cycle down. Values: −1, −0.5, 0,
        +0.5, +1.
      </div>

      {/* header row of colors */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `24px repeat(${K}, 24px)`,
          gap: 4,
          alignItems: "center",
        }}
      >
        <div />
        {Array.from({ length: K }, (_, j) => (
          <div
            key={`h${j}`}
            title={`col ${j}`}
            style={{
              width: 22,
              height: 22,
              borderRadius: "50%",
              background: colors[j % colors.length],
              border: "1px solid #222",
            }}
          />
        ))}

        {/* matrix rows */}
        {Array.from({ length: K }, (_, i) => (
          <React.Fragment key={`row${i}`}>
            <div
              title={`row ${i}`}
              style={{
                width: 22,
                height: 22,
                borderRadius: "50%",
                background: colors[i % colors.length],
                border: "1px solid #222",
              }}
            />
            {Array.from({ length: K }, (_, j) => (
              <div key={`c${i}-${j}`} style={{ position: "relative" }}>
                <div
                  onClick={(e) => handleClick(i, j, e)}
                  onContextMenu={(e) => handleClick(i, j, e)}
                  title={`A[${i}][${j}] = ${A[i][j].toFixed(2)}`}
                  style={{ ...cellBtn, background: valueToSwatch(A[i][j]) }}
                />
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

// ========================= Hooks =========================
function useAnimationFrame(callback: (t: number) => void, paused: boolean) {
  const ref = useRef<number | null>(null)
  useEffect(() => {
    function loop(t: number) {
      if (!paused) {
        callback(t)
        ref.current = requestAnimationFrame(loop)
      }
    }
    if (!paused) ref.current = requestAnimationFrame(loop)
    return () => {
      if (ref.current !== null) cancelAnimationFrame(ref.current)
      ref.current = null
    }
  }, [callback, paused])
}
function useRenderer(pixels: number) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rendererRef = useRef<Renderer | null>(null)
  useEffect(() => {
    function setup() {
      const canvas = canvasRef.current
      if (!canvas) return
      rendererRef.current = setupCanvas(canvas, pixels)
    }
    setup()
    const onResize = () => setup()
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [pixels])
  return { canvasRef, rendererRef }
}

// ========================= Component =========================
export default function App() {
  const [spec, setSpec] = useState<Spec>({ ...SPEC, A: genRingPreset(SPEC.K) })
  const [seed, setSeed] = useState<number>(SPEC.seed)
  const [paused, setPaused] = useState(false)
  const [showRules, setShowRules] = useState(true)
  const [fps, setFps] = useState(0)

  const simRef = useRef<Sim | null>(null)
  const { canvasRef, rendererRef } = useRenderer(spec.pixelScale)

  // init / re-init when core spec changes (except A)
  useEffect(() => {
    // if K changed, ensure A has correct shape: prefer LS → ring
    const fromLS = loadMatrixFromLS(spec.K)
    if (fromLS) {
      setSpec((s) => ({ ...s, A: fromLS, genMatrix: false }))
      // sim will be created in next effect run
      return
    }
    if (
      !spec.A ||
      spec.A.length !== spec.K ||
      spec.A.some((r) => r.length !== spec.K)
    ) {
      setSpec((s) => ({ ...s, A: genRingPreset(spec.K), genMatrix: false }))
      return
    }
    simRef.current = initSim(spec, seed)
  }, [
    spec.K,
    spec.seed,
    spec.cellSize,
    spec.wrap,
    spec.N,
    spec.dt,
    spec.drag,
    spec.vMax,
    seed,
  ]) // core fields

  // draw loop
  useAnimationFrame(() => {
    const sim = simRef.current
    const r = rendererRef.current
    if (!sim || !r) return
    step(sim)
    draw(sim, r)
  }, paused)

  // FPS meter
  useEffect(() => {
    let last = performance.now(),
      frames = 0,
      raf = 0 as unknown as number
    function tick() {
      frames++
      const now = performance.now()
      if (now - last >= 500) {
        setFps((frames * 1000) / (now - last))
        frames = 0
        last = now
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  // persist A
  useEffect(() => {
    saveMatrixToLS(spec.A, spec.K)
  }, [spec.A, spec.K])

  // apply matrix to running sim without full reset
  function applyMatrix(A: number[][]) {
    setSpec((s) => ({ ...s, A, genMatrix: false }))
    if (simRef.current && simRef.current.K === A.length) {
      // shallow clone so React state stays immutable, but sim uses same content
      simRef.current.A = A.map((row) => row.slice())
    }
  }

  // handlers
  const togglePause = () => setPaused((p) => !p)
  const handleReset = () => {
    simRef.current = initSim(spec, seed)
  }
  const handleRandomizeSeed = () => setSeed(Math.floor(Math.random() * 1e9))
  const incN = (delta: number) => {
    const newN = clamp(spec.N + delta, 100, 50000)
    setSpec({ ...spec, N: newN })
  }
  const incK = (delta: number) => {
    const maxK = TYPE_COLORS.length
    const newK = clamp(spec.K + delta, 2, maxK)
    setSpec((s) => ({ ...s, K: newK }))
  }
  const applyRingPreset = () =>
    applyMatrix(genRingPreset(spec.K, 0.9, 0.6, 0.0))

  return (
    <div
      style={{
        display: "grid",
        gridTemplateRows: "auto 1fr auto",
        minHeight: "100vh",
        background: "#0a0a0a",
        color: "#eaeaea",
        fontFamily:
          "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial",
      }}
    >
      {/* Header */}
      <div style={{ padding: "12px 16px", borderBottom: "1px solid #222" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <strong>Particle Interaction Prototype · PL + Rules</strong>
          <span style={{ opacity: 0.7 }}>FPS: {fps.toFixed(0)}</span>
          <button
            onClick={togglePause}
            style={buttonStyle}
            title="Pause or resume"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={handleReset}
            style={buttonStyle}
            title="Reset with current seed/spec"
          >
            Reset
          </button>
          <button
            onClick={handleRandomizeSeed}
            style={buttonStyle}
            title="Randomize seed and reset"
          >
            New Seed
          </button>

          <div style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
            <span style={{ opacity: 0.7 }}>N:</span>
            <button onClick={() => incN(-500)} style={chipStyle}>
              −500
            </button>
            <span>{spec.N}</span>
            <button onClick={() => incN(+500)} style={chipStyle}>
              +500
            </button>
          </div>

          <div style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
            <span style={{ opacity: 0.7 }}>K:</span>
            <button onClick={() => incK(-1)} style={chipStyle}>
              −1
            </button>
            <span>{spec.K}</span>
            <button onClick={() => incK(+1)} style={chipStyle}>
              +1
            </button>
          </div>

          {/* Overlay toggles */}
          <div
            style={{
              display: "inline-flex",
              gap: 8,
              alignItems: "center",
              marginLeft: 12,
            }}
          >
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              <input
                type="checkbox"
                checked={spec.overlays.showVel}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    overlays: { ...spec.overlays, showVel: e.target.checked },
                  })
                }
              />
              Vel
            </label>
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              <input
                type="checkbox"
                checked={spec.overlays.showGrid}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    overlays: { ...spec.overlays, showGrid: e.target.checked },
                  })
                }
              />
              Grid
            </label>
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              <input
                type="checkbox"
                checked={spec.wrap}
                onChange={(e) => setSpec({ ...spec, wrap: e.target.checked })}
              />
              Wrap
            </label>
          </div>

          <button
            onClick={() => setShowRules((s) => !s)}
            style={chipStyle}
            title="Show/Hide rules editor"
          >
            {showRules ? "Hide Rules" : "Show Rules"}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ display: "grid", placeItems: "center", padding: 8 }}>
        <canvas
          ref={canvasRef}
          style={{ borderRadius: 8, background: "#0a0a0a" }}
        />
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "10px 16px",
          borderTop: "1px solid #222",
          fontSize: 12,
          color: "#cfcfcf",
        }}
      >
        <span style={{ opacity: 0.85 }}>
          rMin={spec.rMin} · R={spec.R} · dt={spec.dt} · drag={spec.drag} ·
          vMax={spec.vMax} · cell={spec.cellSize} · seed={seed}
        </span>
      </div>

      {/* Rules toolbar */}
      {showRules && (
        <MatrixToolbar
          K={spec.A.length}
          A={spec.A}
          onChange={applyMatrix}
          onRingPreset={applyRingPreset}
          colors={TYPE_COLORS}
        />
      )}
    </div>
  )
}

// ========================= Styles =========================
const buttonStyle: React.CSSProperties = {
  background: "#1f2937",
  color: "#e5e7eb",
  border: "1px solid #374151",
  padding: "6px 10px",
  borderRadius: 6,
  cursor: "pointer",
}
const chipStyle: React.CSSProperties = {
  background: "#111827",
  color: "#e5e7eb",
  border: "1px solid #374151",
  padding: "2px 6px",
  borderRadius: 6,
  cursor: "pointer",
}
