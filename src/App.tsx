import React, { useEffect, useRef, useState } from "react"

/**
 * Particle Life backend + editable ruleset (A matrix) with toolbar.
 * - World in [-1,1]^2; optional wrap.
 * - Pairwise accelerator with rMin (Particle-Life style).
 * - Second-order dynamics with drag and vMax clamp.
 * - Uniform-grid neighbor search.
 * - Toolbar shows a K×K clickable grid to edit A.
 *   * Left-click cycles value: -1 → 0 → +1 → …
 *   * Right-click cycles the other direction.
 *   * Values are saved to localStorage and applied live.
 * - “Ring preset”: each type attracts itself and its next color (i→i, i→i+1).
 *
 * Rendering change:
 * - Square viewport with letterboxing. No stretching. The canvas matches the container,
 *   but simulation content draws inside a centered square viewport.
 *
 * Notes on stability:
 * - Re-initialize simulation buffers whenever core layout changes (N, K, cellSize, etc.).
 * - Never loop past typed-array lengths; use safe N.
 * - Do not early-return from re-init when A exists in localStorage.
 *
 * Non-reciprocal update:
 * - Interactions are now non-reciprocal: F_ij and F_ji are computed from A[i][j] and A[j][i] separately.
 *   This allows self-propelled clusters when A is asymmetric.
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

  pixelScale: number // retained for backwards compatibility; unused in responsive mode
  genMatrix: boolean

  overlays: { showVel: boolean; showGrid: boolean }

  mutualOnly: boolean
  settleEnabled: boolean
  settleK: number // damping coefficient (1/s). Will be clamped for stability.
  settleR: number // radius under which settling damping applies (> rMin, ≤ R)
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
  "#ffffff", // white
]

// Default spec configured for a 2-type chasing/fleeing lump.
// Type 0 chases 1; type 1 flees 0; both self-attract.
const SPEC: Spec = {
  N: 500,
  K: 5,
  seed: 1337,

  A: [
    [0.8, 0.6],
    [0.6, 0.8],
  ],
  rMin: 0.12,
  R: 0.75,

  dt: 0.03,
  drag: 1,
  vMax: 1.5,

  wrap: false,
  cellSize: 0.1,

  pixelScale: 800,
  genMatrix: true,

  overlays: { showVel: false, showGrid: true },

  mutualOnly: false,
  settleEnabled: true,
  settleK: 0.2,
  settleR: 0.2,
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
  // hard-core repulsion regardless of 'a'
  if (r < rMin) return r / rMin - 1
  const denom = Math.max(1e-6, 1 - rMin)
  // smooth bell between rMin..1
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

  // choose matrix: localStorage → preset (ring) → spec.A
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
  // Safe N prevents writing past typed-array bounds after spec changes.
  const N = Math.min(sim.spec.N, sim.x.length, sim.next.length)
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
function smoothstep01(t: number) {
  const x = clamp(t, 0, 1)
  return x * x * (3 - 2 * x)
}

function step(sim: Sim) {
  const sp = sim.spec
  // Safe N: never iterate beyond buffer lengths.
  const N = Math.min(sp.N, sim.x.length, sim.vx.length, sim.type.length)
  const { R, rMin, dt } = { R: sp.R, rMin: sp.rMin, dt: sp.dt }

  rebuildGrid(sim)
  // Reset only the portion actually used.
  sim.fx.fill(0, 0, N)
  sim.fy.fill(0, 0, N)

  // non-reciprocal accumulation
  for (let i = 0; i < N; i++) {
    const ti = sim.type[i] | 0
    forEachNeighbor(sim, i, (j) => {
      if (j <= i) return

      // geometry
      let dx = sp.wrap ? torusDelta(sim.x[i], sim.x[j]) : sim.x[j] - sim.x[i]
      let dy = sp.wrap ? torusDelta(sim.y[i], sim.y[j]) : sim.y[j] - sim.y[i]
      const r2 = dx * dx + dy * dy
      if (r2 === 0) return
      const r = Math.sqrt(r2)
      if (r > R) return
      const invr = 1 / r
      const ux = dx * invr
      const uy = dy * invr

      // interaction weights (allow non-reciprocity)
      const tj = (sim.type[j] | 0) as number
      let aij = sim.A[ti][tj]
      let aji = sim.A[tj][ti]

      // optional gating: only allow attraction if mutual
      if (sp.mutualOnly) {
        const mutualPos = aij > 0 && aji > 0
        if (!mutualPos) {
          aij = 0
          aji = 0
        }
      }

      // force on i from j (along +u) and on j from i (along -u)
      const fij = accelMag(aij, r, rMin)
      const fji = accelMag(aji, r, rMin)

      if (fij !== 0) {
        sim.fx[i] += fij * ux
        sim.fy[i] += fij * uy
      }
      if (fji !== 0) {
        sim.fx[j] -= fji * ux
        sim.fy[j] -= fji * uy
      }

      // settling: radial dashpot for mutually attracted neighbors within [rMin, settleR]
      if (sp.settleEnabled) {
        const mutualPos = aij > 0 && aji > 0
        if (mutualPos && r > rMin && r < sp.settleR) {
          const vRelRad =
            (sim.vx[i] - sim.vx[j]) * ux + (sim.vy[i] - sim.vy[j]) * uy

          const cCrit = 2 / dt
          const c = Math.min(Math.max(sp.settleK, 0), cCrit)

          const t = (r - rMin) / Math.max(1e-6, sp.settleR - rMin)
          const w = 1 - smoothstep01(t) // 1 near rMin, 0 at settleR
          if (w > 0) {
            const fDamp = -c * vRelRad * w
            const fx = fDamp * ux
            const fy = fDamp * uy
            sim.fx[i] += fx
            sim.fy[i] += fy
            sim.fx[j] -= fx
            sim.fy[j] -= fy
          }
        }
      }
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
  dpr: number
  // square viewport centered within the canvas
  viewX: number
  viewY: number
  viewSize: number
  scale: number // viewSize / WORLD_SIZE
}

/** Canvas matches container; content uses a centered square viewport (no stretch). */
function setupCanvasViewport(
  canvas: HTMLCanvasElement,
  widthPx: number,
  heightPx: number
): Renderer {
  const dpr = Math.max(1, (window.devicePixelRatio as number) || 1)

  canvas.style.width = `${widthPx}px`
  canvas.style.height = `${heightPx}px`
  canvas.width = Math.floor(widthPx * dpr)
  canvas.height = Math.floor(heightPx * dpr)

  const ctx = canvas.getContext("2d")!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const viewSize = Math.min(widthPx, heightPx)
  const viewX = Math.floor((widthPx - viewSize) / 2)
  const viewY = Math.floor((heightPx - viewSize) / 2)
  const scale = viewSize / WORLD_SIZE

  return {
    ctx,
    width: widthPx,
    height: heightPx,
    dpr,
    viewX,
    viewY,
    viewSize,
    scale,
  }
}

function worldToScreen(x: number, y: number, r: Renderer) {
  const sx = r.viewX + (x + 1) * 0.5 * r.viewSize
  const sy = r.viewY + (y + 1) * 0.5 * r.viewSize
  return [sx, sy] as const
}

function draw(sim: Sim, r: Renderer) {
  const { ctx, width, height, viewX, viewY, viewSize } = r
  const { showVel, showGrid } = sim.spec.overlays

  // clear whole canvas
  ctx.fillStyle = "#0a0a0a"
  ctx.fillRect(0, 0, width, height)

  // optional grid inside square viewport
  if (showGrid) {
    ctx.save()
    ctx.beginPath()
    ctx.rect(viewX, viewY, viewSize, viewSize)
    ctx.clip()

    ctx.strokeStyle = "rgba(255,255,255,0.06)"
    ctx.lineWidth = 1
    const step = (sim.spec.cellSize * viewSize) / WORLD_SIZE

    for (let x = viewX; x <= viewX + viewSize + 0.5; x += step) {
      ctx.beginPath()
      ctx.moveTo(x, viewY)
      ctx.lineTo(x, viewY + viewSize)
      ctx.stroke()
    }
    for (let y = viewY; y <= viewY + viewSize + 0.5; y += step) {
      ctx.beginPath()
      ctx.moveTo(viewX, y)
      ctx.lineTo(viewX + viewSize, y)
      ctx.stroke()
    }
    ctx.restore()
  }

  // particles
  const radiusPx = 2
  const N = Math.min(sim.spec.N, sim.x.length)
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

  // HUD text inside viewport
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
    `mutual=${sim.spec.mutualOnly ? 1 : 0}  settle=${
      sim.spec.settleEnabled ? 1 : 0
    }  k=${sim.spec.settleK.toFixed(2)}  sR=${sim.spec.settleR.toFixed(2)}`,
    `A[0,*]=[${A0 ?? ""}]`,
  ]
  let ty = viewY + 16
  for (const ln of lines) {
    ctx.fillText(ln, viewX + 8, ty)
    ty += 14
  }

  // viewport border (optional)
  ctx.strokeStyle = "rgba(255,255,255,0.08)"
  ctx.strokeRect(viewX + 0.5, viewY + 0.5, viewSize - 1, viewSize - 1)
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
  // map -1..1 to blue → black → red
  const t = (v + 1) / 2 // 0..1
  let r, g, b
  if (t < 0.5) {
    const f = t / 0.5
    r = 0
    g = 0
    b = Math.round(255 * (1 - f))
  } else {
    const f = (t - 0.5) / 0.5
    r = Math.round(255 * f)
    g = 0
    b = 0
  }
  return `rgb(${r}, ${g}, ${b})`
}

function cycleValue(v: number, dir: number) {
  const steps = [-1, 0, 1]
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

      <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 8 }}>
        Left-click: cycle up. Right-click: cycle down. Values: −1, 0, +1.
      </div>

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
                  style={{
                    ...cellBtn,
                    background:
                      Math.abs(A[i][j]) > 1e-6
                        ? valueToSwatch(A[i][j])
                        : "transparent",
                    border: "1px solid #ffffff",
                  }}
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

/** Responsive renderer tied to container via ResizeObserver; no stretch (square viewport). */
function useRendererViewport() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const rendererRef = useRef<Renderer | null>(null)

  useEffect(() => {
    const refresh = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      const rect = container.getBoundingClientRect()
      rendererRef.current = setupCanvasViewport(canvas, rect.width, rect.height)
    }

    const ro = new ResizeObserver(() => refresh())
    if (containerRef.current) ro.observe(containerRef.current)
    // initial sizing
    refresh()
    return () => ro.disconnect()
  }, [])

  return { canvasRef, containerRef, rendererRef }
}

// ========================= Component =========================
export default function App() {
  // Use SPEC directly; do not override A on first render.
  const [spec, setSpec] = useState<Spec>({ ...SPEC })
  const [seed, setSeed] = useState<number>(SPEC.seed)
  const [paused, setPaused] = useState(false)
  const [showRules, setShowRules] = useState(true)
  const [fps, setFps] = useState(0)

  const simRef = useRef<Sim | null>(null)
  const { canvasRef, containerRef, rendererRef } = useRendererViewport()

  // -------- Resolve matrix exactly once per K or when genMatrix flag is true.
  // Replace your “Resolve matrix…” effect with this:
  useEffect(() => {
    const need =
      spec.genMatrix ||
      spec.A.length !== spec.K ||
      spec.A.some((r) => r.length !== spec.K)

    if (!need) return

    const fromLS = loadMatrixFromLS(spec.K)
    const nextA = fromLS ?? genRingPreset(spec.K)

    setSpec((s) => ({ ...s, A: nextA, genMatrix: false }))
  }, [spec.K, spec.genMatrix])

  // -------- Initialize / Re-initialize simulation when core layout changes.
  useEffect(() => {
    simRef.current = initSim(spec, seed)
  }, [
    seed,
    spec.N,
    spec.K,
    spec.cellSize,
    spec.wrap,
    spec.R,
    spec.rMin,
    spec.dt,
    spec.drag,
    spec.vMax,
  ])

  // keep runtime sim reading latest spec without forcing reset
  useEffect(() => {
    if (simRef.current) simRef.current.spec = spec
  }, [spec])

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
    const newN = clamp(spec.N + delta, 0, 50000) // allow 0 safely
    setSpec((s) => ({ ...s, N: newN }))
  }
  const incK = (delta: number) => {
    const maxK = TYPE_COLORS.length
    const newK = clamp(spec.K + delta, 2, maxK)
    setSpec((s) => ({ ...s, K: newK, genMatrix: true }))
  }

  const applyRingPreset = () =>
    applyMatrix(genRingPreset(spec.K, 0.9, 0.6, 0.0))

  const inc = (v: number, d: number, lo: number, hi: number) =>
    clamp(v + d, lo, hi)

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
            <button onClick={() => incN(-100)} style={chipStyle}>
              −100
            </button>
            <span>{spec.N}</span>
            <button onClick={() => incN(+100)} style={chipStyle}>
              +100
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
              title="Draw velocity vectors"
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
              title="Show spatial grid"
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
              title="World wraps on edges"
            >
              <input
                type="checkbox"
                checked={spec.wrap}
                onChange={(e) => setSpec({ ...spec, wrap: e.target.checked })}
              />
              Wrap
            </label>
          </div>

          {/* Behavior toggles */}
          <div
            style={{
              display: "inline-flex",
              gap: 10,
              alignItems: "center",
              marginLeft: 12,
            }}
          >
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="Only allow attraction when A[i][j] and A[j][i] are both > 0"
            >
              <input
                type="checkbox"
                checked={spec.mutualOnly}
                onChange={(e) =>
                  setSpec({ ...spec, mutualOnly: e.target.checked })
                }
              />
              Mutual
            </label>
            <label
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="Enable settling (radial damping for mutually attracted pairs)"
            >
              <input
                type="checkbox"
                checked={spec.settleEnabled}
                onChange={(e) =>
                  setSpec({ ...spec, settleEnabled: e.target.checked })
                }
              />
              Settle
            </label>

            <div
              style={{ display: "inline-flex", gap: 6, alignItems: "center" }}
            >
              <span style={{ opacity: 0.7 }}>k:</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleK: inc(s.settleK, -0.01, 0, 20),
                  }))
                }
                style={chipStyle}
              >
                −
              </button>
              <span>{spec.settleK.toFixed(2)}</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleK: inc(s.settleK, +0.01, 0, 20),
                  }))
                }
                style={chipStyle}
              >
                +
              </button>
            </div>

            <div
              style={{ display: "inline-flex", gap: 6, alignItems: "center" }}
            >
              <span style={{ opacity: 0.7 }}>sR:</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleR: inc(s.settleR, -0.01, 0.05, 1),
                  }))
                }
                style={chipStyle}
              >
                −
              </button>
              <span>{spec.settleR.toFixed(2)}</span>
              <button
                onClick={() =>
                  setSpec((s) => ({
                    ...s,
                    settleR: inc(s.settleR, +0.01, 0.05, 1),
                  }))
                }
                style={chipStyle}
              >
                +
              </button>
            </div>
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

      {/* Canvas area (middle row) */}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "100%",
            display: "block",
            background: "#0a0a0a",
          }}
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
