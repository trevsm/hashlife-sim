import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // use docs for build instead of dist
  build: {
    outDir: "docs",
  },
})
