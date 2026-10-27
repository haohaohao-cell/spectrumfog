import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" —— GitHub Pages 通常部署在 /<repo>/ 子路径下，相对路径才能在任意前缀下工作。
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    assetsDir: "assets",
  },
  // vitest：只跑 src 下的 *.test.ts(x)
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
