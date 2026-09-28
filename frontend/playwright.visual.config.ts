import { defineConfig } from '@playwright/test'

// Screenshot comparisons. Fonts and rendering differ between operating systems,
// so these run only inside the Playwright Docker image, locally
// (npm run test:visual:docker) and in CI. Keep the image tag in package.json and
// .github/workflows/ci.yml in step with the installed @playwright/test version.
export default defineConfig({
  testDir: './visual',
  testMatch: '*.visual.ts',
  snapshotPathTemplate: '{testDir}/__screenshots__/{arg}{ext}',
  use: { baseURL: 'http://localhost:5173' },
  expect: {
    toHaveScreenshot: { animations: 'disabled', caret: 'hide', maxDiffPixelRatio: 0.01 },
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 5173 --strictPort',
    url: 'http://localhost:5173',
    reuseExistingServer: false,
  },
})
