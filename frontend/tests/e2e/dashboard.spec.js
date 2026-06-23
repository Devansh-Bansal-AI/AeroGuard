import { test, expect } from '@playwright/test';

test.describe('AeroGuard Dashboard E2E', () => {
  test('should load the dashboard and display fleet overview', async ({ page }) => {
    // Navigate to the app
    await page.goto('/');

    // Check title
    await expect(page).toHaveTitle(/AeroGuard/i);

    // Verify main navigation
    const nav = page.locator('nav.nav-tabs');
    await expect(nav).toBeVisible();

    // Verify Fleet Overview tab is active by default
    await expect(page.locator('a.nav-tab.active').first()).toContainText('Fleet Dashboard');

    // Verify Fleet Overview stats are visible
    const overview = page.locator('.fleet-overview');
    await expect(overview).toBeVisible();
    await expect(overview.locator('.fleet-stat').first()).toBeVisible();
  });

  test('should navigate to Diagnostic Cockpit', async ({ page }) => {
    await page.goto('/');

    // Click on Diagnostic Cockpit tab
    await page.locator('a.nav-tab:has-text("Diagnostics")').click();

    // Verify URL change
    await expect(page).toHaveURL(/.*\/diagnostics/);

    // Verify page content loads
    await expect(page.locator('h1')).toContainText('Diagnostic Cockpit');
    await expect(page.locator('.diagnostic-grid')).toBeVisible();
  });

  test('should navigate to Edge Metrics', async ({ page }) => {
    await page.goto('/');

    // Click on Edge Metrics tab
    await page.locator('a.nav-tab:has-text("Edge Metrics")').click();

    // Verify URL change
    await expect(page).toHaveURL(/.*\/edge/);

    // Verify page content loads
    await expect(page.locator('h1')).toContainText('Edge Intelligence Metrics');
  });
});
