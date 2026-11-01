/// <reference types="node" />
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';

describe('navigation architecture and route structure', () => {
  const rootAppDir = path.resolve('src/app');
  const tabsDir = path.resolve('src/app/(tabs)');

  it('1. Board Review has a registered root-stack route in src/app/_layout.tsx', () => {
    const rootLayoutPath = path.join(rootAppDir, '_layout.tsx');
    assert.equal(fs.existsSync(rootLayoutPath), true, 'Root _layout.tsx must exist');

    const content = fs.readFileSync(rootLayoutPath, 'utf8');
    assert.match(content, /Stack/, 'Root layout must use Stack navigator');
    assert.match(content, /<Stack\.Screen\s+name="\(tabs\)"/, 'Root stack must register (tabs)');
    assert.match(content, /<Stack\.Screen\s+name="board-review"/, 'Root stack must register board-review');
    assert.equal(fs.existsSync(path.join(rootAppDir, 'board-review.tsx')), true, 'src/app/board-review.tsx must exist');
  });

  it('2. The five existing tabs remain registered and retain their public paths', () => {
    const tabsLayoutPath = path.join(tabsDir, '_layout.tsx');
    assert.equal(fs.existsSync(tabsLayoutPath), true, 'Tabs _layout.tsx must exist');

    const appTabsPath = path.resolve('src/components/app-tabs.tsx');
    assert.equal(fs.existsSync(appTabsPath), true, 'src/components/app-tabs.tsx must exist');

    const appTabsContent = fs.readFileSync(appTabsPath, 'utf8');

    // Expected 5 tabs in AppTabs
    const expectedTabs = [
      { name: 'index', label: 'Board', publicPath: '/' },
      { name: 'camera-setup', label: 'Setup', publicPath: '/camera-setup' },
      { name: 'live-game', label: 'Live', publicPath: '/live-game' },
      { name: 'camera-lab', label: 'Lab', publicPath: '/camera-lab' },
      { name: 'study', label: 'Study', publicPath: '/study' },
    ];

    for (const tab of expectedTabs) {
      assert.match(
        appTabsContent,
        new RegExp(`<NativeTabs\\.Trigger\\s+name="${tab.name}"`),
        `Tab trigger "${tab.name}" must be registered in NativeTabs`,
      );
      assert.match(
        appTabsContent,
        new RegExp(`<NativeTabs\\.Trigger\\.Label>${tab.label}</NativeTabs\\.Trigger\\.Label>`),
        `Tab label "${tab.label}" must match for trigger "${tab.name}"`,
      );
      assert.equal(
        fs.existsSync(path.join(tabsDir, `${tab.name}.tsx`)),
        true,
        `Tab file src/app/(tabs)/${tab.name}.tsx must exist`,
      );
    }
  });

  it('3. Board Review is not an extra tab trigger in NativeTabs', () => {
    const appTabsContent = fs.readFileSync(path.resolve('src/components/app-tabs.tsx'), 'utf8');
    assert.doesNotMatch(
      appTabsContent,
      /name="board-review"/,
      'board-review must NOT be declared as a tab trigger in NativeTabs',
    );
  });

  it('4. Camera Setup navigation targets /board-review', () => {
    const cameraSetupContent = fs.readFileSync(path.join(tabsDir, 'camera-setup.tsx'), 'utf8');
    // Manual review button
    assert.match(
      cameraSetupContent,
      /router\.push\('\/board-review'\)/,
      'Camera Setup must route to /board-review for manual review',
    );
    // Captured image review button
    assert.match(
      cameraSetupContent,
      /pathname:\s*['"]\/board-review['"]/,
      'Camera Setup must route to /board-review with imageUri param',
    );
  });

  it('5. Review-to-Study navigation targets /study with validated FEN', () => {
    const boardReviewContent = fs.readFileSync(path.join(rootAppDir, 'board-review.tsx'), 'utf8');
    assert.match(
      boardReviewContent,
      /pathname:\s*['"]\/study['"]/,
      'Board Review must route to /study on handoff',
    );
    assert.match(
      boardReviewContent,
      /params:\s*\{\s*fen:\s*validationResult\.fen\s*\}/,
      'Board Review must pass validated FEN in route params',
    );
  });

  it('6. Back navigation from Board Review and Camera Setup uses router.back()', () => {
    const cameraSetupContent = fs.readFileSync(path.join(tabsDir, 'camera-setup.tsx'), 'utf8');
    assert.match(
      cameraSetupContent,
      /router\.back\(\)/,
      'Camera Setup must support router.back() to return to chessboard',
    );

    // In a stack navigator, native back or router.back() pops the stack
    const rootLayoutContent = fs.readFileSync(path.join(rootAppDir, '_layout.tsx'), 'utf8');
    assert.match(rootLayoutContent, /<Stack/, 'Root Stack allows natural back navigation to previous screen');
  });

  it('7. Public paths mapping conforms to Expo Router route-group rules', () => {
    // Under Expo Router:
    // (tabs)/index.tsx       -> /
    // (tabs)/camera-setup.tsx -> /camera-setup
    // (tabs)/live-game.tsx   -> /live-game
    // (tabs)/camera-lab.tsx  -> /camera-lab
    // (tabs)/study.tsx       -> /study
    // board-review.tsx       -> /board-review
    function resolvePublicPath(filePath: string): string {
      const normalized = filePath.replace(/\\/g, '/');
      const relative = normalized.replace(/^.*src\/app\//, '').replace(/\.tsx$/, '');
      const segments = relative.split('/').filter((s) => !s.startsWith('(') || !s.endsWith(')'));
      if (segments.length === 1 && segments[0] === 'index') {
        return '/';
      }
      return '/' + segments.join('/');
    }

    assert.equal(resolvePublicPath('src/app/(tabs)/index.tsx'), '/');
    assert.equal(resolvePublicPath('src/app/(tabs)/camera-setup.tsx'), '/camera-setup');
    assert.equal(resolvePublicPath('src/app/(tabs)/live-game.tsx'), '/live-game');
    assert.equal(resolvePublicPath('src/app/(tabs)/camera-lab.tsx'), '/camera-lab');
    assert.equal(resolvePublicPath('src/app/(tabs)/study.tsx'), '/study');
    assert.equal(resolvePublicPath('src/app/board-review.tsx'), '/board-review');
  });
});
