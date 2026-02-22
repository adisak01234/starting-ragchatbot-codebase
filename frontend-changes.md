# Frontend Changes: Dark/Light Mode Toggle Button & Light Theme

## Features
1. Sun/moon icon toggle button for switching between dark (default) and light themes
2. Full light theme variant with WCAG AA–compliant colors and component-level overrides

---

## Files Modified

### `frontend/index.html`
- Added a `<button id="themeToggle">` element with two inline SVG icons:
  - **Sun icon** (`.icon-sun`) — visible in dark mode; click to go light
  - **Moon icon** (`.icon-moon`) — visible in light mode; click to go dark
- Button has `aria-label` and `title` attributes for accessibility

### `frontend/style.css`

#### 1. `[data-theme="light"]` CSS variable block
Overrides all `:root` custom properties for the light palette:

| Token | Value | Notes |
|---|---|---|
| `--background` | `#f8fafc` | Slate-50 page bg |
| `--surface` | `#ffffff` | Card / sidebar bg |
| `--surface-hover` | `#f1f5f9` | Hover tint |
| `--text-primary` | `#0f172a` | Contrast ≈ 17:1 on bg |
| `--text-secondary` | `#475569` | Contrast ≈ 5.9:1 on bg |
| `--primary-color` | `#2563eb` | Same as dark (brand) |
| `--primary-hover` | `#1d4ed8` | Darker hover shade |
| `--border-color` | `#cbd5e1` | Slate-300 |
| `--shadow` | multi-layer subtle | Lighter than dark shadow |
| `--user-message` | `#2563eb` | Blue bubble (unchanged) |
| `--assistant-message` | `#f1f5f9` | Light gray bubble |
| `--focus-ring` | `rgba(37,99,235,0.25)` | Accessible focus outline |
| `--welcome-bg` | `#eff6ff` | Blue-50 banner |
| `--welcome-border` | `#bfdbfe` | Blue-200 border |

#### 2. Component-level light-theme overrides
Hardcoded dark-specific colors that CSS variables alone can't fix:

- **`.source-tag`** — text `#1e40af` (contrast ≈ 7.2:1 on white), lighter blue bg/border tints
- **`a.source-tag:hover`** — slightly stronger bg/border on hover, text `#1d4ed8`
- **`.message-content code`** — bg `rgba(15,23,42,0.06)` (light gray tint), text `#1e293b`
- **`.message-content pre`** — bg `#f1f5f9`, adds visible `border: 1px solid #e2e8f0`
- **`.message.welcome-message .message-content`** — softer shadow, border uses `--border-color`

#### 3. Smooth theme transitions
`transition: background-color, color, border-color (0.3s ease)` on body, sidebar, chat areas, messages, input, buttons, and sidebar items.

#### 4. Theme toggle button
Fixed `top: 1rem; right: 1rem` (z-index 200), 40×40 px circle. Icons use `position: absolute` with `opacity` + `rotate/scale` transitions for a smooth crossfade animation.

### `frontend/index.html` (JS functionality update)
- Added an **anti-FOUC inline script** in `<head>` (runs before first paint):
  ```js
  if (localStorage.getItem('theme') === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
  }
  ```
  This prevents a brief flash of dark theme when a returning user has light mode saved.

### `frontend/script.js`

#### `initThemeToggle()`
Called from `DOMContentLoaded`. Reads the `data-theme` attribute already set by the `<head>` script (avoids a second `localStorage` read) to sync the button label, then attaches a click handler that:
1. Reads current `data-theme` from `document.documentElement`
2. Sets `data-theme` to the opposite value (`"light"` ↔ `"dark"`) — always explicit, never just removes the attribute
3. Persists the new value to `localStorage`
4. Calls `syncToggleLabel()` to update `aria-label` and `title`

#### `syncToggleLabel(btn)`
Single-responsibility helper that sets the button's `aria-label` and `title` to match the active theme. Called both on init and after every toggle click.

#### Theme switching flow (end-to-end)
1. Page loads → `<head>` script applies `data-theme` from `localStorage` (no flash)
2. CSS `[data-theme="light"]` block overrides all custom properties instantly
3. `DOMContentLoaded` → `initThemeToggle()` syncs the button label
4. User clicks toggle → `data-theme` flips → CSS transitions animate the color change → new value saved to `localStorage`
