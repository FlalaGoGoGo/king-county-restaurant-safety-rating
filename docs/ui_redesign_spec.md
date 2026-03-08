# Unified UI Redesign Spec

Updated: 2026-03-04

## 1. Design Direction
- Style family: Modern SaaS Bento
- Selected variant: Calm Teal Bento (Final, confirmed 2026-03-04)
- Scope: HTML export + Streamlit UI alignment

## 2. Design Tokens
- Colors:
  - `--bg`: `#e8f4f5`
  - `--bg2`: `#d7ecee`
  - `--surface`: `#ffffff`
  - `--surface-soft`: `#f5fbfb`
  - `--ink`: `#0f2d36`
  - `--muted`: `#5d7d85`
  - `--brand`: `#0d8a98`
  - `--brand-strong`: `#08727f`
  - `--brand2`: `#27c2a8`
  - `--line`: `#cae0e4`
  - `--line-strong`: `#9fc8cf`
- Radius:
  - `8px`, `12px`, `18px`, `24px`
- Shadow:
  - soft cards, medium panel, strong hero elevation

## 3. Typography
- Primary stack: Avenir Next, IBM Plex Sans, Trebuchet MS, Segoe UI
- Heading: 700+ weight, mild tracking (`0.01em`)
- Body: compact but readable line-height (`1.5`)

## 4. Component Rules
- Cards: gradient-white surface, subtle border, soft shadow.
- Tabs: container-backed with active gradient state.
- Filters: consistent input/select radius and focus rings.
- Tables: sticky header, hover feedback, selected-row accent.
- KPI cards: concise title + stronger value emphasis.

## 5. Motion
- Use restrained transitions (`140ms` to `180ms`) on hover/select interactions.
- Avoid heavy animation in data-dense tables.

## 5.1 Mobile refinement (2026-03-04)
- HTML: added small-screen breakpoint (`max-width: 640px`) for tighter shell spacing, 1-column tab stack, 1-column KPI grid, compact table density, and reduced map height.
- Streamlit: added small-screen CSS (`max-width: 768px`) for compact tab controls, full-width action buttons, and tighter card/table radius and spacing.

## 6. Accessibility Baseline
- Preserve readable contrast between text and background.
- Keep clear focus indicators for form controls.
- Avoid low-contrast pastel text on tinted cards.

## 7. Two-Surface Alignment
- HTML and Streamlit share same visual token intent.
- Streamlit uses CSS injection to mirror key token choices.
