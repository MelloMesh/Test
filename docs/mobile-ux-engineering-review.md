# Mobile UX Engineering Review
**Project:** Infant Vaccine Evidence Atlas (MelloMesh/Test)  
**Date:** 2026-10-08 / publication UTC 2026-10-09  
**Reviewer provenance:** AI-assisted senior UI engineering critique based on Apple-inspired usability principles. **Not** an external human review, and **not** performed by a former Apple, Google, or other company employee.

## Intended experience
- Primary task: understand the research within a screen or two, then drill into underlying studies.
- Core navigation: persistent four-destination mobile tab bar, clear menu for nine deeper dashboard sections, dedicated comparative/global/holistic research pages.
- Visual direction: typography-led, restrained palette, structured white space, clear evidence qualification, minimal ornamental UI.

## Critical findings and dispositions
| Severity | Finding | Disposition |
|---|---|---|
| P0 | Intermediate overview revision accidentally omitted closing stylesheet/head/body markup | **Fixed** by reconstructing from verified earlier HTML and integrating executive overview. |
| P1 | Nine horizontally scrolling section tabs were difficult on narrow phones | **Fixed** with bottom quick navigation and a scannable full menu. |
| P1 | Reaching comparative analysis required discovering a deep link | **Fixed** with persistent Compare tab and explicit main Overview link. |
| P1 | Source context was too far from graphics | **Fixed** with inline original research / regulator references on four overview visualizations. |
| P1 | Touch targets were small or dense | **Improved** 44px utility controls, ≥52px tab items and ≥58px menu links. |
| P2 | Decorative glyphs could be read as text by screen readers | **Fixed** with aria-hidden on purely decorative mobile icons. |
| P2 | Keyboard access lacked a bypass for long navigation | **Fixed** with visible-on-focus Skip to research content link. |
| P2 | Dismissal behavior for an open menu via keyboard was unclear | **Fixed** with Escape dismissal and focus return. |
| P2 | Risk of dynamic scripts overwriting server-rendered overview content | **Fixed** so script no longer regenerates summary metric HTML. |
| P2 | Scientific figures could be mistaken for directly comparable probabilities | **Improved** with chart-specific endpoints, denominators and interpretive warnings. |

## Automated source-level checks completed
- HTML retains a complete document skeleton and one main document element.
- Embedded JavaScript passes syntax compilation.
- Four overview charts and three evidence interpretation panels present.
- All nine primary dashboard sections represented in the mobile menu.
- Main links lead to comparative, global trial/CMC and holistic sections.
- Source citations attached to chart panels and summary.
- CSS contains safe-area bottom padding and a narrow-screen responsive breakpoint.
- Keyboard skip link and reduced-motion preference are supported.
- GitHub's native Pages deployment has reported successful builds for the previous and initial mobile updates.

## Remaining QA caveats
- **Actual Safari/iOS device validation has not been completed in this review.** The checks above are structural/logic tests rather than full real-device interaction evidence.
- Before considering accessibility conformance established, run VoiceOver screen-reader inspection, contrast measurement in both themes, landscape/small-screen zoom, keyboard traversal and screen-reader link purpose audits.
- Additional manual acceptance tests: verify first-screen readability at 375x812, 390x844 and 320x568; rotate display; test large system font setting; open research menu and click into every destination; return from a separate research page; use dark mode.
- Independent external human senior-engineer review has **not** occurred.

## Release acceptance
Code-level mobile navigation review: **Pass with open real-device QA items**.  
Independent expert credentialed sign-off: **Not performed**.  
Research causality/clinical audit remains separate from this visual UX review.
