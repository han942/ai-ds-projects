"""Respectful Playwright crawler for public DiningCode restaurant pages.

The crawler checks robots.txt before browsing and blocks any first-party
request disallowed there. It does not log in, evade access controls, or use
stealth/proxy techniques. CSV rows are flushed as each review batch appears;
an unfinished crawl can be resumed from its checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.async_api import Error as PlaywrightError, async_playwright


BASE_URL = "https://www.diningcode.com"
DEFAULT_START_URL = "https://www.diningcode.com/list.dc?query=%EA%B7%BC%EC%B2%98"
NATIONAL_REGION_NAMES = (
    "서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)
NATIONAL_REGIONS_IDENTITY = "diningcode:national-foodrank-17-regions-v1"
USER_AGENT = "rating-recsys-research-crawler"
SEOUL = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "data"
CHECKPOINT_SCHEMA_VERSION = 7
REVIEW_CARD_SELECTOR = (
    "div.latter-graph, [data-review-id], "
    'article[class*="review"], li[class*="review"]'
)
REVIEW_TEXT_SELECTOR = (
    "div.review_contents.btxt, p.review_contents.btxt, "
    '[class*="review_contents"], [class*="reviewContents"]'
)

# Keep legacy DiningCode columns and append source identifiers.
CSV_FIELDS = [
    "item_name",
    "item_area",
    "item_avg_rating",
    "item_spec_area",
    "user_name",
    "user_tot_avg_rating",
    "user_tot_rating_num",
    "user_tot_follow_num",
    "user_rating",
    "user_query",
    "taste",
    "price",
    "service",
    "menu",
    "date",
    "restaurant_id",
    "source_url",
    "crawl_timestamp",
    "review_text_complete",
]


class CrawlStopped(RuntimeError):
    """Raised when the source blocks access or robots.txt disallows a URL."""


def now_iso() -> str:
    return datetime.now(SEOUL).isoformat(timespec="seconds")


def national_region_sources() -> list[tuple[str, str]]:
    return [
        (region, f"{BASE_URL}/foodrank/{urllib.parse.quote(region)}")
        for region in NATIONAL_REGION_NAMES
    ]


def load_robots() -> urllib.robotparser.RobotFileParser:
    robots_url = f"{BASE_URL}/robots.txt"
    request = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 200:
                raise CrawlStopped(f"robots.txt returned HTTP {response.status}")
            contents = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError) as exc:
        raise CrawlStopped(f"Could not read {robots_url}: {exc}") from exc

    parser = urllib.robotparser.RobotFileParser(robots_url)
    parser.parse(contents.splitlines())
    return parser


def assert_allowed(robots: urllib.robotparser.RobotFileParser, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "www.diningcode.com":
        raise CrawlStopped(f"Refusing URL outside www.diningcode.com: {url}")
    if not robots.can_fetch(USER_AGENT, url):
        raise CrawlStopped(f"robots.txt disallows {url}")


async def install_robots_guard(
    context, robots, blocked_paths: set[str], blocked_events: list[str]
) -> None:
    async def guard(route):
        url = route.request.url
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname != "www.diningcode.com":
            await route.continue_()
            return
        if robots.can_fetch(USER_AGENT, url):
            await route.continue_()
            return
        blocked_paths.add(parsed.path)
        blocked_events.append(parsed.path)
        await route.abort("blockedbyclient")

    await context.route("**/*", guard)


async def get_allowed_page(page, robots, url: str, *, wait_ms: int):
    assert_allowed(robots, url)
    response = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    if response is None:
        raise CrawlStopped(f"No HTTP response for {url}")
    if response.status == 403:
        raise CrawlStopped(
            f"DiningCode returned HTTP 403 for {url}; stopping without retry or bypass."
        )
    if response.status in {401, 429}:
        raise CrawlStopped(f"DiningCode returned HTTP {response.status} for {url}; stopping.")
    if response.status >= 400:
        raise CrawlStopped(f"DiningCode returned HTTP {response.status} for {url}")
    await page.wait_for_timeout(wait_ms)
    return response


async def collect_visible_profile_links(page, max_scrolls: int, wait_ms: int):
    links: list[str] = []
    known: set[str] = set()
    stable_rounds = 0
    for _ in range(max_scrolls):
        found = await page.locator("a[href*='profile.php?rid=']").evaluate_all(
            "els => els.map(a => a.href)"
        )
        old_count = len(known)
        for url in found:
            if url not in known:
                known.add(url)
                links.append(url)
        if len(known) == old_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
        if stable_rounds >= 3:
            return links, True
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(wait_ms)
    return links, False


async def find_next_listing_url(page, current_url: str) -> str | None:
    return await page.evaluate(
        """currentUrl => {
          const anchors = [...document.querySelectorAll(
            'a[rel~="next"], a[aria-label*="다음"], a[title*="다음"], a'
          )];
          for (const anchor of anchors) {
            const label = [
              anchor.innerText || '',
              anchor.getAttribute('aria-label') || '',
              anchor.getAttribute('title') || ''
            ].join(' ').replace(/\\s+/g, ' ').trim();
            const disabled = anchor.getAttribute('aria-disabled') === 'true' ||
              /disabled/.test(anchor.className || '');
            if (disabled) continue;
            const isNext = anchor.rel.split(/\\s+/).includes('next') ||
              /(^|\\s)다음(\\s|$)/.test(label);
            if (!isNext || !anchor.href) continue;
            const next = new URL(anchor.href, currentUrl);
            if (next.origin !== location.origin) continue;
            if (next.href !== currentUrl) return next.href;
          }
          return null;
        }""",
        current_url,
    )


async def collect_profile_links(
    page,
    robots,
    *,
    start_url: str,
    max_scrolls: int,
    max_pages: int,
    wait_ms: int,
):
    links: list[str] = []
    known: set[str] = set()
    visited_pages: set[str] = set()
    all_pages_scrolled = True
    listing_complete = False
    listing_error = None
    current_url = start_url
    pages_visited = 0

    while current_url and current_url not in visited_pages and pages_visited < max_pages:
        visited_pages.add(current_url)
        pages_visited += 1
        print(f"Listing page {pages_visited}/{max_pages}: {current_url}", file=sys.stderr, flush=True)
        if pages_visited > 1:
            try:
                await get_allowed_page(page, robots, current_url, wait_ms=wait_ms)
            except (CrawlStopped, PlaywrightError) as exc:
                listing_error = str(exc)
                break

        page_links, scrolled_to_stable = await collect_visible_profile_links(
            page, max_scrolls=max_scrolls, wait_ms=wait_ms
        )
        all_pages_scrolled = all_pages_scrolled and scrolled_to_stable
        for url in page_links:
            if url not in known:
                known.add(url)
                links.append(url)

        next_url = await find_next_listing_url(page, current_url)
        if not next_url:
            listing_complete = all_pages_scrolled
            break
        try:
            assert_allowed(robots, next_url)
        except CrawlStopped as exc:
            listing_error = str(exc)
            break
        if next_url in visited_pages:
            listing_complete = all_pages_scrolled
            break
        if pages_visited >= max_pages:
            listing_error = f"Reached --max-list-pages={max_pages} while a next page was available."
            break
        current_url = next_url

    return links, listing_complete, pages_visited, listing_error


async def read_profile(page, url: str, crawl_timestamp: str) -> list[dict[str, str]]:
    """Extract one structured row from each visitor review card currently in the DOM."""
    return await page.evaluate(
        r"""({profileUrl, crawlTimestamp}) => {
          const clean = value => (value || '').replace(/\s+/g, ' ').trim();
          const text = (root, selectors) => {
            for (const selector of selectors) {
              const node = root.querySelector(selector);
              const value = clean(node?.innerText || node?.textContent || '');
              if (value) return value;
            }
            return '';
          };
          const restaurant = {
            name: text(document, ['h1']),
            area: text(document, ['a.area', '[class*="location"]', '[class*="area"]']),
            average: text(document, ['span.point > strong', '[itemprop="ratingValue"]']),
            address: text(document, ['span.profile_jibun', 'address', '[itemprop="streetAddress"]']),
          };

          // The legacy page used div.latter-graph for one visitor review.
          // Prefer that known card selector, then fall back to review-shaped nodes.
          const legacyCards = [...document.querySelectorAll('div.latter-graph')];
          const fallbackCards = [...document.querySelectorAll(
            '[data-review-id], article[class*="review"], li[class*="review"]'
          )];
          const cards = legacyCards.length ? legacyCards : fallbackCards;

          return cards.map(card => {
            const userStats = [...card.querySelectorAll('p > span.info > span')]
              .map(node => clean(node.innerText || node.textContent));
            const scoreDetails = [...card.querySelectorAll('span.sub_title')]
              .map(node => clean(node.innerText || node.textContent));
            const reviewTextNode = card.querySelector(
              'div.review_contents.btxt, p.review_contents.btxt, ' +
              '[class*="review_contents"], [class*="reviewContents"]'
            );
            const reviewText = clean(
              reviewTextNode?.innerText || reviewTextNode?.textContent || ''
            );
            const visibleTextMoreControl = reviewTextNode && [...reviewTextNode.querySelectorAll(
              'a.more-btn-inline, button, [role="button"], [onclick]'
            )].some(node => {
              const style = getComputedStyle(node);
              const visible = style.display !== 'none' && style.visibility !== 'hidden' &&
                node.getClientRects().length > 0;
              return visible && /^(?:\.{3}|…)?\s*더보기$/.test(
                clean(node.innerText || node.textContent || '')
              );
            });
            const truncatedEnding = /(?:\.{3}|…)\s*더보기$/.test(reviewText);
            return {
              item_name: restaurant.name,
              item_area: restaurant.area,
              item_avg_rating: restaurant.average,
              item_spec_area: restaurant.address,
              user_name: text(card, [
                'p > span:not(.info)',
                '[class*="nickname"]',
                '[class*="userName"]',
                '[class*="user_name"]'
              ]),
              user_tot_avg_rating: userStats[0] || '',
              user_tot_rating_num: userStats[1] || '',
              user_tot_follow_num: userStats[2] || '',
              user_rating: text(card, [
                'span.total_score',
                '[class*="total_score"]',
                '[itemprop="ratingValue"]'
              ]),
              user_query: reviewText,
              taste: scoreDetails[0] || '',
              price: scoreDetails[1] || '',
              service: scoreDetails[2] || '',
              menu: text(card, ['p.ordered_menu_list', '[class*="ordered_menu"]']),
              date: text(card, ['span.date', 'time', '[class*="date"]']),
              restaurant_id: new URL(profileUrl).searchParams.get('rid') || '',
              source_url: profileUrl,
              crawl_timestamp: crawlTimestamp,
              review_text_complete: String(
                !visibleTextMoreControl && !truncatedEnding
              ),
            };
          }).filter(row => row.item_name);
        }""",
        {"profileUrl": url, "crawlTimestamp": crawl_timestamp},
    )

async def expand_review_texts(page, robots, blocked_paths, blocked_events) -> int:
    """Expand each visible, card-local review body control."""
    card_xpath = (
        "xpath=ancestor::div[contains(concat(' ', normalize-space(@class), ' '), "
        "' latter-graph ')][1]"
    )
    candidates_selector = (
        "div.latter-graph div.review_contents a.more-btn-inline, "
        "div.latter-graph p.review_contents a.more-btn-inline, "
        "div.latter-graph [class*='review_contents'] a.more-btn-inline, "
        "div.latter-graph [class*='reviewContents'] a.more-btn-inline"
    )
    more_pattern = re.compile(r"(?:\.{3}|…)\s*더보기\s*$")
    attempted: set[str] = set()
    expanded = 0

    for _ in range(1000):
        clicked = False
        candidates = page.locator(candidates_selector)
        for index in range(await candidates.count()):
            candidate = candidates.nth(index)
            if not await candidate.is_visible():
                continue
            try:
                label = re.sub(r"\s+", "", await candidate.inner_text(timeout=500))
            except PlaywrightError:
                continue
            if not more_pattern.fullmatch(label):
                continue

            card = candidate.locator(card_xpath)
            if not await card.count():
                continue
            card_id = (
                await card.get_attribute("data-review-id")
                or await card.get_attribute("data-id")
                or await card.get_attribute("id")
                or str(index)
            )
            attempt_key = f"{card_id}:{label}"
            if attempt_key in attempted:
                continue
            attempted.add(attempt_key)

            metadata = await review_button_metadata(candidate)
            href = str(metadata.get("href", "")).strip()
            if href and re.match(r"^https?://", href, flags=re.IGNORECASE):
                parsed = urllib.parse.urlparse(href)
                if parsed.hostname != "www.diningcode.com":
                    continue
                if not robots.can_fetch(USER_AGENT, href):
                    blocked_paths.add(parsed.path)
                    blocked_events.append(parsed.path)
                    continue

            try:
                await candidate.click(timeout=3_000)
            except PlaywrightError:
                continue
            expanded += 1
            clicked = True
            try:
                await candidate.wait_for(state="hidden", timeout=500)
            except PlaywrightError:
                pass
            break

        if not clicked:
            break
    return expanded

async def expected_review_count(page) -> int | None:
    """Read a visible visitor evaluation/review count when the page exposes one."""
    return await page.evaluate(
        r"""() => {
          const parseCount = value => {
            const text = (value || '').replace(/\s+/g, ' ').trim();
            if (!/(평가|리뷰)/.test(text)) return null;
            const match = text.match(/([\d,]+)\s*명(?:의)?\s*(?:방문자\s*)?(?:평가|리뷰)/) ||
              text.match(/(?:방문자\s*)?(?:평가|리뷰)\s*[:：(\[]?\s*([\d,]+)/) ||
              text.match(/([\d,]+)\s*(?:개)?\s*(?:방문자\s*)?(?:평가|리뷰)/);
            return match ? Number(match[1].replace(/,/g, '')) : null;
          };
          const pointCount = document.querySelector('span.point > span');
          const fromPoint = parseCount(pointCount?.innerText || pointCount?.textContent);
          if (fromPoint !== null) return fromPoint;

          const headings = [...document.querySelectorAll(
            'h1, h2, h3, [class*="review"][class*="title"], [class*="tit"]'
          )];
          for (const node of headings) {
            const count = parseCount(
              (node.innerText || node.textContent || '') + ' ' +
              (node.parentElement?.innerText || '')
            );
            if (count !== null) return count;
          }
          return null;
        }"""
    )

def row_key(row: dict[str, str]) -> str:
    stable = "\x1f".join(
        row.get(field, "").strip()
        for field in (
            "restaurant_id",
            "user_name",
            "date",
            "user_rating",
            "user_query",
            "taste",
            "price",
            "service",
        )
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


class IncrementalCsv:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()
        if path.exists() and path.stat().st_size:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                self.seen.update(row_key(row) for row in csv.DictReader(handle))
        else:
            self._write_header()

    def _write_header(self):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("\ufeff")
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, rows: list[dict[str, str]]) -> int:
        additions = []
        for row in rows:
            key = row_key(row)
            if key in self.seen:
                continue
            self.seen.add(key)
            additions.append(row)
        if additions:
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writerows(additions)
                handle.flush()
                os.fsync(handle.fileno())
        return len(additions)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    checkpoint["updated_at"] = now_iso()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def find_resume_checkpoint(output_dir: Path, start_url: str) -> tuple[Path, dict]:
    candidates = sorted(
        output_dir.glob("diningcode_playwright_national_*.checkpoint.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if checkpoint.get("start_url") == start_url and not checkpoint.get("crawl_complete"):
            return path, checkpoint
    raise CrawlStopped(
        f"No unfinished checkpoint for this start URL in {output_dir}. "
        "Start a new crawl without --resume."
    )


async def find_more_review_button(page):
    """Find the visible control whose label is 평가 더보기."""
    text_candidates = page.get_by_text(re.compile(r"평가\s*더보기"), exact=False)
    for index in range(await text_candidates.count()):
        text_node = text_candidates.nth(index)
        if not await text_node.is_visible():
            continue
        try:
            label = re.sub(r"\s+", "", await text_node.inner_text(timeout=500))
        except PlaywrightError:
            # The review list can rerender between visibility and text reads.
            # Skip this stale candidate and let the next scan retry it.
            continue
        if "평가더보기" not in label:
            continue
        clickable = text_node.locator(
            "xpath=ancestor-or-self::*[self::button or self::a or "
            "@role='button' or @onclick or @id='div_more_review'][1]"
        )
        if await clickable.count() and await clickable.is_visible():
            return clickable
        return text_node

    # Some DiningCode versions keep the click handler on the legacy wrapper.
    legacy = page.locator("#div_more_review")
    if await legacy.count() and await legacy.is_visible():
        try:
            label = re.sub(r"\s+", "", await legacy.inner_text(timeout=500))
        except PlaywrightError:
            return None
        if "평가더보기" in label:
            return legacy
    return None

async def review_button_metadata(button) -> dict[str, object]:
    return await button.evaluate(
        """el => {
          const actionable = el.closest('button, a, [role="button"], [onclick]') || el;
          const className = typeof actionable.className === 'string'
            ? actionable.className : '';
          const style = getComputedStyle(actionable);
          const disabled = Boolean(actionable.disabled) ||
            actionable.hasAttribute('disabled') ||
            actionable.getAttribute('aria-disabled') === 'true' ||
            /(^|\\s)(disabled|is-disabled|off)(\\s|$)/i.test(className) ||
            style.pointerEvents === 'none';
          const link = actionable.closest('a[href]') || actionable;
          return {
            href: link.href || link.getAttribute('href') || '',
            label: (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim(),
            disabled
          };
        }"""
    )

async def collect_profile_reviews(
    page,
    robots,
    *,
    url: str,
    crawl_timestamp: str,
    max_clicks: int,
    review_wait_ms: int,
    blocked_paths: set[str],
    blocked_events: list[str],
    on_batch,
    on_click,
):
    rows = await read_profile(page, url, crawl_timestamp)
    known_keys: set[str] = set()
    new_count = 0

    def accept_batch(batch):
        nonlocal new_count
        fresh = []
        for row in batch:
            key = row_key(row)
            if key not in known_keys:
                known_keys.add(key)
                fresh.append(row)
        if fresh:
            new_count += len(fresh)
            on_batch(fresh)

    text_expansion_clicks = await expand_review_texts(
        page, robots, blocked_paths, blocked_events
    )
    rows = await read_profile(page, url, crawl_timestamp)
    accept_batch(rows)
    expected_count = await expected_review_count(page)
    clicks = 0
    more_button_seen = False
    reason = None
    status = "incomplete"

    while True:
        button = await find_more_review_button(page)
        if button is None:
            # Let the last async batch settle, then make one final card/button check.
            await page.wait_for_timeout(1_200)
            final_rows = await read_profile(page, url, crawl_timestamp)
            if any(row_key(row) not in known_keys for row in final_rows):
                text_expansion_clicks += await expand_review_texts(
                    page, robots, blocked_paths, blocked_events
                )
                final_rows = await read_profile(page, url, crawl_timestamp)
            accept_batch(final_rows)
            button = await find_more_review_button(page)
            if button is not None:
                continue

            count_reached = expected_count is not None and len(known_keys) >= expected_count
            exhausted_after_clicks = more_button_seen and clicks > 0
            if known_keys and count_reached:
                status = "complete"
                reason = "Collected the visitor evaluation count shown in the page summary."
            elif known_keys and expected_count is None and exhausted_after_clicks:
                status = "complete"
                reason = "No clickable 평가 더보기 control remains after repeated clicks."
            elif known_keys and expected_count is not None and exhausted_after_clicks:
                reason = (
                    f"No clickable 평가 더보기 control remains, but the page shows "
                    f"{expected_count} evaluations and only {len(known_keys)} rows were collected."
                )
            elif known_keys:
                reason = "No clickable 평가 더보기 control remains, but the visible count does not match."
            else:
                reason = "No visitor review cards or 평가 더보기 control were found."
            break
        more_button_seen = True
        metadata = await review_button_metadata(button)
        href = str(metadata.get("href", "")).strip()
        if metadata.get("disabled"):
            count_reached = expected_count is not None and len(known_keys) >= expected_count
            if known_keys and count_reached:
                status = "complete"
                reason = "Collected the visitor evaluation count shown in the page summary."
            elif known_keys and expected_count is None and clicks > 0:
                status = "complete"
                reason = "The 평가 더보기 control is disabled after repeated clicks."
            elif known_keys and expected_count is not None:
                reason = (
                    f"The 평가 더보기 control is disabled, but the page shows "
                    f"{expected_count} evaluations and only {len(known_keys)} rows were collected."
                )
            else:
                reason = "The 평가 더보기 control is disabled before review completeness was confirmed."
            break
        if clicks >= max_clicks:
            reason = f"Reached --max-review-clicks={max_clicks}."
            break

        if href:
            parsed = urllib.parse.urlparse(href)
            if parsed.hostname != "www.diningcode.com":
                reason = f"평가 더보기 control points outside DiningCode: {href}"
                break
            if not robots.can_fetch(USER_AGENT, href):
                blocked_paths.add(parsed.path)
                blocked_events.append(parsed.path)
                reason = f"robots.txt disallows 평가 더보기 URL {href}"
                break

        blocked_before_click = len(blocked_events)
        try:
            await button.click(timeout=10_000)
        except PlaywrightError as exc:
            reason = f"Could not click the visible 평가 더보기 control: {exc}"
            break
        clicks += 1
        on_click(clicks)
        deadline = asyncio.get_running_loop().time() + review_wait_ms / 1000
        saw_growth = False
        last_growth = None
        while asyncio.get_running_loop().time() < deadline:
            if len(blocked_events) > blocked_before_click:
                reason = "robots.txt blocked a DiningCode request; review expansion stopped."
                break
            current_rows = await read_profile(page, url, crawl_timestamp)
            has_new_card = any(row_key(row) not in known_keys for row in current_rows)
            if has_new_card:
                text_expansion_clicks += await expand_review_texts(
                    page, robots, blocked_paths, blocked_events
                )
                current_rows = await read_profile(page, url, crawl_timestamp)
            before = len(known_keys)
            accept_batch(current_rows)
            if len(known_keys) > before:
                saw_growth = True
                last_growth = asyncio.get_running_loop().time()
            elif saw_growth and last_growth is not None:
                if asyncio.get_running_loop().time() - last_growth >= 1.5:
                    break
            await page.wait_for_timeout(300)

        if reason and "robots.txt" in reason:
            break
        if not saw_growth:
            reason = "Click produced no new review rows before the wait timed out."
            break

    if status != "complete" and reason is None:
        reason = "Review expansion stopped before the end of the list."
    return {
        "status": status,
        "visible_review_rows": len(known_keys),
        "new_review_rows": new_count,
        "review_more_clicks": clicks,
        "expected_review_rows": expected_count,
        "review_text_expansion_clicks": text_expansion_clicks,
        "reason": reason,
    }


async def crawl(args: argparse.Namespace) -> dict:
    robots = load_robots()
    crawl_identity = (
        args.profile_url
        or (NATIONAL_REGIONS_IDENTITY if args.national_regions else args.start_url)
    )
    if args.profile_url:
        if args.national_regions:
            raise CrawlStopped("--profile-url and --national-regions cannot be combined.")
        assert_allowed(robots, args.profile_url)
    elif args.national_regions:
        for _, source_url in national_region_sources():
            assert_allowed(robots, source_url)
    else:
        assert_allowed(robots, args.start_url)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint_path, checkpoint = find_resume_checkpoint(args.output_dir, crawl_identity)
        profile_urls = list(checkpoint["profile_urls"])
        if checkpoint.get("schema_version", 0) < CHECKPOINT_SCHEMA_VERSION:
            # Prior checkpoints used different button and row extraction rules.
            # Start a fresh output while retaining the old CSV and checkpoint.
            previous_checkpoint = str(checkpoint_path)
            stamp = datetime.now(SEOUL).strftime("%Y%m%d_%H%M%S")
            stem = f"diningcode_playwright_national_refresh_{stamp}"
            partial_csv = args.output_dir / f"{stem}.csv.partial"
            final_csv = args.output_dir / f"{stem}.csv"
            checkpoint_path = args.output_dir / f"{stem}.checkpoint.json"
            previous = checkpoint
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "start_url": crawl_identity,
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "partial_csv": str(partial_csv.resolve()),
                "final_csv": str(final_csv.resolve()),
                "profile_urls": profile_urls,
                "listing_pages": previous.get("listing_pages", 0),
                "listing_complete": previous.get("listing_complete", False),
                "listing_error": previous.get("listing_error"),
                "listing_sources": previous.get("listing_sources", []),
                "selection_limited": previous.get("selection_limited", False),
                "max_restaurants": previous.get("max_restaurants"),
                "crawl_complete": False,
                "stop_reason": None,
                "refreshed_from_checkpoint": previous_checkpoint,
                "restaurants": {url: {"status": "pending"} for url in profile_urls},
            }
            IncrementalCsv(partial_csv)
            save_checkpoint(checkpoint_path, checkpoint)
            print(
                "Checkpoint uses the old extraction schema; starting a fresh capture "
                f"and preserving the prior files. New checkpoint: {checkpoint_path}",
                file=sys.stderr,
                flush=True,
            )
        else:
            partial_csv = Path(checkpoint["partial_csv"])
            final_csv = Path(checkpoint["final_csv"])
            checkpoint["stop_reason"] = None
            print(f"Resuming from {checkpoint_path}", file=sys.stderr, flush=True)
    else:
        checkpoint_path = None
        checkpoint = None
        partial_csv = None
        final_csv = None
        profile_urls = []

    skip_incomplete_urls: set[str] = set()
    if args.resume and args.skip_incomplete:
        skip_incomplete_urls = set(checkpoint.get("skip_incomplete_urls", []))
        skip_incomplete_urls.update(
            url
            for url, entry in checkpoint["restaurants"].items()
            if entry.get("status") in {"incomplete", "blocked", "running"}
        )
        checkpoint["skip_incomplete_urls"] = sorted(skip_incomplete_urls)
        checkpoint["skip_incomplete_mode"] = True
        save_checkpoint(checkpoint_path, checkpoint)
        complete_profiles = sum(
            entry.get("status") == "complete"
            for entry in checkpoint["restaurants"].values()
        )
        print(
            f"Skipping {complete_profiles} complete and "
            f"{len(skip_incomplete_urls)} previously attempted unfinished profiles.",
            file=sys.stderr,
            flush=True,
        )

    blocked_paths: set[str] = set()
    blocked_events: list[str] = []
    mode = "headless" if args.headless else "headed (visible browser window)"
    print(f"Starting Playwright in {mode} mode: {crawl_identity}", file=sys.stderr, flush=True)

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=args.headless)
        except PlaywrightError as exc:
            message = str(exc)
            if "Executable doesn't exist" in message:
                raise CrawlStopped(
                    "Chromium is missing. Run .venv/bin/python -m playwright install chromium."
                ) from exc
            if "missing dependencies" in message or "error while loading shared libraries" in message:
                raise CrawlStopped(
                    "Chromium Linux libraries are missing. On Ubuntu/WSL run "
                    "sudo .venv/bin/python -m playwright install-deps chromium."
                ) from exc
            raise

        try:
            context = await browser.new_context(user_agent=USER_AGENT, locale="ko-KR")
            await install_robots_guard(context, robots, blocked_paths, blocked_events)

            if not args.resume:
                listing_sources = []
                if args.profile_url:
                    discovered = [args.profile_url]
                    listing_complete = True
                    listing_pages = 0
                    listing_error = None
                elif args.national_regions:
                    discovered = []
                    listing_complete = True
                    listing_pages = 0
                    listing_errors = []
                    known_profiles: set[str] = set()
                    sources = national_region_sources()
                    for source_index, (region, source_url) in enumerate(sources, start=1):
                        listing = await context.new_page()
                        source_record = {
                            "region": region,
                            "url": source_url,
                            "profiles": 0,
                            "complete": False,
                        }
                        try:
                            await get_allowed_page(
                                listing, robots, source_url, wait_ms=args.wait_ms
                            )
                            title = await listing.title()
                            if region not in title:
                                raise CrawlStopped(
                                    f"Expected the {region} foodrank page, but received {title!r}."
                                )
                            region_links, source_complete, source_pages, source_error = (
                                await collect_profile_links(
                                    listing,
                                    robots,
                                    start_url=source_url,
                                    max_scrolls=args.max_list_scrolls,
                                    max_pages=args.max_list_pages,
                                    wait_ms=args.wait_ms,
                                )
                            )
                            for profile_url in region_links:
                                if profile_url not in known_profiles:
                                    known_profiles.add(profile_url)
                                    discovered.append(profile_url)
                            listing_pages += source_pages
                            source_record["profiles"] = len(region_links)
                            source_record["complete"] = source_complete
                            if source_error:
                                source_record["error"] = source_error
                                listing_errors.append(f"{region}: {source_error}")
                            listing_complete = listing_complete and source_complete
                            print(
                                f"Region {source_index}/{len(sources)} {region}: "
                                f"{len(region_links)} profiles; complete={source_complete}",
                                file=sys.stderr,
                                flush=True,
                            )
                        except (CrawlStopped, PlaywrightError) as exc:
                            source_record["error"] = str(exc)
                            listing_errors.append(f"{region}: {exc}")
                            listing_complete = False
                            print(
                                f"Region {source_index}/{len(sources)} {region} failed: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if any(code in str(exc) for code in ("401", "403", "429", "robots.txt")):
                                listing_sources.append(source_record)
                                await listing.close()
                                break
                        finally:
                            if not listing.is_closed():
                                await listing.close()
                        listing_sources.append(source_record)
                        if source_index < len(sources):
                            await asyncio.sleep(args.delay_seconds)
                    listing_error = " | ".join(listing_errors) or None
                else:
                    listing = await context.new_page()
                    await get_allowed_page(listing, robots, args.start_url, wait_ms=args.wait_ms)
                    discovered, listing_complete, listing_pages, listing_error = (
                        await collect_profile_links(
                            listing,
                            robots,
                            start_url=args.start_url,
                            max_scrolls=args.max_list_scrolls,
                            max_pages=args.max_list_pages,
                            wait_ms=args.wait_ms,
                        )
                    )
                    listing_sources = [{
                        "region": None,
                        "url": args.start_url,
                        "profiles": len(discovered),
                        "complete": listing_complete,
                    }]
                if not discovered:
                    raise CrawlStopped(
                        "No restaurant profile links were found. The current listing markup "
                        "may have changed or the source denied access."
                    )
                selection_limited = bool(args.max_restaurants and len(discovered) > args.max_restaurants)
                profile_urls = (
                    discovered[: args.max_restaurants] if args.max_restaurants else discovered
                )
                stamp = datetime.now(SEOUL).strftime("%Y%m%d_%H%M%S")
                stem = f"diningcode_playwright_national_{stamp}"
                partial_csv = args.output_dir / f"{stem}.csv.partial"
                final_csv = args.output_dir / f"{stem}.csv"
                checkpoint_path = args.output_dir / f"{stem}.checkpoint.json"
                checkpoint = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "start_url": crawl_identity,
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                    "partial_csv": str(partial_csv.resolve()),
                    "final_csv": str(final_csv.resolve()),
                    "profile_urls": profile_urls,
                    "listing_pages": listing_pages,
                    "listing_complete": listing_complete,
                    "listing_error": listing_error,
                    "listing_sources": listing_sources,
                    "selection_limited": selection_limited,
                    "max_restaurants": args.max_restaurants,
                    "crawl_complete": False,
                    "stop_reason": None,
                    "restaurants": {url: {"status": "pending"} for url in profile_urls},
                }
                IncrementalCsv(partial_csv)
                save_checkpoint(checkpoint_path, checkpoint)
                print(
                    f"Discovered {len(discovered)} profiles; selected {len(profile_urls)}. "
                    f"Listing complete: {listing_complete}",
                    file=sys.stderr,
                    flush=True,
                )

            csv_writer = IncrementalCsv(partial_csv)
            detail = await context.new_page()
            stop_reason = checkpoint.get("stop_reason")
            for index, url in enumerate(profile_urls, start=1):
                existing = checkpoint["restaurants"].get(url, {})
                if existing.get("status") == "complete":
                    if not args.skip_incomplete:
                        print(
                            f"[{index}/{len(profile_urls)}] already complete: {url}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                if args.skip_incomplete and url in skip_incomplete_urls:
                    continue

                report = {
                    "status": "running",
                    "started_at": now_iso(),
                    "name": "",
                    "visible_review_rows": 0,
                    "new_review_rows": 0,
                    "review_more_clicks": 0,
                }
                checkpoint["restaurants"][url] = report
                save_checkpoint(checkpoint_path, checkpoint)
                print(f"[{index}/{len(profile_urls)}] opening {url}", file=sys.stderr, flush=True)
                if index > 1:
                    await asyncio.sleep(args.delay_seconds)

                try:
                    await get_allowed_page(detail, robots, url, wait_ms=args.wait_ms)
                    await detail.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await detail.wait_for_timeout(args.wait_ms)
                    report["name"] = await detail.evaluate(
                        "document.querySelector('h1')?.innerText?.trim() || document.title"
                    )
                    crawl_timestamp = now_iso()

                    def save_batch(batch):
                        added = csv_writer.append(batch)
                        report["new_review_rows"] += added
                        report["visible_review_rows"] += len(batch)
                        save_checkpoint(checkpoint_path, checkpoint)

                    def save_click_progress(click_count):
                        report["review_more_clicks"] = click_count
                        report["last_click_at"] = now_iso()
                        save_checkpoint(checkpoint_path, checkpoint)
                        print(
                            f"[{index}/{len(profile_urls)}] {report['name']} | "
                            f"평가 더보기 클릭 {click_count}회, 현재 수집 행 "
                            f"{report['visible_review_rows']}",
                            file=sys.stderr,
                            flush=True,
                        )

                    review_report = await collect_profile_reviews(
                        detail,
                        robots,
                        url=url,
                        crawl_timestamp=crawl_timestamp,
                        max_clicks=args.max_review_clicks,
                        review_wait_ms=args.review_wait_ms,
                        blocked_paths=blocked_paths,
                        blocked_events=blocked_events,
                        on_batch=save_batch,
                        on_click=save_click_progress,
                    )
                    report.update(review_report)
                    report["finished_at"] = now_iso()
                    checkpoint["restaurants"][url] = report
                    if args.skip_incomplete and report.get("status") != "complete":
                        skip_incomplete_urls.add(url)
                        checkpoint["skip_incomplete_urls"] = sorted(skip_incomplete_urls)
                    save_checkpoint(checkpoint_path, checkpoint)
                    print(
                        f"[{index}/{len(profile_urls)}] {report['name']} | "
                        f"{report['visible_review_rows']} visible reviews, "
                        f"{report['review_more_clicks']} more clicks | {report['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
                except (CrawlStopped, PlaywrightError) as exc:
                    report["status"] = "blocked"
                    report["reason"] = str(exc)
                    report["finished_at"] = now_iso()
                    checkpoint["restaurants"][url] = report
                    checkpoint["stop_reason"] = str(exc)
                    if args.skip_incomplete:
                        skip_incomplete_urls.add(url)
                        checkpoint["skip_incomplete_urls"] = sorted(skip_incomplete_urls)
                    save_checkpoint(checkpoint_path, checkpoint)
                    stop_reason = str(exc)
                    print(f"Crawl stopped at {url}: {exc}", file=sys.stderr, flush=True)
                    break

            if blocked_paths:
                checkpoint["robots_blocked_paths"] = sorted(blocked_paths)
            if stop_reason:
                checkpoint["stop_reason"] = stop_reason
            rows_total = len(csv_writer.seen)
            every_restaurant_complete = all(
                entry.get("status") == "complete" or url in skip_incomplete_urls
                for url, entry in checkpoint["restaurants"].items()
            )
            complete = (
                checkpoint.get("listing_complete", False)
                and not checkpoint.get("selection_limited", False)
                and every_restaurant_complete
                and rows_total > 0
            )
            checkpoint["crawl_complete"] = complete
            checkpoint["total_unique_review_rows"] = rows_total
            checkpoint["finished_at"] = now_iso() if complete else None
            if complete:
                os.replace(partial_csv, final_csv)
                checkpoint["output_csv"] = str(final_csv)
            else:
                checkpoint["output_csv"] = str(partial_csv)
            save_checkpoint(checkpoint_path, checkpoint)
        finally:
            await browser.close()

    return {
        "output": checkpoint["output_csv"],
        "checkpoint": str(checkpoint_path),
        "restaurants": len(profile_urls),
        "rows": checkpoint.get("total_unique_review_rows", 0),
        "complete": checkpoint.get("crawl_complete", False),
        "skipped_incomplete": len(skip_incomplete_urls),
        "stop_reason": checkpoint.get("stop_reason"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument(
        "--national-regions",
        action="store_true",
        help="Discover restaurants from all 17 public regional foodrank pages.",
    )
    parser.add_argument(
        "--profile-url",
        help="Crawl one restaurant profile directly, useful for inspecting its visitor evaluations.",
    )
    parser.add_argument(
        "--max-restaurants",
        type=int,
        default=500,
        help="Maximum profile pages; 0 means all profiles found on listing pages.",
    )
    parser.add_argument("--max-list-scrolls", type=int, default=40)
    parser.add_argument("--max-list-pages", type=int, default=10)
    parser.add_argument("--max-review-clicks", "--max-evaluation-clicks", dest="max_review_clicks", type=int, default=1000)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--wait-ms", type=int, default=1600)
    parser.add_argument("--review-wait-ms", type=int, default=8000)
    parser.add_argument("--resume", action="store_true", help="Resume the latest unfinished crawl.")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help=(
            "With --resume, defer previously incomplete/blocked/interrupted profiles "
            "and continue with new pending profiles."
        ),
    )
    browser_mode = parser.add_mutually_exclusive_group()
    browser_mode.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run without a visible browser window. Default is headless=False.",
    )
    browser_mode.add_argument(
        "--show-browser",
        dest="headless",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(headless=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.max_restaurants < 0:
        parser.error("--max-restaurants must be >= 0")
    if args.national_regions and args.profile_url:
        parser.error("--national-regions and --profile-url cannot be combined")
    if args.skip_incomplete and not args.resume:
        parser.error("--skip-incomplete requires --resume")
    if args.max_list_scrolls < 1 or args.max_list_pages < 1:
        parser.error("--max-list-scrolls and --max-list-pages must be >= 1")
    if args.max_review_clicks < 1 or args.review_wait_ms < 1000:
        parser.error("--max-review-clicks must be >= 1 and --review-wait-ms must be >= 1000")
    if args.delay_seconds < 2:
        parser.error("--delay-seconds must be at least 2 to keep request volume low")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(crawl(args))
    except CrawlStopped as exc:
        print(f"Crawl stopped: {exc}", file=sys.stderr)
        return 2
    print(f"CSV: {result['output']}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(
        f"Restaurants: {result['restaurants']}; unique review rows: {result['rows']}; "
        f"crawl complete: {result['complete']}"
    )
    if result["skipped_incomplete"]:
        print(f"Deferred previously incomplete profiles: {result['skipped_incomplete']}")
    if result["stop_reason"]:
        print(f"Stopped at: {result['stop_reason']}", file=sys.stderr)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
