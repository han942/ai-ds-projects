"""Pure, testable transformations from legacy CSV rows to database records."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


SEOUL = ZoneInfo("Asia/Seoul")
NULL_LIKE = {"", "nan", "none", "null", "n/a"}

REGION_BY_TOKEN = {
    "seoul": "서울",
    "gyeonggi": "경기",
    "busan": "부산",
    "daegu": "대구",
}

TASTE_MAP = {
    "맛: 좋음": 2,
    "맛: 맛있음": 2,
    "맛: 보통": 1,
    "맛: 부족": 0,
    "맛: 맛없음": 0,
}
PRICE_MAP = {
    "가격: 만족": 2,
    "가격: 보통": 1,
    "가격: 불만": 0,
}
SERVICE_MAP = {
    "응대: 좋음": 2,
    "응대: 친절함": 2,
    "응대: 보통": 1,
    "응대: 나쁨": 0,
    "응대: 불친절": 0,
}


class RowRejected(ValueError):
    """Raised when a source row cannot satisfy the core schema."""


@dataclass(frozen=True)
class ParsedDate:
    value: date | None
    precision: str


@dataclass(frozen=True)
class TransformedRow:
    restaurant_key: str
    canonical_name: str
    area: str | None
    address: str
    region: str
    item_avg_rating: float | None
    restaurant_metadata: dict[str, Any]
    user_key: str
    user_metadata: dict[str, Any]
    rating: float
    review_text: str | None
    taste: int | None
    price: int | None
    service: int | None
    menu: str | None
    reviewed_at: date | None
    reviewed_at_precision: str
    raw_date: str | None
    scraped_at: datetime
    content_hash: str
    source_row_number: int
    source_file: str
    source_payload: dict[str, Any]


@dataclass(frozen=True)
class FileBatch:
    path: Path
    file_sha256: str
    region: str
    scraped_at: datetime
    total_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int]
    rows: list[TransformedRow]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\ufeff", "").split())
    return "" if text.casefold() in NULL_LIKE else text


def optional_float(value: Any) -> float | None:
    text = normalize_text(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    return int(number) if number is not None else None


def sha256_text(*parts: Any) -> str:
    payload = "\x1f".join(normalize_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scraped_at(path: Path) -> datetime:
    match = re.search(r"_(\d{8})_(\d{4})(\d{2})?(?:\D|$)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer crawl timestamp from filename: {path.name}")
    timestamp = "".join(part for part in match.groups() if part)
    pattern = "%Y%m%d%H%M%S" if match.group(3) else "%Y%m%d%H%M"
    return datetime.strptime(timestamp, pattern).replace(tzinfo=SEOUL)


def infer_region(path: Path) -> str:
    filename = path.name.casefold()
    for token, region in REGION_BY_TOKEN.items():
        if token in filename:
            return region
    return "전국"


ADDRESS_REGION_BY_PREFIX = {
    "서울특별시": "서울", "서울시": "서울", "서울": "서울",
    "경기도": "경기", "경기": "경기",
    "부산광역시": "부산", "부산시": "부산", "부산": "부산",
    "인천광역시": "인천", "인천시": "인천", "인천": "인천",
    "대구광역시": "대구", "대구시": "대구", "대구": "대구",
    "광주광역시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}


def infer_region_from_address(address: str) -> str:
    """Use unambiguous province prefixes; retain 전국 when an address is vague."""
    first = normalize_text(address).split(" ", 1)[0]
    return ADDRESS_REGION_BY_PREFIX.get(first, "전국")


def parse_reviewed_at(raw_value: Any, scraped_at: datetime) -> ParsedDate:
    raw = normalize_text(raw_value)
    if not raw:
        return ParsedDate(None, "unknown")

    exact = re.fullmatch(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", raw)
    if exact:
        try:
            return ParsedDate(
                date(int(exact.group(1)), int(exact.group(2)), int(exact.group(3))),
                "exact",
            )
        except ValueError:
            return ParsedDate(None, "unknown")

    month_day = re.fullmatch(r"(\d{1,2})월\s*(\d{1,2})일", raw)
    if month_day:
        month, day = int(month_day.group(1)), int(month_day.group(2))
        try:
            candidate = date(scraped_at.year, month, day)
            if candidate > scraped_at.date():
                candidate = date(scraped_at.year - 1, month, day)
            return ParsedDate(candidate, "inferred_year")
        except ValueError:
            return ParsedDate(None, "unknown")

    days_ago = re.fullmatch(r"(\d+)일\s*전", raw)
    if days_ago:
        return ParsedDate(
            scraped_at.date() - timedelta(days=int(days_ago.group(1))),
            "relative",
        )

    hours_ago = re.fullmatch(r"(\d+)시간\s*전", raw)
    if hours_ago:
        return ParsedDate(
            (scraped_at - timedelta(hours=int(hours_ago.group(1)))).date(),
            "relative",
        )

    if raw.startswith("어제"):
        return ParsedDate(scraped_at.date() - timedelta(days=1), "relative")
    if raw.startswith("오늘"):
        return ParsedDate(scraped_at.date(), "relative")

    return ParsedDate(None, "unknown")


def normalize_user_name(raw_name: Any) -> tuple[str, bool]:
    raw = normalize_text(raw_name)
    is_daco_gourmand = "다코미식가" in raw
    return normalize_text(raw.replace("다코미식가", "")), is_daco_gourmand


def map_ordinal(raw_value: Any, mapping: dict[str, int]) -> int | None:
    return mapping.get(normalize_text(raw_value))


def transform_row(
    row: dict[str, Any],
    *,
    source_row_number: int,
    source_file: str,
    region: str,
    scraped_at: datetime,
    user_hash_salt: str,
) -> TransformedRow:
    item_name = normalize_text(row.get("item_name"))
    address = normalize_text(row.get("item_spec_area"))
    user_name, is_daco_gourmand = normalize_user_name(row.get("user_name"))
    rating = optional_float(row.get("user_rating"))

    missing: list[str] = []
    if not item_name:
        missing.append("item_name")
    if not address:
        missing.append("item_spec_area")
    if not user_name:
        missing.append("user_name")
    if rating is None:
        missing.append("user_rating")
    if missing:
        raise RowRejected(f"missing required fields: {', '.join(missing)}")
    if not 0 <= rating <= 5:
        raise RowRejected("user_rating outside [0, 5]")

    restaurant_key = sha256_text("diningcode", item_name, address)
    user_key = sha256_text("diningcode", user_hash_salt, user_name)
    review_text = normalize_text(row.get("user_query")) or None
    raw_date = normalize_text(row.get("date")) or None
    raw_crawl_timestamp = normalize_text(row.get("crawl_timestamp"))
    if "crawl_timestamp" in row:
        try:
            scraped_at = datetime.fromisoformat(raw_crawl_timestamp)
        except ValueError as exc:
            raise RowRejected("invalid crawl_timestamp") from exc
        if scraped_at.tzinfo is None:
            raise RowRejected("crawl_timestamp must include timezone")
    parsed_date = parse_reviewed_at(raw_date, scraped_at)
    if region == "전국":
        region = infer_region_from_address(address)

    content_hash = sha256_text(
        "diningcode-review",
        restaurant_key,
        user_key,
        raw_date,
        f"{rating:.1f}",
        review_text,
    )

    sanitized_payload = {
        key: value
        for key, value in row.items()
        if key not in {"", "user_name"}
    }

    return TransformedRow(
        restaurant_key=restaurant_key,
        canonical_name=item_name,
        area=normalize_text(row.get("item_area")) or None,
        address=address,
        region=region,
        item_avg_rating=optional_float(row.get("item_avg_rating")),
        restaurant_metadata={"source": "diningcode"},
        user_key=user_key,
        user_metadata={
            "source_total_avg_rating": optional_float(
                row.get("user_tot_avg_rating")
            ),
            "source_total_rating_count": optional_int(
                row.get("user_tot_rating_num")
            ),
            "source_total_follow_count": optional_int(
                row.get("user_tot_follow_num")
            ),
            "daco_gourmand": is_daco_gourmand,
        },
        rating=rating,
        review_text=review_text,
        taste=map_ordinal(row.get("taste"), TASTE_MAP),
        price=map_ordinal(row.get("price"), PRICE_MAP),
        service=map_ordinal(row.get("service"), SERVICE_MAP),
        menu=normalize_text(row.get("menu")) or None,
        reviewed_at=parsed_date.value,
        reviewed_at_precision=parsed_date.precision,
        raw_date=raw_date,
        scraped_at=scraped_at,
        content_hash=content_hash,
        source_row_number=source_row_number,
        source_file=source_file,
        source_payload=sanitized_payload,
    )


def transform_file(path: Path, user_hash_salt: str) -> FileBatch:
    scraped_at = parse_scraped_at(path)
    region = infer_region(path)
    rows: list[TransformedRow] = []
    rejection_reasons: dict[str, int] = {}
    total_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row_number, raw_row in enumerate(reader, start=2):
            total_rows += 1
            try:
                rows.append(
                    transform_row(
                        raw_row,
                        source_row_number=source_row_number,
                        source_file=path.name,
                        region=region,
                        scraped_at=scraped_at,
                        user_hash_salt=user_hash_salt,
                    )
                )
            except RowRejected as exc:
                reason = str(exc)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    return FileBatch(
        path=path,
        file_sha256=file_sha256(path),
        region=region,
        scraped_at=scraped_at,
        total_rows=total_rows,
        rejected_rows=sum(rejection_reasons.values()),
        rejection_reasons=rejection_reasons,
        rows=rows,
    )


def discover_csv_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.csv") if path.is_file())


def unique_content_hashes(rows: Iterable[TransformedRow]) -> set[str]:
    return {row.content_hash for row in rows}
