#!/usr/bin/env python3
"""
Q5 Command Matrix — macro_agent.py v2.1
========================================
מערכת מודיעין פיננסית קונטרריאנית.

Pipeline:
  Phase 1: RADAR — סריקה דינמית של 80+ מניות
  Phase 2: AI TRIAGE — Gemini בוחר Top 3 לכל עמודה (קריאה 1)
  Phase 3: DEEP FETCH — נתוני עומק ל-~12 מניות נבחרות
  Phase 4: AI X-RAY — 4 קריאות נפרדות לג'מיני, כל אחת עם Pydantic Model ייחודי
  Phase 5: VALIDATE & SAVE — master_data.json + history

כל הפלט בעברית. מפתחות JSON באנגלית.
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Tuple, Dict, List

from pydantic import BaseModel, Field, validator

# Load .env for local runs; GitHub Actions injects secrets via environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================
# LOGGING
# ==============================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("Q5")

# ==============================================================
# GEMINI SDK (new → legacy fallback)
# ==============================================================
try:
    from google import genai as genai_sdk
    from google.genai import types as genai_types
    GENAI_NEW_SDK = True
    log.info("Using google.genai (new SDK)")
except ImportError:
    try:
        import google.generativeai as genai_sdk
        GENAI_NEW_SDK = False
        log.info("Using google.generativeai (legacy SDK)")
    except ImportError:
        log.critical("No Gemini SDK found. Install: pip install google-genai")
        sys.exit(1)

import yfinance as yf


# ==============================================================
# PYDANTIC MODELS — per-column X-Ray schemas
# ==============================================================

class XRayParam(BaseModel):
    """Single X-Ray parameter: score + Hebrew analysis text."""
    score: int = Field(ge=1, le=100, description="Contrarian opportunity score 1-100")
    analysis: str = Field(min_length=10, description="Hebrew analysis text, 2-3 sentences")

class AnalystRatings(BaseModel):
    """Analyst consensus section."""
    consensus: str = Field(description="קנייה/החזק/מכירה")
    summary: str = Field(description="2-3 sentences in Hebrew")
    bull_case: str = Field(description="One sentence bullish thesis in Hebrew")
    bear_case: str = Field(description="One sentence bearish thesis in Hebrew")

# ── Day Trading X-Ray ──
class DayTradingXRay(BaseModel):
    semantic_panic: XRayParam
    short_trap: XRayParam
    volume_abnormality: XRayParam
    float_choke: XRayParam

# ── Swing X-Ray ──
class SwingXRay(BaseModel):
    event_horizon: XRayParam
    options_flow: XRayParam
    insider_moves: XRayParam
    narrative_shift: XRayParam

# ── Position X-Ray ──
class PositionXRay(BaseModel):
    institutional_stealth: XRayParam
    supply_bottleneck: XRayParam
    analyst_exhaustion: XRayParam
    macro_tailwind: XRayParam

# ── Investment X-Ray ──
class InvestmentXRay(BaseModel):
    hostage_power: XRayParam
    debt_asymmetry: XRayParam
    esg_premium: XRayParam
    capital_iq: XRayParam

class CompositeScore(BaseModel):
    total: int = Field(ge=0, le=100)

class Price(BaseModel):
    current: float

class StockPick(BaseModel):
    """A single stock pick with all analysis fields."""
    ticker: str
    company_name: str
    sector: str
    price: Price
    composite_score: CompositeScore
    action_signal: str = Field(description="המלצת ביצוע קצרה: 2-3 מילים בעברית")
    thesis_summary: str
    company_description: str = ""
    analyst_ratings: Optional[AnalystRatings] = None
    xray: Dict[str, XRayParam]  # dynamic keys per column

class ColumnResult(BaseModel):
    """Result for a single column — 0 to 3 picks (0 if no actionable opportunities)."""
    top_picks: List[StockPick] = Field(min_length=0, max_length=3)

# Map column key → Pydantic XRay model class
XRAY_MODELS: Dict[str, type] = {
    "day_trading": DayTradingXRay,
    "swing": SwingXRay,
    "position": PositionXRay,
    "investment": InvestmentXRay,
}


# ==============================================================
# CONFIG
# ==============================================================
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"
    OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "master_data.json"
    HISTORY_DIR = Path(__file__).parent.parent / "docs" / "history"
    CONFIG_FILE = Path(__file__).parent.parent / "docs" / "config.json"
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    RATE_LIMIT_DELAY = 65  # seconds to wait on 429 rate limit

    # 4 X-Ray parameters PER COLUMN (unique per time horizon)
    XRAY_KEYS = {
        "day_trading": ["semantic_panic", "short_trap", "volume_abnormality", "float_choke"],
        "swing": ["event_horizon", "options_flow", "insider_moves", "narrative_shift"],
        "position": ["institutional_stealth", "supply_bottleneck", "analyst_exhaustion", "macro_tailwind"],
        "investment": ["hostage_power", "debt_asymmetry", "esg_premium", "capital_iq"],
    }


# ==============================================================
# PHASE 1: RADAR SCANNER
# ==============================================================
class RadarScanner:
    """Dynamically discovers ~80 interesting stocks from contrarian sectors."""

    CONTRARIAN_UNIVERSE = {
        "shipping": ["STNG", "EGLE", "SBLK", "ZIM", "DAC", "GSL"],
        "coal_energy": ["BTU", "AMR", "HCC", "ARLP", "CTRA"],
        "tobacco_sin": ["MO", "PM", "BTI", "IMBBY"],
        "defense": ["LMT", "RTX", "NOC", "GD", "HII"],
        "nuclear_uranium": ["CCJ", "UEC", "LEU", "NNE", "SMR"],
        "oil_gas": ["ET", "EPD", "MPLX", "PBF", "DK"],
        "infrastructure": ["URI", "FLR", "PWR", "EME", "STRL"],
        "rare_earth_mining": ["MP", "LAC", "ALB", "UUUU"],
        "reits_value": ["STWD", "ABR", "BXMT", "NLY"],
        "pharma_value": ["PFE", "BMY", "TEVA", "VTRS"],
        "telecoms_boring": ["T", "VZ", "LUMN"],
        "ag_commodities": ["ADM", "BG", "CTVA", "MOS", "NTR"],
        "banks_value": ["C", "WFC", "USB", "KEY", "RF"],
        "auto_old_economy": ["F", "GM", "STLA"],
        "industrial_boring": ["CMI", "CAT", "DE", "PCAR"],
    }

    EXTRA_TICKERS = [
        "INTC", "BA", "PYPL", "DIS", "NCLH", "CCL",
        "CLF", "X", "AA", "FCX",
        "OXY", "DVN", "HAL", "SLB",
        "KMI", "WMB", "OKE",
    ]

    def scan(self) -> List[str]:
        all_tickers = set()
        for sector, tickers in self.CONTRARIAN_UNIVERSE.items():
            for t in tickers:
                all_tickers.add(t)
        for t in self.EXTRA_TICKERS:
            all_tickers.add(t)
        log.info(f"Total universe tickers: {len(all_tickers)}")
        return sorted(all_tickers)

    def fetch_light_data(self, tickers: List[str]) -> List[Dict]:
        log.info(f"Fetching light data for {len(tickers)} tickers...")
        results = []
        try:
            data = yf.download(tickers, period="5d", group_by="ticker", threads=True, progress=False)
        except Exception as e:
            log.error(f"Batch download failed: {e}")
            return results

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data
                else:
                    df = data[ticker] if ticker in data.columns.get_level_values(0) else None
                if df is None or df.empty:
                    continue
                df = df.dropna()
                if len(df) < 2:
                    continue
                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                change_pct = ((last_close - prev_close) / prev_close) * 100
                volume = int(df["Volume"].iloc[-1])
                avg_volume = int(df["Volume"].mean())
                results.append({
                    "ticker": ticker,
                    "price": round(last_close, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "vol_ratio": round(volume / max(avg_volume, 1), 2),
                })
            except Exception:
                continue
        log.info(f"Light data fetched for {len(results)} tickers")
        return results


# ==============================================================
# PHASE 3: DEEP DATA FETCHER
# ==============================================================
class DeepDataFetcher:
    def fetch_deep(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            result = {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "price": {
                    "current": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                    "target_mean": info.get("targetMeanPrice", 0),
                    "52w_high": info.get("fiftyTwoWeekHigh", 0),
                    "52w_low": info.get("fiftyTwoWeekLow", 0),
                },
                "fundamentals": {
                    "pe_ratio": info.get("trailingPE", None),
                    "forward_pe": info.get("forwardPE", None),
                    "pb_ratio": info.get("priceToBook", None),
                    "dividend_yield": info.get("dividendYield", None),
                    "payout_ratio": info.get("payoutRatio", None),
                    "free_cash_flow": info.get("freeCashflow", None),
                    "revenue": info.get("totalRevenue", None),
                    "ebitda": info.get("ebitda", None),
                    "profit_margin": info.get("profitMargins", None),
                    "roe": info.get("returnOnEquity", None),
                    "debt_to_equity": info.get("debtToEquity", None),
                    "total_debt": info.get("totalDebt", None),
                    "total_cash": info.get("totalCash", None),
                },
                "analyst": {},
                "business_description": info.get("longBusinessSummary", ""),
                "analyst_ratings": {},
                "insiders": [],
            }

            try:
                recs = stock.recommendations
                if recs is not None and not recs.empty:
                    result["analyst"]["recent_recs"] = recs.tail(5).to_dict("records")
            except Exception:
                pass

            try:
                rec_summary = stock.recommendations_summary
                if rec_summary is not None:
                    result["analyst_ratings"] = {
                        "strongBuy": int(rec_summary.get("strongBuy", 0)),
                        "buy": int(rec_summary.get("buy", 0)),
                        "hold": int(rec_summary.get("hold", 0)),
                        "sell": int(rec_summary.get("sell", 0)),
                        "strongSell": int(rec_summary.get("strongSell", 0)),
                    }
            except Exception:
                pass

            try:
                insiders = stock.insider_transactions
                if insiders is not None and not insiders.empty:
                    result["insiders"] = insiders.head(10).to_dict("records")
            except Exception:
                pass

            try:
                cal = stock.calendar
                if cal is not None and isinstance(cal, dict):
                    result["earnings_date"] = str(cal.get("Earnings Date", ["Unknown"])[0]) if "Earnings Date" in cal else "Unknown"
                else:
                    result["earnings_date"] = "Unknown"
            except Exception:
                result["earnings_date"] = "Unknown"

            return result
        except Exception as e:
            log.warning(f"Deep fetch failed for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}


# ==============================================================
# CONTRARIAN AI ENGINE — 4 separate Gemini calls
# ==============================================================
class ContrarianAIEngine:
    """Interfaces with Gemini. Phase 2 = 1 triage call, Phase 4 = 4 separate X-Ray calls."""

    STRATEGY_DEFINITIONS = """
## 4 זירות הפעולה:

### 1. day_trading — יירוט טקטי: ארביטראז' סמנטי
זיהוי חברה שחווה פער מחיר שלילי (Gap Down) כתוצאה מ"רעש" ולא מפגיעה פונדמנטלית. רווח מהיר מפאניקה.

### 2. swing — תקיפת קטליזטור: רכיבה על אירוע
זיהוי חברה המתקרבת לאירוע מכונן שהשוק מתמחר בחסר.

### 3. position — מארב אסטרטגי: צווארי בקבוק וכסף חכם
זיהוי סקטור שבו מתחיל להיכנס "כסף חכם" עקב חוסר מהותי בתשתית או חומר גלם.

### 4. investment — נכסי ברזל: מונופול וחפיר פיזי
חברה בעלת חפיר כלכלי שלא ניתן לשכפול, מאזן חסין אינפלציה, ויכולת ייצור תזרים גם במיתון.
"""

    # ── Per-column X-Ray definitions with detailed instructions ──
    XRAY_DEFINITIONS_PER_COLUMN = {
        "day_trading": """
## 4 פרמטרי רנטגן — יירוט טקטי:

1. **semantic_panic** (מדד הפאניקה הסמנטית): מדוד את הפער בין הסנטימנט השלילי ברשתות/כותרות לבין הנזק הפונדמנטלי בפועל. ציון גבוה = פאניקה רבה מתוך רעש בלבד, ללא פגיעה עסקית אמיתית.
2. **short_trap** (מלכודת שורטיסטים): בדוק את יחס השורט, עלות ההשאלה, וימים לכיסוי. ציון גבוה = שורטיסטים חשופים לסקוויז קרוב.
3. **volume_abnormality** (אנומליית מחזורים): השווה נפח מסחר נוכחי לממוצע 20 יום. ציון גבוה = נפח חריג שמצביע על פעילות לא-אורגנית.
4. **float_choke** (חנק היצע צף): בדוק את ה-Float מול ההחזקות המוסדיות. ציון גבוה = היצע צף נמוך שמגביר תנודתיות.
""",
        "swing": """
## 4 פרמטרי רנטגן — תקיפת קטליזטור:

1. **event_horizon** (אופק האירוע): זהה את הקטליזטור הקרוב — דו"ח, אישור FDA, החלטת ריבית. ציון גבוה = אירוע קרוב שהשוק מתמחר בחסר.
2. **options_flow** (זרימת כסף חכם — אופציות): נתח זרימת אופציות חריגה, יחס Put/Call, ו-Open Interest. ציון גבוה = כסף חכם מהמר בגדול.
3. **insider_moves** (פעילות בעלי עניין): בדוק רכישות/מכירות של בכירים, Form 4. ציון גבוה = אינסיידרים קונים.
4. **narrative_shift** (שינוי נרטיב סקטוריאלי): זהה האם הסיפור הציבורי עומד להשתנות. ציון גבוה = נרטיב שלילי שעומד להתהפך.
""",
        "position": """
## 4 פרמטרי רנטגן — מארב אסטרטגי:

1. **institutional_stealth** (איסוף מוסדי שקט): עקוב אחרי שינויי 13F ורכישות מוסדיות שמתחת לרדאר. ציון גבוה = מוסדיים צוברים בשקט.
2. **supply_bottleneck** (צווארי בקבוק באספקה): זהה חוסרים מבניים בשרשרת האספקה. ציון גבוה = צוואר בקבוק שיגרום לעליית מחירים.
3. **analyst_exhaustion** (מיצוי אנליסטים והיפוך): בדוק האם קונצנזוס הגיע לקיצון. כולם ממליצים מכירה? אות קנייה קונטרריאני. ציון גבוה = הזדמנות היפוך.
4. **macro_tailwind** (רוח גבית מאקרו): נתח מגמות ריבית, אינפלציה, מדיניות ממשלתית שמעניקות רוח גבית נסתרת. ציון גבוה = רוח גבית שהשוק טרם תמחר.
""",
        "investment": """
## 4 פרמטרי רנטגן — נכסי ברזל:

1. **hostage_power** (תופס ערובה — כוח מיקוח): האם החברה חוליה קריטית בשרשרת האספקה? ציון גבוה = חפיר עמוק שנועל לקוחות.
2. **debt_asymmetry** (אסימטריית חוב אינפלציונית): חוב ארוך בריבית קבועה נמוכה שנשחק באינפלציה. ציון גבוה = חוב שהופך מנטל לנכס.
3. **esg_premium** (פרדוקס האתיקה — ESG): לחץ ESG שחוסם מתחרים ושומר על היצע נמוך. ציון גבוה = חפיר ESG הפוך.
4. **capital_iq** (מנת משכל הקצאת הון): הנהלה שמכווצת מניות ומחלקת דיבידנדים. ציון גבוה = הנהלה חכמה.
""",
    }

    def __init__(self):
        if not Config.GEMINI_API_KEY:
            log.critical("GEMINI_API_KEY not set!")
            sys.exit(1)
        if GENAI_NEW_SDK:
            self.client = genai_sdk.Client(api_key=Config.GEMINI_API_KEY)
        else:
            genai_sdk.configure(api_key=Config.GEMINI_API_KEY)
            self.client = None

    def call_gemini(self, prompt: str, temperature: float = 0.7) -> str:
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                if GENAI_NEW_SDK:
                    response = self.client.models.generate_content(
                        model=Config.GEMINI_MODEL,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=8000,
                        ),
                    )
                    return response.text or ""
                else:
                    model = genai_sdk.GenerativeModel(Config.GEMINI_MODEL)
                    response = model.generate_content(
                        prompt,
                        generation_config={"temperature": temperature, "max_output_tokens": 8000},
                    )
                    return response.text or ""
            except Exception as e:
                err_str = str(e)
                log.warning(f"Gemini attempt {attempt}/{Config.MAX_RETRIES} failed: {err_str[:200]}")
                if attempt < Config.MAX_RETRIES:
                    # On rate limit (429), wait longer
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = Config.RATE_LIMIT_DELAY
                        log.info(f"Rate limited — waiting {wait}s before retry...")
                    else:
                        wait = Config.RETRY_DELAY * attempt
                    time.sleep(wait)
                else:
                    raise

    def build_triage_prompt(self, radar_data: List[Dict]) -> str:
        compact = json.dumps(radar_data, ensure_ascii=False, indent=None)
        return f"""אתה אנליסט השקעות קונטרריאני (Contrarian).

{self.STRATEGY_DEFINITIONS}

## נתוני הראדר (מחירים, נפחים, שינויים):
{compact}

## המשימה שלך:
מתוך רשימת המניות, בחר בדיוק 3 מניות לכל אחת מ-4 הזירות.
מניה יכולה להופיע ביותר מזירה אחת אם היא מתאימה.

## פורמט הפלט — JSON בלבד:
```json
{{
  "day_trading": ["TICK1", "TICK2", "TICK3"],
  "swing": ["TICK1", "TICK2", "TICK3"],
  "position": ["TICK1", "TICK2", "TICK3"],
  "investment": ["TICK1", "TICK2", "TICK3"]
}}
```

החזר JSON בלבד. ללא טקסט נוסף."""

    # ── Action signal examples per column ──
    ACTION_SIGNAL_EXAMPLES = {
        "day_trading": "קנייה אגרסיבית / שורט טקטי / כניסה בפריצה / לונג ספקולטיבי",
        "swing": "כניסה לפני דוח / רכישה בתיקון / המתנה לאישור / סווינג לונג",
        "position": "צבירה בחלקים / כניסה אסטרטגית / הגדלת פוזיציה / המתנה לתיקון",
        "investment": "קנייה לטווח ארוך / בנייה הדרגתית / רכישת ליבה / תוספת לתיק",
    }

    def build_xray_prompt(self, column_key: str, column_desc: str, deep_data: List[Dict]) -> str:
        """Build a focused prompt for ONE column only (up to 3 stocks, 4 xray params)."""
        data_str = json.dumps(deep_data, ensure_ascii=False, default=str)
        xray_defs = self.XRAY_DEFINITIONS_PER_COLUMN.get(column_key, "")
        xray_keys = Config.XRAY_KEYS.get(column_key, [])
        signal_examples = self.ACTION_SIGNAL_EXAMPLES.get(column_key, "קנייה / מכירה / המתנה")

        # Build JSON schema snippet
        xray_schema_lines = []
        for k in xray_keys:
            xray_schema_lines.append(f'        "{k}": {{ "score": 75, "analysis": "ניתוח בעברית 2-3 משפטים" }}')
        xray_schema_str = ",\n".join(xray_schema_lines)

        return f"""# תפקיד: מנהל קרן גידור — דסק מסחר קונטרריאני

אתה מנהל קרן גידור קונטרריאנית עם סטנדרטים נוקשים של ניהול סיכונים.
אתה אינך אנליסט מחקר שמספק סקירות כלליות — אתה כלי תומך-החלטות (Decision Support System) עבור דסק מסחר.

## עיקרון ברזל — סינון קשיח:
**אם למניה אין יחס סיכוי-סיכון (Risk/Reward) מצוין ואין לך המלצה מבצעית חדה לגביה — אל תכלול אותה בפלט כלל.**
עדיף להחזיר מערך ריק של `top_picks` מאשר להמליץ על נכסים בינוניים.
כל מניה שנכנסת למטריצה חייבת לייצג הזדמנות אסימטרית אמיתית — לא "מעניין לעקוב" אלא "יש פה כסף על השולחן".

## הזירה: **{column_desc}**

{xray_defs}

## הנתונים הפיננסיים:
{data_str}

## המשימה:
בחן את 3 המועמדים. לכל מניה שעוברת את מבחן ה-Risk/Reward שלך, ייצר ניתוח X-Ray מלא.
**אם מניה לא עוברת את הסף — פשוט אל תכלול אותה.** מותר להחזיר 0, 1, 2, או 3 מניות.

### כללים קריטיים:
1. **כל הטקסט בעברית בלבד** — תקציר, ניתוח, שמות סקטורים, תיאור חברה, דעת אנליסטים.
2. **מפתחות JSON באנגלית בלבד** — בדיוק כפי שמופיע בסכמה למטה.
3. **ציון 1-100** לכל פרמטר — ציון גבוה = הזדמנות קונטרריאנית חזקה.
4. **ניתוח 2-3 משפטים** לכל פרמטר — ספציפי, עם נתונים מוחשיים. לא כלליות.
5. **composite_score** — ממוצע משוקלל של 4 הפרמטרים.
6. **action_signal** — המלצת ביצוע קצרה וחדה ב-2-3 מילים בעברית. דוגמאות: {signal_examples}
7. **thesis_summary** — חייב להתחיל בשורה התחתונה (Bottom Line): ההיגיון הכלכלי העומד בבסיס הפעולה. משפט ראשון = למה לפעול עכשיו. משפט שני-שלישי = הנתונים התומכים.
8. **company_description** — תיאור 3-4 משפטים בעברית.
9. **analyst_ratings** — consensus, summary, bull_case, bear_case — הכל בעברית.

## פורמט פלט — JSON בלבד:
```json
{{
  "top_picks": [
    {{
      "ticker": "XXX",
      "company_name": "שם בעברית",
      "sector": "סקטור בעברית",
      "price": {{ "current": 0.0 }},
      "composite_score": {{ "total": 75 }},
      "action_signal": "קנייה אגרסיבית",
      "thesis_summary": "[שורה תחתונה: למה עכשיו] — [ניתוח תומך 2-3 משפטים בעברית]",
      "company_description": "תיאור החברה — 3-4 משפטים בעברית",
      "analyst_ratings": {{
        "consensus": "קנייה/החזק/מכירה",
        "summary": "סיכום דעת האנליסטים — 2-3 משפטים בעברית",
        "bull_case": "התזה החיובית — משפט אחד",
        "bear_case": "התזה השלילית — משפט אחד"
      }},
      "xray": {{
{xray_schema_str}
      }}
    }}
  ]
}}
```

**אם אין אף מניה שעוברת את הסף, החזר:** `{{ "top_picks": [] }}`

חשוב מאוד: השתמש בדיוק ב-4 מפתחות ה-xray: {', '.join(xray_keys)}. אל תוסיף ואל תשנה מפתחות.
החזר JSON בלבד. ללא טקסט נוסף."""


# ==============================================================
# JSON EXTRACTION
# ==============================================================
def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from Gemini response with multiple fallback methods."""
    if not text or not text.strip():
        return None

    # Method 1: Direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 2: Strip markdown code fences
    cleaned = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        if fence in cleaned:
            parts = cleaned.split(fence, 1)
            if len(parts) > 1:
                cleaned = parts[1]
                break
    if "```" in cleaned:
        cleaned = cleaned.split("```")[0]
    try:
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 3: Brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    # Method 4: Last resort
    last_brace = text.rfind("}")
    if last_brace > start:
        try:
            return json.loads(text[start:last_brace + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    log.error(f"All JSON extraction methods failed. Preview: {text[:200]}")
    return None


# ==============================================================
# PYDANTIC VALIDATION
# ==============================================================
def validate_column_with_pydantic(column_key: str, raw_picks: List[Dict]) -> List[Dict]:
    """Validate and clean picks using the column's Pydantic model."""
    xray_model_class = XRAY_MODELS.get(column_key)
    if not xray_model_class:
        return raw_picks

    validated = []
    for pick_data in raw_picks:
        try:
            # Validate xray against the column-specific model
            xray_data = pick_data.get("xray", {})
            xray_obj = xray_model_class(**xray_data)

            # Validate analyst_ratings if present
            ar_data = pick_data.get("analyst_ratings")
            if ar_data and isinstance(ar_data, dict):
                try:
                    AnalystRatings(**ar_data)
                except Exception:
                    pass  # Non-critical — keep raw data

            # Convert validated xray back to dict
            pick_data["xray"] = xray_obj.model_dump()
            validated.append(pick_data)
            log.info(f"    ✓ {pick_data.get('ticker', '?')} — Pydantic validation passed")

        except Exception as e:
            log.warning(f"    ✗ {pick_data.get('ticker', '?')} — Pydantic validation failed: {e}")
            # Still include the pick but log the issue
            validated.append(pick_data)

    return validated


# ==============================================================
# OUTPUT VALIDATOR
# ==============================================================
class OutputValidator:
    COLUMNS = ["day_trading", "swing", "position", "investment"]

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        errors = []
        if "meta" not in data:
            errors.append("Missing 'meta'")
        if "matrix" not in data:
            errors.append("Missing 'matrix'")
            return False, errors

        matrix = data["matrix"]
        total_picks = 0
        for col in self.COLUMNS:
            if col not in matrix:
                errors.append(f"Missing column '{col}'")
                continue
            picks = matrix[col].get("top_picks", [])
            total_picks += len(picks)
            if len(picks) > 3:
                errors.append(f"'{col}' has {len(picks)} picks (max 3)")
            if len(picks) == 0:
                log.info(f"  '{col}': no actionable opportunities (strict filtering)")

            col_xray_keys = Config.XRAY_KEYS.get(col, [])
            for i, pick in enumerate(picks):
                if "ticker" not in pick:
                    errors.append(f"'{col}' pick {i}: missing ticker")
                if "action_signal" not in pick:
                    errors.append(f"'{col}' {pick.get('ticker','?')}: missing action_signal")
                if "xray" not in pick:
                    errors.append(f"'{col}' pick {i}: missing xray")
                    continue
                xray = pick["xray"]
                for key in col_xray_keys:
                    if key not in xray:
                        errors.append(f"'{col}' {pick.get('ticker','?')}: missing xray.{key}")
                    elif "score" not in xray[key]:
                        errors.append(f"'{col}' {pick.get('ticker','?')}: xray.{key} missing score")

        log.info(f"  Total picks across all columns: {total_picks}")
        return len(errors) == 0, errors


# ==============================================================
# MARKET STATUS HELPER
# ==============================================================
def get_market_status() -> str:
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    weekday = now_et.weekday()
    hour = now_et.hour
    minute = now_et.minute

    if weekday >= 5:
        return "weekend"
    current_minutes = hour * 60 + minute
    if current_minutes < 4 * 60:
        return "closed"
    elif current_minutes < 9 * 60 + 30:
        return "pre_market"
    elif current_minutes < 16 * 60:
        return "open"
    elif current_minutes < 20 * 60:
        return "after_hours"
    else:
        return "closed"


# ==============================================================
# MAIN PIPELINE
# ==============================================================
def main():
    log.info("=" * 60)
    log.info("Q5 SIGNAL MATRIX v2.1 — macro_agent.py")
    log.info("=" * 60)
    start_time = time.time()

    # ─── Phase 1: RADAR ───
    log.info("PHASE 1: RADAR SCAN")
    radar = RadarScanner()
    tickers = radar.scan()
    light_data = radar.fetch_light_data(tickers)

    if not light_data:
        log.error("Radar returned no data. Aborting.")
        sys.exit(1)
    log.info(f"Radar complete: {len(light_data)} stocks with data")

    # ─── Phase 2: AI TRIAGE (1 Gemini call) ───
    log.info("PHASE 2: AI TRIAGE — 1 Gemini call")
    ai = ContrarianAIEngine()

    triage_prompt = ai.build_triage_prompt(light_data)
    triage_response = ai.call_gemini(triage_prompt, temperature=0.5)
    triage_result = extract_json(triage_response)

    if not triage_result:
        log.error("Failed to parse triage response.")
        log.error(triage_response[:500])
        sys.exit(1)

    selected_tickers = set()
    column_tickers: Dict[str, List[str]] = {}
    for col in OutputValidator.COLUMNS:
        picks = triage_result.get(col, [])
        column_tickers[col] = picks[:3]
        for t in picks[:3]:
            selected_tickers.add(t)

    log.info(f"Triage selected {len(selected_tickers)} unique tickers: {selected_tickers}")

    # ─── Phase 3: DEEP FETCH ───
    log.info("PHASE 3: DEEP DATA FETCH")
    deep_fetcher = DeepDataFetcher()
    deep_data_map: Dict[str, Dict] = {}

    for ticker in selected_tickers:
        log.info(f"  Deep fetch: {ticker}")
        deep_data_map[ticker] = deep_fetcher.fetch_deep(ticker)
        time.sleep(0.3)

    # ─── Phase 4: AI X-RAY — 4 SEPARATE Gemini calls ───
    log.info("PHASE 4: AI X-RAY — 4 separate Gemini calls (one per column)")

    column_descriptions = {
        "day_trading": "יירוט טקטי — ארביטראז' סמנטי",
        "swing": "תקיפת קטליזטור — רכיבה על אירוע",
        "position": "מארב אסטרטגי — צווארי בקבוק וכסף חכם",
        "investment": "נכסי ברזל — מונופול וחפיר פיזי",
    }

    final_matrix = {}

    for col in OutputValidator.COLUMNS:
        log.info(f"  ── Gemini call for: {col} ──")
        tickers_for_col = column_tickers.get(col, [])
        deep_for_col = [deep_data_map.get(t, {"ticker": t}) for t in tickers_for_col]

        xray_prompt = ai.build_xray_prompt(col, column_descriptions[col], deep_for_col)

        log.info(f"  Sending {len(tickers_for_col)} stocks to Gemini for {col}...")
        xray_response = ai.call_gemini(xray_prompt, temperature=0.6)
        xray_result = extract_json(xray_response)

        if xray_result and "top_picks" in xray_result:
            # Validate each pick with Pydantic
            validated_picks = validate_column_with_pydantic(col, xray_result["top_picks"])
            final_matrix[col] = {"top_picks": validated_picks}
            log.info(f"  ✓ {col}: {len(validated_picks)} picks validated")
        else:
            log.error(f"  ✗ Failed to parse X-Ray for {col}")
            log.error(f"  Raw: {xray_response[:300]}")
            # Fallback
            col_keys = Config.XRAY_KEYS.get(col, [])
            final_matrix[col] = {
                "top_picks": [
                    {
                        "ticker": t,
                        "company_name": deep_data_map.get(t, {}).get("company_name", t),
                        "sector": deep_data_map.get(t, {}).get("sector", ""),
                        "price": {"current": deep_data_map.get(t, {}).get("price", {}).get("current", 0)},
                        "composite_score": {"total": 0},
                        "action_signal": "ניתוח נכשל",
                        "thesis_summary": "ניתוח לא זמין — נסה שנית",
                        "company_description": "",
                        "analyst_ratings": {"consensus": "לא זמין", "summary": "", "bull_case": "", "bear_case": ""},
                        "xray": {k: {"score": 0, "analysis": "לא זמין"} for k in col_keys},
                    }
                    for t in tickers_for_col
                ]
            }

        time.sleep(1)  # Rate limiting between columns

    # ─── Phase 5: VALIDATE & SAVE ───
    log.info("PHASE 5: VALIDATE & SAVE")

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_status": get_market_status(),
            "pipeline_version": "2.1",
            "radar_stats": {
                "total_scanned": len(tickers),
                "with_data": len(light_data),
                "selected_for_analysis": len(selected_tickers),
            },
        },
        "matrix": final_matrix,
    }

    validator = OutputValidator()
    is_valid, errors = validator.validate(output)

    if not is_valid:
        log.warning(f"Validation warnings ({len(errors)}):")
        for err in errors:
            log.warning(f"  - {err}")
    else:
        log.info("Validation PASSED — all 4 columns × 3 picks × 4 X-Ray params each")

    # Save
    Config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Archive to history
    Config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history_file = Config.HISTORY_DIR / f"{date_str}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info(f"Archived to {history_file}")

    # Update history index
    history_index = []
    for hf in sorted(Config.HISTORY_DIR.glob("*.json"), reverse=True):
        if hf.name == "index.json":
            continue
        history_index.append({"date": hf.stem, "file": f"history/{hf.name}"})
    with open(Config.HISTORY_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(history_index, f, ensure_ascii=False, indent=2)
    log.info(f"History index updated: {len(history_index)} entries")

    elapsed = time.time() - start_time
    log.info(f"Saved to {Config.OUTPUT_FILE}")
    log.info(f"Total time: {elapsed:.1f}s")
    log.info("=" * 60)
    log.info("Q5 SIGNAL v2.1 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
