import os
import re
import csv
import time
import string
import spacy
from bs4 import BeautifulSoup
from datetime import datetime

# Load spaCy English model for NER
nlp = spacy.load('en_core_web_sm')

RAW_DATA_DIR = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\raw_data"
OUTPUT_CSV = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\structured_data.csv"
DUPLICATES_CSV = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\structured_data_duplicates.csv"

# Institutions to filter out from company extraction
FILTER_ORGS = {
    'european commission', 'commission', 'eu', 'european union',
    'wto', 'ec', 'european parliament', 'council', 'the commission',
    'the european commission',
}

# Case type keyword mapping — searched across the whole title
CASE_TYPE_KEYWORDS = {
    'merger': [
        'merger', 'mergers', 'merge'],
    'state aid': [
        'state aid', 'state aids',
        'aid scheme', 'aid schemes',
    ],
    'antitrust': [
        'antitrust', 'anti-trust', 'anti trust',
        ],
}

def none_if_empty(val):
    if val is None or str(val).strip() == '':
        return None
    return val


def clean_text(text):
    """Clean text: remove special characters, punctuation, and extra whitespace."""
    if not text:
        return text
    # Remove special characters (keep letters, digits, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u024F]', ' ', text)
    # Remove extra whitespace (multiple spaces, tabs, newlines → single space)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_article_id(file_path):
    """Extract article ID from the filename (e.g. 'ip_00_1000.html' → 'ip_00_1000')."""
    return os.path.splitext(os.path.basename(file_path))[0].lower()


def get_title(soup):
    h1 = soup.select_one('.ecl-page-header__title h1')
    if h1:
        return re.sub(r'\s+', ' ', h1.get_text(strip=True)).lower()
    og = soup.find('meta', attrs={'property': 'og:description'})
    if og and og.get('content', '').strip().lower() != 'null':
        return re.sub(r'\s+', ' ', og['content'].strip()).lower()
    return None


def _is_skip_line(text):
    """Return True if the text looks like an article-ref or city/date metadata line."""
    t = text.strip()
    # Article reference line  e.g.  IP/00/1000 , MEMO/01/234
    if re.match(r'^(IP|MEMO|SPEECH|PRES|DN|DOC|P|SC|CJE|STAT)[/_\-]\d', t, re.IGNORECASE):
        return True
    # City + date line  e.g.  Brussels, 13 September 2000
    if re.match(r'^[A-Za-z\u00C0-\u024F\s]{2,25},\s*\d{1,2}\s+\w+\s+\d{4}', t):
        return True
    return False


def get_body(soup):
    """Get only actual body paragraphs — skip article ref, city/date, and h1."""
    container = soup.select_one('.ecl-paragraph')
    if not container:
        return None
    parts = []
    for tag in container.children:
        # skip right-aligned <p> (article ID line, city/date line)
        if tag.name == 'p' and tag.get('align') == 'right':
            continue
        # skip duplicate <h1> title
        if tag.name == 'h1':
            continue
        if hasattr(tag, 'get_text'):
            text = tag.get_text(separator=' ', strip=True)
            if not text:
                continue
            # skip article-ref and city/date lines by content pattern
            if _is_skip_line(text):
                continue
            parts.append(text)
    body = re.sub(r'\s+', ' ', ' '.join(parts)).strip().lower()
    return body if len(body) > 10 else None


def get_date(soup):
    meta = soup.find('meta', attrs={'name': 'Date'}) or soup.find('meta', attrs={'name': 'date'})
    if meta:
        ds = meta.get('content', '').strip()
        for fmt in ('%Y-%m-%d', '%d %B %Y'):
            try:
                dt = datetime.strptime(ds, fmt)
                return dt.strftime('%d/%m/%Y'), dt.strftime('%B').lower(), str(dt.year)
            except ValueError:
                continue
    for item in soup.select('.ecl-meta__item'):
        try:
            dt = datetime.strptime(item.get_text(strip=True), '%d %B %Y')
            return dt.strftime('%d/%m/%Y'), dt.strftime('%B').lower(), str(dt.year)
        except ValueError:
            continue
    return None, None, None


def get_published_at(soup):
    """Extract publication city from body metadata (e.g. 'Brussels, 13 September 2000')."""
    container = soup.select_one('.ecl-paragraph')
    if not container:
        return None
    for p in container.find_all('p'):
        text = p.get_text(strip=True)
        # Match pattern: City, DD Month YYYY  or  City,DD Month YYYY
        m = re.match(r'^([A-Za-z\u00C0-\u024F\s]+?)\s*,\s*\d{1,2}\s+\w+\s+\d{4}', text)
        if m:
            city = m.group(1).strip().lower()
            if city and len(city) < 30:
                return city
    return None


def get_language(soup):
    """Extract article language from <html lang='...'> attribute."""
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        return html_tag['lang'].strip().lower()
    # Fallback: ecl-lang-select-sites__code-text span
    code = soup.select_one('.ecl-lang-select-sites__code-text')
    if code:
        return code.get_text(strip=True).lower()
    return None


def get_available_languages(soup):
    """Extract list of available language codes from the language dropdown."""
    select = soup.select_one('select.ecpr-lang-dropdown')
    if not select:
        return None
    langs = [opt.get('value', '').strip().lower()
             for opt in select.find_all('option')
             if opt.get('value')]
    return '; '.join(langs) if langs else None


def get_language_links(soup):
    """Build URLs for each available language version using the canonical URL as base."""
    base_url = None
    og = soup.find('meta', attrs={'property': 'og:url'})
    if og:
        base_url = og.get('content', '').strip()
    if not base_url:
        dc = soup.find('meta', attrs={'property': 'dcterms.identifier'})
        if dc:
            base_url = dc.get('content', '').strip()
    if not base_url:
        return None
    select = soup.select_one('select.ecpr-lang-dropdown')
    if not select:
        return None
    # base_url looks like: .../detail/en/ip_00_1000
    # replace /en/ (or current lang code) with each available lang code
    links = []
    for opt in select.find_all('option'):
        lang = opt.get('value', '').strip().lower()
        if not lang:
            continue
        # replace the language segment in the URL
        link = re.sub(r'/detail/[a-z]{2}/', f'/detail/{lang}/', base_url)
        links.append(f"{lang}:{link}")
    return '; '.join(links) if links else None


def get_pdf_url(soup):
    """Extract the PDF download URL from the print-friendly PDF section."""
    a = soup.select_one('a.ecl-file__download[href]')
    if a:
        return a['href'].strip()
    return None


def extract_entities_spacy(text):
    """Use spaCy NER to extract companies and countries from text."""
    if not text:
        return [], []

    doc = nlp(text)

    companies = []
    countries = []

    for ent in doc.ents:
        label = ent.label_
        value = ent.text.strip().lower()
        if not value:
            continue
        # Skip EU institutions
        if value in FILTER_ORGS:
            continue
        if label == 'ORG':
            companies.append(value)
        elif label == 'GPE':
            # Treat GPE as country (user wants country, not city)
            countries.append(value)

    # Deduplicate while preserving order
    companies = list(dict.fromkeys(companies))
    countries = list(dict.fromkeys(countries))

    return companies, countries


def extract_monetary_amounts(text):
    if not text:
        return []

    results = []

    # ── Pattern 1: CURRENCY + NUMBER + SCALE (required) ──────────
    #   e.g. "eur 1.2 million", "€ 3 billion", "usd 500 million"
    p1 = (r'(?:eur|euro|euros|€|\$|usd|gbp|£)'
          r'\s{0,3}'
          r'(\d[\d,.\s]{0,20}\d|\d+)'
          r'\s{0,3}'
          r'(million|billion|mln|bln|mn|bn)')
    for m in re.finditer(p1, text, re.IGNORECASE):
        amount = _clean_num(m.group(1))
        scale = m.group(2).strip().lower()
        if amount:
            results.append(_fmt_amount(amount, scale))

    # ── Pattern 2: CURRENCY + LARGE NUMBER (no scale word) ──────
    #   e.g. "€ 1,200,000", "eur 500000"
    p2 = (r'(?:eur|euro|euros|€|\$|usd|gbp|£)'
          r'\s{0,3}'
          r'(\d[\d,.\s]{0,20}\d|\d+)'
          r'(?!\s{0,3}(?:million|billion|mln|bln|mn|bn))')
    for m in re.finditer(p2, text, re.IGNORECASE):
        amount = _clean_num(m.group(1))
        if amount:
            try:
                val = float(amount)
                if val >= 100_000:  # only keep genuinely large numbers
                    results.append(_fmt_amount(amount, ''))
            except ValueError:
                pass

    # ── Pattern 3: NUMBER + SCALE + CURRENCY (number first) ─────
    #   e.g. "1.2 million euro", "3 billion euros"
    p3 = (r'(\d[\d,.\s]{0,20}\d|\d+)'
          r'\s{0,3}'
          r'(million|billion|mln|bln|mn|bn)'
          r'\s{0,3}'
          r'(?:euro|euros|eur|€|dollar|dollars|usd|pound|pounds|gbp|£)')
    for m in re.finditer(p3, text, re.IGNORECASE):
        amount = _clean_num(m.group(1))
        scale = m.group(2).strip().lower()
        if amount:
            results.append(_fmt_amount(amount, scale))

    # ── Pattern 4: LARGE NUMBER + CURRENCY (no scale word) ──────
    #   e.g. "1,200,000 euros", "500000 eur"
    p4 = (r'(\d[\d,.\s]{0,20}\d|\d+)'
          r'\s{0,3}'
          r'(?:euro|euros|eur|€|dollar|dollars|usd|pound|pounds|gbp|£)'
          r'(?!\s{0,3}(?:million|billion|mln|bln|mn|bn))')
    for m in re.finditer(p4, text, re.IGNORECASE):
        amount = _clean_num(m.group(1))
        if amount:
            try:
                val = float(amount)
                if val >= 100_000:
                    results.append(_fmt_amount(amount, ''))
            except ValueError:
                pass

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def _clean_num(raw):
    """Normalise a captured number string → clean float-parsable string.
    Returns None if the result looks invalid."""
    s = re.sub(r'[\s,]+', '', raw).strip('.')
    if not s or not re.fullmatch(r'\d+\.?\d*', s):
        return None
    return s


def _fmt_amount(num_str, scale):
    """Format an extracted amount into a numeric-only value in EUR."""
    scale_map = {'mln': 'million', 'mn': 'million', 'bln': 'billion', 'bn': 'billion'}
    scale = scale_map.get(scale, scale)
    try:
        val = float(num_str)
        if scale == 'million':
            val = val * 1_000_000
        elif scale == 'billion':
            val = val * 1_000_000_000
        return f'{val:.2f}'
    except ValueError:
        return num_str


def extract_references(text, own_id):
    if not text:
        return []
    pattern = r'\b(IP)[/_ -](\d{2,4})[/_ -](\d+)\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    refs = []
    for prefix, mid, num in matches:
        # Normalise to same format as article_id: lowercase, underscore-separated
        ref_id = f"{prefix.lower()}_{mid}_{num}"
        if ref_id != own_id and ref_id not in refs:
            refs.append(ref_id)
    return refs


def get_case_type(title):
    if not title:
        return None
    title_lower = title.lower()
    for case_type, keywords in CASE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return case_type
    colon_match = re.match(r'^([^:]+?)\s*:', title_lower)
    if colon_match:
        prefix = colon_match.group(1).strip()
        # Check if any known case-type keyword appears anywhere in the title
        for case_type, keywords in CASE_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in title_lower:
                    return case_type
        words = prefix.split()
        return ' '.join(words[:2]) if words else None

    return None


def process_file(file_path):
    # Try multiple encodings for robust reading
    encodings = ['utf-8', 'latin-1', 'windows-1252']
    soup = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                soup = BeautifulSoup(f, 'html.parser')
            break
        except Exception as e:
            last_err = e
    if soup is None:
        print(f"  [ERROR] {file_path}: {last_err}")
        return [file_path] + [None] * 11

    article_id = none_if_empty(get_article_id(file_path))
    title_raw = get_title(soup)
    body_raw = get_body(soup)
    # Remove files with empty body
    if not body_raw or not body_raw.strip():
        return None
    date, month, year = get_date(soup)
    published_at = get_published_at(soup)
    case_type = get_case_type(title_raw)
    language       = get_language(soup)
    avail_langs    = get_available_languages(soup)
    pdf_url        = get_pdf_url(soup)

    # spaCy NER entity extraction from title (before cleaning)
    t_companies, t_countries = extract_entities_spacy(title_raw)

    country = t_countries[0] if t_countries else None
    company = '; '.join(t_companies) if t_companies else None

    # Remove duplicate title from body text
    if title_raw and body_raw:
        title_norm = re.sub(r'\s+', ' ', title_raw).strip().lower()
        # Remove title if it appears at the start of the body
        if body_raw.startswith(title_norm):
            body_raw = body_raw[len(title_norm):].strip()
        # Also remove title if it appears anywhere in the body
        elif title_norm in body_raw:
            body_raw = body_raw.replace(title_norm, '', 1).strip()
        # Collapse any resulting double spaces
        body_raw = re.sub(r'\s+', ' ', body_raw).strip()
        if not body_raw or len(body_raw) <= 10:
            return None

    # Extract related article references from body text
    related_refs = extract_references(body_raw, article_id)
    related_to = '; '.join(related_refs) if related_refs else None

    # Extract monetary amounts from body text
    money = extract_monetary_amounts(body_raw)
    monetary_amounts = '; '.join(money) if money else None

    # Store raw text — cleaning will happen after all data is collected
    title = none_if_empty(title_raw)
    summary = none_if_empty(body_raw)

    return [file_path, article_id, title, summary, date, month, year,
            published_at, country, company, case_type, related_to, monetary_amounts,
            language, avail_langs, pdf_url]


def main():
    rows = []
    seen = {}          # (article_id, title) -> row index in rows list
    duplicate_rows = [] # rows identified as duplicates
    count = 0

    headers = ['file_path', 'article_id', 'title', 'summary',
               'date', 'month', 'year', 'published_at',
               'country', 'company', 'case_type', 'related_to',
               'monetary_amounts', 'language', 'available_languages', 'pdf_url']

    TITLE_IDX = headers.index('title')
    SUMMARY_IDX = headers.index('summary')

    print(f"Scanning: {RAW_DATA_DIR}")
    start_time = time.time()
    for root, _, files in os.walk(RAW_DATA_DIR):
        html_files = sorted([f for f in files if f.lower().endswith('.html')])
        if not html_files:
            continue
        folder = os.path.basename(root)
        total = len(html_files)
        print(f"\n  {folder}: {total} files")
        for idx, f in enumerate(html_files, 1):
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 and count > 0 else 0
            print(f"    [{folder}] {idx}/{total}  (done: {count}, {rate:.1f} files/s)  {f}", flush=True)
            row = process_file(os.path.join(root, f))
            if row is None:
                continue  # skip files with empty body

            aid   = str(row[1] or '').strip().lower()
            title = str(row[2] or '').strip().lower()
            key = (aid, title)

            if key in seen:
                duplicate_rows.append(row)
                continue  # duplicate — skip
            seen[key] = True

            rows.append(row)
            count += 1

    # --- Post-collection cleaning ---
    for row in rows:
        row[TITLE_IDX] = none_if_empty(clean_text(row[TITLE_IDX]))
        row[SUMMARY_IDX] = none_if_empty(clean_text(row[SUMMARY_IDX]))

    # Also clean duplicates before saving
    for row in duplicate_rows:
        row[TITLE_IDX] = none_if_empty(clean_text(row[TITLE_IDX]))
        row[SUMMARY_IDX] = none_if_empty(clean_text(row[SUMMARY_IDX]))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Save main dataset
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    # Save duplicates to separate CSV
    with open(DUPLICATES_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(duplicate_rows)

    print(f"\nDone! {len(rows)} rows saved → {OUTPUT_CSV}")
    print(f"  Duplicates removed: {len(duplicate_rows)} → {DUPLICATES_CSV}")
    for i, h in enumerate(headers):
        filled = sum(1 for r in rows if r[i] is not None)
        print(f"  {h:15s}: {filled:>5d}/{len(rows)}  ({filled/len(rows)*100:.1f}%)" if rows else '')


if __name__ == "__main__":
    main()