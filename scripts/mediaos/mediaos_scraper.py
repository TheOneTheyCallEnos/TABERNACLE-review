#!/usr/bin/env python3
"""
MediaOS CRM Scraper
Extracts account data from MediaOS and outputs structured JSON.

Uses Playwright for JavaScript rendering (MediaOS is a Vue.js SPA).

Usage:
    python mediaos_scraper.py --account-id 2878 --output universal_ventilation.json
    python mediaos_scraper.py --account-id 2878  # outputs to stdout
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import browser_cookie3
except ImportError:
    print("ERROR: browser_cookie3 not installed. Run: pip install browser-cookie3", file=sys.stderr)
    sys.exit(1)

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: beautifulsoup4 not installed. Run: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)


MEDIAOS_BASE_URL = "https://app.mediaos.com"


def get_chrome_cookies(domain: str = "mediaos.com") -> list:
    """Extract cookies from Chrome for the specified domain and format for Playwright."""
    try:
        cj = browser_cookie3.chrome(domain_name=domain)
        cookies = []
        for cookie in cj:
            playwright_cookie = {
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
            }
            if cookie.secure:
                playwright_cookie["secure"] = True
            if hasattr(cookie, 'expires') and cookie.expires:
                playwright_cookie["expires"] = cookie.expires
            cookies.append(playwright_cookie)
        return cookies
    except Exception as e:
        print(f"ERROR: Failed to extract Chrome cookies: {e}", file=sys.stderr)
        print("Make sure Chrome is closed or try running with sudo if permission denied.", file=sys.stderr)
        sys.exit(1)


def extract_text(element, default: str = "") -> str:
    """Safely extract text from a BeautifulSoup element."""
    if element is None:
        return default
    return element.get_text(strip=True)


def parse_header(soup: BeautifulSoup) -> dict:
    """Parse the account header section using MediaOS-specific selectors."""
    header = {
        "company_name": "",
        "phone": "",
        "email": "",
        "address": "",
        "website": "",
        "description": "",
        "tags": [],
        "last_insertion_date": None,
        "last_contract_date": None,
        "last_activity_date": None,
    }

    # Company name - MediaOS uses h2 inside media-body, or title tag
    title_tag = soup.select_one("title")
    if title_tag:
        title_text = extract_text(title_tag)
        if title_text and title_text.lower() != "mediaos":
            header["company_name"] = title_text

    # Also try h2 which might have the display name
    h2_candidates = soup.select("h2.mar-no, .media-body h2")
    for h2 in h2_candidates:
        text = extract_text(h2)
        if text and len(text) > 2:
            header["company_name"] = text
            break

    # Contact data is in .contactDataContainer elements
    contact_containers = soup.select(".contactDataContainer")
    for container in contact_containers:
        text_el = container.select_one("span.text-primary")
        if text_el:
            text = extract_text(text_el)
            # Detect phone vs email vs address
            if re.match(r'^\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}', text):
                if not header["phone"]:
                    header["phone"] = text
            elif "@" in text and "." in text:
                if not header["email"]:
                    header["email"] = text
            elif re.match(r'^#?\d+.*[A-Za-z]', text) or any(x in text.lower() for x in ["ave", "st", "road", "drive", "blvd"]):
                if not header["address"]:
                    header["address"] = text

    # Additional address parsing - look for full address blocks
    address_divs = soup.select(".additional-contact-data div")
    address_parts = []
    for div in address_divs:
        container = div.select_one(".contactDataContainer")
        if container:
            text = extract_text(container.select_one("span.text-primary") or container)
            # Skip phone/email
            if text and "@" not in text and not re.match(r'^\(?\d{3}\)?[\s\-\.]?\d{3}', text):
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                if text not in address_parts:  # Avoid duplicates
                    address_parts.append(text)
    if address_parts and not header["address"]:
        header["address"] = ", ".join(address_parts)

    # Clean up address if it has issues
    if header["address"]:
        # Remove duplicates and clean whitespace
        header["address"] = re.sub(r'\s+', ' ', header["address"]).strip()
        # Add comma between street and city if missing (e.g., "SECalgaryt" -> "SE, Calgaryt")
        header["address"] = re.sub(r'([A-Z]{2})([A-Z][a-z])', r'\1, \2', header["address"])

    # Description - in .pad-btm div under advertiserSummary
    desc_el = soup.select_one("#advertiserSummary .pad-btm")
    if desc_el:
        header["description"] = extract_text(desc_el)

    # Tags - MediaOS uses label classes for tags
    # Look for tags in the tag input area (SMCAA, SMCAA-members style badges)
    tag_input_area = soup.select_one(".vue-tags-input")
    if tag_input_area:
        tags = tag_input_area.select(".ti-tag")
        header["tags"] = [extract_text(t) for t in tags if extract_text(t)]

    # Also look for pipeline tags like "PAST ADVERTISER"
    pipeline_labels = soup.select(".label-table.label-primary, .label-table.label-success, .label-table.label-warning")
    for label in pipeline_labels:
        text = extract_text(label)
        if text and text not in header["tags"] and len(text) < 50:
            header["tags"].append(text)

    # Date fields - MediaOS shows these in a specific format
    # Look for "Last Insertion", "Last Contract", "Last Activity" labels
    page_text = soup.get_text()

    # More specific date extraction
    date_sections = soup.select(".col-lg-4, .col-md-3, .col-sm-3")
    for section in date_sections:
        text = extract_text(section)
        if "Last Insertion" in text:
            match = re.search(r'Last Insertion\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})', text)
            if match:
                header["last_insertion_date"] = match.group(1)
        elif "Last Contract" in text:
            match = re.search(r'Last Contract\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})', text)
            if match:
                header["last_contract_date"] = match.group(1)
        elif "Last Activity" in text:
            match = re.search(r'Last Activity\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})', text)
            if match:
                header["last_activity_date"] = match.group(1)

    # Fallback regex on full page text
    if not header["last_insertion_date"]:
        match = re.search(r'Last\s+Insertion\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})', page_text)
        if match:
            header["last_insertion_date"] = match.group(1)
    if not header["last_contract_date"]:
        match = re.search(r'Last\s+Contract\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})', page_text)
        if match:
            header["last_contract_date"] = match.group(1)
    if not header["last_activity_date"]:
        match = re.search(r'Last\s+Activity\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})', page_text)
        if match:
            header["last_activity_date"] = match.group(1)

    return header


def parse_contacts(soup: BeautifulSoup) -> list:
    """Parse contacts from the sidebar using MediaOS-specific selectors."""
    contacts = []
    seen_names = set()

    # MediaOS uses .cui-contact-list for the contacts sidebar
    contact_list = soup.select_one(".cui-contact-list")
    if contact_list:
        contact_rows = contact_list.select(".contactRow")
        for row in contact_rows:
            # Name is in .cui-contact-name
            name_el = row.select_one(".cui-contact-name")
            if name_el:
                name = extract_text(name_el)
                if name and name not in seen_names and name.lower() != "add":
                    seen_names.add(name)
                    contact = {"name": name}

                    # Try to get title/role from .media-body div after name
                    title_el = row.select_one(".media-body > div:not(.cui-contact-name)")
                    if title_el:
                        title = extract_text(title_el)
                        if title and title != "-" and title != name:
                            contact["title"] = title

                    contacts.append(contact)

    # Also check for contacts in the main contact display areas
    contact_displays = soup.select("[data-v-3f0f5a5f] .media-body")
    for display in contact_displays:
        name_el = display.select_one(".cui-contact-name")
        if name_el:
            name = extract_text(name_el)
            if name and name not in seen_names:
                seen_names.add(name)
                contact = {"name": name}
                contacts.append(contact)

    return contacts


def parse_contracts(soup: BeautifulSoup) -> list:
    """Parse the contracts table using MediaOS-specific selectors."""
    contracts = []

    # MediaOS uses .asg-table for tables - look for contracts tab content
    tables = soup.select(".asg-table, table.bootstrap-table, table")

    for table in tables:
        # Get headers to understand columns
        headers = table.select("th")
        header_names = [extract_text(h).lower() for h in headers]

        # Check if this looks like a contracts table
        if not any(k in " ".join(header_names) for k in ["contract", "proposal", "no.", "total"]):
            continue

        # Build column mapping based on actual MediaOS contract table headers
        col_map = {}
        for i, h in enumerate(header_names):
            h_clean = h.strip()
            if h_clean in ["no.", "no", "#"] or "contract" in h_clean:
                col_map["contract_number"] = i
            elif "account" in h_clean:
                col_map["account_name"] = i
            elif "pipeline" in h_clean:
                col_map["pipeline_status"] = i
            elif "created" in h_clean:
                col_map["created_date"] = i
            elif "last line" in h_clean or "line item" in h_clean:
                col_map["last_line_item_date"] = i
            elif "signed" in h_clean:
                col_map["signed_date"] = i
            elif "sold by" in h_clean or "rep" in h_clean or "owner" in h_clean:
                col_map["sold_by"] = i
            elif h_clean == "total" or (h_clean.startswith("total") and "cancel" not in h_clean):
                col_map["total"] = i
            elif "cancel" in h_clean:
                col_map["cancelled_total"] = i
            elif "status" in h_clean and "pipeline" not in h_clean:
                col_map["sold_status"] = i

        # Parse rows
        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if not cells:
                continue

            contract = {
                "contract_number": "",
                "account_name": "",
                "pipeline_status": "",
                "created_date": "",
                "last_line_item_date": "",
                "signed_date": "",
                "sold_by": "",
                "total": "",
                "cancelled_total": "",
                "sold_status": "",  # Won/Lost
            }

            for field, idx in col_map.items():
                if idx < len(cells):
                    cell = cells[idx]
                    # Check for labels (badges) in cells
                    label = cell.select_one(".label")
                    if label:
                        contract[field] = extract_text(label)
                    else:
                        contract[field] = extract_text(cell)

            # Try to infer sold_status from labels if not explicitly found
            if not contract["sold_status"]:
                for cell in cells:
                    labels = cell.select(".label")
                    for label in labels:
                        text = extract_text(label).lower()
                        if text in ["won", "lost", "pending", "draft"]:
                            contract["sold_status"] = text.capitalize()
                            break

            if contract["contract_number"] or contract["total"]:
                contracts.append(contract)

        if contracts:
            break

    return contracts


def parse_activity(soup: BeautifulSoup) -> list:
    """Parse recent activity items using MediaOS timeline structure."""
    activities = []
    seen_activities = set()

    # MediaOS uses .timeline-entry for activity items
    timeline_entries = soup.select(".timeline-entry")

    for entry in timeline_entries:
        activity = {
            "date": "",
            "type": "",
            "user": "",
            "action": "",
            "description": "",
        }

        # Date is in .pull-right.mar-rgt
        date_el = entry.select_one(".pull-right.mar-rgt")
        if date_el:
            activity["date"] = extract_text(date_el)

        # Type is determined by icon in .timeline-icon
        icon_el = entry.select_one(".timeline-icon i")
        if icon_el:
            classes = " ".join(icon_el.get("class", []))
            if "fa-phone" in classes:
                activity["type"] = "call"
            elif "fa-envelope" in classes:
                activity["type"] = "email"
            elif "fa-sticky-note" in classes or "fa-comment" in classes:
                activity["type"] = "note"
            elif "fa-calendar" in classes:
                activity["type"] = "meeting"
            elif "fa-repeat" in classes:
                activity["type"] = "follow-up"
            elif "fa-check" in classes:
                activity["type"] = "task"
            elif "fa-file" in classes:
                activity["type"] = "file"

        # User name - look for span inside the user link
        user_link = entry.select_one(".timeline-label-line a.text-primary")
        if user_link:
            user_span = user_link.select_one("span")
            if user_span:
                activity["user"] = extract_text(user_span)
            else:
                activity["user"] = extract_text(user_link)

        # Description/action - get text after "about" or the main content
        label_line = entry.select_one(".timeline-label-line")
        if label_line:
            full_text = extract_text(label_line)
            # Remove date from text
            if activity["date"]:
                full_text = full_text.replace(activity["date"], "").strip()

            # Clean up concatenated words (e.g., "DruwécalledUniversal" -> "Druwé called Universal")
            # Add space before capital letters that follow lowercase (including accented chars)
            full_text = re.sub(r'([a-zéèêëàâäùûüîïôöç])([A-Z])', r'\1 \2', full_text)
            # Add space before lowercase that follows accented char directly followed by verb
            full_text = re.sub(r'([éèêëàâäùûüîïôöç])([a-z]{4,}ed|[a-z]{4,}ing|scheduled|called|emailed|noted)', r'\1 \2', full_text)
            # Add space around "about" if missing (handles ".about " and ".aboutX" cases)
            full_text = re.sub(r'\.about\s', '. about ', full_text, flags=re.IGNORECASE)
            full_text = re.sub(r'\.about(\w)', r'. about \1', full_text, flags=re.IGNORECASE)
            # Clean up multiple spaces
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Try to separate action from description
            if " about " in full_text.lower():
                parts = re.split(r'\s+about\s+', full_text, maxsplit=1, flags=re.IGNORECASE)
                activity["action"] = parts[0].strip()
                activity["description"] = parts[1].strip() if len(parts) > 1 else parts[0].strip()
            else:
                # For actions without "about", use the full text as description
                activity["description"] = full_text

        # Create a unique key to avoid duplicates
        activity_key = f"{activity['date']}|{activity['type']}|{activity['description'][:50]}"
        if activity_key not in seen_activities and activity["description"]:
            seen_activities.add(activity_key)
            activities.append(activity)

    return activities


def scrape_account(account_id: int, debug: bool = False, wait_time: int = 5000) -> dict:
    """Main scraping function using Playwright for JS rendering."""

    print(f"Extracting Chrome cookies for mediaos.com...", file=sys.stderr)
    cookies = get_chrome_cookies()
    print(f"Found {len(cookies)} cookies", file=sys.stderr)

    url = f"{MEDIAOS_BASE_URL}/accounts/account/{account_id}"
    print(f"Fetching account page: {url}", file=sys.stderr)

    activity_html = ""
    contracts_html = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        context.add_cookies(cookies)
        page = context.new_page()

        try:
            page.goto(url, wait_until="networkidle", timeout=30000)

            print(f"Waiting for content to render...", file=sys.stderr)
            page.wait_for_timeout(wait_time)

            # Wait for specific MediaOS elements
            try:
                page.wait_for_selector("#advertiserSummary, .timeline-entry, h2.mar-no", timeout=10000)
            except PlaywrightTimeout:
                print("Warning: Could not find expected MediaOS elements", file=sys.stderr)

            # Capture Activity tab HTML (default view)
            activity_html = page.content()

            if debug:
                debug_path = Path(f"/tmp/mediaos_debug_{account_id}_activity.html")
                debug_path.write_text(activity_html)
                print(f"DEBUG: Activity HTML saved to {debug_path}", file=sys.stderr)

                screenshot_path = f"/tmp/mediaos_debug_{account_id}_activity.png"
                page.screenshot(path=screenshot_path, full_page=True)
                print(f"DEBUG: Activity screenshot saved to {screenshot_path}", file=sys.stderr)

            # Click Contracts tab to get contracts data
            print("Clicking Contracts tab...", file=sys.stderr)
            try:
                # Try multiple selectors for the Contracts tab
                contracts_tab = page.locator("text=Contracts").first
                if contracts_tab.is_visible():
                    contracts_tab.click()
                    page.wait_for_timeout(2000)  # Wait for tab content to load

                    # Wait for table to appear
                    try:
                        page.wait_for_selector("table, .asg-table", timeout=5000)
                    except PlaywrightTimeout:
                        print("Warning: Contracts table not found after clicking tab", file=sys.stderr)

                    contracts_html = page.content()

                    if debug:
                        debug_path = Path(f"/tmp/mediaos_debug_{account_id}_contracts.html")
                        debug_path.write_text(contracts_html)
                        print(f"DEBUG: Contracts HTML saved to {debug_path}", file=sys.stderr)

                        screenshot_path = f"/tmp/mediaos_debug_{account_id}_contracts.png"
                        page.screenshot(path=screenshot_path, full_page=True)
                        print(f"DEBUG: Contracts screenshot saved to {screenshot_path}", file=sys.stderr)
                else:
                    print("Warning: Contracts tab not visible", file=sys.stderr)
                    contracts_html = activity_html
            except Exception as e:
                print(f"Warning: Could not click Contracts tab: {e}", file=sys.stderr)
                contracts_html = activity_html

        except PlaywrightTimeout as e:
            print(f"ERROR: Page load timed out: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            browser.close()

    # Check for login page
    if "login" in activity_html.lower() and "password" in activity_html.lower() and "sign in" in activity_html.lower():
        print("ERROR: Received login page. Your session may have expired.", file=sys.stderr)
        print("Please log into MediaOS in Chrome and try again.", file=sys.stderr)
        sys.exit(1)

    # Parse activity page for header, contacts, activity
    activity_soup = BeautifulSoup(activity_html, "html.parser")

    # Parse contracts page for contracts
    contracts_soup = BeautifulSoup(contracts_html, "html.parser") if contracts_html else activity_soup

    print("Parsing account data...", file=sys.stderr)

    result = {
        "account_id": account_id,
        "url": url,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "header": parse_header(activity_soup),
        "contacts": parse_contacts(activity_soup),
        "contracts": parse_contracts(contracts_soup),
        "activity": parse_activity(activity_soup),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Scrape MediaOS account data and output JSON"
    )
    parser.add_argument(
        "--account-id", "-a",
        type=int,
        required=True,
        help="MediaOS account ID to scrape"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (defaults to stdout)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Save rendered HTML and screenshot for debugging"
    )
    parser.add_argument(
        "--wait", "-w",
        type=int,
        default=5000,
        help="Milliseconds to wait for Vue to render (default: 5000)"
    )

    args = parser.parse_args()

    result = scrape_account(args.account_id, debug=args.debug, wait_time=args.wait)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output)
        print(f"Output written to {output_path}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
