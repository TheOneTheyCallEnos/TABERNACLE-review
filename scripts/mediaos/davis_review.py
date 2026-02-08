#!/usr/bin/env python3
"""
Davis Review Generator
Synthesizes account data into actionable sales materials.

Takes MediaOS scraper JSON output and generates:
- Deal Description (prospect summary, activity, won/lost contracts)
- Call Script (opening, discovery, pivot, objection handling, close)
- First Email (adaptive tone based on last contact direction)

Usage:
    python davis_review.py --input universal_ventilation.json
    python mediaos_scraper.py -a 2878 | python davis_review.py
    python davis_review.py --input universal_ventilation.json --no-search  # skip web search
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: beautifulsoup4 not installed. Run: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)


OUTPUT_DIR = Path.home() / "TABERNACLE" / "outputs" / "davis-reviews"


def slugify(text: str) -> str:
    """Convert text to a URL/filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def search_company_identity(company_name: str, location: str = "") -> dict:
    """
    Search for company identity information using DuckDuckGo HTML search.

    Returns dict with:
    - website: company website if found
    - snippet: brief description from search results
    - industry: inferred industry
    - additional_info: any other relevant info
    """
    result = {
        "website": "",
        "snippet": "",
        "industry": "",
        "additional_info": "",
    }

    if not company_name:
        return result

    # Build search query
    query = company_name
    if location:
        query += f" {location}"

    try:
        # Use DuckDuckGo HTML search
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract search results
        results = soup.select(".result")
        snippets = []

        for r in results[:5]:  # Check first 5 results
            title_el = r.select_one(".result__title")
            snippet_el = r.select_one(".result__snippet")
            url_el = r.select_one(".result__url")

            if title_el and snippet_el:
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True)
                result_url = url_el.get_text(strip=True) if url_el else ""

                # Check if this looks like the company's website
                company_words = company_name.lower().split()
                if any(word in result_url.lower() for word in company_words if len(word) > 3):
                    if not result.get("website"):
                        result["website"] = result_url

                snippets.append(snippet)

        # Combine snippets for company description
        if snippets:
            result["snippet"] = " | ".join(snippets[:2])

        # Try to infer industry from snippets
        industry_keywords = {
            "hvac": "HVAC / Mechanical",
            "heating": "HVAC / Mechanical",
            "ventilation": "HVAC / Mechanical",
            "air conditioning": "HVAC / Mechanical",
            "construction": "Construction",
            "contractor": "Construction / Contracting",
            "plumbing": "Plumbing",
            "electrical": "Electrical",
            "manufacturing": "Manufacturing",
            "restaurant": "Food Service",
            "hotel": "Hospitality",
            "retail": "Retail",
            "software": "Technology",
            "healthcare": "Healthcare",
            "medical": "Healthcare",
            "legal": "Legal Services",
            "accounting": "Financial Services",
            "real estate": "Real Estate",
        }

        combined_text = " ".join(snippets).lower()
        for keyword, industry in industry_keywords.items():
            if keyword in combined_text:
                result["industry"] = industry
                break

    except Exception as e:
        print(f"Warning: Web search failed: {e}", file=sys.stderr)

    return result


def analyze_contact_direction(activity: list) -> str:
    """
    Analyze the activity feed to determine last contact direction.

    Returns:
    - "inbound" if customer reached out
    - "outbound" if we reached out
    - "ghosted" if multiple outbound with no response
    - "unknown" if can't determine
    """
    if not activity:
        return "unknown"

    # Count recent outbound attempts
    outbound_count = 0
    has_inbound = False

    for item in activity[:5]:  # Check last 5 activities
        desc = item.get("description", "").lower()
        activity_type = item.get("type", "").lower()

        # Outbound indicators
        if activity_type in ["call", "email"] and any(x in desc for x in ["called", "emailed", "na ", "lvm", "no answer", "voicemail"]):
            outbound_count += 1

        # Inbound indicators
        if any(x in desc for x in ["received", "inbound", "they called", "replied", "responded"]):
            has_inbound = True
            break

    if has_inbound:
        return "inbound"
    elif outbound_count >= 3:
        return "ghosted"
    elif outbound_count > 0:
        return "outbound"

    return "unknown"


def format_currency(value: str) -> str:
    """Clean up currency formatting."""
    if not value:
        return "$0"
    # Already formatted
    if value.startswith("$"):
        return value
    # Try to format
    try:
        num = float(re.sub(r'[^\d.]', '', value))
        return f"${num:,.2f}"
    except:
        return value


def generate_deal_description(data: dict, web_info: dict) -> str:
    """Generate the Deal Description section."""
    header = data.get("header", {})
    contracts = data.get("contracts", [])
    activity = data.get("activity", [])

    # Part (i): Prospect Summary
    company = header.get("company_name", "Unknown Company")
    industry = web_info.get("industry") or "Industry unknown"
    description = header.get("description", "")
    web_snippet = web_info.get("snippet", "")

    prospect_summary = f"**{company}** is a {industry} company"
    if header.get("address"):
        # Extract city from address
        address = header.get("address", "")
        city_match = re.search(r',\s*([A-Za-z\s]+),?\s*[A-Z]{2}', address)
        if city_match:
            prospect_summary += f" based in {city_match.group(1).strip()}"

    prospect_summary += "."

    if description:
        prospect_summary += f" {description[:200]}..."
    elif web_snippet:
        prospect_summary += f" {web_snippet[:200]}..."

    # Part (ii): Recent Activity Summary
    activity_summary = "No recent activity recorded."
    if activity:
        recent = activity[:3]
        activity_lines = []
        for a in recent:
            date = a.get("date", "")
            atype = a.get("type", "activity")
            desc = a.get("description", "")[:100]
            if desc:
                activity_lines.append(f"- {date}: {atype.capitalize()} - {desc}")
        if activity_lines:
            activity_summary = "\n".join(activity_lines)

    # Part (iii): Last Closed Sales (Won contracts)
    won_contracts = [c for c in contracts if c.get("sold_status", "").lower() == "won"]
    won_summary = "No won contracts on record."
    if won_contracts:
        won_lines = []
        for c in won_contracts[-3:]:  # Last 3 won
            num = c.get("contract_number", "?")
            total = format_currency(c.get("total", ""))
            signed = c.get("signed_date", "")
            sold_by = c.get("sold_by", "")
            won_lines.append(f"- Contract #{num}: {total} (Signed: {signed}, Rep: {sold_by})")
        won_summary = "\n".join(won_lines)

    # Part (iv): Last Lost Proposals
    lost_contracts = [c for c in contracts if c.get("sold_status", "").lower() == "lost"]
    lost_summary = "No lost proposals on record."
    if lost_contracts:
        lost_lines = []
        for c in lost_contracts[-3:]:  # Last 3 lost
            num = c.get("contract_number", "?")
            total = format_currency(c.get("total", ""))
            created = c.get("created_date", "")
            sold_by = c.get("sold_by", "")
            lost_lines.append(f"- Proposal #{num}: {total} (Created: {created}, Rep: {sold_by})")
        lost_summary = "\n".join(lost_lines)

    return f"""### (i) Prospect Summary
{prospect_summary}

### (ii) Recent Activity Summary
{activity_summary}

### (iii) Last Closed Sales (Won)
{won_summary}

### (iv) Last Lost Proposals
{lost_summary}"""


def generate_call_script(data: dict, web_info: dict, contact_direction: str) -> str:
    """Generate the Call Script section."""
    header = data.get("header", {})
    contacts = data.get("contacts", [])
    contracts = data.get("contracts", [])
    activity = data.get("activity", [])

    company = header.get("company_name", "the company")
    contact_name = contacts[0].get("name", "there") if contacts else "there"
    first_name = contact_name.split()[0] if contact_name != "there" else "there"

    # Determine if we have history
    has_won = any(c.get("sold_status", "").lower() == "won" for c in contracts)
    has_lost = any(c.get("sold_status", "").lower() == "lost" for c in contracts)

    # Opening based on contact direction
    if contact_direction == "inbound":
        opening = f"Hi {first_name}, thanks for reaching out! I'm calling to follow up on your inquiry."
    elif contact_direction == "ghosted":
        opening = f"Hi {first_name}, this is [Name] from Davis Media. I know we've been playing phone tag — I wanted to try one more time to connect."
    elif has_won:
        opening = f"Hi {first_name}, this is [Name] from Davis Media. We've worked together before on some advertising and I wanted to check in."
    else:
        opening = f"Hi {first_name}, this is [Name] from Davis Media. I'm reaching out to folks in the {web_info.get('industry', 'industry')} space."

    # Connection point
    if has_won:
        connection = f"We helped {company} with advertising back in [YEAR] — wanted to see how things have been going."
    elif web_info.get("industry"):
        connection = f"We work with a lot of {web_info.get('industry')} companies on their marketing."
    else:
        connection = "We help companies like yours get in front of decision-makers in your industry."

    # Discovery questions
    discovery = """- "What's your current approach to reaching new customers?"
- "Are you doing any advertising or sponsorships right now?"
- "What's worked well for you in the past? What hasn't?"
- "Who typically handles marketing decisions there?\""""

    # Pivot to opportunity
    pivot = """If there's interest: "We have some great opportunities coming up with [PUBLICATION/EVENT].
Would it make sense for me to send over some info on what that could look like for you?\""""

    # Objection handling
    objections = """- **"Not interested"**: "I understand — just curious, is it a budget thing or timing?"
- **"Send me info"**: "Happy to! What specifically would be most useful — rates, audience data, or examples?"
- **"We tried it before"**: "What happened? I'd love to understand what didn't work so we can do better.\""""

    # Close
    close = """- **If warm**: "Great, I'll send that over. Can I follow up [DAY] to walk through it together?"
- **If cold**: "No problem — mind if I check back in [TIMEFRAME] when you're planning next year?"\""""

    return f"""**Opening:**
{opening}

**Connection Point:**
{connection}

**Discovery Questions:**
{discovery}

**Pivot to Opportunity:**
{pivot}

**Objection Handling:**
{objections}

**Close:**
{close}"""


def generate_first_email(data: dict, web_info: dict, contact_direction: str) -> str:
    """Generate the First Email section."""
    header = data.get("header", {})
    contacts = data.get("contacts", [])
    contracts = data.get("contracts", [])

    company = header.get("company_name", "your company")
    contact_name = contacts[0].get("name", "") if contacts else ""
    first_name = contact_name.split()[0] if contact_name else ""

    has_won = any(c.get("sold_status", "").lower() == "won" for c in contracts)
    industry = web_info.get("industry", "your industry")

    # Adaptive tone based on contact direction
    if contact_direction == "inbound":
        subject = f"Following up on your inquiry"
        greeting = f"Hi {first_name}," if first_name else "Hi there,"
        opener = "Thanks for reaching out! I wanted to follow up on your inquiry and share some options."
        tone = "warm"
    elif contact_direction == "ghosted":
        subject = f"Quick note from Davis Media"
        greeting = f"Hi {first_name}," if first_name else "Hi,"
        opener = f"I've tried reaching you a few times — I know how busy things get. Just wanted to send a quick note in case email works better."
        tone = "humble"
    elif has_won:
        subject = f"Reconnecting — Davis Media"
        greeting = f"Hi {first_name}," if first_name else "Hi,"
        opener = f"It's been a while since we last worked together, and I wanted to reconnect. Hope things are going well at {company}!"
        tone = "reconnect"
    else:
        subject = f"Advertising opportunity for {company}"
        greeting = f"Hi {first_name}," if first_name else "Hi,"
        opener = f"I'm reaching out from Davis Media — we work with companies in {industry} on targeted advertising and sponsorship opportunities."
        tone = "cold"

    # Body based on tone
    if tone == "warm":
        body = f"""I'd love to learn more about what you're looking for and share some ideas on how we can help {company} reach your target audience.

Would you have 15 minutes this week for a quick call? I can also send over some info on our current opportunities if that's easier."""

    elif tone == "humble":
        body = f"""I don't want to be a pest — I'll keep this short. We have some upcoming opportunities that I think could be a great fit for {company}.

If now's not the right time, no worries at all. But if there's any interest, I'd be happy to send over details or jump on a quick call."""

    elif tone == "reconnect":
        body = f"""We have some exciting new opportunities coming up that I think could be valuable for {company}, especially given your previous advertising with us.

Would love to catch up and share what's new. Do you have 15 minutes this week?"""

    else:  # cold
        body = f"""We help companies like yours get in front of decision-makers through our publications and events. I thought there might be some synergy worth exploring.

Would you be open to a brief call to see if there's a fit? I can also send over a media kit if you'd like to review on your own time."""

    closing = """Let me know what works best for you.

Best,
[Your Name]
Davis Media"""

    return f"""**Subject:** {subject}

---

{greeting}

{opener}

{body}

{closing}"""


def generate_review(data: dict, skip_search: bool = False) -> str:
    """Generate the complete Davis Review markdown document."""
    header = data.get("header", {})
    company = header.get("company_name", "Unknown Company")
    activity = data.get("activity", [])

    # Web search for company identity
    web_info = {"website": "", "snippet": "", "industry": "", "additional_info": ""}
    if not skip_search:
        print("Searching for company identity...", file=sys.stderr)
        location = ""
        if header.get("address"):
            # Extract city/province from address
            match = re.search(r'([A-Za-z\s]+),?\s*[A-Z]{2}\s*[A-Z0-9]{3}', header.get("address", ""))
            if match:
                location = match.group(1).strip()
        web_info = search_company_identity(company, location)

    # Analyze contact direction
    contact_direction = analyze_contact_direction(activity)
    print(f"Contact direction: {contact_direction}", file=sys.stderr)

    # Generate sections
    deal_description = generate_deal_description(data, web_info)
    call_script = generate_call_script(data, web_info, contact_direction)
    first_email = generate_first_email(data, web_info, contact_direction)

    # Build document
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""# Davis Review: {company}

**Generated:** {timestamp}
**Account URL:** {data.get("url", "N/A")}
**Contact Direction:** {contact_direction.capitalize()}

---

## Deal Description

{deal_description}

---

## Call Script

{call_script}

---

## First Email

{first_email}

---

## Raw Data Summary

- **Phone:** {header.get("phone", "N/A")}
- **Email:** {header.get("email", "N/A")}
- **Address:** {header.get("address", "N/A")}
- **Tags:** {", ".join(header.get("tags", [])) or "None"}
- **Last Insertion:** {header.get("last_insertion_date", "N/A")}
- **Last Contract:** {header.get("last_contract_date", "N/A")}
- **Last Activity:** {header.get("last_activity_date", "N/A")}
- **Web Search Industry:** {web_info.get("industry", "N/A")}
- **Web Search Website:** {web_info.get("website", "N/A")}
"""

    return doc


def main():
    parser = argparse.ArgumentParser(
        description="Generate Davis Review from MediaOS scraper JSON"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON file (or reads from stdin if not provided)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output markdown file (defaults to ~/TABERNACLE/outputs/davis-reviews/{slug}.md)"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip web search for company identity"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output to stdout instead of file"
    )

    args = parser.parse_args()

    # Load JSON data
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            # Try looking in the outputs directory
            alt_path = OUTPUT_DIR / args.input
            if alt_path.exists():
                input_path = alt_path
            else:
                print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
                sys.exit(1)
        data = json.loads(input_path.read_text())
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("ERROR: No input file provided and no data on stdin", file=sys.stderr)
            print("Usage: python davis_review.py --input file.json", file=sys.stderr)
            print("   or: python mediaos_scraper.py -a 2878 | python davis_review.py", file=sys.stderr)
            sys.exit(1)
        data = json.load(sys.stdin)

    # Generate review
    review = generate_review(data, skip_search=args.no_search)

    # Determine output path
    if args.stdout:
        print(review)
    else:
        company = data.get("header", {}).get("company_name", "unknown")
        slug = slugify(company)

        if args.output:
            output_path = Path(args.output)
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / f"{slug}.md"

        output_path.write_text(review)
        print(f"Review written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
