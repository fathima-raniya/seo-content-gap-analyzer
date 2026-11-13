# SEO Content Gap Analyzer - Advanced

One-file Streamlit app for any website. Enter your site and competitors to get page-health points,
competitor comparison charts, ranked content gaps and content briefs.

## Run
```
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m streamlit run app.py
```
Press Enter if Streamlit asks for an email. Keep the `.streamlit` folder next to `app.py` (theme).

## What is new
- **Page-health points (0-100)** for every site: title length, meta description, one H1, 2+ H2s, 300+ words, 3+ internal links.
- **Key points**: auto-written takeaways on the Overview tab.
- **Comparison charts**: health points, pass rate per check, topic coverage by competitor,
  who-covers-what heatmap, words / H2 / internal-link averages, keyword gaps.
- **Opportunity points**: each gap is scored out of 100 (relevance 35, competitor coverage 25, gap size 20,
  commercial intent 20) with a stacked breakdown chart, intent donut and opportunity map.
- **Page audit**: your pages ranked weakest first, with the exact checks each one fails.

## New: page changes and who is better
- **Fix your pages**: every page of your site ranked by points you can gain, with exact changes
  (title length, meta description, H1, H2s, word count, internal links, and gap topics to add to that page),
  plus new pages to create. Download the plan as CSV.
- **Who is better**: overall comparison score for every site, score by area (health, snippets, depth,
  structure, linking, topic breadth), head-to-head table against you and a verdict.

## Tabs
Overview - Who is better - Fix your pages - Content gaps - Competitor comparison - Content briefs

## Tips
- Use 2-4 competitors in the same niche; start with 10-15 pages per site.
- Too many unrelated topics: raise Minimum relevance. Too few gaps: lower Match threshold.
- Sites that block crawlers or build pages with JavaScript may return too few pages.
