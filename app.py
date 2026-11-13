"""SEO Content Gap Analyzer - Advanced edition (works for ANY website).

Adds to the base app: page-health points for every site, competitor comparison charts,
topic-coverage heatmap, opportunity score breakdown and auto-generated key points.

Install:  pip install -r requirements.txt
Run:      python -m streamlit run app.py
"""
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import altair as alt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UA = "SEOGapBot/1.0 (educational SEO analysis; respects robots.txt)"
HEADERS = {"User-Agent": UA}
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z\+\-]{2,}\b"
GENERIC = {"home", "contact us", "contact", "about us", "about", "privacy policy", "our services",
           "read more", "faq", "faqs", "key features", "features", "benefits", "overview",
           "why choose us", "get in touch", "our team", "blog", "testimonials", "our clients",
           "related posts", "leave a reply", "subscribe", "newsletter", "request a demo"}
GENERIC_START = re.compile(r"^(why choose|our |get in touch|follow us|subscribe|related|share|"
                           r"leave a|comments?|recent posts|quick links|call us|email us|copyright)", re.I)

# On-page health checks: key, label, points (total = 100)
CHECKS = [("title", "Title is 30-65 characters", 15),
          ("meta", "Meta description is 70-165 characters", 15),
          ("h1", "Exactly one H1 heading", 15),
          ("h2", "At least two H2 subheadings", 15),
          ("words", "300 or more words", 20),
          ("links", "Three or more internal links", 20)]
CHECK_LABEL = {k: l for k, l, _ in CHECKS}
SCORE_PARTS = {"pts_relevance": "Relevance", "pts_coverage": "Competitor coverage",
               "pts_gap": "Gap size", "pts_intent": "Commercial intent"}


# ================================================================ analysis engine
def fix(url):
    url = url.strip()
    return url if url.startswith("http") else "https://" + url


def domain(url):
    return urlparse(fix(url)).netloc.lower().replace("www.", "")


def valid_topic(t):
    t = re.sub(r"\s+", " ", t).strip()
    low = t.lower().strip(" .:!?")
    return (2 <= len(t.split()) <= 12 and len(t) >= 8 and low not in GENERIC
            and not GENERIC_START.match(low))


def intent_of(topic):
    low = topic.lower()
    if re.search(r"\b(price|pricing|cost|buy|quote|hire|services?|near me|company|agency|"
                 r"consultants?|book|demo|trial|packages?)\b", low):
        return "Transactional", 1.0
    if re.search(r"\b(best|top|vs|versus|alternatives?|review|reviews|compare|comparison)\b", low):
        return "Comparison", 0.85
    return "Informational", 0.55


def page_checks(raw_title, meta, h1_count, h2_count, words, links):
    c = {"title": 30 <= len(raw_title) <= 65, "meta": 70 <= len(meta) <= 165, "h1": h1_count == 1,
         "h2": h2_count >= 2, "words": words >= 300, "links": links >= 3}
    return c, sum(p for k, _, p in CHECKS if c[k])


def crawl(start, max_pages, delay, on_page=None):
    start = fix(start).rstrip("/")
    site = domain(start)
    root = "{0.scheme}://{0.netloc}".format(urlparse(start))
    robots, robots_ok = RobotFileParser(), False
    try:
        r = requests.get(root + "/robots.txt", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            robots.parse(r.text.splitlines())
            robots_ok = True
            cd = robots.crawl_delay(UA)
            delay = max(delay, float(cd)) if cd else delay
    except requests.RequestException:
        pass

    queue, seen, pages = deque([start]), set(), []
    while queue and len(pages) < max_pages:
        url = queue.popleft().split("#")[0].split("?")[0].rstrip("/") or root
        if url in seen or domain(url) != site:
            continue
        if re.search(r"\.(jpg|jpeg|png|gif|svg|webp|pdf|zip|css|js|xml|json|mp4|docx?|xlsx?)$", url, re.I):
            continue
        seen.add(url)
        if robots_ok and not robots.can_fetch(UA, url):
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
        except requests.RequestException:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        meta_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        meta = (meta_tag.get("content") or "").strip() if meta_tag else ""
        for t in soup(["script", "style", "nav", "footer", "aside", "form", "noscript", "svg"]):
            t.decompose()
        raw_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        parts = [p for p in re.split(r"\s[\|\-–—:•»]\s", raw_title) if len(p.split()) >= 2]
        title = parts[0] if parts else raw_title
        h1 = [h.get_text(" ", strip=True) for h in soup.find_all("h1")]
        h2 = [h.get_text(" ", strip=True) for h in soup.find_all("h2")]
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        words = len(text.split())
        internal = []
        for a in soup.find_all("a", href=True):
            link = urljoin(url + "/", a["href"]).split("#")[0].split("?")[0].rstrip("/")
            if link.startswith("http") and domain(link) == site:
                internal.append(link)
                queue.append(link)
        n_links = len({l for l in internal if l != url})
        if words >= 30:
            checks, points = page_checks(raw_title, meta, len(h1), len(h2), words, n_links)
            pages.append({"site": site, "url": url, "title": title, "topics": [title] + h1 + h2,
                          "subheads": h2, "text": text[:4000], "words": words, "h2_count": len(h2),
                          "links": n_links, "points": points, "title_len": len(raw_title),
                          "meta_len": len(meta), "h1_count": len(h1),
                          **{f"c_{k}": v for k, v in checks.items()}})
        if on_page:
            on_page(site, len(pages))
        time.sleep(delay)
    return pages


def auto_focus_terms(pages, n=25):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
                          token_pattern=TOKEN, max_features=3000)
    X = vec.fit_transform([p["text"] for p in pages])
    scores = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    return [terms[i] for i in scores.argsort()[::-1][:n]]


def analyze(my_pages, comp_pages, threshold, focus_terms, min_relevance):
    """Returns (gaps DataFrame, per-competitor topic stats DataFrame)."""
    empty = (pd.DataFrame(), pd.DataFrame())
    f_units = [(p["text"], p["url"]) for p in my_pages]
    f_units += [(t, p["url"]) for p in my_pages for t in p["topics"] if valid_topic(t)]
    topics = {}
    for p in comp_pages:
        for t in p["topics"]:
            if valid_topic(t):
                clean = re.sub(r"\s+", " ", t).strip()
                d = topics.setdefault(clean.lower(), {"topic": clean, "sites": set(), "urls": set()})
                d["sites"].add(p["site"])
                d["urls"].add(p["url"])
    if not topics or not f_units:
        return empty

    items = list(topics.values())
    profile = " ".join(p["text"] for p in my_pages)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
                          token_pattern=TOKEN).fit([u[0] for u in f_units] + [i["topic"] for i in items])

    TT = cosine_similarity(vec.transform([i["topic"] for i in items]))        # merge near-duplicates
    merged, absorbed = [], set()
    for a in range(len(items)):
        if a in absorbed:
            continue
        for b in range(a + 1, len(items)):
            if b not in absorbed and TT[a, b] >= 0.6:
                items[a]["sites"] |= items[b]["sites"]
                items[a]["urls"] |= items[b]["urls"]
                absorbed.add(b)
        merged.append(items[a])
    items = merged

    T = vec.transform([i["topic"] for i in items])
    sims = cosine_similarity(T, vec.transform([u[0] for u in f_units]))
    prof_sim = cosine_similarity(T, vec.transform([profile])).ravel()
    focus_words = {w for t in focus_terms for w in re.findall(r"[a-z]{3,}", t.lower())}
    page_text = {p["url"]: p["text"] for p in comp_pages}
    page_heads = {p["url"]: p["subheads"] for p in comp_pages}
    profile_vec = vec.transform([profile])
    n_comp = len({p["site"] for p in comp_pages})

    stats, rows = {}, []
    for i, it in enumerate(items):
        best = float(sims[i].max())
        for s in it["sites"]:
            stats.setdefault(s, {"covered": 0, "gap": 0, "other": 0})
        if best >= threshold:                                   # already covered by your site
            for s in it["sites"]:
                stats[s]["covered"] += 1
            continue
        words = set(re.findall(r"[a-z]{3,}", it["topic"].lower()))
        kw = min(1.0, len(words & focus_words) / 2)
        ctx = cosine_similarity(vec.transform([page_text[u] for u in it["urls"]]), profile_vec).max()
        relevance = round(0.4 * kw + 0.25 * min(1.0, prof_sim[i] / 0.2) + 0.35 * min(1.0, ctx / 0.25), 2)
        if relevance < min_relevance:                           # off-topic for your site
            for s in it["sites"]:
                stats[s]["other"] += 1
            continue
        for s in it["sites"]:
            stats[s]["gap"] += 1
        intent, commercial = intent_of(it["topic"])
        coverage = len(it["sites"]) / max(n_comp, 1)
        gap_size = max(0.0, 1 - best / threshold)
        parts = {"pts_relevance": 35 * relevance, "pts_coverage": 25 * coverage,
                 "pts_gap": 20 * gap_size, "pts_intent": 20 * commercial}
        score = round(sum(parts.values()), 1)
        near = f_units[int(sims[i].argmax())][1]
        subs = []
        for u in it["urls"]:
            for h in page_heads.get(u, []):
                if valid_topic(h) and h.lower() != it["topic"].lower() and h not in subs:
                    subs.append(h)
        improve = best >= threshold * 0.7
        rows.append({
            "topic": it["topic"], "priority": "High" if score >= 65 else "Medium" if score >= 45 else "Low",
            "score": score, "intent": intent, "relevance": relevance, "gap_size": round(gap_size, 2),
            "competitors": len(it["sites"]), "covered_by": ", ".join(sorted(it["sites"])),
            "sites": sorted(it["sites"]),
            "action": "Improve existing page" if improve else "Create new page",
            "closest_page": near if improve else "", "similarity": round(best, 2),
            "subtopics": "; ".join(subs[:6]), "reference": sorted(it["urls"])[0],
            **{k: round(v, 1) for k, v in parts.items()}})
    stats_df = pd.DataFrame([{"site": s, **v} for s, v in stats.items()])
    gaps = (pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
            if rows else pd.DataFrame())
    return gaps, stats_df


def keyword_gaps(my_pages, comp_pages, top_n=40):
    docs = [p["text"] for p in my_pages] + [p["text"] for p in comp_pages]
    if len(docs) < 3 or not my_pages or not comp_pages:
        return pd.DataFrame()
    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", sublinear_tf=True,
                          min_df=2, token_pattern=TOKEN, max_features=20000)
    X = vec.fit_transform(docs)
    k = len(my_pages)
    mine = (X[:k] > 0).sum(axis=0).A1
    used = (X[k:] > 0).sum(axis=0).A1
    avg = X[k:].mean(axis=0).A1
    terms = vec.get_feature_names_out()
    mask = (mine == 0) & (used >= 2)
    df = pd.DataFrame({"keyword": terms[mask], "competitor pages": used[mask], "weight": avg[mask].round(4)})
    return df.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)


def scoreboard(pages_df, my_site, topic_counts):
    rows = []
    for site, g in pages_df.groupby("site"):
        rows.append({"site": site, "role": "You" if site == my_site else "Competitor", "pages": len(g),
                     "health_points": round(g.points.mean(), 1), "avg_words": int(g.words.mean()),
                     "avg_h2": round(g.h2_count.mean(), 1), "avg_links": round(g.links.mean(), 1),
                     "unique_topics": topic_counts.get(site, 0)})
    return pd.DataFrame(rows).sort_values("health_points", ascending=False).reset_index(drop=True)


def pass_rates(pages_df):
    rows = []
    for site, g in pages_df.groupby("site"):
        for k, label, _ in CHECKS:
            rows.append({"site": site, "check": label, "rate": round(100 * g[f"c_{k}"].mean(), 0)})
    return pd.DataFrame(rows)


WEIGHT = {k: p for k, _, p in CHECKS}
CATS = [("On-page health", "health_points", 0.30), ("Search snippets (title + meta)", "snippet", 0.15),
        ("Content depth (avg words)", "avg_words", 0.15), ("Content structure (avg H2)", "avg_h2", 0.10),
        ("Internal linking (avg links)", "avg_links", 0.10), ("Topic breadth (distinct topics)", "unique_topics", 0.20)]


def page_fixes(r, bench):
    """Concrete changes for one page: list of (check key, instruction, points gained)."""
    fx = []
    t, m, h1, h2, w, ln = r["title_len"], r["meta_len"], r["h1_count"], r["h2_count"], r["words"], r["links"]
    if not r["c_title"]:
        fx.append(("title", "Add a title tag of 30-65 characters that names the page topic." if t == 0 else
                   f"Lengthen the title from {t} to 30-65 characters: add the main keyword and a benefit." if t < 30 else
                   f"Shorten the title from {t} to 65 characters or fewer so search results do not cut it off.",
                   WEIGHT["title"]))
    if not r["c_meta"]:
        fx.append(("meta", "Add a meta description of 70-165 characters that summarises the page and invites the click."
                   if m == 0 else f"Extend the meta description from {m} to 70-165 characters." if m < 70 else
                   f"Shorten the meta description from {m} to 165 characters or fewer.", WEIGHT["meta"]))
    if not r["c_h1"]:
        fx.append(("h1", "Add one H1 heading that states the page topic." if h1 == 0 else
                   f"Keep a single H1 and change the other {h1 - 1} H1 headings to H2.", WEIGHT["h1"]))
    if not r["c_h2"]:
        need = 2 - h2
        fx.append(("h2", f"Add {need} more H2 subheading{'s' if need > 1 else ''} to split the content into clear "
                   f"sections (competitors average {bench['h2']:.1f} per page).", WEIGHT["h2"]))
    if not r["c_words"]:
        target = max(300, int(round(bench["words"], -1)))
        fx.append(("words", f"Expand the content from {w} to about {target} words (the minimum is 300; competitors "
                   f"average {int(bench['words'])}). Add detail, examples and answers to common questions.",
                   WEIGHT["words"]))
    if not r["c_links"]:
        need = 3 - ln
        fx.append(("links", f"Add {need} more internal link{'s' if need > 1 else ''} to related pages on your site.",
                   WEIGHT["links"]))
    return fx


def build_page_plan(pages_df, gaps, my_site, bench):
    gap_map = {}
    if len(gaps):
        for g in gaps[(gaps.action == "Improve existing page") & (gaps.closest_page != "")].itertuples():
            gap_map.setdefault(g.closest_page, []).append(g.topic)
    rows = []
    for r in pages_df[pages_df.site == my_site].to_dict("records"):
        fx = page_fixes(r, bench)
        topics = gap_map.get(r["url"], [])
        rows.append({"url": r["url"], "title": r["title"], "points": int(r["points"]),
                     "gain": sum(p for _, _, p in fx), "fixes": fx, "add_topics": topics,
                     "n_changes": len(fx) + (1 if topics else 0)})
    df = pd.DataFrame(rows)
    return df.sort_values(["gain", "n_changes"], ascending=False).reset_index(drop=True) if len(df) else df


def compare_sites(sb, rates, my_site):
    snip = rates[rates.check.isin([CHECK_LABEL["title"], CHECK_LABEL["meta"]])].groupby("site").rate.mean()
    raw = sb.set_index("site").copy()
    raw["snippet"] = snip
    wide = pd.DataFrame({label: raw[col] for label, col, _ in CATS}).T.astype(float)   # rows: categories
    norm = wide.div(wide.max(axis=1).replace(0, 1), axis=0) * 100                     # best site = 100
    weights = pd.Series({label: w for label, _, w in CATS})
    overall = (norm.mul(weights, axis=0).sum() / weights.sum()).round(1).sort_values(ascending=False)
    rows = []
    for c in [x for x in wide.columns if x != my_site]:
        diff = (norm[c] - norm[my_site]).sort_values()
        a, b = overall[c], overall[my_site]
        rows.append({"site": c, "overall": a, "vs_you": round(a - b, 1),
                     "beats_you_in": int((wide[c] > wide[my_site]).sum()),
                     "you_lead_in": int((wide[c] < wide[my_site]).sum()),
                     "verdict": "Better than you" if a - b > 2 else "Behind you" if a - b < -2 else "About equal",
                     "their_edge": diff.index[-1] if diff.iloc[-1] > 0 else "-",
                     "your_edge": diff.index[0] if diff.iloc[0] < 0 else "-"})
    h2h = pd.DataFrame(rows).sort_values("overall", ascending=False).reset_index(drop=True)
    return {"wide": wide, "norm": norm, "overall": overall, "h2h": h2h,
            "best": wide.idxmax(axis=1), "winner": overall.index[0]}


def build_results(my_pages, comp_pages, my_site, threshold, focus, min_rel):
    pages_df = pd.DataFrame([{k: v for k, v in p.items() if k not in ("text", "topics", "subheads")}
                             for p in my_pages + comp_pages])
    gaps, stats = analyze(my_pages, comp_pages, threshold, focus, min_rel)
    counts = {}
    for p in my_pages + comp_pages:
        counts.setdefault(p["site"], set()).update(t.lower() for t in p["topics"] if valid_topic(t))
    counts = {s: len(v) for s, v in counts.items()}
    sb, rates = scoreboard(pages_df, my_site, counts), pass_rates(pages_df)
    cb = sb[sb.site != my_site]
    bench = {"words": float(cb.avg_words.mean()), "h2": float(cb.avg_h2.mean()), "links": float(cb.avg_links.mean())}
    return {"gaps": gaps, "stats": stats, "kw": keyword_gaps(my_pages, comp_pages), "focus": focus,
            "pages": pages_df, "sb": sb, "rates": rates,
            "my_site": my_site, "comps": sorted({p["site"] for p in comp_pages}),
            "my_topics": counts.get(my_site, 0), "bench": bench,
            "plan": build_page_plan(pages_df, gaps, my_site, bench),
            "cmp": compare_sites(sb, rates, my_site)}


def build_insights(res):
    sb, gaps, rates, stats, kw = res["sb"], res["gaps"], res["rates"], res["stats"], res["kw"]
    my_site = res["my_site"]
    me = sb[sb.site == my_site].iloc[0]
    comps = sb[sb.site != my_site]
    out = []
    cmp_, plan = res["cmp"], res["plan"]
    rank = list(cmp_["overall"].index).index(my_site) + 1
    if cmp_["winner"] == my_site:
        out.append(f"<b>You are the strongest site overall</b> ({cmp_['overall'][my_site]:.0f}/100 comparison score) "
                   f"among the {len(sb)} sites compared.")
    else:
        w = cmp_["winner"]
        edge = (cmp_["norm"][w] - cmp_["norm"][my_site]).sort_values(ascending=False).index[:2]
        out.append(f"<b>{w} is the strongest site overall</b> ({cmp_['overall'][w]:.0f}/100). You rank "
                   f"<b>#{rank} of {len(sb)}</b> ({cmp_['overall'][my_site]:.0f}/100). It leads you most on "
                   f"{edge[0].lower()} and {edge[1].lower()}.")
    if len(plan):
        need = plan[plan.gain > 0]
        top = need.head(5)
        lift = top.gain.sum() / len(plan)
        if len(need):
            out.append(f"<b>{len(need)} of your {len(plan)} analyzed pages need changes.</b> Fixing the "
                       f"{len(top)} most urgent page{'s' if len(top) > 1 else ''} alone lifts your average "
                       f"page-health from {me.health_points:.0f} to about {min(100, me.health_points + lift):.0f} "
                       f"points. See the Fix your pages tab.")
    best = comps.iloc[0]
    diff = me.health_points - best.health_points
    out.append(f"Your pages average <b>{me.health_points:.0f} of 100</b> page-health points. "
               + (f"That is {diff:.0f} points ahead of the strongest competitor, {best.site} ({best.health_points:.0f})."
                  if diff >= 0 else
                  f"{best.site} leads with {best.health_points:.0f}, so you are {abs(diff):.0f} points behind."))
    mine = rates[rates.site == my_site].set_index("check").rate
    avg_c = rates[rates.site != my_site].groupby("check").rate.mean()
    delta = (mine - avg_c).sort_values()
    if len(delta) and delta.iloc[0] < -10:
        w = delta.index[0]
        out.append(f"Biggest on-page weakness: <b>{w.lower()}</b>. {mine[w]:.0f}% of your pages pass, "
                   f"against {avg_c[w]:.0f}% for competitors on average.")
    elif len(delta):
        out.append("Your on-page basics are in line with competitors. The opportunity is in content coverage.")
    top_c = comps.sort_values("unique_topics", ascending=False).iloc[0]
    out.append(f"{top_c.site} covers <b>{int(top_c.unique_topics)}</b> distinct topics in the pages read, "
               f"compared with <b>{res['my_topics']}</b> on your site.")
    if len(gaps):
        t = gaps.iloc[0]
        out.append(f"Top opportunity: <b>{t.topic}</b> (score {t.score:.0f}), covered by "
                   f"{int(t.competitors)} of {len(res['comps'])} competitors.")
        quick = int((gaps.action == "Improve existing page").sum())
        if quick:
            out.append(f"<b>{quick}</b> gaps can be closed by improving a page you already have, "
                       "which is usually the fastest win.")
        mix = gaps.intent.value_counts()
        out.append(f"Most gaps are <b>{mix.index[0].lower()}</b> topics ({int(mix.iloc[0])} of {len(gaps)}).")
    else:
        out.append("No relevant content gaps were found. Try a lower match threshold or add a competitor.")
    if len(kw):
        out.append("Terms competitors use that you do not: <b>" + ", ".join(kw.keyword.head(4)) + "</b>.")
    return out


# =================================================================== charts
BLUE, INK, MUTED, GRID = "#2F5BEA", "#0F1B2D", "#5B6B82", "#E3E8F0"
COMP_COLORS = ["#0E8A7D", "#C2680A", "#6B4FBB", "#B42318", "#3D7EA6", "#7A8B2E"]
PRIORITY_RANGE = {"High": "#B42318", "Medium": "#B54708", "Low": "#8A97AB"}


def site_scale(my_site, comps):
    return alt.Scale(domain=[my_site] + comps,
                     range=[BLUE] + [COMP_COLORS[i % len(COMP_COLORS)] for i in range(len(comps))])


def finish(chart):
    return (chart.configure(font="IBM Plex Sans", background="transparent")
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID,
                            tickColor=GRID, labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_view(strokeWidth=0)
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, orient="bottom"))


def chart_health(sb, scale):
    order = sb.sort_values("health_points", ascending=False).site.tolist()
    base = alt.Chart(sb).encode(y=alt.Y("site:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)))
    bars = base.mark_bar(size=24).encode(
        x=alt.X("health_points:Q", title="Average page-health points (out of 100)",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("site:N", scale=scale, legend=None),
        tooltip=["site", "role", "pages", "health_points", "avg_words", "avg_h2", "avg_links"])
    text = base.mark_text(align="left", dx=6, color=INK, fontWeight=600).encode(
        x="health_points:Q", text=alt.Text("health_points:Q", format=".0f"))
    return finish((bars + text).properties(height=max(100, 52 * len(sb))))


def chart_checks(rates, scale, sites):
    order = [l for _, l, _ in CHECKS]
    ch = alt.Chart(rates).mark_bar().encode(
        y=alt.Y("check:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)),
        yOffset=alt.YOffset("site:N", sort=sites),
        x=alt.X("rate:Q", title="Share of pages that pass (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("site:N", scale=scale, legend=alt.Legend(title=None)),
        tooltip=["site", "check", "rate"])
    return finish(ch.properties(height=max(260, 24 * len(sites) * len(order))))


def chart_overall(overall, scale):
    df = overall.rename_axis("site").reset_index(name="score")
    base = alt.Chart(df).encode(y=alt.Y("site:N", sort=df.site.tolist(), title=None, axis=alt.Axis(labelLimit=240)))
    bars = base.mark_bar(size=26).encode(
        x=alt.X("score:Q", title="Overall comparison score (best site in each area = 100)",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("site:N", scale=scale, legend=None), tooltip=["site", "score"])
    text = base.mark_text(align="left", dx=6, color=INK, fontWeight=600).encode(
        x="score:Q", text=alt.Text("score:Q", format=".0f"))
    return finish((bars + text).properties(height=max(100, 54 * len(df))))


def chart_categories(norm, scale, sites):
    long = norm.reset_index(names="category").melt(id_vars="category", var_name="site", value_name="score")
    long["score"] = long.score.round(0)
    ch = alt.Chart(long).mark_bar().encode(
        y=alt.Y("category:N", sort=[c for c, _, _ in CATS], title=None, axis=alt.Axis(labelLimit=280)),
        yOffset=alt.YOffset("site:N", sort=sites),
        x=alt.X("score:Q", title="Score (best site in each area = 100)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("site:N", scale=scale, legend=alt.Legend(title=None)), tooltip=["site", "category", "score"])
    return finish(ch.properties(height=max(280, 24 * len(sites) * len(CATS))))


def chart_metric(sb, col, title, scale, fmt=".0f"):
    order = sb.sort_values(col, ascending=False).site.tolist()
    base = alt.Chart(sb).encode(y=alt.Y("site:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)))
    bars = base.mark_bar(size=18).encode(x=alt.X(f"{col}:Q", title=title),
                                         color=alt.Color("site:N", scale=scale, legend=None),
                                         tooltip=["site", col])
    text = base.mark_text(align="left", dx=5, color=INK).encode(x=f"{col}:Q", text=alt.Text(f"{col}:Q", format=fmt))
    return finish((bars + text).properties(height=max(90, 34 * len(sb))))


def chart_topic_split(stats):
    long = stats.rename(columns={"covered": "Already on your site", "gap": "Relevant gap",
                                 "other": "Not relevant to you"}).melt(
        id_vars="site", var_name="type", value_name="topics")
    order = stats.assign(t=stats.gap).sort_values("t", ascending=False).site.tolist()
    ch = alt.Chart(long).mark_bar(size=24).encode(
        y=alt.Y("site:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("topics:Q", title="Topics found on competitor pages", stack="zero"),
        color=alt.Color("type:N", title=None, sort=["Already on your site", "Relevant gap", "Not relevant to you"],
                        scale=alt.Scale(domain=["Already on your site", "Relevant gap", "Not relevant to you"],
                                        range=["#B8C4D6", "#B42318", "#E6EAF0"])),
        tooltip=["site", "type", "topics"])
    return finish(ch.properties(height=max(100, 52 * len(stats))))


def chart_heatmap(gaps, my_site, comps, n=12):
    top = gaps.head(n)
    rows = [{"topic": r.topic, "site": s, "status": "Covers" if s in r.sites else "Missing"}
            for r in top.itertuples() for s in [my_site] + comps]
    order = top.topic.tolist()
    ch = alt.Chart(pd.DataFrame(rows)).mark_rect(stroke="white", strokeWidth=3, cornerRadius=3).encode(
        x=alt.X("site:N", sort=[my_site] + comps, title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelLimit=140)),
        y=alt.Y("topic:N", sort=order, title=None, axis=alt.Axis(labelLimit=340)),
        color=alt.Color("status:N", title=None,
                        scale=alt.Scale(domain=["Covers", "Missing"], range=["#0E8A7D", "#EEF1F6"])),
        tooltip=["topic", "site", "status"])
    return finish(ch.properties(height=max(120, 30 * len(top))))


def chart_scatter(gaps):
    ch = alt.Chart(gaps).mark_circle(opacity=0.8, stroke="white", strokeWidth=1).encode(
        x=alt.X("gap_size:Q", title="Gap size (how far from your current content)", scale=alt.Scale(domain=[0, 1.05])),
        y=alt.Y("relevance:Q", title="Relevance to your site", scale=alt.Scale(domain=[0, 1.05])),
        size=alt.Size("competitors:Q", title="Competitors covering", scale=alt.Scale(range=[80, 520])),
        color=alt.Color("priority:N", title="Priority", scale=alt.Scale(
            domain=list(PRIORITY_RANGE), range=list(PRIORITY_RANGE.values()))),
        tooltip=["topic", "score", "priority", "competitors", "relevance", "gap_size"])
    return finish(ch.properties(height=340))


def chart_breakdown(gaps, n=10):
    top = gaps.head(n)
    long = top.melt(id_vars="topic", value_vars=list(SCORE_PARTS), var_name="part", value_name="points")
    long["part"] = long.part.map(SCORE_PARTS)
    ch = alt.Chart(long).mark_bar(size=20).encode(
        y=alt.Y("topic:N", sort=top.topic.tolist(), title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("points:Q", title="Opportunity points (out of 100)", stack="zero"),
        color=alt.Color("part:N", title=None, sort=list(SCORE_PARTS.values()), scale=alt.Scale(
            domain=list(SCORE_PARTS.values()), range=[BLUE, "#0E8A7D", "#8FA3BF", "#C2680A"])),
        tooltip=["topic", "part", "points"])
    return finish(ch.properties(height=max(140, 34 * len(top))))


def chart_intent(gaps):
    counts = gaps.intent.value_counts().rename_axis("intent").reset_index(name="topics")
    ch = alt.Chart(counts).mark_arc(innerRadius=58, stroke="white", strokeWidth=2).encode(
        theta="topics:Q",
        color=alt.Color("intent:N", title=None, scale=alt.Scale(
            domain=["Transactional", "Comparison", "Informational"], range=[BLUE, "#0E8A7D", "#8FA3BF"])),
        tooltip=["intent", "topics"])
    return finish(ch.properties(height=260))


def chart_keywords(kw, n=15):
    top = kw.head(n)
    ch = alt.Chart(top).mark_bar(size=16, color=BLUE).encode(
        y=alt.Y("keyword:N", sort=top.keyword.tolist(), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("weight:Q", title="Importance on competitor pages"),
        tooltip=["keyword", "competitor pages", "weight"])
    return finish(ch.properties(height=max(120, 24 * len(top))))


# ===================================================================== interface
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button { font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
.hero { background: #0F1B2D; color: #FFFFFF; border-radius: 6px; padding: 30px 34px; margin-bottom: 22px; }
.hero h1 { color: #FFFFFF; font-size: 1.85rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 8px 0; padding: 0; }
.hero p { color: #B8C4D6; font-size: 1.02rem; line-height: 1.55; margin: 0; max-width: 66ch; }
.panel-title { font-size: 1.05rem; font-weight: 600; color: #0F1B2D; margin: 14px 0 2px 0; }
.panel-note { font-size: 0.9rem; color: #5B6B82; margin: 0 0 8px 0; }
.kpis { display: flex; background: #FFFFFF; border: 1px solid #D9E0EA; border-radius: 6px; margin: 6px 0 16px 0; }
.kpi { flex: 1; padding: 16px 20px; border-right: 1px solid #D9E0EA; }
.kpi:last-child { border-right: none; }
.kpi .v { display: block; font-size: 1.9rem; font-weight: 600; color: #0F1B2D; font-variant-numeric: tabular-nums; line-height: 1.15; }
.kpi .l { font-size: 0.88rem; color: #5B6B82; }
.kpi.alert .v { color: #B42318; }
.kpi.good .v { color: #0E8A7D; }
.note { font-size: 0.88rem; color: #5B6B82; margin: 4px 0 10px 0; }
.verdict { background: #0F1B2D; color: #FFFFFF; border-radius: 6px; padding: 18px 24px; margin: 4px 0 14px 0; }
.verdict .big { font-size: 1.35rem; font-weight: 600; line-height: 1.35; margin: 0 0 6px 0; }
.verdict .sub { color: #B8C4D6; font-size: 0.97rem; line-height: 1.55; margin: 0; }
.fix-n { color: #2F5BEA; font-weight: 600; }
.insights { background: #FFFFFF; border: 1px solid #D9E0EA; border-left: 3px solid #2F5BEA; border-radius: 4px; padding: 6px 22px 6px 14px; margin: 6px 0 14px 0; }
.insights ul { margin: 8px 0; padding-left: 20px; }
.insights li { margin: 7px 0; line-height: 1.55; color: #0F1B2D; font-size: 0.98rem; }
.stButton > button[kind="primary"], .stDownloadButton > button { background: #2F5BEA; border: 1px solid #2F5BEA; color: #FFFFFF; font-weight: 500; border-radius: 5px; padding: 0.55rem 1.4rem; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background: #2349C4; border-color: #2349C4; color: #FFFFFF; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #D9E0EA; }
.stTabs [data-baseweb="tab"] { font-weight: 500; color: #5B6B82; padding: 10px 14px; }
.stTabs [aria-selected="true"] { color: #2F5BEA; }
@media (max-width: 700px) { .kpis { flex-direction: column; } .kpi { border-right: none; border-bottom: 1px solid #D9E0EA; } .hero { padding: 22px 20px; } }
</style>
"""

PRIORITY_COLOR = {"High": "color:#B42318;font-weight:600", "Medium": "color:#B54708;font-weight:600",
                  "Low": "color:#475467;font-weight:600"}


def style_priority(df):
    styler = df.style
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(lambda v: PRIORITY_COLOR.get(v, ""), subset=["priority"])


def kpi_strip(items):
    cells = "".join(f'<div class="kpi {cls}"><span class="v">{v}</span><span class="l">{l}</span></div>'
                    for v, l, cls in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def head(title, note=""):
    st.markdown(f'<p class="panel-title">{title}</p>' + (f'<p class="panel-note">{note}</p>' if note else ""),
                unsafe_allow_html=True)


def show(chart):
    st.altair_chart(chart, width="stretch", theme=None)


def csv_bytes(df):
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, list)).any():
            out[c] = out[c].map(lambda v: " | ".join(map(str, v)) if isinstance(v, list) else v)
    return out.to_csv(index=False).encode("utf-8")


def render(res):
    gaps, sb, stats, rates = res["gaps"], res["sb"], res["stats"], res["rates"]
    my_site, comps = res["my_site"], res["comps"]
    scale = site_scale(my_site, comps)
    sites = [my_site] + comps
    me = sb[sb.site == my_site].iloc[0]
    high = int((gaps.priority == "High").sum()) if len(gaps) else 0
    kpi_strip([(len(res["pages"]), "Pages analyzed", ""), (len(comps), "Competitors compared", ""),
               (len(gaps), "Content gaps found", ""), (high, "High-priority gaps", "alert" if high else ""),
               (f"{me.health_points:.0f}/100", "Your page-health points",
                "good" if me.health_points >= 70 else "")])
    st.markdown('<p class="note">Topics used to judge relevance: ' + ", ".join(res["focus"][:12]) + "</p>",
                unsafe_allow_html=True)

    t1, t6, t5, t2, t3, t4 = st.tabs(["Overview", "Who is better", "Fix your pages", "Content gaps",
                                      "Competitor comparison", "Content briefs"])

    with t1:
        head("Key points", "Generated from this analysis.")
        st.markdown('<div class="insights"><ul>' + "".join(f"<li>{x}</li>" for x in build_insights(res))
                    + "</ul></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            head("Page-health points", "Each page earns up to 100 points for six on-page checks. "
                 "Shown as the average per site.")
            show(chart_health(sb, scale))
        with c2:
            head("Where each site passes or fails", "Share of pages passing each on-page check.")
            show(chart_checks(rates, scale, sites))
        head("Site scoreboard")
        board = sb.rename(columns={"site": "Site", "role": "Role", "pages": "Pages", "health_points": "Health points",
                                   "avg_words": "Avg words", "avg_h2": "Avg H2 per page",
                                   "avg_links": "Avg internal links", "unique_topics": "Distinct topics"})
        st.dataframe(board, width="stretch", hide_index=True,
                     column_config={"Health points": st.column_config.ProgressColumn(
                         "Health points", min_value=0, max_value=100, format="%.0f")})
        st.download_button("Download scoreboard (CSV)", csv_bytes(board), "site_scoreboard.csv", "text/csv")

    with t2:
        if gaps.empty:
            st.info("No gaps found with these settings. Lower the match threshold or minimum relevance, "
                    "or add another competitor.")
        else:
            f1, f2 = st.columns(2)
            pri = f1.multiselect("Priority", ["High", "Medium", "Low"], ["High", "Medium", "Low"])
            ints = f2.multiselect("Search intent", sorted(gaps.intent.unique()), sorted(gaps.intent.unique()))
            view = gaps[gaps.priority.isin(pri) & gaps.intent.isin(ints)]
            if view.empty:
                st.info("No gaps match these filters.")
            else:
                st.dataframe(style_priority(view[["topic", "priority", "score", "intent", "action",
                                                  "competitors", "relevance"]]),
                             width="stretch", hide_index=True,
                             column_config={"topic": st.column_config.TextColumn("Topic", width="large"),
                                            "priority": "Priority",
                                            "score": st.column_config.ProgressColumn("Opportunity points", min_value=0,
                                                                                     max_value=100, format="%.0f"),
                                            "intent": "Search intent", "action": "Recommended action",
                                            "competitors": st.column_config.NumberColumn("Competitors covering"),
                                            "relevance": st.column_config.NumberColumn("Relevance", format="%.2f")})
                st.download_button("Download gaps (CSV)", csv_bytes(view.drop(columns=["sites"])),
                                   "seo_content_gaps.csv", "text/csv")
                c1, c2 = st.columns([3, 2], gap="large")
                with c1:
                    head("How each top opportunity earned its points",
                         "Relevance 35, competitor coverage 25, gap size 20, commercial intent 20.")
                    show(chart_breakdown(view))
                with c2:
                    head("Gaps by search intent")
                    show(chart_intent(view))
                head("Opportunity map", "Top right is best: highly relevant topics far from your current content. "
                     "Larger circles are covered by more competitors.")
                show(chart_scatter(view))
            st.caption("Points are estimates from relevance, competitor coverage, gap size and search intent. "
                       "Review each recommendation before publishing.")

    with t3:
        if stats.empty:
            st.info("Not enough topic data to compare.")
        else:
            head("Topic coverage by competitor",
                 "How many of each competitor's topics you already cover, which are real gaps, and which are unrelated.")
            show(chart_topic_split(stats))
        if len(gaps):
            head("Who covers the top gaps", "Rows are your highest-scoring gaps. Dark cells show competitors that cover them.")
            show(chart_heatmap(gaps, my_site, comps))
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            head("Average words per page")
            show(chart_metric(sb, "avg_words", "Words", scale))
        with c2:
            head("Average H2 headings per page")
            show(chart_metric(sb, "avg_h2", "H2 headings", scale, ".1f"))
        with c3:
            head("Average internal links per page")
            show(chart_metric(sb, "avg_links", "Links", scale, ".1f"))
        head("Keyword gaps", "Words and phrases used on at least two competitor pages and on none of yours.")
        if res["kw"].empty:
            st.info("Not enough pages to compare keywords. Increase pages per site.")
        else:
            show(chart_keywords(res["kw"]))
            with st.expander("Show keyword table"):
                st.dataframe(res["kw"], width="stretch", hide_index=True)

    with t4:
        if gaps.empty:
            st.info("Briefs appear here once gaps are found.")
        for r in gaps.head(15).itertuples():
            with st.expander(f"{r.topic}  ({r.priority}, {r.score:.0f} points)"):
                st.markdown(f"**Action:** {r.action}" + (f" ({r.closest_page})" if r.closest_page else ""))
                st.markdown(f"**Search intent:** {r.intent}  \n**Covered by:** {r.covered_by}")
                st.markdown("**Suggested sections:** " + (r.subtopics.replace("; ", ", ") if r.subtopics
                                                           else "Review the competitor page below."))
                st.markdown(f"**Competitor reference:** {r.reference}")
                st.markdown("**Checklist:** use the topic in the title, H1 and first paragraph; write a "
                            "meta description of about 155 characters; add an FAQ section; link from related pages.")

    with t6:
        cmp_ = res["cmp"]
        overall, winner = cmp_["overall"], cmp_["winner"]
        rank = list(overall.index).index(my_site) + 1
        if winner == my_site:
            big = f"{my_site} is the strongest site overall"
            sub = (f"You score {overall[my_site]:.0f} of 100 and lead the {len(overall) - 1} competitor"
                   f"{'s' if len(overall) > 2 else ''} compared. Keep widening the gap with the content gaps found.")
        else:
            edge = (cmp_["norm"][winner] - cmp_["norm"][my_site]).sort_values(ascending=False)
            big = f"{winner} is the strongest site overall"
            sub = (f"It scores {overall[winner]:.0f} of 100. You rank #{rank} of {len(overall)} with "
                   f"{overall[my_site]:.0f}. {winner} is furthest ahead on {edge.index[0].lower()}"
                   f"{' and ' + edge.index[1].lower() if len(edge) > 1 and edge.iloc[1] > 0 else ''}.")
        st.markdown(f'<div class="verdict"><p class="big">{big}</p><p class="sub">{sub}</p></div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            head("Overall ranking", "Weighted from the six areas on the right.")
            show(chart_overall(overall, scale))
        with c2:
            head("Score by area", "In each area the best site scores 100 and the others are shown relative to it.")
            show(chart_categories(cmp_["norm"], scale, sites))
        head("Head-to-head against you")
        h = cmp_["h2h"].rename(columns={"site": "Competitor", "overall": "Overall score", "vs_you": "Difference vs you",
                                        "beats_you_in": "Areas it beats you", "you_lead_in": "Areas you lead",
                                        "verdict": "Verdict", "their_edge": "Its biggest advantage",
                                        "your_edge": "Your biggest advantage"})
        st.dataframe(h, width="stretch", hide_index=True,
                     column_config={"Overall score": st.column_config.NumberColumn(format="%.0f"),
                                    "Difference vs you": st.column_config.NumberColumn(format="%+.0f")})
        head("Area by area", "Green cells mark the best site in each area. Higher is better in every row.")
        wide = cmp_["wide"]
        table = wide.reset_index(names="Area")
        table["Best site"] = cmp_["best"].values
        cols = list(wide.columns)
        styler = (table.style.format("{:.1f}", subset=cols)
                  .apply(lambda row: ["background-color:#DDF2EE;font-weight:600" if v == row[cols].max() else ""
                                      for v in row], axis=1, subset=cols))
        st.dataframe(styler, width="stretch", hide_index=True)
        st.caption("Areas and weights: on-page health 30%, topic breadth 20%, search snippets 15%, content depth 15%, "
                   "structure 10%, internal linking 10%. Sites are compared on the same number of pages, "
                   "so topic breadth reflects the pages read.")
        st.download_button("Download comparison (CSV)", wide.reset_index(names="Area").to_csv(index=False).encode("utf-8"),
                           "site_comparison.csv", "text/csv")

    with t5:
        plan = res["plan"]
        bench = res["bench"]
        need = plan[plan.gain > 0] if len(plan) else plan
        head("Which of your pages to change, and what to change",
             "Pages are ordered by how many points the changes would earn. Competitor averages are used as targets.")
        if plan.empty or need.empty:
            st.success("All of your analyzed pages pass the on-page checks. Focus on the content gaps instead.")
        else:
            total_gain = need.gain.sum() / len(plan)
            st.markdown(f"<p class=\"note\"><b>{len(need)} of {len(plan)}</b> pages need changes. Completing every "
                        f"change would raise your average page-health from <b>{me.health_points:.0f}</b> to "
                        f"<b>{min(100, me.health_points + total_gain):.0f}</b> points.</p>", unsafe_allow_html=True)
            summary = pd.DataFrame({
                "Priority": range(1, len(need) + 1), "Page": need.url, "Current points": need.points,
                "Points to gain": need.gain, "Changes": need.n_changes,
                "Biggest fix": need.fixes.map(lambda f: max(f, key=lambda x: x[2])[1] if f else
                                              "Add content for a competitor gap topic")})
            st.dataframe(summary, width="stretch", hide_index=True,
                         column_config={"Page": st.column_config.LinkColumn("Page"),
                                        "Biggest fix": st.column_config.TextColumn(width="large"),
                                        "Current points": st.column_config.ProgressColumn(
                                            "Current points", min_value=0, max_value=100, format="%.0f")})
            export = need.assign(changes=need.apply(lambda r: " | ".join(
                [f"{x[1]} (+{x[2]})" for x in r.fixes] + ([f"Add content about: {', '.join(r.add_topics)}"]
                                                          if r.add_topics else [])), axis=1))[
                ["url", "points", "gain", "changes"]].rename(columns={"gain": "points_to_gain"})
            st.download_button("Download page change plan (CSV)", export.to_csv(index=False).encode("utf-8"),
                               "page_change_plan.csv", "text/csv")
            head("Step-by-step changes for each page")
            for r in need.head(20).itertuples():
                with st.expander(f"{r.url}   |   {r.points} points, +{r.gain} available"):
                    st.markdown(f"**Page title:** {r.title or '(none)'}")
                    for n, (_, text, pts) in enumerate(r.fixes, 1):
                        st.markdown(f"{n}. {text}  **(+{pts} points)**")
                    if r.add_topics:
                        st.markdown(f"{len(r.fixes) + 1}. Add sections that cover these competitor topics: "
                                    f"**{', '.join(r.add_topics)}**")
        if len(gaps):
            new = gaps[gaps.action == "Create new page"].head(10)
            if len(new):
                head("New pages to create", "Topics competitors cover that no page on your site comes close to.")
                st.dataframe(new[["topic", "priority", "score", "intent", "covered_by", "subtopics"]].rename(
                    columns={"topic": "Page to create", "priority": "Priority", "score": "Opportunity points",
                             "intent": "Search intent", "covered_by": "Covered by", "subtopics": "Suggested sections"}),
                    width="stretch", hide_index=True)
        with st.expander("All pages analyzed (every site)"):
            st.dataframe(res["pages"][["site", "url", "title", "points", "words"]], width="stretch", hide_index=True)


st.set_page_config(page_title="SEO Content Gap Analyzer", page_icon="🔎", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>SEO Content Gap Analyzer</h1>'
            '<p>Compare your website with its competitors. See where you fall behind on page quality, '
            'which topics you are missing, and what to publish first.</p></div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    head("Your website", "The site you want to improve.")
    my_url = st.text_input("Website URL", placeholder="https://yourwebsite.com", label_visibility="collapsed")
    focus_in = st.text_input("Focus keywords (optional)", placeholder="e.g. dental implants, teeth whitening",
                             help="Leave empty and the app detects your topics from your own pages.")
with right:
    head("Competitors", "One website per line. Two to four in your niche works best.")
    comps_in = st.text_area("Competitor URLs", height=118, label_visibility="collapsed",
                            placeholder="https://competitor1.com\nhttps://competitor2.com")

with st.expander("Advanced settings"):
    a1, a2, a3, a4 = st.columns(4)
    max_pages = a1.slider("Pages per site", 5, 60, 15)
    delay = a2.slider("Delay (seconds)", 0.5, 3.0, 0.8, 0.1)
    threshold = a3.slider("Match threshold", 0.10, 0.80, 0.35, 0.05,
                          help="Lower finds fewer gaps. Higher finds more.")
    min_rel = a4.slider("Minimum relevance", 0.0, 1.0, 0.30, 0.05,
                        help="Raise this to hide topics unrelated to your site.")
    st.caption("The crawler follows robots.txt and only reads public pages. Check each site's terms first.")

if st.button("Analyze content gaps", type="primary"):
    comp_list = [c.strip() for c in comps_in.splitlines() if c.strip()]
    if not my_url.strip() or not comp_list:
        st.error("Enter your website URL and at least one competitor URL.")
        st.stop()
    bar, status = st.progress(0.0), st.empty()
    sites_in = [my_url] + list(dict.fromkeys(c for c in comp_list if domain(c) != domain(my_url)))
    crawled = {}
    for i, u in enumerate(sites_in):
        crawled[u] = crawl(u, max_pages, delay,
                           lambda s, n, i=i: (status.text(f"Reading {s}: {n} pages"),
                                              bar.progress(min((i + n / max_pages) / len(sites_in), 0.95))))
    my_pages = crawled[my_url]
    comp_pages = [p for u in sites_in[1:] for p in crawled[u]]
    if not my_pages or not comp_pages:
        bar.empty()
        status.empty()
        st.error("Not enough pages could be read. The site may block crawlers, build its pages with "
                 "JavaScript, or be unreachable. Check the URLs or try a different competitor.")
        st.stop()
    status.text("Analyzing topics...")
    focus = [f.strip() for f in focus_in.split(",") if f.strip()] or auto_focus_terms(my_pages)
    st.session_state["res"] = build_results(my_pages, comp_pages, domain(my_url), threshold, focus, min_rel)
    bar.empty()
    status.empty()

if st.session_state.get("res"):
    render(st.session_state["res"])
else:
    st.markdown('<p class="note">Enter your website and competitors above, then select '
                '<b>Analyze content gaps</b>. A first run with 15 pages per site takes a few minutes.</p>',
                unsafe_allow_html=True)
