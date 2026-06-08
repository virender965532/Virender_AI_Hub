() => {
  function absUrl(href) {
    if (!href) return "";
    href = String(href).trim();
    if (href.startsWith("//")) return "https:" + href;
    if (href.startsWith("/")) return "https://www.naukri.com" + href;
    return href;
  }

  function text(el) {
    return el && el.innerText ? el.innerText.trim() : "";
  }

  const selectors = [
    "div.srp-jobtuple-wrapper",
    "div.cust-job-tuple",
    "div.jobTuple",
    "article.jobTuple",
    "[class*='jobTuple']",
  ];
  let cards = [];
  for (const sel of selectors) {
    cards = Array.from(document.querySelectorAll(sel));
    if (cards.length) break;
  }

  function parseMetaUl(card) {
    const out = { location: "", salary: "", experience: "", skills: "" };
    const uls = card.querySelectorAll("ul.mb-3, ul");
    for (const ul of uls) {
      if (!ul.querySelector("li img[alt]")) continue;
      for (const li of ul.querySelectorAll(":scope > li")) {
        const img = li.querySelector("img[alt]");
        if (!img) continue;
        const alt = (img.getAttribute("alt") || "").trim().toLowerCase();
        let val = "";
        const spans = li.querySelectorAll("span.text-body14R, span[class*='body']");
        for (const sp of spans) {
          const t = sp.innerText ? sp.innerText.trim() : "";
          if (t) {
            val = t;
            break;
          }
        }
        if (!val && li.innerText) val = li.innerText.trim();
        if (alt === "location") out.location = val;
        else if (alt === "salary") out.salary = val;
        else if (alt === "skills") out.skills = val;
        else if (alt === "experience") out.experience = val;
      }
      break;
    }
    return out;
  }

  function pickPosted(card) {
    const postedRe = /\d+\s*(h|d|w|mo)\s*ago|today|yesterday|just\s+now/i;
    const sels = [
      "p.flex.items-center.text-body12R",
      "p.flex.items-center.pt-1.text-body12R",
      "p.text-body12R.text-n400",
      "span.job-post-day",
      "span[class*='job-post']",
      ".type br2 fleft grey-text",
    ];
    for (const css of sels) {
      for (const el of card.querySelectorAll(css)) {
        const t = text(el);
        if (t && postedRe.test(t)) return t;
      }
    }
    return "";
  }

  const jobs = [];
  for (const card of cards) {
    let title = "";
    for (const s of ["a.title", "h2 a", ".title a", "h3 a", "a[href*='/job-listings/']"]) {
      const el = card.querySelector(s);
      title = text(el);
      if (title) break;
    }
    let company = "";
    for (const s of ["a.comp-name", "span.comp-name", ".comp-name", "[class*='comp-name']"]) {
      const el = card.querySelector(s);
      company = text(el);
      if (company) break;
    }
    let location = "";
    for (const s of ["span.loc-wrap", ".loc-wrap", "[class*='loc-wrap']"]) {
      const el = card.querySelector(s);
      location = text(el);
      if (location) break;
    }
    let experience = "";
    for (const s of ["span.exp-wrap", ".exp-wrap", "[class*='exp-wrap']"]) {
      const el = card.querySelector(s);
      experience = text(el);
      if (experience) break;
    }
    let salary = "";
    for (const s of ["span.sal-wrap", ".sal-wrap", "[class*='sal-wrap']"]) {
      const el = card.querySelector(s);
      salary = text(el);
      if (salary) break;
    }
    if (!salary) {
      const salEl = card.querySelector("[class*='salary'], .salary, span[class*='package']");
      salary = text(salEl);
    }

    const metaUl = parseMetaUl(card);
    if (metaUl.location && !location) location = metaUl.location;
    if (metaUl.experience && !experience) experience = metaUl.experience;
    if (metaUl.salary && !salary) salary = metaUl.salary;

    let link = "";
    const aa = card.querySelectorAll("a[href*='job-listings']");
    for (const a of aa) {
      const href = absUrl(a.getAttribute("href") || "").split("?")[0].replace(/\/$/, "");
      if (href) {
        link = href;
        break;
      }
    }

    let skills = metaUl.skills || "";
    const tagLis = card.querySelectorAll("ul.tags-gt li, ul[class*='tags-gt'] li");
    if (tagLis.length) {
      skills = Array.from(tagLis)
        .map((li) => text(li))
        .filter(Boolean);
    }

    if (!title && !company) continue;
    jobs.push({
      title: title || "—",
      company: company || "—",
      experience: experience || "—",
      location: location || "—",
      salary,
      skills,
      posted: pickPosted(card),
      link,
    });
  }
  return jobs;
}
