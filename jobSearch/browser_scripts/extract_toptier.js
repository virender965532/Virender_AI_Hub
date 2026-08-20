() => {
  function absUrl(href) {
    if (!href) return "";
    href = String(href).trim();
    if (href.startsWith("//")) return "https:" + href;
    if (href.startsWith("/")) return "https://www.naukri.com" + href;
    return href;
  }

  function pickTitle(right) {
    const sels = [
      "div.box-border.py-5.text-headline24Sb",
      "div.text-headline24Sb.text-n100",
      "div.text-title18Sb.text-n100",
    ];
    for (const s of sels) {
      const el = right.querySelector(s);
      const t = el && el.innerText ? el.innerText.trim() : "";
      if (t) return t;
    }
    return "";
  }

  function pickCompany(left) {
    const sels = [
      "h4.text-title18Sb.text-n200",
      "h4 div.text-title16Sb.text-n200",
      "h4 div.line-clamp-1.text-title16Sb",
      "h4 div.truncate.text-title16Sb",
      "h4 div.text-title16Sb",
    ];
    for (const s of sels) {
      const el = left.querySelector(s);
      const t = el && el.innerText ? el.innerText.trim() : "";
      if (t) return t;
    }
    const pb = left.querySelector("p.text-body14R.text-n400");
    return pb && pb.innerText ? pb.innerText.trim() : "";
  }

  function parseMeta(right) {
    const out = { location: "", salary: "", experience: "", skills: "" };
    const uls = right.querySelectorAll("ul.mb-3");
    let ul = null;
    for (const u of uls) {
      if (u.querySelector("li img[alt]")) {
        ul = u;
        break;
      }
    }
    if (!ul) return out;
    for (const li of ul.querySelectorAll(":scope > li")) {
      const img = li.querySelector("img[alt]");
      if (!img) continue;
      const alt = (img.getAttribute("alt") || "").trim().toLowerCase();
      let val = "";
      const spans = li.querySelectorAll("span.text-body14R");
      for (const sp of spans) {
        const t = sp.innerText ? sp.innerText.trim() : "";
        if (t) {
          val = t;
          break;
        }
      }
      if (alt === "location") out.location = val;
      else if (alt === "salary") out.salary = val;
      else if (alt === "skills") out.skills = val;
      else if (alt === "experience") out.experience = val;
    }
    return out;
  }

  function pickPosted() {
    const postedRe =
      /\d+\s*(h|hrs?|hours?|d|days?|w|weeks?|mo|months?)\s*ago|few\s+hours?\s*ago|few\s+days?\s*ago|today|yesterday|just\s+now/i;
    for (let a = 0; a < arguments.length; a++) {
      const scope = arguments[a];
      if (!scope) continue;
      const dayEl = scope.querySelector("span.job-post-day, .job-post-day");
      if (dayEl) {
        const direct = dayEl.innerText ? dayEl.innerText.trim() : "";
        if (direct) return direct;
      }
    }
    const sels = [
      "p.flex.items-center.text-body12R",
      "p.flex.items-center.pt-1.text-body12R",
      "p.text-body12R.text-n400",
      "span[class*='job-post']",
      "div.row6 span",
    ];
    for (let a = 0; a < arguments.length; a++) {
      const scope = arguments[a];
      if (!scope) continue;
      for (const css of sels) {
        for (const p of scope.querySelectorAll(css)) {
          const t = p.innerText ? p.innerText.trim() : "";
          if (t && postedRe.test(t)) return t;
        }
      }
    }
    return "";
  }

  function jobLink(card, right) {
    const roots = [right, card];
    for (const root of roots) {
      const anchors = root.querySelectorAll("a[href*='job-listings']");
      for (const a of anchors) {
        let href = a.getAttribute("href") || "";
        if (!href.includes("job-listings")) continue;
        href = absUrl(href).split("?")[0].replace(/\/$/, "");
        if (href) return href;
      }
    }
    return "";
  }

  const xpath =
    "//div[contains(@class,'rounded-3xl') and contains(@class,'bg-n800') " +
    "and contains(@class,'cursor-pointer') and contains(@class,'flex') " +
    "and contains(@class,'min-h-')]";
  const snap = document.evaluate(
    xpath,
    document,
    null,
    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
    null,
  );
  const jobs = [];
  for (let i = 0; i < snap.snapshotLength; i++) {
    const card = snap.snapshotItem(i);
    if (!(card instanceof Element)) continue;
    if (card.id === "load-more-btn") continue;
    const cls = card.className || "";
    if (cls.includes("animate-pulse")) continue;

    let left = card;
    try {
      const lx = document.evaluate(
        ".//div[contains(@class,'border-r') and contains(@class,'w-[220px]')]",
        card,
        null,
        XPathResult.FIRST_ORDERED_NODE_TYPE,
        null,
      ).singleNodeValue;
      if (lx instanceof Element) left = lx;
    } catch (e) {}

    let right = card;
    try {
      const rx = document.evaluate(
        ".//div[contains(@class,'min-w-[480px]')]",
        card,
        null,
        XPathResult.FIRST_ORDERED_NODE_TYPE,
        null,
      ).singleNodeValue;
      if (rx instanceof Element) right = rx;
    } catch (e) {}

    const title = pickTitle(right);
    if (!title) continue;

    let company = pickCompany(left);
    if (!company) company = "—";
    const meta = parseMeta(right);
    jobs.push({
      title,
      company,
      experience: meta.experience || "—",
      location: meta.location || "—",
      salary: meta.salary || "",
      skills: meta.skills || "",
      posted: pickPosted(card, right, left),
      link: jobLink(card, right),
    });
  }
  return jobs;
}
