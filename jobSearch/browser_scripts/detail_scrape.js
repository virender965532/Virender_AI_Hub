() => {
  const root = document.getElementById("jobs-desc");
  if (!root) {
    return { description: "", skills: [], postedRaw: "", location: "" };
  }

  function parseMeta(ul) {
    const out = { location: "", salary: "", skills: "", experience: "" };
    if (!ul) return out;
    for (const li of ul.querySelectorAll(":scope > li")) {
      const img = li.querySelector("img[alt]");
      if (!img) continue;
      const alt = (img.getAttribute("alt") || "").trim().toLowerCase();
      let val = "";
      for (const sp of li.querySelectorAll("span.text-body14R")) {
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

  let ul = null;
  for (const sel of ["ul.mb-3.space-y-2", "ul.mb-3"]) {
    const uls = root.querySelectorAll(sel);
    for (const u of uls) {
      if (u.querySelector("li img[alt]")) {
        ul = u;
        break;
      }
    }
    if (ul) break;
  }
  const meta = parseMeta(ul);

  let description = "";
  const hdrs = root.querySelectorAll("p");
  for (const hdr of hdrs) {
    const txt = (hdr.innerText || "").trim().replace(/\s+/g, " ");
    if (txt === "Job Description" || txt.indexOf("Job Description") === 0) {
      let sib = hdr.nextElementSibling;
      while (sib && sib.tagName === "SCRIPT") {
        sib = sib.nextElementSibling;
      }
      if (sib && sib.innerText) {
        description = sib.innerText.trim();
        break;
      }
    }
  }
  if (!description) {
    for (const div of root.querySelectorAll("div.text-title16R.text-n300")) {
      const t = (div.innerText || "").trim();
      if (t.length > 120) {
        description = t;
        break;
      }
    }
  }

  let postedRaw = "";
  const postedRe = /\d+\s*(h|d|w|mo)\s*ago|today|yesterday|just\s+now/i;
  for (const css of [
    "p.flex.items-center.text-body12R",
    "p.flex.items-center.pt-1.text-body12R",
    "p.text-body12R.text-n400",
  ]) {
    for (const p of root.querySelectorAll(css)) {
      const t = (p.innerText || "").trim();
      if (t && postedRe.test(t)) {
        postedRaw = t;
        break;
      }
    }
    if (postedRaw) break;
  }

  const skillsStr = meta.skills || "";
  const skills = skillsStr
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  return {
    description,
    skills,
    postedRaw,
    location: meta.location || "",
  };
}
