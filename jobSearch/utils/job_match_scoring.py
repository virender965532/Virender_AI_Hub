"""Weighted job–profile skill matching for ATS, job search, and recommendation flows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

# ---------------------------------------------------------------------------
# Configurable profile skill weights (canonical display name -> weight)
# ---------------------------------------------------------------------------

SKILL_WEIGHTS: dict[str, float] = {
    # Strong experience (1.0)
    "React": 1.0,
    "Next": 1.0,
    "Node": 1.0,
    "JavaScript": 1.0,
    "TypeScript": 1.0,
    "Backend": 1.0,
    "Frontend": 1.0,
    "Development": 1.0,
    "Full Stack": 1.0,
    "MERN": 1.0,
    "SQL": 1.0,
    "Vector Database": 1.0,
    "MongoDB": 1.0,
    "NoSQL": 1.0,
    "MySQL": 1.0,
    "Lambda": 1.0,
    "AI": 1.0,
    "Code Quality": 1.0,
    "Prompt Engineering": 1.0,
    "System Design": 1.0,
    "REST API": 1.0,
    "EC2": 1.0,
    "Solution Architect": 1.0,
    "GraphQL": 1.0,
    "RDBMS": 1.0,
    "PostgreSQL": 1.0,
    "Redux": 1.0,
    "DB": 1.0,
    "LangChain": 1.0,
    "Programming": 1.0,
    "Neo4j": 1.0,
    "HTML": 1.0,
    "CSS": 1.0,
    "Tailwind": 1.0,
    "Docker": 1.0,
    "Copilot": 1.0,
    "Cursor": 1.0,
    "Senior": 1.0,
    "OpenAI": 1.0,
    "API": 1.0,
    "OpenSearch": 1.0,
    "Express": 1.0,
    "CI/CD": 1.0,
    "GitHub": 1.0,
    # Knowledge (0.7)
    "Gen AI": 1,
    "Generative AI": 1,
    "Agentic AI": 1,
    "RAG": 1,
    "LangGraph": 1,
    "LLM": 1,
    "Transformer": 1,
    "AWS": 1,
    # Basic / intermediate (0.5)
    "Python": 0.5,
    "Microservices": 0.5,
}

# Alias token (normalized) -> canonical SKILL_WEIGHTS key
# Matching is always on lowercased / normalized tokens.
SKILL_ALIASES: dict[str, str] = {
    # JavaScript
    "js": "JavaScript",
    "javascript": "JavaScript",
    # TypeScript
    "ts": "TypeScript",
    "typescript": "TypeScript",
    # React
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    # Next
    "nextjs": "Next",
    "next.js": "Next",
    "next js": "Next",
    # Node
    "nodejs": "Node",
    "node.js": "Node",
    "node js": "Node",
    # RAG
    "retrieval augmented generation": "RAG",
    # AI
    "artificial intelligence": "AI",
    # LLM
    "llms": "LLM",
    "large language model": "LLM",
    "large language models": "LLM",
    # AWS / Lambda / EC2
    "amazon web service": "AWS",
    "amazon web services": "AWS",
    "aws lambda": "Lambda",
    "lambda": "Lambda",
    "amazon lambda": "Lambda",
    "ec2": "EC2",
    "amazon ec2": "EC2",
    "aws ec2": "EC2",
    # Full stack / MERN / backend
    "fullstack": "Full Stack",
    "full stack": "Full Stack",
    "full-stack": "Full Stack",
    "mern": "MERN",
    "mern stack": "MERN",
    "backend": "Backend",
    "back end": "Backend",
    "back-end": "Backend",
    "frontend": "Frontend",
    "front end": "Frontend",
    "front-end": "Frontend",
    "development": "Development",
    # Data stores
    "sql": "SQL",
    "mysql": "MySQL",
    "nosql": "NoSQL",
    "no sql": "NoSQL",
    "no-sql": "NoSQL",
    "mongodb": "MongoDB",
    "mongo db": "MongoDB",
    "mongo": "MongoDB",
    "rdbms": "RDBMS",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "postgress": "PostgreSQL",
    "postgre sql": "PostgreSQL",
    "vector database": "Vector Database",
    "vector databases": "Vector Database",
    "vectordb": "Vector Database",
    "vector db": "Vector Database",
    "db": "DB",
    "database": "DB",
    "databases": "DB",
    "neo4j": "Neo4j",
    "neo 4j": "Neo4j",
    # Frontend / tooling
    "redux": "Redux",
    "redux toolkit": "Redux",
    "html": "HTML",
    "html5": "HTML",
    "css": "CSS",
    "css3": "CSS",
    "tailwind": "Tailwind",
    "tailwindcss": "Tailwind",
    "tailwind css": "Tailwind",
    "docker": "Docker",
    "docker compose": "Docker",
    # AI tools / platforms
    "copilot": "Copilot",
    "github copilot": "Copilot",
    "cursor": "Cursor",
    "cursor ai": "Cursor",
    "cursor ide": "Cursor",
    "openai": "OpenAI",
    "open ai": "OpenAI",
    "opensearch": "OpenSearch",
    "open search": "OpenSearch",
    "amazon opensearch": "OpenSearch",
    # Express / API / CI / GitHub / senior
    "express": "Express",
    "expressjs": "Express",
    "expresjs": "Express",
    "express.js": "Express",
    "express js": "Express",
    "api": "API",
    "apis": "API",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "ci/cd": "CI/CD",
    "github": "GitHub",
    "git hub": "GitHub",
    "senior": "Senior",
    # LangChain / programming
    "langchain": "LangChain",
    "lang chain": "LangChain",
    "programming": "Programming",
    # APIs / design / quality
    "graphql": "GraphQL",
    "graph ql": "GraphQL",
    "rest api": "REST API",
    "restapi": "REST API",
    "rest-api": "REST API",
    "rest apis": "REST API",
    "system design": "System Design",
    "code quality": "Code Quality",
    "prompt engineering": "Prompt Engineering",
    "solution architect": "Solution Architect",
    "solutions architect": "Solution Architect",
    # Other
    "genai": "Gen AI",
    "gen ai": "Gen AI",
    "rag": "RAG",
    "micro-services": "Microservices",
    "micro services": "Microservices",
    "lang graph": "LangGraph",
    "agentic ai": "Agentic AI",
    "generative ai": "Generative AI",
}

# Regex patterns applied to normalized skill text (pattern -> canonical key)
# Order matters: more specific patterns must appear before broader ones.
# Patterns enable partial matches (e.g. "full stack developer" -> Full Stack).
_SKILL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bretrieval\s+augmented\s+generation\b"), "RAG"),
    (re.compile(r"\brag\b"), "RAG"),
    (re.compile(r"\bartificial\s+intelligence\b"), "AI"),
    (re.compile(r"\bgen\s*ai\b"), "Gen AI"),
    (re.compile(r"\bgenerative\s+ai\b"), "Generative AI"),
    (re.compile(r"\bagentic\s+ai\b"), "Agentic AI"),
    (re.compile(r"\blarge\s+language\s+models?\b"), "LLM"),
    (re.compile(r"\bllms?\b"), "LLM"),
    (re.compile(r"\bprompt\s+engineering\b"), "Prompt Engineering"),
    (re.compile(r"\bsystem\s+design\b"), "System Design"),
    (re.compile(r"\bcode\s+quality\b"), "Code Quality"),
    (re.compile(r"\bsolutions?\s+architect\b"), "Solution Architect"),
    (re.compile(r"\bvector\s*databases?\b"), "Vector Database"),
    (re.compile(r"\bvectordb\b"), "Vector Database"),
    (re.compile(r"\bvector\s+db\b"), "Vector Database"),
    (re.compile(r"\blang\s*chain\b"), "LangChain"),
    (re.compile(r"\blangchain\b"), "LangChain"),
    (re.compile(r"\bneo\s*4j\b"), "Neo4j"),
    (re.compile(r"\bredux(?:\s+toolkit)?\b"), "Redux"),
    (re.compile(r"\btailwind(?:\s*css)?\b"), "Tailwind"),
    (re.compile(r"\bdocker(?:\s+compose)?\b"), "Docker"),
    (re.compile(r"\bhtml5?\b"), "HTML"),
    (re.compile(r"\bcss3?\b"), "CSS"),
    (re.compile(r"\bgithub\s+copilot\b"), "Copilot"),
    (re.compile(r"\bcopilot\b"), "Copilot"),
    (re.compile(r"\bcursor(?:\s+(?:ai|ide))?\b"), "Cursor"),
    (re.compile(r"\bopen\s*ai\b"), "OpenAI"),
    (re.compile(r"\bamazon\s+open\s*search\b"), "OpenSearch"),
    (re.compile(r"\bopen\s*search\b"), "OpenSearch"),
    (re.compile(r"\bci\s*/?\s*cd\b"), "CI/CD"),
    (re.compile(r"\bcicd\b"), "CI/CD"),
    (re.compile(r"\bgit\s*hub\b"), "GitHub"),
    (re.compile(r"\bexpres+s?\.?js\b"), "Express"),
    (re.compile(r"\bexpress\s+js\b"), "Express"),
    (re.compile(r"\bexpress\b"), "Express"),
    (re.compile(r"\bsenior\b"), "Senior"),
    (re.compile(r"\brest[\s-]?apis?\b"), "REST API"),
    (re.compile(r"\bgraph\s*ql\b"), "GraphQL"),
    (re.compile(r"\bgraphql\b"), "GraphQL"),
    (re.compile(r"\bapis?\b"), "API"),
    (re.compile(r"\bfull[\s-]?stack\b"), "Full Stack"),
    (re.compile(r"\bmern(?:\s+stack)?\b"), "MERN"),
    (re.compile(r"\bback[\s-]?end\b"), "Backend"),
    (re.compile(r"\bfront[\s-]?end\b"), "Frontend"),
    (re.compile(r"\bmongo(?:db|\s*db)?\b"), "MongoDB"),
    (re.compile(r"\bno[\s-]?sql\b"), "NoSQL"),
    (re.compile(r"\bmysql\b"), "MySQL"),
    (re.compile(r"\bpostgre(?:s|sql|ss)?\b"), "PostgreSQL"),
    (re.compile(r"\bpostgre\s+sql\b"), "PostgreSQL"),
    (re.compile(r"\brdbms\b"), "RDBMS"),
    (re.compile(r"\bsql\b"), "SQL"),
    (re.compile(r"\bdatabases?\b"), "DB"),
    (re.compile(r"\bdb\b"), "DB"),
    (re.compile(r"\bamazon\s+ec2\b"), "EC2"),
    (re.compile(r"\baws\s+ec2\b"), "EC2"),
    (re.compile(r"\bec2\b"), "EC2"),
    (re.compile(r"\bamazon\s+web\s+services?\b"), "AWS"),
    (re.compile(r"\baws\s+lambda\b"), "Lambda"),
    (re.compile(r"\bamazon\s+lambda\b"), "Lambda"),
    (re.compile(r"\blambda\b"), "Lambda"),
    (re.compile(r"\baws\b"), "AWS"),
    (re.compile(r"\bjavascript\b"), "JavaScript"),
    (re.compile(r"\bjs\b"), "JavaScript"),
    (re.compile(r"\btypescript\b"), "TypeScript"),
    (re.compile(r"\bts\b"), "TypeScript"),
    (re.compile(r"\bnode\.?js\b"), "Node"),
    (re.compile(r"\bnode\s+js\b"), "Node"),
    (re.compile(r"\breact\.?js\b"), "React"),
    (re.compile(r"\breact\s+js\b"), "React"),
    (re.compile(r"\breactjs\b"), "React"),
    (re.compile(r"\bnext\.?js\b"), "Next"),
    (re.compile(r"\bnext\s+js\b"), "Next"),
    (re.compile(r"\bnextjs\b"), "Next"),
    (re.compile(r"\bmicro[\s-]?services\b"), "Microservices"),
    (re.compile(r"\bprogramming\b"), "Programming"),
    (re.compile(r"\bdevelopment\b"), "Development"),
    (re.compile(r"\bai\b"), "AI"),
]

# Stacks that reduce relevance (pattern -> display label)
EXCLUDED_TECH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bjava\b"), "Java"),
    (re.compile(r"\bc#\b|\bc\s*sharp\b|\bc-sharp\b"), "C#"),
    (re.compile(r"\.net\b|\bdotnet\b|\basp\.net\b"), ".NET"),
    (re.compile(r"\bgolang\b|\bgo\s+lang\b"), "Golang"),
]

# Points subtracted from the base score for each distinct excluded stack found
EXCLUDED_TECH_PENALTY: float = 20.0


class MatchCategory(str, Enum):
    EXCELLENT = "Excellent Match"
    GOOD = "Good Match"
    POSSIBLE = "Possible Match"
    LOW = "Low Match"


class SkillBreakdownEntry(TypedDict):
    job_skill: str
    canonical_skill: str | None
    weight: float
    matched: bool
    contribution: float


class JobMatchResult(TypedDict):
    score: float
    base_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    excluded_technologies: list[str]
    exclusion_penalty: float
    match_category: str
    total_job_skills: int
    match_percentage: float
    skill_breakdown: list[SkillBreakdownEntry]


@dataclass(frozen=True)
class ScoringConfig:
    """Immutable scoring configuration; extend for required/nice-to-have tiers later."""

    skill_weights: dict[str, float] = field(default_factory=lambda: dict(SKILL_WEIGHTS))
    aliases: dict[str, str] = field(default_factory=lambda: dict(SKILL_ALIASES))
    patterns: list[tuple[re.Pattern[str], str]] = field(
        default_factory=lambda: list(_SKILL_PATTERNS)
    )
    excluded_patterns: list[tuple[re.Pattern[str], str]] = field(
        default_factory=lambda: list(EXCLUDED_TECH_PATTERNS)
    )
    excluded_penalty_per_tech: float = EXCLUDED_TECH_PENALTY
    # Future: required_skill_multiplier: float = 1.0
    # Future: nice_to_have_multiplier: float = 0.6


def _normalize_token(value: str) -> str:
    """Lowercase, collapse whitespace, unify common separators."""
    text = (value or "").strip().lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _skill_blob(skills: list[str]) -> str:
    return " ".join(str(s).strip() for s in skills if s and str(s).strip()).lower()


def _category_for_score(score: float) -> MatchCategory:
    if score >= 85:
        return MatchCategory.EXCELLENT
    if score >= 70:
        return MatchCategory.GOOD
    if score >= 50:
        return MatchCategory.POSSIBLE
    return MatchCategory.LOW


class JobMatchScorer:
    """Score job skill lists against a weighted profile. Injectable for tests and extensions."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._config = config or ScoringConfig()
        self._weight_by_norm: dict[str, tuple[str, float]] = {}
        for name, weight in self._config.skill_weights.items():
            self._weight_by_norm[_normalize_token(name)] = (name, weight)

    def resolve_canonical(self, raw_skill: str) -> str | None:
        """Map a single job skill string to a canonical profile skill, if recognized."""
        normalized = _normalize_token(raw_skill)
        if not normalized:
            return None

        dotless = normalized.replace(".", " ")
        dotless = re.sub(r"\s+", " ", dotless).strip()

        alias_target = self._config.aliases.get(normalized) or self._config.aliases.get(dotless)
        if alias_target and alias_target in self._config.skill_weights:
            return alias_target

        direct = self._weight_by_norm.get(normalized) or self._weight_by_norm.get(dotless)
        if direct:
            return direct[0]

        for pattern, canonical in self._config.patterns:
            if pattern.search(dotless) and canonical in self._config.skill_weights:
                return canonical

        return None

    def find_excluded_technologies(self, job_skills: list[str]) -> list[str]:
        """Return distinct excluded stack labels detected in the job skill list."""
        blob = _skill_blob(job_skills)
        if not blob:
            return []

        dotless = blob.replace(".", " ")
        found: list[str] = []
        seen: set[str] = set()
        for pattern, label in self._config.excluded_patterns:
            if label in seen:
                continue
            if pattern.search(blob) or pattern.search(dotless):
                found.append(label)
                seen.add(label)
        return found

    def weight_for(self, canonical: str | None) -> float:
        if not canonical:
            return 0.0
        return float(self._config.skill_weights.get(canonical, 0.0))

    def calculate(self, job_skills: list[str]) -> JobMatchResult:
        """
        Compare ``job_skills`` against the configured profile weights.

        Base score = (sum of matched skill weights) / (deduplicated job skill count) * 100.
        Final score = base score minus ``excluded_penalty_per_tech`` for each excluded stack
        (Java, C#, .NET, Golang, etc.), floored at 0.
        """
        breakdown: list[SkillBreakdownEntry] = []
        seen_keys: set[str] = set()
        matched_display: list[str] = []
        missing_display: list[str] = []
        earned = 0.0
        deduped_count = 0

        for raw in job_skills:
            raw_text = str(raw or "").strip()
            if not raw_text:
                continue

            canonical = self.resolve_canonical(raw_text)
            weight = self.weight_for(canonical)
            matched = canonical is not None and weight > 0

            dedupe_key = canonical if canonical else f"__unknown__:{_normalize_token(raw_text)}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped_count += 1

            contribution = weight if matched else 0.0
            earned += contribution

            entry: SkillBreakdownEntry = {
                "job_skill": raw_text,
                "canonical_skill": canonical,
                "weight": weight,
                "matched": matched,
                "contribution": contribution,
            }
            breakdown.append(entry)

            if matched and canonical:
                matched_display.append(canonical)
            else:
                missing_display.append(raw_text)

        base_score = round((earned / deduped_count) * 100.0, 1) if deduped_count else 0.0
        excluded = self.find_excluded_technologies(job_skills)
        penalty = round(
            self._config.excluded_penalty_per_tech * len(excluded),
            1,
        )
        score = round(max(0.0, base_score - penalty), 1)
        category = _category_for_score(score)

        return JobMatchResult(
            score=score,
            base_score=base_score,
            matched_skills=matched_display,
            missing_skills=missing_display,
            excluded_technologies=excluded,
            exclusion_penalty=penalty,
            match_category=category.value,
            total_job_skills=deduped_count,
            match_percentage=score,
            skill_breakdown=breakdown,
        )


_DEFAULT_SCORER = JobMatchScorer()


def calculate_job_match(job_skills: list[str]) -> JobMatchResult:
    """
    Calculate how relevant a job is for the default profile skill weights.

    Args:
        job_skills: Skill strings extracted from a job description or listing.

    Returns:
        Dict with score (0–100), matched/missing skills, category, and per-skill breakdown.

    Example:
        >>> calculate_job_match(["React", "Node.js", "Python", "Kubernetes"])["score"]
        83.3
    """
    return _DEFAULT_SCORER.calculate(job_skills)


def _example_usage() -> None:
    """Print sample inputs and outputs for manual verification."""
    samples: list[tuple[str, list[str]]] = [
        ("Full-stack JS role", ["React", "Next.js", "Node.js", "TypeScript", "AWS"]),
        ("With aliases", ["js", "ts", "nodejs", "genai", "micro-services"]),
        ("Skill variants", [
            "Reactjs",
            "next js",
            "Retrieval Augmented Generation",
            "Artificial Intelligence",
            "large language model",
            "amazon web services",
            "aws lambda",
        ]),
        ("Partial match", ["React", "Java", "Kubernetes"]),
        ("Excluded stack penalty", ["React", "TypeScript", "Java", "Spring Boot"]),
        ("Golang penalty", ["React", "Node.js", "Golang", "PostgreSQL"]),
        ("Empty", []),
    ]
    for label, skills in samples:
        result = calculate_job_match(skills)
        print(f"\n--- {label} ---")
        print(f"Input:  {skills}")
        print(
            f"Score:  {result['score']}% ({result['match_category']}) "
            f"[base {result['base_score']}% - {result['exclusion_penalty']}% penalty]"
        )
        if result["excluded_technologies"]:
            print(f"Excluded: {result['excluded_technologies']}")
        print(f"Matched: {result['matched_skills']}")
        for row in result["skill_breakdown"]:
            if row["matched"]:
                print(f"  · {row['job_skill']!r} -> {row['canonical_skill']!r}")


if __name__ == "__main__":
    _example_usage()
