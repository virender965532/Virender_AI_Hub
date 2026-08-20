from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoleConfig:
    id: str
    label: str
    retrieval_priorities: tuple[str, ...]
    answer_sections: tuple[str, ...]
    terminology: str
    depth: str
    system_prompt: str


ROLES: dict[str, RoleConfig] = {
    "CEO": RoleConfig(
        id="CEO",
        label="CEO",
        retrieval_priorities=("business value", "ROI", "risk", "cost savings", "competitive advantage"),
        answer_sections=("Executive Summary", "Business Impact", "Risks", "Recommendations", "Expected ROI"),
        terminology="executive, strategic, board-ready",
        depth="high-level strategic — no implementation detail",
        system_prompt=(
            "You are advising a CEO. Focus on business value, ROI, risk, cost savings, "
            "and competitive positioning. Use executive language. Avoid technical jargon unless "
            "briefly explained in business terms."
        ),
    ),
    "CTO": RoleConfig(
        id="CTO",
        label="CTO",
        retrieval_priorities=("architecture", "scalability", "technology stack", "build vs buy", "technical debt"),
        answer_sections=("Architecture Overview", "Technology Stack", "Scalability", "Tradeoffs", "Recommendations"),
        terminology="technical leadership, platform strategy",
        depth="architecture and technology decisions with strategic rationale",
        system_prompt=(
            "You are advising a CTO. Focus on architecture, scalability, technology decisions, "
            "platform strategy, and engineering tradeoffs. Balance innovation with operational excellence."
        ),
    ),
    "CIO": RoleConfig(
        id="CIO",
        label="CIO",
        retrieval_priorities=("IT governance", "integration", "compliance", "vendor management", "digital transformation"),
        answer_sections=("Strategic Alignment", "Integration & Governance", "Compliance", "Vendor Considerations", "Roadmap"),
        terminology="enterprise IT, governance, transformation",
        depth="enterprise IT strategy with governance and compliance focus",
        system_prompt=(
            "You are advising a CIO. Emphasize IT governance, enterprise integration, compliance, "
            "vendor strategy, and digital transformation alignment with business goals."
        ),
    ),
    "Enterprise Architect": RoleConfig(
        id="Enterprise Architect",
        label="Enterprise Architect",
        retrieval_priorities=("system design", "patterns", "integration", "standards", "reference architecture"),
        answer_sections=("System Design", "Architecture Patterns", "Integration Points", "Tradeoffs", "Reference Model"),
        terminology="TOGAF-aligned, pattern-oriented, standards-based",
        depth="deep architectural analysis with patterns and tradeoffs",
        system_prompt=(
            "You are advising an Enterprise Architect. Provide system design, architecture patterns, "
            "integration considerations, standards alignment, and explicit tradeoff analysis."
        ),
    ),
    "Solution Architect": RoleConfig(
        id="Solution Architect",
        label="Solution Architect",
        retrieval_priorities=("solution design", "requirements mapping", "components", "APIs", "deployment"),
        answer_sections=("Solution Overview", "Component Design", "Integration", "Deployment", "Risks"),
        terminology="solution-focused, component-level, delivery-oriented",
        depth="detailed solution design with component breakdown",
        system_prompt=(
            "You are advising a Solution Architect. Focus on solution design, requirement mapping, "
            "component architecture, API design, and deployment considerations."
        ),
    ),
    "Engineering Manager": RoleConfig(
        id="Engineering Manager",
        label="Engineering Manager",
        retrieval_priorities=("team impact", "delivery", "process", "quality", "resource planning"),
        answer_sections=("Summary", "Team Impact", "Delivery Plan", "Quality & Process", "Action Items"),
        terminology="people + delivery, pragmatic engineering leadership",
        depth="balanced technical and people/process perspective",
        system_prompt=(
            "You are advising an Engineering Manager. Cover team impact, delivery timelines, "
            "process improvements, quality gates, and actionable next steps for the team."
        ),
    ),
    "Technical Lead": RoleConfig(
        id="Technical Lead",
        label="Technical Lead",
        retrieval_priorities=("implementation approach", "code quality", "patterns", "technical decisions", "mentoring"),
        answer_sections=("Technical Summary", "Implementation Approach", "Key Decisions", "Code Patterns", "Next Steps"),
        terminology="hands-on technical leadership",
        depth="implementation-focused with design rationale",
        system_prompt=(
            "You are advising a Technical Lead. Provide implementation approach, code patterns, "
            "technical decision rationale, and guidance the team can execute on."
        ),
    ),
    "Senior Software Engineer": RoleConfig(
        id="Senior Software Engineer",
        label="Senior Software Engineer",
        retrieval_priorities=("implementation", "code examples", "APIs", "debugging", "best practices"),
        answer_sections=("Direct Answer", "Implementation", "Code Guidance", "Patterns", "Pitfalls"),
        terminology="developer-focused, concrete, code-oriented",
        depth="deep implementation detail with code examples when relevant",
        system_prompt=(
            "You are advising a Senior Software Engineer. Provide concrete implementation details, "
            "code patterns, API usage, and practical best practices. Include code examples when helpful."
        ),
    ),
    "Product Manager": RoleConfig(
        id="Product Manager",
        label="Product Manager",
        retrieval_priorities=("user value", "features", "prioritization", "metrics", "roadmap"),
        answer_sections=("Product Summary", "User Value", "Feature Implications", "Metrics", "Recommendations"),
        terminology="user-centric, outcome-driven, prioritization-focused",
        depth="product strategy with user impact and prioritization",
        system_prompt=(
            "You are advising a Product Manager. Focus on user value, feature implications, "
            "prioritization, success metrics, and product roadmap considerations."
        ),
    ),
    "Project Manager": RoleConfig(
        id="Project Manager",
        label="Project Manager",
        retrieval_priorities=("timeline", "dependencies", "risks", "milestones", "stakeholders"),
        answer_sections=("Summary", "Timeline & Milestones", "Dependencies", "Risks", "Action Plan"),
        terminology="delivery-focused, milestone-driven",
        depth="project planning with timelines and risk management",
        system_prompt=(
            "You are advising a Project Manager. Emphasize timelines, dependencies, milestones, "
            "stakeholder impact, and risk mitigation with clear action items."
        ),
    ),
    "Business Analyst": RoleConfig(
        id="Business Analyst",
        label="Business Analyst",
        retrieval_priorities=("requirements", "process", "stakeholders", "gap analysis", "acceptance criteria"),
        answer_sections=("Business Context", "Requirements", "Process Impact", "Gap Analysis", "Recommendations"),
        terminology="requirements-driven, process-oriented",
        depth="business analysis with clear requirements mapping",
        system_prompt=(
            "You are advising a Business Analyst. Focus on requirements, business process impact, "
            "stakeholder needs, gap analysis, and acceptance criteria."
        ),
    ),
    "HR Manager": RoleConfig(
        id="HR Manager",
        label="HR Manager",
        retrieval_priorities=("workforce impact", "skills", "training", "policy", "culture"),
        answer_sections=("Workforce Impact", "Skills Required", "Training Needs", "Policy Considerations", "Recommendations"),
        terminology="people operations, organizational impact",
        depth="HR and organizational perspective",
        system_prompt=(
            "You are advising an HR Manager. Address workforce impact, required skills, "
            "training needs, policy implications, and organizational change considerations."
        ),
    ),
    "Recruiter": RoleConfig(
        id="Recruiter",
        label="Recruiter",
        retrieval_priorities=("skills", "qualifications", "role requirements", "market trends", "interview topics"),
        answer_sections=("Role Summary", "Required Skills", "Qualifications", "Interview Topics", "Market Context"),
        terminology="talent acquisition, skills mapping",
        depth="recruiting-focused with skill and qualification mapping",
        system_prompt=(
            "You are advising a Recruiter. Map required skills, qualifications, interview topics, "
            "and market context for talent acquisition related to the question."
        ),
    ),
    "Data Scientist": RoleConfig(
        id="Data Scientist",
        label="Data Scientist",
        retrieval_priorities=("data", "models", "metrics", "experiments", "statistical validity"),
        answer_sections=("Analysis Summary", "Data Requirements", "Modeling Approach", "Metrics", "Validation"),
        terminology="data science, statistical, experimental",
        depth="analytical with modeling and validation focus",
        system_prompt=(
            "You are advising a Data Scientist. Focus on data requirements, modeling approaches, "
            "evaluation metrics, experimental design, and statistical validity."
        ),
    ),
    "AI Engineer": RoleConfig(
        id="AI Engineer",
        label="AI Engineer",
        retrieval_priorities=("LLM", "RAG", "agents", "MLOps", "prompt engineering", "evaluation"),
        answer_sections=("AI Architecture", "Implementation", "Model Selection", "Evaluation", "Production Considerations"),
        terminology="AI/ML engineering, production AI systems",
        depth="deep AI engineering with production focus",
        system_prompt=(
            "You are advising an AI Engineer. Cover LLM/RAG architecture, agent design, "
            "prompt engineering, evaluation strategies, and production deployment considerations."
        ),
    ),
    "DevOps Engineer": RoleConfig(
        id="DevOps Engineer",
        label="DevOps Engineer",
        retrieval_priorities=("CI/CD", "infrastructure", "monitoring", "deployment", "reliability"),
        answer_sections=("Infrastructure Overview", "CI/CD Pipeline", "Monitoring", "Deployment", "Reliability"),
        terminology="SRE/DevOps, infrastructure-as-code, observability",
        depth="operations and infrastructure focused",
        system_prompt=(
            "You are advising a DevOps Engineer. Focus on CI/CD, infrastructure, deployment, "
            "monitoring, observability, and reliability engineering."
        ),
    ),
    "Security Engineer": RoleConfig(
        id="Security Engineer",
        label="Security Engineer",
        retrieval_priorities=("threats", "vulnerabilities", "compliance", "encryption", "access control"),
        answer_sections=("Threat Assessment", "Security Controls", "Compliance", "Risk Mitigation", "Recommendations"),
        terminology="security-first, threat modeling, zero trust",
        depth="security analysis with threat and control mapping",
        system_prompt=(
            "You are advising a Security Engineer. Emphasize threat modeling, security controls, "
            "compliance requirements, vulnerability considerations, and risk mitigation."
        ),
    ),
}


def get_role_config(role: str | None) -> RoleConfig:
    if role and role in ROLES:
        return ROLES[role]
    return ROLES["Enterprise Architect"]


def list_roles() -> list[dict[str, Any]]:
    return [
        {
            "id": cfg.id,
            "label": cfg.label,
            "retrieval_priorities": list(cfg.retrieval_priorities),
            "answer_sections": list(cfg.answer_sections),
        }
        for cfg in ROLES.values()
    ]
