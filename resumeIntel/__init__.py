"""Resume Intelligence — JD-specific resume gap analysis (Product A, spec part 11).

Given an uploaded resume (or the default profile resume) and a pasted job
description, returns matched/missing skills, weak vs. strong evidence, and
truthful, non-fabricated resume improvement suggestions. Self-contained
module: no other package imports from here, and this package does not
import from jobEngine/jobApplyPro to avoid coupling.
"""
