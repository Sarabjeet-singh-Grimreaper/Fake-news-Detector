# Ponytail AI Ruleset (Enforce YAGNI & Minimalism)

Before writing or modifying any code, you must evaluate the request against this exact ladder:
1. Does this feature actually need to exist? If not, skip it.
2. Can the problem be solved by deleting code or simplifying existing code?
3. Can the programming language's Standard Library solve it natively?
4. Is there a native web platform, browser API, or framework feature that handles it?
5. Can an already installed dependency solve it?
6. Can it be written cleanly in one or two lines?

Never compromise on security, accessibility, data validation, or error handling, but aggressively reject unnecessary architectural layers, new dependencies, or bloated wrapper components.
