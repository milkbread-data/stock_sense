# gemini-cli rule file: Python Performance and Security Guidelines

rules:
  - id: performance_01
    category: performance
    description: "Optimize loops and comprehensions"
    directives:
      - "Prefer list comprehensions or generator expressions over explicit loops when appropriate."
      - "Avoid unnecessary nested loops; consider algorithmic improvements."
      - "Use built-in functions and libraries for optimized performance."
  
  - id: performance_02
    category: performance
    description: "Efficient memory management"
    directives:
      - "Use generators to handle large datasets instead of loading entire lists in memory."
      - "Avoid global variables to reduce memory footprint and improve function purity."
      - "Release resources promptly using context managers (with statements)."

  - id: performance_03
    category: performance
    description: "Concurrency and parallelism"
    directives:
      - "Use asyncio for IO-bound concurrency."
      - "Use multiprocessing for CPU-bound parallelism."
      - "Avoid blocking calls in asynchronous code."

  - id: security_01
    category: security
    description: "Input validation and sanitization"
    directives:
      - "Validate all external inputs rigorously."
      - "Sanitize inputs to prevent injection attacks."
      - "Never trust user-provided data without validation."

  - id: security_02
    category: security
    description: "Avoid dangerous functions"
    directives:
      - "Avoid use of eval(), exec(), or other dynamic code execution functions."
      - "Do not use pickle for untrusted data deserialization."
      - "Use safe alternatives for serialization like JSON."

  - id: security_03
    category: security
    description: "Secure database access"
    directives:
      - "Use parameterized queries or ORM methods to prevent SQL injection."
      - "Avoid constructing SQL queries by string concatenation."

  - id: security_04
    category: security
    description: "Cryptography best practices"
    directives:
      - "Use Python’s built-in cryptography libraries (e.g., hashlib, secrets)."
      - "Avoid implementing custom cryptographic algorithms."
      - "Store sensitive data securely, avoid hardcoding secrets."

  - id: python_best_practices
    category: python-specific
    description: "Python idioms and maintainability"
    directives:
      - "Use type hints for function signatures."
      - "Write modular, reusable functions."
      - "Use logging instead of print statements for diagnostics."
      - "Write tests for critical functions."