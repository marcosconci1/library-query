# Security policy

library-query is a local tool: it reads your library folder, stores
vectors in a local Qdrant instance, and talks to one network endpoint
that you configure for embeddings. There is no telemetry and no hosted
component.

## Reporting a vulnerability

Open a private security advisory through
[GitHub security advisories](https://github.com/marcosconci1/library-query/security/advisories/new)
rather than a public issue. Include the version, the command you ran,
and what you expected to stay local.

The maintainer treats the following as security issues:

* Anything that sends file contents, file names, or library structure
  anywhere other than the configured embeddings endpoint.
* Anything that writes to the library folder outside the two opt-in
  commands.
* Anything that executes code from library file content.

A report that the tool returns an imprecise search result is a normal
issue, not a security report.
