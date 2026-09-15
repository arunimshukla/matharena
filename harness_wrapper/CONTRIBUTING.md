# Contributing

## Development checks

Run the formatter, static checks, and test suite from this directory:

```bash
.venv/bin/ruff format .
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest -q
```

Keep changes focused, preserve explicit credential handling, and add tests for
new behavior. API keys, OAuth tokens, request headers, and process environments
must not be written to logs or traces.

## Adding a sandbox

New execution environments subclass `Sandbox` and normally implement:

- `enabled`: return `True` when the environment provides OS-level isolation.
- `prepare_command()`: translate the command, working directory, and explicit
  environment into the host process that launches the sandbox.
- `expose_host_service()`: make one invocation-local host proxy reachable from
  inside the sandbox and yield a `HostServiceRoute`.

The host-service route is part of the sandbox security contract. Built-in agent
adapters always send model traffic through a host-side proxy so requests can be
captured and protected headers or credentials can be injected. Adapters must not
special-case a concrete sandbox type or silently bypass capture.

`network_enabled=False` means that sandboxed code has no direct external network
route. It does not prohibit a narrowly scoped connection to a registered host
service. For example, `PodmanSandbox` creates a temporary internal bridge that
can reach only the protected host proxy; the proxy performs the external model
request and the bridge is removed afterward.

An isolated sandbox must override `expose_host_service()`. Its implementation
must:

1. Create the transport before yielding the route.
2. Set `listen_host` to an address on which the host proxy can listen.
3. Set `client_host` to the corresponding address visible inside the sandbox.
4. Set `allow_remote_clients=True` only when the sandbox cannot appear as a
   loopback peer. The proxy's random URL capability or one-time API key remains
   mandatory authentication.
5. When a direct host listener is impossible, provide both `unix_socket` and
   `client_port`: the proxy listens on that Unix socket while the sandbox-owned
   transport exposes it at `client_host:client_port`.
6. Tear down temporary networks, forwards, sockets, and other resources in a
   `finally` block.
7. Keep secrets out of command arguments and exception messages.

A minimal non-isolating implementation inherits the default loopback route:

```python
from harness_wrapper import Sandbox


class LocalEnvironment(Sandbox):
    pass
```

An isolated implementation supplies its own scoped transport:

```python
from collections.abc import Iterator
from contextlib import contextmanager

from harness_wrapper import HostServiceRoute, Sandbox


class IsolatedEnvironment(Sandbox):
    @property
    def enabled(self) -> bool:
        return True

    @contextmanager
    def expose_host_service(self) -> Iterator[HostServiceRoute]:
        transport = self._create_temporary_transport()
        try:
            yield HostServiceRoute(
                listen_host=transport.host_address,
                client_host=transport.sandbox_address,
                allow_remote_clients=True,
            )
        finally:
            transport.close()
```

The transport may be an internal bridge, VM port forward, Unix-socket relay, or
another mechanism appropriate to the sandbox. If no safe route exists, raise a
clear error rather than enabling broad host networking.

`PodmanSandbox` illustrates both route forms. Rootful Podman lets the proxy bind
directly to the internal bridge. Rootless Podman keeps that bridge in a private
network namespace, so a small sandbox-owned relay exposes a TCP port there and
forwards it to the proxy's protected Unix socket. This relay runs on the host;
the container image does not need Python or any other helper binary.

Tests for a sandbox should verify command translation, secret handling, route
visibility, disabled-network behavior, cleanup after success and exceptions,
and rejection of concurrent route activation when applicable.
