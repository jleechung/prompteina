# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.

# Community models package containing external model implementations.
#
# Submodules are NOT imported here. openfold and colabdesign pull in large,
# optional dependency stacks (torch, jax, haiku), and importing this package
# must not require any of them. Import what you need directly.
