# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""Bridges to other libraries. Each submodule needs its own optional extra
and is imported explicitly (``spectral_trust`` itself imports none of them):

- :mod:`spectral_trust.integrations.lm_polygraph`: spectral-trust metrics as
  LM-Polygraph uncertainty estimators (``pip install spectral_trust[polygraph]``).
"""
