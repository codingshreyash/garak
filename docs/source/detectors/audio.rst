garak.detectors.audio
=====================

.. automodule:: garak.detectors.audio
   :members:
   :undoc-members:
   :show-inheritance:

The deterministic tool-risk detector is deliberately conservative and cannot
resolve every descriptive-versus-actionable response. The optional judge
handles that semantic distinction. Judge failures and malformed verdicts are
reported as unscored, not as successful attacks.
