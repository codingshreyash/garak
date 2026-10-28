garak.probes.audio_injection
============================

.. automodule:: garak.probes.audio_injection
   :members:
   :undoc-members:
   :show-inheritance:

``Suffix`` renders a benign carrier and an intent through the configured PETTS
helper, inserts a short silence, and sends the resulting WAV to the target. The
probe exercises audio concatenation directly; it does not add unused transform
infrastructure.
