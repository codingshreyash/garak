garak.probes.audio
==================

.. automodule:: garak.probes.audio
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::

Reliability precursors
----------------------

``TTSReliability`` checks whether short benign strings survive synthesis and
the target's audio-input path. ``SemanticReliability`` checks whether the same
path preserves simple spoken requests. Both are calibration probes: run them
before interpreting results from adversarial audio probes, and treat missing
target text as unscorable rather than as a safety pass.
