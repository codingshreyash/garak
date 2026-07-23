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

Spoken tool-risk probes
-----------------------

``ToolRiskPETTS`` and its semantic variants send fixed, non-executing
tool-action requests through the audio channel. They score target-provided text
for actionable content; garak never executes a returned command or tool call.
``NativeToolRiskPETTS`` remains unscored unless the user has separately
validated that the target can emit native tool calls.
